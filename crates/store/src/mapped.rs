//! Read-only, memory-mapped segments.
//!
//! A [`Segment`] maps a segment file and implements
//! [`GraphView`] over it. Search then runs
//! the *same* traversal code as the in-memory index — `graph::search_with_scratch`
//! is generic over the view — so a recall figure measured while building applies
//! to the served index, and there is no second implementation of the search loop
//! to keep in sync.
//!
//! # What is and is not validated on open
//!
//! `open` always validates structure: the header and footer checksums, every
//! block's bounds, and the graph invariants a traversal relies on to be safe
//! (CSR offsets monotone and in range, every neighbour index in range, no links
//! above a node's own top level). Those checks are what make it impossible for a
//! corrupt link to be dereferenced silently.
//!
//! Checksums over the bulk blocks are *not* verified on open: hashing 256 MB of
//! vectors (plus the adjacency) would cost more than mapping them, and the
//! structural checks already prevent unsafe reads. [`Segment::verify`] runs the
//! full checksum pass on demand, which is what `lodestar verify` and the
//! corruption tests use.

use std::path::{Path, PathBuf};

use bytemuck::try_cast_slice;
use lodestar_ann_core::{Candidate, Metric};
use lodestar_ann_index::graph::{GraphView, SearchScratch, search_with_scratch};
use lodestar_ann_index::hnsw::HnswConfig;
use memmap2::{Mmap, MmapOptions};

use crate::error::{Error, Result};
use crate::segment::{
    DIRECTORY_ENTRIES, DIRECTORY_ENTRY_LEN, DirectoryEntry, FOOTER_LEN, Footer, HEADER_LEN, Header,
    SegmentInfo, crc32,
};

/// A mapped, immutable HNSW segment.
#[derive(Debug)]
pub struct Segment {
    path: PathBuf,
    map: Mmap,
    header: Header,
    footer: Footer,
    directory: Vec<DirectoryEntry>,
}

impl Segment {
    /// Maps and structurally validates a segment file.
    ///
    /// # Errors
    ///
    /// * [`Error::Io`] if the file cannot be opened or mapped.
    /// * [`Error::Corrupt`] if structural validation fails.
    /// * [`Error::UnsupportedVersion`] if the format version is newer.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let len = file
            .metadata()
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?
            .len();
        if len < (HEADER_LEN + FOOTER_LEN) as u64 {
            return Err(Error::corrupt(
                "segment",
                format!("{len} bytes is shorter than the header and footer"),
            ));
        }
        // SAFETY: the mapping is read-only and the file handle is not exposed,
        // so nothing in this process can write through it. The structural
        // validation below refuses to hand out any slice that is not inside the
        // mapping, which is what keeps unsafe reads out of the traversal.
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;

        let header = Header::decode(&map[..HEADER_LEN])?;
        let footer = Footer::decode(&map[map.len() - FOOTER_LEN..], map.len() as u64)?;
        if footer.nodes != header.nodes || footer.live != header.live {
            return Err(Error::corrupt(
                "segment",
                format!(
                    "header and footer disagree: {} / {} nodes",
                    header.nodes, footer.nodes
                ),
            ));
        }
        if footer.header_crc != crc32(&map[..HEADER_LEN - 4]) {
            return Err(Error::corrupt(
                "segment",
                "footer does not describe this header",
            ));
        }
        let reserved = &map[112..HEADER_LEN - 4];
        if reserved.iter().any(|byte| *byte != 0) {
            return Err(Error::corrupt(
                "segment header",
                "reserved bytes are not zero, which means a field this build does not know",
            ));
        }

        let directory = Self::read_directory(&map, &header)?;
        let segment = Self {
            path,
            map,
            header,
            footer,
            directory,
        };
        segment.validate_graph()?;
        Ok(segment)
    }

    /// Decodes and bounds-checks the level directory.
    fn read_directory(map: &[u8], header: &Header) -> Result<Vec<DirectoryEntry>> {
        let start = usize::try_from(header.directory_at).map_err(|_| {
            Error::corrupt(
                "segment",
                format!("directory offset {}", header.directory_at),
            )
        })?;
        let end = start
            .checked_add(DIRECTORY_ENTRIES * DIRECTORY_ENTRY_LEN)
            .ok_or_else(|| Error::corrupt("segment", "directory offset overflows"))?;
        let bytes = map
            .get(start..end)
            .ok_or_else(|| Error::corrupt("segment", "directory block is out of bounds"))?;
        let mut directory = Vec::with_capacity(DIRECTORY_ENTRIES);
        for index in 0..DIRECTORY_ENTRIES {
            let entry = &bytes[index * DIRECTORY_ENTRY_LEN..(index + 1) * DIRECTORY_ENTRY_LEN];
            directory.push(DirectoryEntry::decode(entry)?);
        }
        let levels = header.max_level as usize + 1;
        if header.nodes > 0 {
            // Every level up to the graph's top must have a directory entry,
            // and levels above it must be empty.
            for (level, entry) in directory.iter().enumerate() {
                if level < levels {
                    continue;
                }
                if entry.links != 0 || entry.offsets_at != 0 || entry.links_at != 0 {
                    return Err(Error::corrupt(
                        "segment directory",
                        format!("level {level} has entries above the graph's top level"),
                    ));
                }
            }
        }
        Ok(directory)
    }

    /// Checks the graph invariants a traversal depends on.
    fn validate_graph(&self) -> Result<()> {
        let nodes = self.header.nodes as usize;
        let dim = self.header.dim as usize;
        let len = self.map.len() as u64;
        let end_of = |offset: u64, bytes: u64, what: &str| -> Result<()> {
            let end = offset
                .checked_add(bytes)
                .ok_or_else(|| Error::corrupt("segment", format!("{what} offset overflows")))?;
            if end > len {
                return Err(Error::corrupt(
                    "segment",
                    format!("{what} ends at {end}, past the file's {len} bytes"),
                ));
            }
            Ok(())
        };
        end_of(self.header.levels_at, nodes as u64, "level block")?;
        end_of(
            self.header.vectors_at,
            self.header.vector_bytes(),
            "vector block",
        )?;
        end_of(self.header.ids_at, nodes as u64 * 8, "id block")?;
        end_of(self.header.live_at, nodes as u64, "liveness block")?;
        if self.header.vectors_at % 16 != 0
            || self.header.ids_at % 4 != 0
            || self.header.live_at % 4 != 0
            || self.header.levels_at % 4 != 0
        {
            return Err(Error::corrupt(
                "segment",
                "a block starts at an offset its element type cannot be read from",
            ));
        }
        if dim == 0 {
            return Err(Error::corrupt("segment", "zero dimensionality"));
        }

        let levels = self.node_levels()?;
        for (node, top) in levels.iter().enumerate() {
            if usize::from(*top) > self.header.max_level as usize {
                return Err(Error::corrupt(
                    "segment",
                    format!(
                        "node {node}'s level {top} exceeds the graph's top level {}",
                        self.header.max_level
                    ),
                ));
            }
        }
        let mut live_count = 0u64;
        for flag in self.live_bytes()? {
            match flag {
                0 => {}
                1 => live_count += 1,
                other => {
                    return Err(Error::corrupt(
                        "segment",
                        format!("liveness flag {other} is neither 0 nor 1"),
                    ));
                }
            }
        }
        if live_count != self.header.live {
            return Err(Error::corrupt(
                "segment",
                format!(
                    "{} live flags for {} live nodes",
                    live_count, self.header.live
                ),
            ));
        }

        let mut links_seen = 0u64;
        for level in 0..=self.header.max_level as usize {
            let entry = self.directory[level];
            // A corrupt directory entry must be rejected before the offsets it
            // names are used to index the link list below.
            end_of(entry.offsets_at, (nodes as u64 + 1) * 4, "offsets block")?;
            // The link count is read straight from the directory, which unlike
            // the header has no checksum at `open` time — so the byte count it
            // implies has to be computed without trusting it not to overflow.
            let links_bytes = entry
                .links
                .checked_mul(4)
                .ok_or_else(|| Error::corrupt("segment adjacency", "links block overflows"))?;
            end_of(entry.links_at, links_bytes, "links block")?;
            let offsets = self.offsets(level)?;
            let links = self.links(level)?;
            if offsets.len() != nodes + 1 {
                return Err(Error::corrupt(
                    "segment adjacency",
                    format!(
                        "level {level} has {} offsets for {nodes} nodes",
                        offsets.len()
                    ),
                ));
            }
            if offsets[0] != 0 {
                return Err(Error::corrupt(
                    "segment adjacency",
                    format!("level {level} offsets do not start at zero"),
                ));
            }
            // The offsets index into the link list, so the two views of the
            // link count have to agree before any neighbour is read.
            if offsets[nodes] as u64 != entry.links {
                return Err(Error::corrupt(
                    "segment adjacency",
                    format!("level {level} offsets and directory disagree on the link count"),
                ));
            }
            for node in 0..nodes {
                let (start, end) = (offsets[node], offsets[node + 1]);
                if start > end || end > offsets[nodes] {
                    return Err(Error::corrupt(
                        "segment adjacency",
                        format!("level {level} offsets are not monotone at node {node}"),
                    ));
                }
                if level > usize::from(levels[node]) && start != end {
                    return Err(Error::corrupt(
                        "segment adjacency",
                        format!("node {node} has links at level {level}, above its top level"),
                    ));
                }
                for &neighbour in &links[start as usize..end as usize] {
                    if neighbour as usize >= nodes {
                        return Err(Error::corrupt(
                            "segment adjacency",
                            format!(
                                "level {level} node {node} links to {neighbour}, outside {nodes} nodes"
                            ),
                        ));
                    }
                    if usize::from(levels[neighbour as usize]) < level {
                        return Err(Error::corrupt(
                            "segment adjacency",
                            format!(
                                "level {level} node {node} links to {neighbour}, which does not reach level {level}"
                            ),
                        ));
                    }
                }
            }
            links_seen += u64::from(offsets[nodes]);
        }
        if links_seen != self.footer.links {
            return Err(Error::corrupt(
                "segment",
                format!(
                    "{links_seen} links walked, footer says {}",
                    self.footer.links
                ),
            ));
        }
        if nodes == 0 {
            if self.header.entry != u32::MAX {
                return Err(Error::corrupt("segment", "empty graph with an entry point"));
            }
            return Ok(());
        }
        let entry = self.header.entry as usize;
        if usize::from(levels[entry]) != self.header.max_level as usize {
            return Err(Error::corrupt(
                "segment",
                format!(
                    "entry point {entry} reaches level {}, graph tops out at {}",
                    levels[entry], self.header.max_level
                ),
            ));
        }
        Ok(())
    }

    /// Runs the full checksum pass over every block.
    ///
    /// This is the expensive, on-demand companion to the structural validation
    /// that `open` always performs. It detects corruption that keeps the graph's
    /// shape, in particular a flipped bit inside a neighbour index.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] naming the block whose checksum disagrees.
    pub fn verify(&self) -> Result<()> {
        let vectors = self.vector_bytes()?;
        if crc32(bytemuck::cast_slice::<f32, u8>(vectors)) != self.footer.vectors_crc {
            return Err(Error::corrupt("segment vector block", "checksum mismatch"));
        }
        let mut chunks: Vec<&[u8]> = Vec::with_capacity(8);
        chunks.push(self.node_levels()?);
        chunks.push(bytemuck::cast_slice::<u64, u8>(self.id_bytes()?));
        chunks.push(self.live_bytes()?);
        for level in 0..=self.header.max_level as usize {
            chunks.push(bytemuck::cast_slice::<u32, u8>(self.offsets(level)?));
            chunks.push(bytemuck::cast_slice::<u32, u8>(self.links(level)?));
        }
        // The directory is hashed in its on-disk encoding, which is what the
        // writer hashed.
        let start = self.header.directory_at as usize;
        let end = start + DIRECTORY_ENTRIES * DIRECTORY_ENTRY_LEN;
        chunks.insert(
            3,
            self.map
                .get(start..end)
                .ok_or_else(|| Error::corrupt("segment directory", "out of bounds"))?,
        );
        let mut hasher = crc32fast::Hasher::new();
        for chunk in chunks {
            hasher.update(chunk);
        }
        if hasher.finalize() != self.footer.graph_crc {
            return Err(Error::corrupt("segment graph blocks", "checksum mismatch"));
        }
        Ok(())
    }

    /// Path this segment was mapped from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Dimensionality of the indexed vectors.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.header.dim as usize
    }

    /// Ranking metric.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.header.metric
    }

    /// Number of nodes, tombstones included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.header.nodes as usize
    }

    /// Whether the segment holds no nodes at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.header.nodes == 0
    }

    /// Number of live nodes.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.header.live as usize
    }

    /// Graph parameters the segment was built with.
    #[must_use]
    pub fn config(&self) -> HnswConfig {
        HnswConfig {
            m: self.header.m as usize,
            m0: self.header.m0 as usize,
            ef_construction: self.header.ef_construction as usize,
            ef_search: HnswConfig::default().ef_search,
            seed: self.header.seed,
        }
    }

    /// Bytes held by the mapping, which is the file size rounded to pages.
    #[must_use]
    pub fn mapped_bytes(&self) -> usize {
        self.map.len()
    }

    /// Everything a caller needs to report about a segment.
    #[must_use]
    pub fn info(&self) -> SegmentInfo {
        SegmentInfo {
            path: self.path.clone(),
            dim: self.dim(),
            metric: self.metric(),
            nodes: self.len(),
            live: self.live_count(),
            max_level: self.header.max_level as usize,
            links: self.footer.links as usize,
            bytes: self.map.len() as u64,
            created_unix_ms: self.header.created_unix_ms,
        }
    }

    /// Zero-copy view of the vector block.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] if the block's alignment or length is wrong, which
    /// `open` has already rejected.
    pub fn vector_bytes(&self) -> Result<&[f32]> {
        let start = self.header.vectors_at as usize;
        let bytes = self.header.vector_bytes() as usize;
        let slice = self
            .map
            .get(
                start..start.checked_add(bytes).ok_or_else(|| {
                    Error::corrupt(
                        "segment vector block",
                        "vector block overflows the address space",
                    )
                })?,
            )
            .ok_or_else(|| Error::corrupt("segment vector block", "out of bounds"))?;
        try_cast_slice(slice).map_err(|_| {
            Error::corrupt(
                "segment vector block",
                "vector block is not aligned and sized for f32",
            )
        })
    }

    /// The stored vector for `node`.
    ///
    /// # Panics
    ///
    /// Panics if `node` is out of range, which callers inside this crate never
    /// do because indices come from the same validated file.
    #[must_use]
    pub fn vector(&self, node: u32) -> &[f32] {
        let dim = self.dim();
        let start = node as usize * dim;
        &self.vector_bytes().expect("validated at open")[start..start + dim]
    }

    /// Zero-copy view of the id block.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] if the block is misaligned, which `open` rejects.
    pub fn id_bytes(&self) -> Result<&[u64]> {
        let start = self.header.ids_at as usize;
        let bytes = self.len() * 8;
        let slice = self
            .map
            .get(
                start..start.checked_add(bytes).ok_or_else(|| {
                    Error::corrupt("segment id block", "id block overflows the address space")
                })?,
            )
            .ok_or_else(|| Error::corrupt("segment id block", "out of bounds"))?;
        try_cast_slice(slice).map_err(|_| {
            Error::corrupt(
                "segment id block",
                "id block is not aligned and sized for u64",
            )
        })
    }

    /// Zero-copy view of the liveness block.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] if the block is out of bounds.
    pub fn live_bytes(&self) -> Result<&[u8]> {
        let start = self.header.live_at as usize;
        self.map
            .get(start..start + self.len())
            .ok_or_else(|| Error::corrupt("segment liveness block", "out of bounds"))
    }

    /// Node levels, one byte per node.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] if the block is out of bounds.
    pub fn node_levels(&self) -> Result<&[u8]> {
        let start = self.header.levels_at as usize;
        self.map
            .get(start..start + self.len())
            .ok_or_else(|| Error::corrupt("segment level block", "out of bounds"))
    }

    /// CSR offsets for `level`.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] if the level has no directory entry or the block is
    /// out of bounds.
    pub fn offsets(&self, level: usize) -> Result<&[u32]> {
        let entry = self.directory.get(level).ok_or_else(|| {
            Error::corrupt("segment directory", format!("no entry for level {level}"))
        })?;
        let start = entry.offsets_at as usize;
        let bytes = (self.len() + 1) * 4;
        let slice = self
            .map
            .get(
                start..start.checked_add(bytes).ok_or_else(|| {
                    Error::corrupt(
                        "segment adjacency",
                        "offsets block overflows the address space",
                    )
                })?,
            )
            .ok_or_else(|| Error::corrupt("segment adjacency", "offsets block is out of bounds"))?;
        try_cast_slice(slice).map_err(|_| {
            Error::corrupt(
                "segment adjacency",
                format!("level {level} offsets are not aligned for u32"),
            )
        })
    }

    /// Neighbour list for `level`.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] if the block is out of bounds or misaligned.
    pub fn links(&self, level: usize) -> Result<&[u32]> {
        let entry = self.directory.get(level).ok_or_else(|| {
            Error::corrupt("segment directory", format!("no entry for level {level}"))
        })?;
        let start = entry.links_at as usize;
        let bytes = usize::try_from(
            entry
                .links
                .checked_mul(4)
                .ok_or_else(|| Error::corrupt("segment adjacency", "links block overflows"))?,
        )
        .map_err(|_| Error::corrupt("segment adjacency", "links block overflows"))?;
        let slice = self
            .map
            .get(
                start..start.checked_add(bytes).ok_or_else(|| {
                    Error::corrupt(
                        "segment adjacency",
                        "links block overflows the address space",
                    )
                })?,
            )
            .ok_or_else(|| Error::corrupt("segment adjacency", "links block is out of bounds"))?;
        try_cast_slice(slice).map_err(|_| {
            Error::corrupt(
                "segment adjacency",
                format!("level {level} links are not aligned for u32"),
            )
        })
    }

    /// Searches the segment.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`](lodestar_ann_core::Error::DimensionMismatch)
    /// if the query does not match the segment's dimensionality.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> lodestar_ann_core::Result<Vec<Candidate>> {
        if query.len() != self.dim() {
            return Err(lodestar_ann_core::Error::DimensionMismatch {
                expected: self.dim(),
                actual: query.len(),
            });
        }
        let mut prepared = query.to_vec();
        self.metric().prepare(&mut prepared);
        let mut scratch = SearchScratch::new(self.len());
        Ok(search_with_scratch(
            self,
            &prepared,
            k,
            ef,
            None,
            &mut scratch,
        ))
    }
}

impl GraphView for Segment {
    fn len(&self) -> usize {
        self.header.nodes as usize
    }

    fn dim(&self) -> usize {
        self.header.dim as usize
    }

    fn metric(&self) -> Metric {
        self.header.metric
    }

    fn entry_point(&self) -> Option<u32> {
        if self.header.entry == u32::MAX {
            None
        } else {
            Some(self.header.entry)
        }
    }

    fn max_level(&self) -> usize {
        self.header.max_level as usize
    }

    fn node_level(&self, node: u32) -> usize {
        self.node_levels()
            .ok()
            .and_then(|levels| levels.get(node as usize).copied())
            .map_or(0, usize::from)
    }

    fn neighbors(&self, node: u32, level: usize) -> &[u32] {
        if level > self.max_level() {
            return &[];
        }
        let Ok(offsets) = self.offsets(level) else {
            return &[];
        };
        let Some((&start, &end)) = offsets
            .get(node as usize)
            .zip(offsets.get(node as usize + 1))
        else {
            return &[];
        };
        let Ok(links) = self.links(level) else {
            return &[];
        };
        links.get(start as usize..end as usize).unwrap_or(&[])
    }

    fn vector(&self, node: u32) -> &[f32] {
        Segment::vector(self, node)
    }

    fn is_live(&self, node: u32) -> bool {
        self.live_bytes()
            .ok()
            .and_then(|flags| flags.get(node as usize).copied())
            .is_some_and(|flag| flag == 1)
    }

    fn external_id(&self, node: u32) -> u64 {
        self.id_bytes()
            .ok()
            .and_then(|ids| ids.get(node as usize).copied())
            .unwrap_or(u64::MAX)
    }
}

//! The immutable segment format.
//!
//! A segment is a self-describing file holding one finished HNSW graph. It is
//! written once, never modified, and opened by mapping it read-only (see
//! [`crate::mapped`]), so the vector block is served straight out of the page
//! cache with no copy and no deserialisation. That is the reason this project
//! has a binary format at all: reopening a million-vector index must cost a few
//! milliseconds, not a graph rebuild.
//!
//! # Layout
//!
//! ```text
//! offset 0      header      512 bytes, fixed field positions, self-crc
//!               levels      one byte per node: that node's top level
//!               vectors     nodes * dim * 4 bytes, each block start 16-aligned
//!               ids         nodes * 8 bytes
//!               live        one byte per node: 1 live, 0 tombstoned
//!               directory   33 entries of 32 bytes, one per level
//!               adjacency   per level: (nodes + 1) u32 offsets, then u32 links
//! last 64 bytes footer      counts, vector checksum, file length, self-crc
//! ```
//!
//! Every integer is little-endian. Blocks are padded to a 16-byte boundary so
//! that `f32` and `u32` reads are aligned — the traversal casts the mapped bytes
//! to slices and would otherwise have to copy them.
//!
//! # Checksums
//!
//! The header, the footer and the adjacency blocks are checksummed and verified
//! on every open: they are small, and a corrupt graph is much worse than a slow
//! start. The vector block is checksummed too, but the check is *not* run on
//! open: reading 256 MB of vectors at start-up would cost more than the mapping
//! it validates. [`SegmentWriter`] writes the checksum into the footer and
//! [`crate::mapped::Segment::verify`] runs it when asked, which is what the CLI's
//! `lodestar verify` command does.
//!
//! # Writing
//!
//! Writing goes through a temporary file in the same directory followed by a
//! rename, so a crash mid-write leaves either the old state or a stray temporary
//! file that the next open removes. The file becomes visible only when it is
//! complete and flushed, which is what makes the manifest's atomic rename a
//! meaningful commit point.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bytemuck::cast_slice;
use crc32fast::Hasher;
use lodestar_ann_core::Metric;
use lodestar_ann_index::hnsw::HnswParts;

use crate::error::{Error, Result};

/// File magic for a segment: `LODSEG01`.
pub const MAGIC: [u8; 8] = *b"LODSEG01";
/// File magic for the footer: `LODEND01`.
pub const FOOTER_MAGIC: [u8; 8] = *b"LODEND01";
/// Format version this build writes and the only one it reads.
pub const FORMAT_VERSION: u32 = 1;
/// Size of the fixed header block.
pub const HEADER_LEN: usize = 512;
/// Size of the trailing footer.
pub const FOOTER_LEN: usize = 64;
/// Size of one level-directory entry.
pub const DIRECTORY_ENTRY_LEN: usize = 32;
/// Number of directory entries: levels 0 through the index's maximum.
pub const DIRECTORY_ENTRIES: usize = 33;
/// Byte alignment applied to the start of every block.
pub const BLOCK_ALIGNMENT: u64 = 16;

/// Offset of the header's own checksum within the header.
const HEADER_CRC_AT: usize = HEADER_LEN - 4;
/// Offset of the footer's own checksum within the footer.
const FOOTER_CRC_AT: usize = FOOTER_LEN - 4;
/// Sentinel for "no entry point" in the header.
const NO_ENTRY: u32 = u32::MAX;

/// Contents of a segment header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Format version.
    pub version: u32,
    /// Vector dimensionality.
    pub dim: u32,
    /// Ranking metric.
    pub metric: Metric,
    /// HNSW `m` the graph was built with.
    pub m: u32,
    /// HNSW `m0` the graph was built with.
    pub m0: u32,
    /// HNSW `ef_construction` the graph was built with.
    pub ef_construction: u32,
    /// HNSW level-generation seed, so a rebuild is reproducible.
    pub seed: u64,
    /// Number of nodes in the graph.
    pub nodes: u64,
    /// Number of live nodes.
    pub live: u64,
    /// Highest populated level.
    pub max_level: u32,
    /// Entry point node, [`NO_ENTRY`] when the graph is empty.
    pub entry: u32,
    /// Creation time, milliseconds since the Unix epoch.
    pub created_unix_ms: u64,
    /// Offset of the node-level block.
    pub levels_at: u64,
    /// Offset of the vector block.
    pub vectors_at: u64,
    /// Offset of the id block.
    pub ids_at: u64,
    /// Offset of the liveness block.
    pub live_at: u64,
    /// Offset of the adjacency directory.
    pub directory_at: u64,
}

impl Header {
    /// Encodes the header, checksum included.
    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut bytes = [0u8; HEADER_LEN];
        bytes[0..8].copy_from_slice(&MAGIC);
        put_u32(&mut bytes, 8, self.version);
        put_u32(&mut bytes, 12, self.dim);
        put_u32(&mut bytes, 16, metric_code(self.metric));
        put_u32(&mut bytes, 20, self.m);
        put_u32(&mut bytes, 24, self.m0);
        put_u32(&mut bytes, 28, self.ef_construction);
        put_u64(&mut bytes, 32, self.seed);
        put_u64(&mut bytes, 40, self.nodes);
        put_u64(&mut bytes, 48, self.live);
        put_u32(&mut bytes, 56, self.max_level);
        put_u32(&mut bytes, 60, self.entry);
        put_u64(&mut bytes, 64, self.created_unix_ms);
        put_u64(&mut bytes, 72, self.levels_at);
        put_u64(&mut bytes, 80, self.vectors_at);
        put_u64(&mut bytes, 88, self.ids_at);
        put_u64(&mut bytes, 96, self.live_at);
        put_u64(&mut bytes, 104, self.directory_at);
        // Bytes 112..HEADER_CRC_AT are reserved for future fields and must stay
        // zero: readers reject unknown flags rather than guess at their meaning.
        let crc = crc32(&bytes[..HEADER_CRC_AT]);
        put_u32(&mut bytes, HEADER_CRC_AT, crc);
        bytes
    }

    /// Decodes and validates a header.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] for a bad magic, checksum or field, and
    /// [`Error::UnsupportedVersion`] for a format this build cannot read.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(Error::Corrupt {
                what: "segment header".to_string(),
                detail: format!("only {} bytes", bytes.len()),
            });
        }
        if bytes[0..8] != MAGIC {
            return Err(Error::Corrupt {
                what: "segment header".to_string(),
                detail: "magic mismatch".to_string(),
            });
        }
        let stored = get_u32(bytes, HEADER_CRC_AT)?;
        if crc32(&bytes[..HEADER_CRC_AT]) != stored {
            return Err(Error::Corrupt {
                what: "segment header".to_string(),
                detail: "checksum mismatch".to_string(),
            });
        }
        let version = get_u32(bytes, 8)?;
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let dim = get_u32(bytes, 12)?;
        if dim == 0 || dim > 65_536 {
            return Err(Error::Corrupt {
                what: "segment header".to_string(),
                detail: format!("implausible dimensionality {dim}"),
            });
        }
        let metric = metric_from_code(get_u32(bytes, 16)?)?;
        let m = get_u32(bytes, 20)?;
        let m0 = get_u32(bytes, 24)?;
        let ef_construction = get_u32(bytes, 28)?;
        if m > 256 || m0 > 1024 || ef_construction == 0 {
            return Err(Error::Corrupt {
                what: "segment header".to_string(),
                detail: format!("implausible graph parameters m={m} m0={m0} ef={ef_construction}"),
            });
        }
        let header = Self {
            version,
            dim,
            metric,
            m,
            m0,
            ef_construction,
            seed: get_u64(bytes, 32)?,
            nodes: get_u64(bytes, 40)?,
            live: get_u64(bytes, 48)?,
            max_level: get_u32(bytes, 56)?,
            entry: get_u32(bytes, 60)?,
            created_unix_ms: get_u64(bytes, 64)?,
            levels_at: get_u64(bytes, 72)?,
            vectors_at: get_u64(bytes, 80)?,
            ids_at: get_u64(bytes, 88)?,
            live_at: get_u64(bytes, 96)?,
            directory_at: get_u64(bytes, 104)?,
        };
        if header.max_level as usize >= DIRECTORY_ENTRIES {
            return Err(Error::Corrupt {
                what: "segment header".to_string(),
                detail: format!("max level {} out of range", header.max_level),
            });
        }
        if header.live > header.nodes {
            return Err(Error::Corrupt {
                what: "segment header".to_string(),
                detail: format!("{} live of {} nodes", header.live, header.nodes),
            });
        }
        if header.nodes == 0 {
            if header.entry != NO_ENTRY {
                return Err(Error::Corrupt {
                    what: "segment header".to_string(),
                    detail: "empty segment with an entry point".to_string(),
                });
            }
        } else if u64::from(header.entry) >= header.nodes {
            return Err(Error::Corrupt {
                what: "segment header".to_string(),
                detail: format!("entry {} outside {} nodes", header.entry, header.nodes),
            });
        }
        Ok(header)
    }

    /// Number of bytes the vector block occupies.
    #[must_use]
    pub const fn vector_bytes(&self) -> u64 {
        self.nodes * self.dim as u64 * 4
    }
}

/// One level's slice of the adjacency block: offsets and links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryEntry {
    /// Byte offset of the `(nodes + 1)` `u32` offsets.
    pub offsets_at: u64,
    /// Byte offset of the `u32` neighbour list.
    pub links_at: u64,
    /// Number of neighbour entries in the list.
    pub links: u64,
}

impl DirectoryEntry {
    fn encode(&self) -> [u8; DIRECTORY_ENTRY_LEN] {
        let mut bytes = [0u8; DIRECTORY_ENTRY_LEN];
        put_u64(&mut bytes, 0, self.offsets_at);
        put_u64(&mut bytes, 8, self.links_at);
        put_u64(&mut bytes, 16, self.links);
        bytes
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            offsets_at: get_u64(bytes, 0)?,
            links_at: get_u64(bytes, 8)?,
            links: get_u64(bytes, 16)?,
        })
    }
}

/// Contents of a segment footer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footer {
    /// Number of nodes, repeated so that a truncated file is detectable.
    pub nodes: u64,
    /// Number of live nodes.
    pub live: u64,
    /// Total neighbour entries across all levels.
    pub links: u64,
    /// Checksum of the vector block.
    pub vectors_crc: u32,
    /// Checksum of every graph block: levels, ids, liveness, directory and
    /// adjacency. Structural validation catches a graph that is *shaped*
    /// wrongly, but only a checksum catches a link whose value was flipped while
    /// keeping its shape, and that kind of corruption degrades recall silently.
    pub graph_crc: u32,
    /// Checksum of the header, repeated for the same reason.
    pub header_crc: u32,
    /// File length in bytes.
    pub file_len: u64,
}

impl Footer {
    fn encode(&self) -> [u8; FOOTER_LEN] {
        let mut bytes = [0u8; FOOTER_LEN];
        bytes[0..8].copy_from_slice(&FOOTER_MAGIC);
        put_u64(&mut bytes, 8, self.nodes);
        put_u64(&mut bytes, 16, self.live);
        put_u64(&mut bytes, 24, self.links);
        put_u32(&mut bytes, 32, self.vectors_crc);
        put_u32(&mut bytes, 36, self.graph_crc);
        put_u32(&mut bytes, 40, self.header_crc);
        put_u64(&mut bytes, 44, self.file_len);
        let crc = crc32(&bytes[..FOOTER_CRC_AT]);
        put_u32(&mut bytes, FOOTER_CRC_AT, crc);
        bytes
    }

    pub(crate) fn decode(bytes: &[u8], file_len: u64) -> Result<Self> {
        if bytes.len() < FOOTER_LEN {
            return Err(Error::Corrupt {
                what: "segment footer".to_string(),
                detail: format!("only {} bytes", bytes.len()),
            });
        }
        if bytes[0..8] != FOOTER_MAGIC {
            return Err(Error::Corrupt {
                what: "segment footer".to_string(),
                detail: "magic mismatch, the file is truncated or not a segment".to_string(),
            });
        }
        let stored = get_u32(bytes, FOOTER_CRC_AT)?;
        if crc32(&bytes[..FOOTER_CRC_AT]) != stored {
            return Err(Error::Corrupt {
                what: "segment footer".to_string(),
                detail: "checksum mismatch".to_string(),
            });
        }
        let footer = Self {
            nodes: get_u64(bytes, 8)?,
            live: get_u64(bytes, 16)?,
            links: get_u64(bytes, 24)?,
            vectors_crc: get_u32(bytes, 32)?,
            graph_crc: get_u32(bytes, 36)?,
            header_crc: get_u32(bytes, 40)?,
            file_len: get_u64(bytes, 44)?,
        };
        if footer.file_len != file_len {
            return Err(Error::Corrupt {
                what: "segment footer".to_string(),
                detail: format!("records {} bytes, file has {file_len}", footer.file_len),
            });
        }
        Ok(footer)
    }
}

/// What a written (or opened) segment contains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentInfo {
    /// Path of the segment file.
    pub path: PathBuf,
    /// Dimensionality.
    pub dim: usize,
    /// Ranking metric.
    pub metric: Metric,
    /// Number of nodes.
    pub nodes: usize,
    /// Number of live nodes.
    pub live: usize,
    /// Highest populated level.
    pub max_level: usize,
    /// Total neighbour entries across all levels.
    pub links: usize,
    /// File size in bytes.
    pub bytes: u64,
    /// Creation time, milliseconds since the Unix epoch.
    pub created_unix_ms: u64,
}

/// Writes HNSW parts out as an immutable segment.
///
/// The file is built beside its final path and renamed into place, so a reader
/// never sees a partially written segment.
///
/// # Errors
///
/// [`Error::Io`] for filesystem failures, [`Error::Core`] if the parts are
/// inconsistent (they have already been validated by the index, but the
/// conversion is fallible in principle).
pub fn write_segment(path: &Path, parts: &HnswParts<'_>) -> Result<SegmentInfo> {
    write_segment_at(path, parts, now_unix_ms())
}

/// [`write_segment`] with an explicit timestamp, for reproducible tests.
///
/// # Errors
///
/// As [`write_segment`].
pub fn write_segment_at(
    path: &Path,
    parts: &HnswParts<'_>,
    created_unix_ms: u64,
) -> Result<SegmentInfo> {
    let nodes = parts.ids.len();
    let dim = u32::try_from(parts.dim).map_err(|_| Error::Corrupt {
        what: "segment".to_string(),
        detail: format!("dimensionality {} does not fit in u32", parts.dim),
    })?;
    // Statistics that shape the layout, computed before anything is written.
    let mut level_links = vec![0u64; parts.max_level + 1];
    for (node, lists) in parts.levels.iter().enumerate() {
        let top = usize::from(parts.node_levels[node]);
        for (level, links) in lists.iter().enumerate() {
            if level > top {
                return Err(Error::Corrupt {
                    what: "segment".to_string(),
                    detail: format!(
                        "node {node} has links at level {level}, above its own top level {top}"
                    ),
                });
            }
            if level >= level_links.len() {
                level_links.resize(level + 1, 0);
            }
            level_links[level] += links.len() as u64;
        }
    }
    let directory_entries = level_links.len().max(1);
    if directory_entries > DIRECTORY_ENTRIES {
        return Err(Error::Corrupt {
            what: "segment".to_string(),
            detail: format!("{directory_entries} levels exceeds the format's {DIRECTORY_ENTRIES}"),
        });
    }
    let live = parts.live.iter().filter(|live| **live).count() as u64;
    let total_links: u64 = level_links.iter().sum();

    // Plan the block layout. Blocks are 16-byte aligned so that the mapped
    // bytes can be read in place as f32 / u32 slices.
    let levels_at = HEADER_LEN as u64;
    let vectors_at = align(levels_at + nodes as u64);
    let ids_at = align(vectors_at + nodes as u64 * u64::from(dim) * 4);
    let live_at = align(ids_at + nodes as u64 * 8);
    let directory_at = align(live_at + nodes as u64);
    let mut cursor = align(directory_at + (DIRECTORY_ENTRIES * DIRECTORY_ENTRY_LEN) as u64);
    let mut directory = Vec::with_capacity(directory_entries);
    for &links in &level_links {
        let offsets_at = cursor;
        cursor = align(offsets_at + (nodes as u64 + 1) * 4);
        let links_at = cursor;
        cursor = align(links_at + links * 4);
        directory.push(DirectoryEntry {
            offsets_at,
            links_at,
            links,
        });
    }
    let file_len = cursor + FOOTER_LEN as u64;

    let header = Header {
        version: FORMAT_VERSION,
        dim,
        metric: parts.metric,
        m: u32::try_from(parts.config.m).unwrap_or(u32::MAX),
        m0: u32::try_from(parts.config.m0).unwrap_or(u32::MAX),
        ef_construction: u32::try_from(parts.config.ef_construction).unwrap_or(u32::MAX),
        seed: parts.config.seed,
        nodes: nodes as u64,
        live,
        max_level: parts.max_level as u32,
        entry: parts.entry.unwrap_or(NO_ENTRY),
        created_unix_ms,
        levels_at,
        vectors_at,
        ids_at,
        live_at,
        directory_at,
    };
    let header_bytes = header.encode();
    let header_crc = get_u32(&header_bytes, HEADER_CRC_AT)?;

    // Write to a temporary neighbour, then rename. The temporary name keeps the
    // process id so that two writers cannot collide.
    let directory_of = path.parent().unwrap_or_else(|| Path::new("."));
    let temporary = directory_of.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "segment".to_string()),
        std::process::id()
    ));
    let file = File::create(&temporary).map_err(|source| Error::Io {
        path: temporary.clone(),
        source,
    })?;
    let mut writer = CrcWriter::new(BufWriter::with_capacity(1 << 20, file), temporary.clone());
    // Checksums cover exactly the bytes that land on disk: the graph checksum
    // takes the meaningful ranges of each graph block (padding excluded), which
    // is the same set of ranges the reader re-hashes in `verify`.
    let mut graph_crc = Hasher::new();
    let mut vectors_crc = Hasher::new();

    // Header.
    writer.write_all(&header_bytes)?;
    // Node levels, padded up to the vector block.
    let mut levels = Vec::with_capacity(nodes);
    for node in 0..nodes {
        levels.push(parts.node_levels[node]);
    }
    graph_crc.update(&levels);
    writer.write_all(&levels)?;
    writer.pad_to(vectors_at)?;
    // Vectors, hashed and written in chunks so a large index never needs a
    // second copy of its data in memory.
    for chunk in parts.data.chunks(1 << 16) {
        let bytes = cast_slice::<f32, u8>(chunk);
        vectors_crc.update(bytes);
        writer.write_all(bytes)?;
    }
    writer.pad_to(ids_at)?;
    let mut ids = Vec::with_capacity(nodes * 8);
    for id in parts.ids {
        ids.extend_from_slice(&id.to_le_bytes());
    }
    graph_crc.update(&ids);
    writer.write_all(&ids)?;
    writer.pad_to(live_at)?;
    let mut live_bytes = Vec::with_capacity(nodes);
    for node in 0..nodes {
        live_bytes.push(u8::from(parts.live[node]));
    }
    graph_crc.update(&live_bytes);
    writer.write_all(&live_bytes)?;
    writer.pad_to(directory_at)?;
    // The directory block is a fixed size on disk: readers do not have to know
    // how many levels a graph happened to use, and unused entries are zero.
    let mut directory_bytes = vec![0u8; DIRECTORY_ENTRIES * DIRECTORY_ENTRY_LEN];
    for (level, entry) in directory.iter().enumerate() {
        let at = level * DIRECTORY_ENTRY_LEN;
        directory_bytes[at..at + DIRECTORY_ENTRY_LEN].copy_from_slice(&entry.encode());
    }
    graph_crc.update(&directory_bytes);
    writer.write_all(&directory_bytes)?;
    writer.pad_to(align(
        directory_at + (DIRECTORY_ENTRIES * DIRECTORY_ENTRY_LEN) as u64,
    ))?;
    for (level, entry) in directory.iter().enumerate() {
        let mut offsets = Vec::with_capacity((nodes + 1) * 4);
        let mut running = 0u32;
        offsets.extend_from_slice(&running.to_le_bytes());
        for node in 0..nodes {
            let count = if level <= usize::from(parts.node_levels[node]) {
                parts.levels[node][level].len()
            } else {
                0
            };
            running += count as u32;
            offsets.extend_from_slice(&running.to_le_bytes());
        }
        graph_crc.update(&offsets);
        writer.write_all(&offsets)?;
        writer.pad_to(align(entry.offsets_at + (nodes as u64 + 1) * 4))?;
        let mut links_bytes = Vec::with_capacity(entry.links as usize * 4);
        for node in 0..nodes {
            if level <= usize::from(parts.node_levels[node]) {
                for neighbour in &parts.levels[node][level] {
                    links_bytes.extend_from_slice(&neighbour.to_le_bytes());
                }
            }
        }
        graph_crc.update(&links_bytes);
        writer.write_all(&links_bytes)?;
        writer.pad_to(align(entry.links_at + entry.links * 4))?;
    }
    let footer = Footer {
        nodes: nodes as u64,
        live,
        links: total_links,
        vectors_crc: vectors_crc.finalize(),
        graph_crc: graph_crc.finalize(),
        header_crc,
        file_len,
    };
    writer.write_all(&footer.encode())?;
    if writer.position() != file_len {
        return Err(Error::Corrupt {
            what: "segment".to_string(),
            detail: format!(
                "layout planned {file_len} bytes but wrote {}",
                writer.position()
            ),
        });
    }
    let file = writer
        .into_inner()
        .into_inner()
        .map_err(|source| Error::Io {
            path: temporary.clone(),
            source: source.into_error(),
        })?;
    // Durability order matters: the data has to be on disk before the name
    // exists, otherwise a crash could leave a visible but empty segment.
    file.sync_all().map_err(|source| Error::Io {
        path: temporary.clone(),
        source,
    })?;
    drop(file);
    std::fs::rename(&temporary, path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    sync_directory(directory_of)?;

    Ok(SegmentInfo {
        path: path.to_path_buf(),
        dim: parts.dim,
        metric: parts.metric,
        nodes,
        live: live as usize,
        max_level: parts.max_level,
        links: total_links as usize,
        bytes: file_len,
        created_unix_ms,
    })
}

/// Removes leftover temporary segment files from an interrupted write.
///
/// Called when a collection is opened: a crash between `create` and `rename`
/// leaves a file that no manifest can reference.
///
/// # Errors
///
/// [`Error::Io`] if the directory cannot be read or a temporary file cannot be
/// removed.
pub fn remove_temporaries(directory: &Path) -> Result<usize> {
    let mut removed = 0;
    let entries = std::fs::read_dir(directory).map_err(|source| Error::Io {
        path: directory.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: directory.to_path_buf(),
            source,
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') && name.contains(".tmp-") {
            std::fs::remove_file(entry.path()).map_err(|source| Error::Io {
                path: entry.path(),
                source,
            })?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Flushes a directory entry, which is what makes a rename durable.
fn sync_directory(directory: &Path) -> Result<()> {
    // Directory handles cannot be opened for writing on every platform; on
    // those the rename itself is the durable operation.
    match File::open(directory) {
        Ok(handle) => handle.sync_all().map_err(|source| Error::Io {
            path: directory.to_path_buf(),
            source,
        }),
        Err(_) => Ok(()),
    }
}

/// Current wall-clock time in milliseconds since the Unix epoch.
#[must_use]
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// Rounds `offset` up to the next [`BLOCK_ALIGNMENT`] boundary.
#[must_use]
pub const fn align(offset: u64) -> u64 {
    offset.div_ceil(BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT
}

/// Maps a metric to its on-disk code.
#[must_use]
pub const fn metric_code(metric: Metric) -> u32 {
    match metric {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::InnerProduct => 2,
    }
}

/// Maps an on-disk code back to a metric.
fn metric_from_code(code: u32) -> Result<Metric> {
    match code {
        0 => Ok(Metric::L2),
        1 => Ok(Metric::Cosine),
        2 => Ok(Metric::InnerProduct),
        other => Err(Error::Corrupt {
            what: "segment header".to_string(),
            detail: format!("unknown metric code {other}"),
        }),
    }
}

/// CRC-32 of a byte slice.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

/// Reads a little-endian `u32` at `at`.
fn get_u32(bytes: &[u8], at: usize) -> Result<u32> {
    let slice = bytes.get(at..at + 4).ok_or_else(|| Error::Corrupt {
        what: "segment".to_string(),
        detail: format!("u32 at offset {at} is out of bounds"),
    })?;
    Ok(u32::from_le_bytes(slice.try_into().expect("four bytes")))
}

/// Reads a little-endian `u64` at `at`.
fn get_u64(bytes: &[u8], at: usize) -> Result<u64> {
    let slice = bytes.get(at..at + 8).ok_or_else(|| Error::Corrupt {
        what: "segment".to_string(),
        detail: format!("u64 at offset {at} is out of bounds"),
    })?;
    Ok(u64::from_le_bytes(slice.try_into().expect("eight bytes")))
}

/// Writes a little-endian `u32` at `at`.
fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// Writes a little-endian `u64` at `at`.
fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

/// A writer that tracks its position and covers every byte written by a CRC.
///
/// The wrapper exists so that the checksums cover exactly what lands on disk:
/// computing them separately from the write would be one more thing to get
/// wrong.
struct CrcWriter<W: Write> {
    inner: W,
    position: u64,
    /// Target path, reported in write errors.
    path: PathBuf,
}

impl<W: Write> CrcWriter<W> {
    fn new(inner: W, path: PathBuf) -> Self {
        Self {
            inner,
            position: 0,
            path,
        }
    }

    fn position(&self) -> u64 {
        self.position
    }

    /// Pads with zeros until `offset`, which must be ahead of the position.
    fn pad_to(&mut self, offset: u64) -> Result<()> {
        let missing = offset.saturating_sub(self.position);
        if missing > 0 {
            let zeros = vec![0u8; missing as usize];
            self.write_all(&zeros)?;
        }
        Ok(())
    }

    fn into_inner(self) -> W {
        self.inner
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.inner.write_all(bytes).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.position += bytes.len() as u64;
        Ok(())
    }
}

/// Truncates a file to `len`, used when a torn tail is rejected on open.
///
/// # Errors
///
/// [`Error::Io`] if the file cannot be truncated.
pub fn truncate_file(path: &Path, len: u64) -> Result<()> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.set_len(len).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    file.sync_all().map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Removes a file if it exists, used by tools and by the collection's
/// compaction path.
///
/// # Errors
///
/// [`Error::Io`] for filesystem failures other than a missing file.
pub fn remove_file_if_present(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(Error::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

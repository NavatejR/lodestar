//! Command implementations for the Lodestar CLI.
//!
//! Each function is one subcommand: open (or create) a collection, do one
//! thing, print one summary. Human output goes to stdout as plain text and
//! progress to stderr, while `--json` makes stdout machine-readable so it can
//! always be piped into `jq` or a script.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use lodestar_ann_core::{Metric, Rng, l2_normalize};
use lodestar_ann_index::hnsw::HnswConfig;
use lodestar_ann_store::{Collection, SegmentInfo};

use crate::io_util::{InsertRecord, parse_metric, parse_vector};

/// How many records the insert loop batches before calling the collection.
const INSERT_BATCH: usize = 1024;

/// The bare file name of a sealed segment, for messages.
fn segment_name(info: &SegmentInfo) -> String {
    info.path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| info.path.to_string_lossy().into_owned())
}

/// Creates an empty collection.
pub fn create(
    root: &Path,
    name: &str,
    dim: usize,
    metric_spec: &str,
    config: HnswConfig,
) -> Result<()> {
    let metric = parse_metric(metric_spec)?;
    Collection::create(root, name, dim, metric, config)
        .with_context(|| format!("creating collection {name}"))?;
    eprintln!(
        "created {name}: dim {dim}, metric {metric_spec}, m {} m0 {} ef_construction {}",
        config.m, config.m0, config.ef_construction
    );
    Ok(())
}

/// Inserts vectors from a JSONL file, one `{"id": N, "vector": [...]}` per line.
pub fn insert(root: &Path, name: &str, file: &Path, flush: bool) -> Result<()> {
    let input = std::fs::File::open(file).with_context(|| format!("opening {}", file.display()))?;
    let mut collection = open(root, name)?;
    let dim = collection.dim();

    let mut ids = Vec::with_capacity(INSERT_BATCH);
    let mut vectors = Vec::with_capacity(INSERT_BATCH * dim);
    let mut total = 0usize;
    let mut line_number = 0usize;
    for line in BufReader::new(input).lines() {
        line_number += 1;
        let line = line.with_context(|| format!("reading {}", file.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: InsertRecord = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: invalid record", file.display(), line_number))?;
        if record.vector.len() != dim {
            bail!(
                "{}:{}: vector has {} dimensions, collection {name} has {dim}",
                file.display(),
                line_number,
                record.vector.len()
            );
        }
        ids.push(record.id);
        vectors.extend(record.vector);
        if ids.len() == INSERT_BATCH {
            collection
                .upsert_batch(&ids, &vectors)
                .with_context(|| format!("{}:{}: upsert failed", file.display(), line_number))?;
            total += ids.len();
            ids.clear();
            vectors.clear();
            eprint!("\rinserted {total}");
        }
    }
    if !ids.is_empty() {
        collection.upsert_batch(&ids, &vectors)?;
        total += ids.len();
    }
    if flush {
        collection.flush()?;
    }
    eprintln!();
    println!("inserted {total} vectors into {name}");
    Ok(())
}

/// Deletes ids from a collection.
pub fn delete(root: &Path, name: &str, ids: &[u64]) -> Result<()> {
    let mut collection = open(root, name)?;
    let removed = collection.delete_batch(ids)?;
    println!("tombstoned {removed} of {} ids in {name}", ids.len());
    Ok(())
}

/// Searches a collection, either for one `--vector` or for every vector in a
/// JSONL file given with `--file`.
pub fn search(
    root: &Path,
    name: &str,
    vector_spec: Option<&str>,
    queries_file: Option<&Path>,
    k: usize,
    ef: usize,
    json: bool,
) -> Result<()> {
    let queries: Vec<Vec<f32>> = match (vector_spec, queries_file) {
        (Some(spec), None) => {
            let vector = parse_vector(spec)?;
            vec![vector]
        }
        (None, Some(file)) => read_queries(file)?,
        _ => bail!("give exactly one of --vector or --file"),
    };
    let dim = open(root, name)?.dim();
    for query in &queries {
        if query.len() != dim {
            bail!(
                "query has {} dimensions, collection {name} has {dim}",
                query.len()
            );
        }
    }

    let collection = open(root, name)?;
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for (index, query) in queries.iter().enumerate() {
        let results = collection.search(query, k, ef)?;
        if json {
            let line = serde_json::json!({
                "query": index,
                "results": results
                    .iter()
                    .map(|candidate| serde_json::json!({
                        "id": candidate.id,
                        "distance": candidate.distance,
                    }))
                    .collect::<Vec<_>>(),
            });
            writeln!(out, "{line}")?;
        } else {
            if queries.len() > 1 {
                writeln!(out, "query {}", index + 1)?;
            }
            writeln!(out, "        id   distance")?;
            for candidate in &results {
                writeln!(out, "{:>9}  {:>10.4}", candidate.id, candidate.distance)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// Seals the in-memory tail into an immutable segment.
pub fn flush(root: &Path, name: &str, json: bool) -> Result<()> {
    let mut collection = open(root, name)?;
    let sealed = collection.flush()?;
    match (sealed, json) {
        (Some(info), true) => println!(
            "{}",
            serde_json::json!({"segment": segment_name(&info), "nodes": info.nodes, "bytes": info.bytes})
        ),
        (Some(info), false) => println!(
            "sealed {} nodes into {} ({} bytes)",
            info.nodes,
            segment_name(&info),
            info.bytes
        ),
        (None, true) => println!("{}", serde_json::json!({"segment": null})),
        (None, false) => println!("nothing to seal"),
    }
    Ok(())
}

/// Rewrites every live vector into a single segment.
pub fn compact(root: &Path, name: &str, json: bool) -> Result<()> {
    let mut collection = open(root, name)?;
    let sealed = collection.compact()?;
    match (sealed, json) {
        (Some(info), true) => println!(
            "{}",
            serde_json::json!({"segment": segment_name(&info), "nodes": info.nodes, "bytes": info.bytes})
        ),
        (Some(info), false) => println!(
            "compacted {} nodes into {} ({} bytes)",
            info.nodes,
            segment_name(&info),
            info.bytes
        ),
        (None, true) => println!("{}", serde_json::json!({"segment": null})),
        (None, false) => println!("collection is empty; nothing to compact"),
    }
    Ok(())
}

/// Runs the full checksum pass over a collection.
pub fn verify(root: &Path, name: &str, json: bool) -> Result<()> {
    let collection = open(root, name)?;
    collection.verify()?;
    let stats = collection.stats();
    if json {
        println!(
            "{}",
            serde_json::json!({"ok": true, "nodes": stats.sealed_nodes + stats.tail_nodes})
        );
    } else {
        println!(
            "ok: {} nodes across {} segment(s), {} mapped bytes",
            stats.sealed_nodes + stats.tail_nodes,
            stats.segments,
            stats.mapped_bytes
        );
    }
    Ok(())
}

/// Prints collection statistics.
pub fn stats(root: &Path, name: &str, json: bool) -> Result<()> {
    let collection = open(root, name)?;
    let stats = collection.stats();
    if json {
        println!(
            "{}",
            serde_json::json!({
                "name": stats.name,
                "dim": stats.dim,
                "metric": stats.metric,
                "segments": stats.segments,
                "sealed_nodes": stats.sealed_nodes,
                "sealed_live": stats.sealed_live,
                "tail_nodes": stats.tail_nodes,
                "tail_live": stats.tail_live,
                "tombstoned_ids": stats.tombstoned_ids,
                "wal_bytes": stats.wal_bytes,
                "mapped_bytes": stats.mapped_bytes,
                "tail_bytes": stats.tail_bytes,
            })
        );
    } else {
        println!("name            {}", stats.name);
        println!("dim             {}", stats.dim);
        println!("metric          {:?}", stats.metric);
        println!("segments        {}", stats.segments);
        println!(
            "sealed nodes    {} ({} live)",
            stats.sealed_nodes, stats.sealed_live
        );
        println!(
            "tail nodes      {} ({} live)",
            stats.tail_nodes, stats.tail_live
        );
        println!("tombstoned ids  {}", stats.tombstoned_ids);
        println!("wal bytes       {}", stats.wal_bytes);
        println!("mapped bytes    {}", stats.mapped_bytes);
        println!("tail heap bytes {}", stats.tail_bytes);
    }
    Ok(())
}

/// Lists the collections under the root directory.
pub fn list(root: &Path, json: bool) -> Result<()> {
    let mut collections = Vec::new();
    let entries = std::fs::read_dir(root).with_context(|| format!("reading {}", root.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let manifest_path = path.join(lodestar_ann_store::collection::MANIFEST_FILE);
        if !manifest_path.is_file() {
            continue;
        }
        let manifest = lodestar_ann_store::Manifest::load(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?;
        collections.push(serde_json::json!({
            "name": manifest.name,
            "dim": manifest.dim,
            "metric": manifest.metric,
        }));
    }
    if json {
        println!("{}", serde_json::json!(collections));
    } else if collections.is_empty() {
        println!("no collections under {}", root.display());
    } else {
        for collection in &collections {
            println!(
                "{:<24} {:>4}  {:?}",
                collection["name"].as_str().unwrap_or("?"),
                collection["dim"],
                serde_json::from_value::<Metric>(collection["metric"].clone())
                    .unwrap_or(Metric::L2)
            );
        }
    }
    Ok(())
}

/// Generates a synthetic, clustered collection for demos and smoke tests.
///
/// Points are drawn from `clusters` uniform random centres plus uniform noise,
/// which gives a corpus with structure worth searching. Cosine collections are
/// normalised, so the centres and points lie on the unit sphere.
pub fn sample(
    root: &Path,
    name: &str,
    count: usize,
    dim: usize,
    metric_spec: &str,
    clusters: usize,
    seed: u64,
) -> Result<()> {
    if count == 0 {
        bail!("--count must be at least 1");
    }
    if clusters == 0 {
        bail!("--clusters must be at least 1");
    }
    let metric = parse_metric(metric_spec)?;
    let mut collection = Collection::open_or_create(root, name, dim, metric, HnswConfig::default())
        .with_context(|| format!("opening collection {name}"))?;
    if collection.dim() != dim {
        bail!(
            "collection {name} has {} dimensions, --dim says {dim}",
            collection.dim()
        );
    }

    let mut rng = Rng::new(seed);
    let spread = 0.35f32;
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| uniform_vector(&mut rng, dim))
        .collect();

    let batch_size = 1024;
    let mut ids = Vec::with_capacity(batch_size);
    let mut vectors = Vec::with_capacity(batch_size * dim);
    for index in 0..count {
        let center = &centers[rng.below(clusters)];
        let mut vector: Vec<f32> = center
            .iter()
            .map(|x| x + (rng.next_f64() * 2.0 - 1.0) as f32 * spread)
            .collect();
        if metric == Metric::Cosine {
            l2_normalize(&mut vector);
        }
        ids.push(index as u64);
        vectors.extend(vector);
        if ids.len() == batch_size {
            collection.upsert_batch(&ids, &vectors)?;
            ids.clear();
            vectors.clear();
            eprint!("\rsampled {}", index + 1);
        }
    }
    if !ids.is_empty() {
        collection.upsert_batch(&ids, &vectors)?;
    }
    collection.flush()?;
    eprintln!();
    println!("sampled {count} vectors into {name}");
    Ok(())
}

/// Opens a collection, mapping the store's error onto the CLI's error type.
fn open(root: &Path, name: &str) -> Result<Collection> {
    Collection::open(root, name).with_context(|| format!("opening collection {name}"))
}

/// One uniform random vector in [-1, 1).
fn uniform_vector(rng: &mut Rng, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
        .collect()
}

/// Reads query vectors from a JSONL file.
fn read_queries(file: &Path) -> Result<Vec<Vec<f32>>> {
    let input = std::fs::File::open(file).with_context(|| format!("opening {}", file.display()))?;
    let mut queries = Vec::new();
    for (line_number, line) in BufReader::new(input).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: InsertRecord = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: invalid record", file.display(), line_number + 1))?;
        queries.push(record.vector);
    }
    if queries.is_empty() {
        bail!("{} contains no queries", file.display());
    }
    Ok(queries)
}

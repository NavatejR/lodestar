//! Lodestar command-line interface.
//!
//! One binary that drives the storage layer directly: create, fill, search,
//! maintain and verify collections. Every command takes `--root`, the parent
//! directory of the collection, so several collections can live side by side;
//! commands that print data take `--json` so scripts get machine-readable
//! output on stdout while progress stays on stderr.
//!
//! Typical session:
//!
//! ```text
//! lodestar create --name docs --dim 384 --metric cosine
//! lodestar insert --name docs --file vectors.jsonl
//! lodestar search --name docs --vector "0.1, 0.2, 0.3" --k 10
//! lodestar verify --name docs
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};
use lodestar_ann_index::hnsw::HnswConfig;

mod commands;
mod io_util;

/// Builds the graph configuration from the optional overrides.
fn graph_config(
    m: Option<usize>,
    m0: Option<usize>,
    ef_construction: Option<usize>,
    seed: Option<u64>,
) -> HnswConfig {
    let mut config = HnswConfig::default();
    if let Some(value) = m {
        config.m = value;
    }
    if let Some(value) = m0 {
        config.m0 = value;
    }
    if let Some(value) = ef_construction {
        config.ef_construction = value;
    }
    if let Some(value) = seed {
        config.seed = value;
    }
    config
}

#[derive(Parser)]
#[command(
    name = "lodestar",
    version,
    about = "Vector search from scratch: create, fill, search and verify collections",
    after_help = "Use --root to point every command at a directory holding your collections."
)]
struct Cli {
    /// Directory that holds the collections.
    #[arg(long, global = true, default_value = ".")]
    root: PathBuf,

    /// Print machine-readable JSON on stdout.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create an empty collection.
    Create {
        /// Collection name, which is also its directory name under --root.
        name: String,
        /// Vector dimensionality, e.g. 384.
        #[arg(long)]
        dim: usize,
        /// Ranking metric: l2, cosine or inner_product.
        #[arg(long, default_value = "l2")]
        metric: String,
        /// Neighbours per node above level 0 (default 16).
        #[arg(long)]
        m: Option<usize>,
        /// Neighbours per node at level 0 (default 32).
        #[arg(long)]
        m0: Option<usize>,
        /// Candidate list size used while building (default 200).
        #[arg(long)]
        ef_construction: Option<usize>,
        /// Seed for level assignment, for reproducible builds.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Insert vectors from a JSONL file of {"id": N, "vector": [...]} records.
    Insert {
        name: String,
        /// JSONL file to read from; `-` reads stdin.
        #[arg(long)]
        file: PathBuf,
        /// Seal the tail into a segment when done.
        #[arg(long)]
        flush: bool,
    },
    /// Delete (tombstone) ids from a collection.
    Delete {
        name: String,
        /// Comma-separated ids to delete.
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u64>,
    },
    /// Search a collection for nearest neighbours.
    Search {
        name: String,
        /// Comma-separated query vector, e.g. "0.1, 0.2, 0.3".
        #[arg(long, conflicts_with = "file")]
        vector: Option<String>,
        /// JSONL file of query vectors; every record's vector is searched.
        #[arg(long)]
        file: Option<PathBuf>,
        /// How many neighbours to return.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Candidate list size; larger is slower and more accurate.
        #[arg(long, default_value_t = 64)]
        ef: usize,
    },
    /// Seal the in-memory tail into an immutable segment.
    Flush { name: String },
    /// Rewrite every live vector into one segment, dropping tombstones.
    Compact { name: String },
    /// Run the full checksum pass over every segment and the log.
    Verify { name: String },
    /// Print collection statistics.
    Stats { name: String },
    /// List the collections under --root.
    List,
    /// Generate a synthetic clustered collection for demos and smoke tests.
    Sample {
        name: String,
        /// How many vectors to generate.
        #[arg(long, default_value_t = 10_000)]
        count: usize,
        /// Vector dimensionality.
        #[arg(long, default_value_t = 64)]
        dim: usize,
        /// Ranking metric: l2, cosine or inner_product.
        #[arg(long, default_value = "l2")]
        metric: String,
        /// How many cluster centres to draw points around.
        #[arg(long, default_value_t = 16)]
        clusters: usize,
        /// Seed for the generator, for reproducible corpora.
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
}

fn run(cli: &Cli) -> Result<()> {
    let root = &cli.root;
    let json = cli.json;
    match &cli.command {
        Command::Create {
            name,
            dim,
            metric,
            m,
            m0,
            ef_construction,
            seed,
        } => commands::create(
            root,
            name,
            *dim,
            metric,
            graph_config(*m, *m0, *ef_construction, *seed),
        ),
        Command::Insert { name, file, flush } => {
            if file.as_os_str() == "-" {
                anyhow::bail!("reading from stdin is not supported yet; pass a file path");
            }
            commands::insert(root, name, file, *flush)
        }
        Command::Delete { name, ids } => commands::delete(root, name, ids),
        Command::Search {
            name,
            vector,
            file,
            k,
            ef,
        } => commands::search(
            root,
            name,
            vector.as_deref(),
            file.as_deref(),
            *k,
            *ef,
            json,
        ),
        Command::Flush { name } => commands::flush(root, name, json),
        Command::Compact { name } => commands::compact(root, name, json),
        Command::Verify { name } => commands::verify(root, name, json),
        Command::Stats { name } => commands::stats(root, name, json),
        Command::List => commands::list(root, json),
        Command::Sample {
            name,
            count,
            dim,
            metric,
            clusters,
            seed,
        } => commands::sample(root, name, *count, *dim, metric, *clusters, *seed),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(1)
        }
    }
}

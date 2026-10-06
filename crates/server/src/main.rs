//! The `lodestar-server` binary.
//!
//! ```text
//! lodestar-server --root ./data --addr :8080
//! ```
//!
//! Every flag also reads an environment variable (`LODESTAR_ROOT`,
//! `LODESTAR_ADDR`, `LODESTAR_CORS`, `LODESTAR_READ_ONLY`, `LODESTAR_LOG`), so
//! the container image in `demo/` needs no command line at all.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use axum::http::HeaderValue;
use clap::Parser;
use lodestar_ann_server::{AppState, router_with_cors};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tracing_subscriber::EnvFilter;

/// Version reported on start-up and by `/healthz`.
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "lodestar-server",
    version,
    about = "HTTP API for Lodestar vector indexes",
    after_help = "Collections are directories under --root, so the CLI, the Python bindings \
                  and this service can all read the same data."
)]
struct Args {
    /// Directory that holds the collections.
    #[arg(long, env = "LODESTAR_ROOT", default_value = "./data")]
    root: PathBuf,

    /// Address to listen on: `host:port`, `:port` or a bare `port`.
    #[arg(long, env = "LODESTAR_ADDR", default_value = "127.0.0.1:8080")]
    addr: String,

    /// Allowed browser origin(s), comma separated. `*` allows any.
    #[arg(long, env = "LODESTAR_CORS", value_delimiter = ',')]
    cors: Vec<String>,

    /// Refuse every mutation with 403.
    #[arg(long, env = "LODESTAR_READ_ONLY")]
    read_only: bool,

    /// Tracing filter, e.g. `info` or `lodestar_ann_server=debug`.
    #[arg(long, env = "LODESTAR_LOG", default_value = "info")]
    log: String,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(1)
        }
    }
}

/// Parses the listen address, accepting the three shapes a container needs.
fn parse_addr(spec: &str) -> Result<SocketAddr> {
    let spec = spec.trim();
    if let Ok(addr) = spec.parse::<SocketAddr>() {
        return Ok(addr);
    }
    let port = spec.strip_prefix(':').unwrap_or(spec);
    if let Ok(port) = port.parse::<u16>() {
        // A bare port means "every interface", which is what a container
        // wants; the default stays on loopback for a local run.
        return Ok(SocketAddr::from(([0, 0, 0, 0], port)));
    }
    anyhow::bail!("`{spec}` is not an address; expected `host:port`, `:port` or `port`")
}

/// Builds the CORS layer, or `None` when no origin was configured.
fn cors_layer(origins: &[String]) -> Result<Option<CorsLayer>> {
    if origins.is_empty() {
        return Ok(None);
    }
    let layer = CorsLayer::new().allow_methods(Any).allow_headers(Any);
    if origins.iter().any(|origin| origin == "*") {
        return Ok(Some(layer.allow_origin(Any)));
    }
    let mut values = Vec::with_capacity(origins.len());
    for origin in origins {
        values.push(origin_value(origin)?);
    }
    Ok(Some(layer.allow_origin(AllowOrigin::list(values))))
}

/// Validates one configured origin.
///
/// A typo here would not fail at start-up on its own — a header value accepts
/// plenty of strings that no browser ever sends as an `Origin` — it would just
/// silently block every browser request, which is the worst way for a
/// configuration mistake to behave.
fn origin_value(origin: &str) -> Result<HeaderValue> {
    if origin != "null" && !origin.starts_with("http://") && !origin.starts_with("https://") {
        anyhow::bail!("`{origin}` is not an origin; expected `https://host[:port]`, `*` or `null`");
    }
    origin
        .parse::<HeaderValue>()
        .with_context(|| format!("`{origin}` is not a valid header value"))
}

/// Installs tracing, letting `RUST_LOG` override the flag.
fn init_tracing(spec: &str) -> Result<()> {
    let directive = std::env::var("RUST_LOG").unwrap_or_else(|_| spec.to_string());
    let filter = EnvFilter::try_new(&directive)
        .with_context(|| format!("invalid log filter `{directive}`"))?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
    Ok(())
}

/// Resolves on SIGINT or SIGTERM, so an in-flight request finishes and the
/// collections are closed cleanly on the way out.
async fn shutdown_signal() {
    let interrupt = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "could not listen for SIGINT");
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => tracing::error!(%error, "could not listen for SIGTERM"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = interrupt => {}
        () = terminate => {}
    }
    tracing::info!("shutdown signal received; finishing in-flight requests");
}

async fn run() -> Result<()> {
    let args = Args::parse();
    init_tracing(&args.log)?;
    let addr = parse_addr(&args.addr)?;
    let cors = cors_layer(&args.cors)?;

    let state = AppState::open(&args.root, args.read_only).with_context(|| {
        format!(
            "opening the data root {}; is it writable, and is every collection in it valid?",
            args.root.display()
        )
    })?;
    tracing::info!(
        version = VERSION,
        root = %state.root().display(),
        collections = state.len(),
        read_only = state.read_only(),
        "starting lodestar-server"
    );

    let app = router_with_cors(state.clone(), cors);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    let local = listener.local_addr()?;

    let mode = if state.read_only() {
        "read-only"
    } else {
        "read-write"
    };
    // Printed as well as logged: the URL is the one thing a person needs from
    // the start-up output, and the log format may be JSON or filtered.
    println!("lodestar-server {VERSION} listening on http://{local} ({mode})");
    println!("  root    {}", state.root().display());
    println!("  docs    http://{local}/docs");
    println!("  metrics http://{local}/metrics");
    println!("  health  http://{local}/healthz");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serving")?;
    tracing::info!("lodestar-server stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_accept_the_shapes_a_container_needs() {
        assert_eq!(parse_addr("127.0.0.1:8080").unwrap().port(), 8080);
        assert_eq!(parse_addr(":9000").unwrap().port(), 9000);
        assert_eq!(parse_addr("9000").unwrap().port(), 9000);
        assert_eq!(parse_addr(" 8081 ").unwrap().port(), 8081);
        assert!(parse_addr(":9000").unwrap().ip().is_unspecified());
        assert!(parse_addr("localhost:8080").is_err());
        assert!(parse_addr("nonsense").is_err());
        assert!(parse_addr(":99999").is_err());
    }

    #[test]
    fn cors_is_off_until_an_origin_is_configured() {
        assert!(cors_layer(&[]).unwrap().is_none());
        assert!(cors_layer(&["*".to_string()]).unwrap().is_some());
        assert!(
            cors_layer(&["https://example.com".to_string()])
                .unwrap()
                .is_some()
        );
        assert!(cors_layer(&["null".to_string()]).unwrap().is_some());
        assert!(
            cors_layer(&["http://localhost:3000".to_string()])
                .unwrap()
                .is_some()
        );
        assert!(origin_value("not an origin").is_err());
        assert!(origin_value("example.com").is_err());
    }
}

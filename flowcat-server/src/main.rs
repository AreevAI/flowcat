// SPDX-License-Identifier: Apache-2.0
//
//! `flowcat-server` binary: load an agent config and serve it over HTTP.
//!
//! ```text
//! flowcat-server --config agent.yaml      # or a positional path, or $FLOWCAT_CONFIG
//! ```
//!
//! Resolves the agent graph from the config, then runs an axum server exposing
//! `/healthz`, `/readyz`, the Plivo media WebSocket, and the Plivo answer XML.
//! Provider keys are read from the environment (see [`flowcat_server::run`]).

use std::path::PathBuf;

use flowcat_server::config::ServerConfig;
use flowcat_server::server::{build_router, AppState};
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config_path = config_path_from_args();
    let config = match ServerConfig::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "flowcat-server: failed to load config {}: {e}",
                config_path.display()
            );
            std::process::exit(1);
        }
    };
    let base_dir = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let graph = match config.resolve_graph(&base_dir) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("flowcat-server: {e}");
            std::process::exit(1);
        }
    };

    let bind = config.server.bind.clone();
    let public_url = std::env::var("FLOWCAT_PUBLIC_URL").ok();
    let app = build_router(AppState::new(config, graph, public_url));

    let listener = match tokio::net::TcpListener::bind(&bind).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("flowcat-server: cannot bind {bind}: {e}");
            std::process::exit(1);
        }
    };
    info!(%bind, "flowcat-server listening");
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("flowcat-server: server error: {e}");
        std::process::exit(1);
    }
}

/// Resolve the config path: `--config <p>` / `-c <p>`, else the first positional
/// arg, else `$FLOWCAT_CONFIG`, else `agent.yaml`.
fn config_path_from_args() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" | "-c" => {
                if let Some(p) = args.next() {
                    return PathBuf::from(p);
                }
            }
            other if !other.starts_with('-') => return PathBuf::from(other),
            _ => {}
        }
    }
    std::env::var("FLOWCAT_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("agent.yaml"))
}

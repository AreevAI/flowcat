// SPDX-License-Identifier: Apache-2.0
//
//! Flowcat developer CLI (`flowcat`).
//!
//! This is the OSS demo surface — the analogue of pipecat's `examples/`. Both
//! demos are credential-free and run with no external service, so they exercise
//! the runtime in CI.
//!
//!   1. **`pipeline`** — an in-process, network-free showcase of the composable
//!      [`FrameProcessor`](flowcat_core::FrameProcessor) pipeline (the product's
//!      core value prop). A synthetic source pumps a 1 s, 16 kHz sine wave through
//!      an identity "echo" processor + a trivial tap, and a
//!      [`FrameObserver`](flowcat_core::FrameObserver) counts the audio frames that
//!      flowed and reports a summary.
//!
//!   2. **`ws-echo`** — real I/O over the generic WebSocket media transport
//!      ([`WsTransport`](flowcat_transports::ws::WsTransport), little-endian i16
//!      mono PCM). `--connect <ws://url>` connects to a peer and echoes every
//!      inbound audio chunk back; the default `--loopback` mode stands up an
//!      in-process `tokio-tungstenite` server, sends a handful of known PCM frames
//!      from a client, runs the echo loop on the server side, and asserts the
//!      client got its frames back byte-for-byte — proving the WS round-trip.

use std::time::Instant;

use clap::{Parser, Subcommand};

mod pipeline_demo;
mod ws_echo;

/// The `flowcat` developer CLI — runnable, credential-free runtime demos.
#[derive(Parser)]
#[command(
    name = "flowcat",
    about = "Flowcat runtime demos (pipeline + ws-echo)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// In-process FrameProcessor pipeline over a synthetic sine-wave source.
    Pipeline,
    /// WebSocket PCM echo: `--connect <url>` to a peer, else a self-contained loopback.
    WsEcho {
        /// Connect to this `ws://`/`wss://` peer and echo its audio back.
        #[arg(long)]
        connect: Option<String>,
        /// Run the self-contained in-process loopback (the default when no `--connect`).
        #[arg(long)]
        loopback: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Pipeline => {
            let started = Instant::now();
            let summary = pipeline_demo::run().await;
            summary.print(started.elapsed());
        }
        Command::WsEcho { connect, loopback } => {
            let result = match connect {
                Some(url) if !loopback => ws_echo::run_connect(&url).await,
                // Default (and explicit `--loopback`): self-contained round-trip.
                _ => ws_echo::run_loopback().await,
            };
            if let Err(e) = result {
                eprintln!("ws-echo failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

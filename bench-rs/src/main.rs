// SPDX-License-Identifier: Apache-2.0
//
//! Flowcat spike — Rust runtime baseline, mirroring bench/pipecat_profile.py.
//!
//! Same shape as the pipecat measurement: a 7-stage passthrough pipeline
//! (source + 5 processors + sink) connected by bounded channels, carrying
//! 160-byte (20ms @ 8kHz μ-law) audio frames. Each stage is one task that
//! forwards frames downstream — the direct analogue of a pipecat FrameProcessor.
//!
//!   throughput [--multi]      per-frame hop cost + frames/s (1 core, or all cores)
//!   mem        --sessions N   resident RAM per idle session
//!
//! Run: cargo run --release --manifest-path bench-rs/Cargo.toml -- throughput

use bytes::Bytes;
use std::process::Command;
use std::time::Instant;
use tokio::runtime::Builder;
use tokio::sync::mpsc;

const STAGES: usize = 7; // source + 5 + sink — matches pipecat's wrapped pipeline
const FRAME_LEN: usize = 160; // one 20ms telephony audio frame
const CHAN_CAP: usize = 64;

#[derive(Clone)]
#[allow(dead_code)] // payload is moved/forwarded, never inspected — the alloc+clone+drop cost is real
enum Frame {
    Audio(Bytes),
    End,
}

/// One pipeline stage: forward every frame downstream (the passthrough analogue
/// of the stubbed transport/LLM processors in the pipecat profile).
async fn stage(mut rx: mpsc::Receiver<Frame>, tx: mpsc::Sender<Frame>) {
    while let Some(f) = rx.recv().await {
        let end = matches!(f, Frame::End);
        if tx.send(f).await.is_err() {
            break;
        }
        if end {
            break;
        }
    }
}

/// Build one session: head sender → STAGES tasks → tail receiver.
fn build_pipeline() -> (
    mpsc::Sender<Frame>,
    mpsc::Receiver<Frame>,
    Vec<tokio::task::JoinHandle<()>>,
) {
    let mut handles = Vec::with_capacity(STAGES);
    let (head_tx, first_rx) = mpsc::channel(CHAN_CAP);
    let mut prev_rx = first_rx;
    for _ in 0..STAGES {
        let (tx, rx) = mpsc::channel(CHAN_CAP);
        handles.push(tokio::spawn(stage(prev_rx, tx)));
        prev_rx = rx;
    }
    (head_tx, prev_rx, handles)
}

fn rss_bytes() -> u64 {
    let pid = std::process::id().to_string();
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .expect("ps failed");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0)
        * 1024 // ps reports KB
}

fn throughput(k: u64, multi: bool) {
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let rt = if multi {
        Builder::new_multi_thread().worker_threads(workers).enable_all().build()
    } else {
        Builder::new_current_thread().enable_all().build()
    }
    .unwrap();

    rt.block_on(async move {
        let (head, mut tail, handles) = build_pipeline();
        let audio = Bytes::from(vec![0xffu8; FRAME_LEN]);

        let sink = tokio::spawn(async move {
            let mut n: u64 = 0;
            while let Some(f) = tail.recv().await {
                if matches!(f, Frame::End) {
                    break;
                }
                n += 1;
            }
            n
        });

        let t0 = Instant::now();
        for _ in 0..k {
            head.send(Frame::Audio(audio.clone())).await.unwrap();
        }
        head.send(Frame::End).await.unwrap();
        let n = sink.await.unwrap();
        let dt = t0.elapsed().as_secs_f64();
        for h in handles {
            let _ = h.await;
        }

        let fps = n as f64 / dt;
        let kind = if multi {
            format!("multi-thread ({workers} cores)")
        } else {
            "current-thread (1 core)".to_string()
        };
        println!("[throughput] runtime={kind} frames={n}");
        println!("  elapsed_s          = {dt:.3}");
        println!("  frames_per_sec     = {fps:.0}");
        println!("  us_per_frame       = {:.4}", dt / n as f64 * 1e6);
        println!("  us_per_proc_hop    = {:.4}", dt / n as f64 / STAGES as f64 * 1e6);
        println!(
            "  -> a live call is ~100 frames/s. This runtime sustains ~{:.0} such \
             calls before CPU-bound on framework alone.",
            fps / 100.0
        );
    });
}

fn mem(n: usize, multi: bool) {
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let rt = if multi {
        Builder::new_multi_thread().worker_threads(workers).enable_all().build()
    } else {
        Builder::new_current_thread().enable_all().build()
    }
    .unwrap();

    rt.block_on(async move {
        // warm up one session so first-touch allocations don't skew the delta
        let _warm = build_pipeline();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        let before = rss_bytes();
        // Keep senders + receivers alive so the STAGES tasks park on recv (idle call).
        let mut keep: Vec<(mpsc::Sender<Frame>, mpsc::Receiver<Frame>, Vec<tokio::task::JoinHandle<()>>)> =
            Vec::with_capacity(n);
        for _ in 0..n {
            keep.push(build_pipeline());
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let after = rss_bytes();

        let per = (after.saturating_sub(before)) as f64 / n as f64;
        println!("[mem] sessions={n} tasks_per_session={STAGES}");
        println!("  rss_before_mb      = {:.1}", before as f64 / 1e6);
        println!("  rss_after_mb       = {:.1}", after as f64 / 1e6);
        println!("  rss_per_session_kb = {:.1}", per / 1024.0);
        println!(
            "  -> 1000 sessions ~ {:.2} GB RAM, {} tokio tasks",
            per * 1000.0 / 1e9,
            STAGES * 1000
        );
        // keep `keep` alive until here
        drop(keep);
    });
}

/// Drive one pipeline end to end: pump `k` frames, count what reaches the sink.
async fn drive(k: u64) -> u64 {
    let (head, mut tail, handles) = build_pipeline();
    let audio = Bytes::from(vec![0xffu8; FRAME_LEN]);
    let sink = tokio::spawn(async move {
        let mut n: u64 = 0;
        while let Some(f) = tail.recv().await {
            if matches!(f, Frame::End) {
                break;
            }
            n += 1;
        }
        n
    });
    for _ in 0..k {
        head.send(Frame::Audio(audio.clone())).await.unwrap();
    }
    head.send(Frame::End).await.unwrap();
    let n = sink.await.unwrap();
    for h in handles {
        let _ = h.await;
    }
    n
}

/// Density test: run `p` pipelines concurrently. On multi-thread this spreads
/// across all cores in ONE process — the thing Python's GIL cannot do without
/// running `p`-many separate processes.
fn concurrent(p: usize, k: u64, multi: bool) {
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let rt = if multi {
        Builder::new_multi_thread().worker_threads(workers).enable_all().build()
    } else {
        Builder::new_current_thread().enable_all().build()
    }
    .unwrap();

    rt.block_on(async move {
        let t0 = Instant::now();
        let tasks: Vec<_> = (0..p).map(|_| tokio::spawn(drive(k))).collect();
        let mut total: u64 = 0;
        for t in tasks {
            total += t.await.unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        let fps = total as f64 / dt;
        let kind = if multi {
            format!("multi-thread ({workers} cores)")
        } else {
            "current-thread (1 core)".to_string()
        };
        println!("[concurrent] runtime={kind} pipelines={p} frames_each={k}");
        println!("  total_frames       = {total}");
        println!("  elapsed_s          = {dt:.3}");
        println!("  frames_per_sec     = {fps:.0}");
        println!("  -> aggregate ~{:.0} live calls' worth of framework work/s", fps / 100.0);
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("throughput");
    let multi = args.iter().any(|a| a == "--multi");

    let val = |flag: &str, default: u64| -> u64 {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };

    match mode {
        "throughput" => throughput(val("--frames", 2_000_000), multi),
        "concurrent" => concurrent(val("--pipelines", 200) as usize, val("--frames", 50_000), multi),
        "mem" => mem(val("--sessions", 1000) as usize, multi),
        other => eprintln!("unknown mode: {other} (use throughput|concurrent|mem)"),
    }
}

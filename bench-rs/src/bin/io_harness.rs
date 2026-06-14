// SPDX-License-Identifier: Apache-2.0
//
//! Flowcat spike — real-I/O harness (WebSocket + μ-law), end-to-end.
//!
//! Two modes, sharing one wire protocol (raw 160-byte μ-law binary WS frames):
//!
//!   serve --addr 127.0.0.1:9099
//!       The Rust SUT. Per connection: WS read → μ-law decode → (pipeline) →
//!       μ-law encode → WS write, full-duplex echo. One process, all cores.
//!
//!   load  --url ws://127.0.0.1:9099 --conns N --secs T --fps 50
//!       The load generator (also drives the Python pipecat SUT). N callers,
//!       each sending one frame every 1000/fps ms, matching returned frames in
//!       order to measure per-frame RTT (p50/p99/max) + achieved throughput.
//!
//! The same `load` binary points at either SUT, so the comparison is identical
//! wire + identical client. Both SUT and load-gen share this laptop, so absolute
//! capacity is understated (conservative); the relative comparison is clean.

use futures_util::{SinkExt, StreamExt};
use std::time::{Duration, Instant};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message};

// ── G.711 μ-law codec (per-frame work both stacks must do) ──────────────────

fn ulaw_decode_sample(byte: u8) -> i16 {
    let u = !byte;
    let sign = (u & 0x80) != 0;
    let exponent = ((u >> 4) & 0x07) as i32;
    let mantissa = (u & 0x0f) as i32;
    let magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84;
    if sign {
        -(magnitude as i32) as i16
    } else {
        magnitude as i16
    }
}

fn ulaw_encode_sample(s: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32635;
    let mut sample = s as i32;
    let sign = if sample < 0 {
        sample = -sample;
        0x80u8
    } else {
        0u8
    };
    if sample > CLIP {
        sample = CLIP;
    }
    sample += BIAS;
    let mut exponent = 7i32;
    let mut mask = 0x4000i32;
    while exponent > 0 && (sample & mask) == 0 {
        exponent -= 1;
        mask >>= 1;
    }
    let mantissa = ((sample >> (exponent + 3)) & 0x0f) as u8;
    !(sign | ((exponent as u8) << 4) | mantissa)
}

fn ulaw_to_pcm(buf: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len() * 2);
    for &u in buf {
        out.extend_from_slice(&ulaw_decode_sample(u).to_le_bytes());
    }
    out
}

fn pcm_to_ulaw(pcm: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() / 2);
    for c in pcm.chunks_exact(2) {
        out.push(ulaw_encode_sample(i16::from_le_bytes([c[0], c[1]])));
    }
    out
}

// ── SUT ─────────────────────────────────────────────────────────────────────

async fn handle_conn(stream: TcpStream) {
    stream.set_nodelay(true).ok();
    let ws = match accept_async(stream).await {
        Ok(w) => w,
        Err(_) => return,
    };
    let (mut write, mut read) = ws.split();
    while let Some(Ok(msg)) = read.next().await {
        if msg.is_close() {
            break;
        }
        if msg.is_binary() {
            let buf = msg.into_data();
            // Real per-frame work: decode → (pipeline, identity here) → encode.
            let pcm = ulaw_to_pcm(buf.as_ref());
            let out = pcm_to_ulaw(&pcm);
            if write.send(Message::binary(out)).await.is_err() {
                break;
            }
        }
    }
}

async fn serve(addr: String) {
    let listener = TcpListener::bind(&addr).await.expect("bind failed");
    eprintln!("[serve] Rust SUT listening on ws://{addr}");
    while let Ok((stream, _)) = listener.accept().await {
        tokio::spawn(handle_conn(stream));
    }
}

// ── Load generator ────────────────────────────────────────────────────────--

async fn run_conn(url: String, secs: u64, fps: u64) -> (Vec<f32>, u64, u64) {
    let (ws, _) = match connect_async(&url).await {
        Ok(x) => x,
        Err(_) => return (vec![], 0, 0),
    };
    let (mut write, mut read) = ws.split();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Instant>();
    let period = Duration::from_micros(1_000_000 / fps.max(1));
    let dur = Duration::from_secs(secs);

    let send_task = tokio::spawn(async move {
        let frame = vec![0xffu8; 160];
        let mut ticker = tokio::time::interval(period);
        let start = Instant::now();
        let mut sent = 0u64;
        loop {
            ticker.tick().await;
            if start.elapsed() >= dur {
                break;
            }
            if write.send(Message::binary(frame.clone())).await.is_err() {
                break;
            }
            let _ = tx.send(Instant::now());
            sent += 1;
        }
        let _ = write.send(Message::Close(None)).await;
        sent
    });

    let mut samples: Vec<f32> = Vec::new();
    let mut recv = 0u64;
    let recv_deadline = tokio::time::Instant::now() + dur + Duration::from_secs(3);
    loop {
        tokio::select! {
            m = read.next() => match m {
                Some(Ok(msg)) => {
                    if msg.is_close() { break; }
                    if msg.is_binary() {
                        if let Ok(t0) = rx.try_recv() {
                            samples.push(t0.elapsed().as_secs_f32() * 1000.0);
                        }
                        recv += 1;
                    }
                }
                _ => break,
            },
            _ = tokio::time::sleep_until(recv_deadline) => break,
        }
    }
    let sent = send_task.await.unwrap_or(0);
    (samples, sent, recv)
}

async fn load(url: String, conns: u64, secs: u64, fps: u64) {
    let tasks: Vec<_> = (0..conns)
        .map(|_| tokio::spawn(run_conn(url.clone(), secs, fps)))
        .collect();

    let mut all: Vec<f32> = Vec::new();
    let (mut sent_tot, mut recv_tot, mut failed) = (0u64, 0u64, 0u64);
    for t in tasks {
        match t.await {
            Ok((mut s, sent, recv)) if sent > 0 || recv > 0 => {
                all.append(&mut s);
                sent_tot += sent;
                recv_tot += recv;
            }
            _ => failed += 1,
        }
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f32 {
        if all.is_empty() {
            0.0
        } else {
            all[((p * (all.len() - 1) as f64) as usize).min(all.len() - 1)]
        }
    };
    let mean = if all.is_empty() {
        0.0
    } else {
        all.iter().map(|&x| x as f64).sum::<f64>() / all.len() as f64
    };
    let target = conns * fps;
    let achieved = recv_tot as f64 / secs as f64;

    println!("[load] conns={conns} secs={secs} fps_per_conn={fps} failed_conns={failed}");
    println!("  target_fps_total   = {target}");
    println!(
        "  achieved_fps_total = {achieved:.0}  ({:.0}% of target)",
        achieved / target as f64 * 100.0
    );
    println!("  sent={sent_tot} recv={recv_tot} samples={}", all.len());
    println!(
        "  rtt_ms  p50={:.2}  p90={:.2}  p99={:.2}  p999={:.2}  max={:.2}  mean={:.2}",
        pct(0.50),
        pct(0.90),
        pct(0.99),
        pct(0.999),
        all.last().copied().unwrap_or(0.0),
        mean
    );
}

// ── entry ─────────────────────────────────────────────────────────────────--

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("serve");
    let sval = |flag: &str, def: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| def.to_string())
    };
    let uval = |flag: &str, def: u64| -> u64 {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(def)
    };

    match mode {
        "serve" => serve(sval("--addr", "127.0.0.1:9099")).await,
        "load" => {
            load(
                sval("--url", "ws://127.0.0.1:9099"),
                uval("--conns", 100),
                uval("--secs", 10),
                uval("--fps", 50),
            )
            .await
        }
        other => eprintln!("unknown mode: {other} (use serve|load)"),
    }
}

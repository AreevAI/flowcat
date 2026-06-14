// SPDX-License-Identifier: Apache-2.0
//
//! Demo 2 — WebSocket PCM echo over the generic [`WsTransport`].
//!
//! [`WsTransport`] is a [`MediaTransport`](flowcat_core::MediaTransport) over a raw
//! WebSocket carrying little-endian i16 mono PCM: `recv()` yields
//! [`MediaIn::Audio`] / [`MediaIn::Stop`], and `send_audio` writes the PCM straight
//! back out as a binary frame.
//!
//!   - [`run_connect`] — connect to a `ws://`/`wss://` peer and echo every inbound
//!     audio chunk back until the peer stops. Credential-free; needs a peer.
//!   - [`run_loopback`] — self-contained: stand up an in-process
//!     `tokio-tungstenite` server bound to `127.0.0.1:0`, run the echo loop on the
//!     server side (through `WsTransport`), and have a client send a handful of
//!     known PCM frames and assert it gets them back byte-for-byte. Runs in CI.

use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

use flowcat_core::transport::{MediaIn, MediaTransport};
use flowcat_transports::ws::WsTransport;

/// The carrier sample rate the loopback tags its PCM with (16 kHz mono).
const CARRIER_RATE: u32 = 16_000;

/// Run the echo loop over any `MediaTransport`: read inbound events and bounce
/// every audio chunk straight back, until the stream stops. Returns the number of
/// chunks echoed. Shared by [`run_connect`] and [`run_loopback`]'s server side.
async fn echo_loop<T: MediaTransport>(transport: &mut T) -> Result<u64, String> {
    let mut echoed = 0u64;
    loop {
        match transport.recv().await {
            Some(MediaIn::StreamStart { call_id }) => {
                println!("ws-echo: stream started (call_id={call_id})");
            }
            Some(MediaIn::Audio(chunk)) => {
                let samples = chunk.pcm.len();
                transport
                    .send_audio(chunk)
                    .await
                    .map_err(|e| format!("send_audio: {e}"))?;
                echoed += 1;
                println!("ws-echo: echoed frame {echoed} ({samples} samples)");
            }
            Some(MediaIn::Stop) | None => break,
        }
    }
    println!("ws-echo: stream stopped after {echoed} echoed frame(s)");
    Ok(echoed)
}

/// `--connect` mode: connect to `url` and echo its audio back until it stops.
pub(crate) async fn run_connect(url: &str) -> Result<(), String> {
    println!("ws-echo: connecting to {url} ...");
    let mut transport = WsTransport::connect(url, CARRIER_RATE, "ws-echo-cli")
        .await
        .map_err(|e| format!("connect: {e}"))?;
    let echoed = echo_loop(&mut transport).await?;
    println!("ws-echo: done — echoed {echoed} frame(s) back to {url}");
    Ok(())
}

/// `--loopback` mode (the default): a full in-process WS round-trip with no peer.
///
/// Binds a server on `127.0.0.1:0`, spawns the echo loop on the accepted socket
/// (wrapped in a server-side `WsTransport`), then a client connects, sends a few
/// known PCM frames, reads the echoes, and asserts they match byte-for-byte.
pub(crate) async fn run_loopback() -> Result<(), String> {
    // The known frames the client sends (and expects back unchanged).
    let frames: Vec<Vec<i16>> = vec![
        vec![0, 1, -1, 1000, -1000, i16::MAX, i16::MIN],
        vec![100, 200, 300, -300, -200, -100],
        vec![7; 64],
    ];

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("bind: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("local_addr: {e}"))?;
    let url = format!("ws://{addr}");
    println!("ws-echo: loopback server listening on {url}");

    // Server task: accept one connection, wrap it as a server-side WsTransport, and
    // run the echo loop until the client closes.
    let server = tokio::spawn(async move {
        let (tcp, _peer) = listener
            .accept()
            .await
            .map_err(|e| format!("accept: {e}"))?;
        let socket = tokio_tungstenite::accept_async(tcp)
            .await
            .map_err(|e| format!("accept_async: {e}"))?;
        let mut transport = WsTransport::new(socket, CARRIER_RATE, "loopback");
        echo_loop(&mut transport).await
    });

    // Client: connect with the raw tungstenite client, send each known frame as a
    // binary PCM frame, then read exactly that many echoes back and compare.
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .map_err(|e| format!("client connect: {e}"))?;

    for pcm in &frames {
        ws.send(Message::binary(pcm16_to_bytes(pcm)))
            .await
            .map_err(|e| format!("client send: {e}"))?;
    }

    let mut got: Vec<Vec<i16>> = Vec::with_capacity(frames.len());
    while got.len() < frames.len() {
        match ws.next().await {
            Some(Ok(Message::Binary(b))) => got.push(bytes_to_pcm16(&b)),
            Some(Ok(_)) => continue, // ignore non-binary control frames
            Some(Err(e)) => return Err(format!("client recv: {e}")),
            None => break,
        }
    }

    // Close the client → the server's recv() sees Close → Stop → echo_loop returns.
    ws.close(None)
        .await
        .map_err(|e| format!("client close: {e}"))?;

    let echoed = server.await.map_err(|e| format!("server join: {e}"))??;

    if got != frames {
        return Err(format!(
            "loopback mismatch: sent {:?}, got back {:?}",
            frames, got
        ));
    }

    println!(
        "ws-echo: loopback OK — {} frame(s) round-tripped byte-for-byte ({} echoed server-side)",
        frames.len(),
        echoed
    );
    Ok(())
}

/// Decode a binary frame of little-endian i16 mono PCM (drops a trailing odd byte).
fn bytes_to_pcm16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Encode i16 mono PCM into a little-endian byte buffer for a binary WS frame.
fn pcm16_to_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full WS round-trip: client frames come back byte-for-byte through the
    /// server-side `WsTransport` echo loop.
    #[tokio::test]
    async fn ws_loopback_round_trips() {
        run_loopback().await.expect("loopback round-trip");
    }

    #[test]
    fn pcm_byte_roundtrip_is_lossless() {
        let pcm = vec![0i16, 1, -1, i16::MAX, i16::MIN, 12345, -12345];
        assert_eq!(bytes_to_pcm16(&pcm16_to_bytes(&pcm)), pcm);
    }
}

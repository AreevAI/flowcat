// SPDX-License-Identifier: Apache-2.0
//
//! Local mic/speaker transport (CLI demo harness) — **stub**.
//!
//! A [`MediaTransport`](flowcat_core::MediaTransport) over the local audio
//! device, for the CLI demo harness. Behind the `local` feature.
//!
//! ## Why this is a stub
//!
//! A real local device leg needs `cpal` (or similar) to open the host's default
//! input/output streams, which is a **heavy, platform-specific** dependency
//! (CoreAudio / WASAPI / ALSA) that the headless server build must not pull.
//! This module ships the intended *shape* (the [`LocalTransport`] type + its
//! planned construction) without the device backend. Filling it in is a
//! follow-up that adds `cpal` **optional + `local`-gated** and wires the input
//! stream through the [`SourcePump`](flowcat_core::SourcePump) exactly like the
//! WebRTC/SIP legs.
//!
//! The `local` feature still pulls `audiopus`: a local harness typically wants
//! Opus loopback for parity with the WebRTC path, and the codec lives in
//! [`crate::webrtc::opus`] once `webrtc-str0m` is also on.

use async_trait::async_trait;

use flowcat_core::error::FlowcatError;
use flowcat_core::transport::{MediaIn, MediaTransport};
use flowcat_core::types::AudioChunk;

/// Planned local mic/speaker transport. Currently a stub: it carries the carrier
/// rate and the [`MediaTransport`] shape but has no device backend, so
/// [`recv`](MediaTransport::recv) ends immediately and
/// [`send_audio`](MediaTransport::send_audio) is a no-op.
///
/// This exists so downstream code can name the type and the feature compiles; a
/// follow-up adds the `cpal` device streams.
pub struct LocalTransport {
    carrier_rate: u32,
    started: bool,
}

impl LocalTransport {
    /// Build a (stub) local transport at `carrier_rate`. A real implementation
    /// would open the default input/output device streams here.
    pub fn new(carrier_rate: u32) -> Self {
        Self {
            carrier_rate,
            started: false,
        }
    }
}

#[async_trait]
impl MediaTransport for LocalTransport {
    async fn recv(&mut self) -> Option<MediaIn> {
        // Emit StreamStart once for protocol parity, then end (no device yet).
        if !self.started {
            self.started = true;
            return Some(MediaIn::StreamStart {
                call_id: "local".to_string(),
            });
        }
        Some(MediaIn::Stop)
    }

    async fn send_audio(&mut self, _chunk: AudioChunk) -> Result<(), FlowcatError> {
        // No device backend yet — silently accept (the CLI harness can still run
        // the pipeline without audible output).
        Ok(())
    }

    async fn send_clear(&mut self) -> Result<(), FlowcatError> {
        Ok(())
    }

    fn carrier_rate(&self) -> u32 {
        self.carrier_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_emits_streamstart_then_stop() {
        let mut t = LocalTransport::new(16000);
        assert_eq!(t.carrier_rate(), 16000);
        assert!(matches!(t.recv().await, Some(MediaIn::StreamStart { .. })));
        assert_eq!(t.recv().await, Some(MediaIn::Stop));
        // send_audio / send_clear are no-ops that never error.
        t.send_audio(AudioChunk::new(vec![0i16; 16], 16000))
            .await
            .unwrap();
        t.send_clear().await.unwrap();
    }
}

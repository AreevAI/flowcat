// SPDX-License-Identifier: Apache-2.0
//
//! Per-call **live-event** bridge (RTVI → browser `rtf-*` frames).
//!
//! flowcat-core's [`RtviObserver`](flowcat_core::observer::RtviObserver) maps
//! pipeline frames → RTVI messages and hands each to an `RtviSink`. [`RtfSink`] is
//! that sink: it remaps each RTVI message to a browser `{type, payload}` `rtf-*`
//! frame and pushes it onto a per-call channel keyed by a `pc_id`. The
//! [`events_ws`] endpoint streams that channel to the browser so the playground can
//! render the live transcript + tool-call + node-transition markers.
//!
//! The channel is an **unbounded mpsc** whose receiver exists from
//! [`EventRegistry::register`] (called BEFORE the SDP answer is returned), so events
//! published before the browser subscribes are **buffered, not lost**.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::debug;

use flowcat_core::observer::{RtviMessage, RtviSink};
use flowcat_core::{AgentBrain, SessionSource};

use crate::server::AppState;

/// One call's event channel: the single consumer `rx` (taken once by [`events_ws`]).
struct CallChannel {
    rx: Mutex<Option<mpsc::UnboundedReceiver<String>>>,
}

/// Per-call live-event channels keyed by `pc_id`. Held in `AppState`.
#[derive(Default)]
pub struct EventRegistry {
    calls: Mutex<HashMap<String, Arc<CallChannel>>>,
}

impl EventRegistry {
    /// A fresh, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a fresh channel for `pc_id`; returns the publish handle + a
    /// drop-guard that deregisters on call end. Call BEFORE replying with the SDP
    /// answer so no opening event is lost to a subscribe race.
    pub fn register(self: &Arc<Self>, pc_id: &str) -> (CallEvents, RegistryGuard) {
        let (tx, rx) = mpsc::unbounded_channel();
        let ch = Arc::new(CallChannel {
            rx: Mutex::new(Some(rx)),
        });
        self.calls.lock().unwrap().insert(pc_id.to_string(), ch);
        (
            CallEvents { tx },
            RegistryGuard {
                registry: Arc::clone(self),
                pc_id: pc_id.to_string(),
            },
        )
    }

    /// Take the single consumer receiver for `pc_id` (once). `None` if no such call
    /// or it was already taken.
    fn take_receiver(&self, pc_id: &str) -> Option<mpsc::UnboundedReceiver<String>> {
        let calls = self.calls.lock().unwrap();
        let taken = calls.get(pc_id)?.rx.lock().unwrap().take();
        taken
    }

    fn deregister(&self, pc_id: &str) {
        self.calls.lock().unwrap().remove(pc_id);
    }
}

/// Drop-guard: removes a call's channel from the registry when the call ends.
pub struct RegistryGuard {
    registry: Arc<EventRegistry>,
    pc_id: String,
}
impl Drop for RegistryGuard {
    fn drop(&mut self) {
        self.registry.deregister(&self.pc_id);
    }
}

/// Publish handle for one call's live-event channel. Fire-and-forget: a send with
/// no live consumer is buffered (unbounded mpsc) or ignored once the call has ended.
#[derive(Clone)]
pub struct CallEvents {
    tx: mpsc::UnboundedSender<String>,
}
impl CallEvents {
    /// Publish a browser `{type, payload}` `rtf-*` frame.
    pub fn publish(&self, type_: &str, payload: Value) {
        let _ = self
            .tx
            .send(json!({ "type": type_, "payload": payload }).to_string());
    }
}

/// flowcat `RtviSink` impl: turns each RTVI message into a browser `rtf-*` frame on
/// the call channel. `send` is synchronous (the observer calls it inline) — the
/// underlying mpsc send never blocks.
pub struct RtfSink {
    events: CallEvents,
}
impl RtfSink {
    /// Wrap a call's publish handle as an `RtviSink`.
    pub fn new(events: CallEvents) -> Self {
        Self { events }
    }
}
impl RtviSink for RtfSink {
    fn send(&self, message: RtviMessage) {
        if let Some((ty, payload)) = map_rtvi_to_rtf(&message) {
            self.events.publish(ty, payload);
        }
    }
}

/// Map an RTVI message → the browser `rtf-*` `{type, payload}` the playground
/// renders. Only the known set is forwarded; everything else is dropped. `data` is
/// passed through (the UI reads the fields it needs and ignores extras).
///
/// `rtf-bot-text` comes from EXACTLY ONE kind — `bot-transcription` — so the
/// realtime and cascaded paths each produce it once, with no doubling from
/// `bot-tts-text`/`bot-output` (deliberately unmapped).
fn map_rtvi_to_rtf(m: &RtviMessage) -> Option<(&'static str, Value)> {
    let payload = m.data.clone().unwrap_or_else(|| json!({}));
    let ty = match m.kind {
        "user-transcription" => "rtf-user-transcription", // {text, final}
        "bot-transcription" => "rtf-bot-text",            // {text}
        "bot-started-speaking" => "rtf-bot-started-speaking",
        "bot-stopped-speaking" => "rtf-bot-stopped-speaking",
        "user-mute-started" => "rtf-user-mute-started",
        "user-mute-stopped" => "rtf-user-mute-stopped",
        "llm-function-call-started" | "llm-function-call-in-progress" => "rtf-function-call-start",
        "llm-function-call-stopped" => "rtf-function-call-end",
        _ => return None,
    };
    Some((ty, payload))
}

/// `GET /webrtc/events/{pc_id}` — stream a call's `rtf-*` frames to the browser.
/// Serves only a registered `pc_id` (unknown → 404). Generic over the embedder's
/// session/brain so it shares the one [`AppState`] the rest of the router uses.
pub async fn events_ws<S, B>(
    State(state): State<AppState<S, B>>,
    Path(pc_id): Path<String>,
    ws: WebSocketUpgrade,
) -> Response
where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
{
    let Some(rx) = state.events.take_receiver(&pc_id) else {
        return (StatusCode::NOT_FOUND, "no such call").into_response();
    };
    debug!(pc_id = %pc_id, "live events subscribed");
    ws.on_upgrade(move |socket| stream_events(socket, rx))
}

async fn stream_events(mut socket: WebSocket, mut rx: mpsc::UnboundedReceiver<String>) {
    while let Some(frame) = rx.recv().await {
        if socket.send(Message::Text(frame.into())).await.is_err() {
            break; // subscriber gone
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(kind: &'static str, data: Option<Value>) -> RtviMessage {
        RtviMessage {
            label: flowcat_core::observer::RTVI_MESSAGE_LABEL,
            kind,
            data,
        }
    }

    #[test]
    fn registry_register_subscribe_publish_and_drop() {
        let reg = Arc::new(EventRegistry::new());
        let (events, guard) = reg.register("pc-1");
        let mut rx = reg.take_receiver("pc-1").expect("receiver");
        // Single consumer: taking again yields None.
        assert!(reg.take_receiver("pc-1").is_none());
        // Publish reaches the consumer.
        events.publish(
            "rtf-user-transcription",
            json!({ "text": "hi", "final": true }),
        );
        let got = rx.try_recv().expect("a frame");
        assert!(got.contains("rtf-user-transcription") && got.contains("\"final\":true"));
        // Drop the guard → the call is deregistered.
        drop(guard);
        assert!(reg.take_receiver("pc-1").is_none());
    }

    #[test]
    fn map_rtvi_to_rtf_matches_the_browser_contract() {
        assert_eq!(
            map_rtvi_to_rtf(&msg(
                "user-transcription",
                Some(json!({"text":"hi","final":true}))
            ))
            .unwrap()
            .0,
            "rtf-user-transcription"
        );
        assert_eq!(
            map_rtvi_to_rtf(&msg("bot-transcription", Some(json!({"text":"hello"}))))
                .unwrap()
                .0,
            "rtf-bot-text"
        );
        assert_eq!(
            map_rtvi_to_rtf(&msg(
                "llm-function-call-in-progress",
                Some(json!({"tool_call_id":"t1"}))
            ))
            .unwrap()
            .0,
            "rtf-function-call-start"
        );
        assert_eq!(
            map_rtvi_to_rtf(&msg(
                "llm-function-call-stopped",
                Some(json!({"tool_call_id":"t1"}))
            ))
            .unwrap()
            .0,
            "rtf-function-call-end"
        );
        // Unmapped kinds are dropped (no double bot bubble, no noise).
        assert!(map_rtvi_to_rtf(&msg("bot-tts-text", Some(json!({"text":"x"})))).is_none());
        assert!(map_rtvi_to_rtf(&msg("metrics", None)).is_none());
    }
}

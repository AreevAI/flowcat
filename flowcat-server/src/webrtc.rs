// SPDX-License-Identifier: Apache-2.0
//
//! Browser WebRTC playground: `GET /` serves the page, `POST /webrtc/offer`
//! accepts a browser SDP offer and runs the configured agent over a str0m
//! [`WebRtcTransport`], and the live transcript streams over
//! `/webrtc/events/{pc_id}` (see [`crate::events`]).
//!
//! This is the "talk to your agent in the browser" path: no control plane, no
//! credentials in the page — the server runs the single configured agent and the
//! browser is just a mic + speaker + transcript view.

use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{error, info, warn};

use flowcat_agent::DeclarativeBrain;
use flowcat_core::observer::{FrameObserver, RtviObserver, RtviSink};
use flowcat_transports::WebRtcTransport;

use crate::events::RtfSink;
use crate::run;
use crate::server::AppState;
use crate::session::StaticSession;

/// str0m carrier rate — matches the realtime input rate (the S2S processors
/// resample to 16 kHz in / 24 kHz out internally).
const WEBRTC_CARRIER_RATE: u32 = 16_000;

/// The static playground page (vanilla JS: mic → WebRTC offer → audio out + a live
/// transcript over the events WS). Embedded so the binary is self-contained.
const PLAYGROUND_HTML: &str = include_str!("playground.html");

/// Browser SDP offer.
#[derive(Debug, Deserialize)]
pub struct OfferRequest {
    /// The browser's SDP offer (post-ICE-gathering, so it carries candidates).
    pub sdp: String,
}

/// SDP answer + the per-call id the browser uses to subscribe to live events.
#[derive(Debug, Serialize)]
pub struct OfferResponse {
    /// The str0m SDP answer.
    pub sdp: String,
    /// Per-call id (`pc-<n>`); the browser opens `/webrtc/events/{pc_id}`.
    pub pc_id: String,
}

/// `GET /` — the browser playground page.
pub async fn playground_page() -> Html<&'static str> {
    Html(PLAYGROUND_HTML)
}

/// `POST /webrtc/offer` — accept the browser SDP offer and run the configured agent.
pub async fn offer(State(state): State<AppState>, Json(body): Json<OfferRequest>) -> Response {
    // Build the brain from the configured graph BEFORE accepting the offer, so a
    // bad graph is a clean 422 with no peer created.
    let brain = match DeclarativeBrain::new(
        state.graph.as_ref(),
        Value::Object(Default::default()),
        state.config.agent.seed_vars.clone(),
    ) {
        Ok(b) => b,
        Err(e) => {
            warn!(error = %e, "webrtc offer: invalid agent graph");
            return (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()).into_response();
        }
    };

    // Bind a UDP socket for the WebRTC media on the configured interface (str0m
    // advertises this as the host ICE candidate and rejects 0.0.0.0).
    let bind = SocketAddr::new(IpAddr::V4(state.webrtc_bind_ip), 0);
    let socket = match tokio::net::UdpSocket::bind(bind).await {
        Ok(s) => s,
        Err(e) => {
            error!(error = %e, "webrtc offer: failed to bind UDP socket");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to bind a media socket",
            )
                .into_response();
        }
    };

    // Accept the browser offer → the str0m transport + the SDP answer.
    let (transport, answer) =
        match WebRtcTransport::accept_offer(&body.sdp, socket, WEBRTC_CARRIER_RATE) {
            Ok(pair) => pair,
            Err(e) => {
                warn!(error = %e, "webrtc offer: accept_offer failed");
                return (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()).into_response();
            }
        };

    let run_id = state.next_pc.fetch_add(1, Ordering::Relaxed) as i64;
    let pc_id = format!("pc-{run_id}");

    // Register the live-event channel BEFORE returning the answer so opening
    // markers aren't lost to a subscribe race.
    let (call_events, guard) = state.events.register(&pc_id);
    let sink: Arc<dyn RtviSink> = Arc::new(RtfSink::new(call_events));
    let observers: Vec<Arc<dyn FrameObserver>> = vec![Arc::new(RtviObserver::new(sink))];

    let session = StaticSession::new(
        (*state.graph).clone(),
        state.config.agent.seed_vars.clone(),
        "webrtc",
    );
    let topology = state.config.topology.clone();
    info!(pc_id = %pc_id, "webrtc offer accepted; running call detached");
    let call_pc = pc_id.clone();
    tokio::spawn(async move {
        let _guard = guard; // deregister the live-event channel on call end
        let res = run::run_call(
            transport,
            &topology,
            brain,
            session,
            run_id,
            String::new(),
            observers,
        )
        .await;
        match res {
            Ok(()) => info!(pc_id = %call_pc, "webrtc call ended cleanly"),
            Err(e) => error!(pc_id = %call_pc, error = %e, "webrtc call ended with error"),
        }
    });

    Json(OfferResponse { sdp: answer, pc_id }).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use crate::server::{build_router, AppState};
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn state() -> AppState {
        let config = ServerConfig::parse(
            r#"{ "agent": { "graph_inline": {"nodes":[{"id":"s","type":"startCall","data":{"prompt":"hi"}},{"id":"e","type":"endCall"}],"edges":[{"id":"x","source":"s","target":"e","label":"done"}]} },
                 "topology": { "mode": "realtime", "provider": "gemini" } }"#,
            false,
        )
        .unwrap();
        let graph = config.resolve_graph(std::path::Path::new(".")).unwrap();
        AppState::new(config, graph, None)
    }

    #[tokio::test]
    async fn playground_page_serves_html() {
        let app = build_router(state());
        let resp = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let html = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            html.contains("/webrtc/offer"),
            "page must POST to /webrtc/offer"
        );
        assert!(html.contains("getUserMedia"), "page must capture the mic");
    }

    #[tokio::test]
    async fn malformed_offer_is_422() {
        let app = build_router(state());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/webrtc/offer")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"sdp":"not-a-valid-sdp"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        // A graph this small is valid, so we get past the brain build; the junk SDP
        // fails str0m's accept_offer → 422 (client error), no peer created.
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
}

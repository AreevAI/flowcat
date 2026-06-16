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

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use flowcat_core::observer::{FrameObserver, RtviObserver, RtviSink};
use flowcat_core::{AgentBrain, FlowcatError, SessionSource};
use flowcat_transports::WebRtcTransport;

use crate::config::TopologyConfig;
use crate::events::RtfSink;
use crate::run::{self, SpecResolver};
use crate::server::{resolve_brain, AppState};

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

/// Inputs for [`handle_offer`], grouped so the helper stays under the
/// argument-count lint. Generic over the brain `B`, session `S`, and a `keepalive`
/// value `G` the spawned call owns for its lifetime.
pub struct OfferParams<B, S, G> {
    /// The browser's SDP offer (post-ICE-gathering, so it carries candidates).
    pub sdp: String,
    /// Concrete IPv4 the media socket binds (str0m advertises it as the host ICE
    /// candidate and rejects 0.0.0.0); the caller chooses the interface — security.
    pub bind_ip: Ipv4Addr,
    /// Pipeline-facing carrier sample rate the str0m transport resamples to.
    pub carrier_rate: u32,
    /// Which providers to run + how their specs (keys) resolve.
    pub topology: TopologyConfig,
    /// Resolves each provider leg's spec (see [`SpecResolver`]).
    pub resolver: SpecResolver,
    /// The conversation brain for this call.
    pub brain: B,
    /// The session bootstrap/finalize source.
    pub session: S,
    /// Run id passed to the pipeline.
    pub run_id: i64,
    /// Per-call token passed to the pipeline (empty for the playground).
    pub token: String,
    /// Pipeline observers (e.g. an `RtviObserver` bridging live events).
    pub observers: Vec<Arc<dyn FrameObserver>>,
    /// A value held by the spawned call and dropped when it ends (e.g. an events
    /// channel drop-guard). Pass `()` if there is nothing to hold.
    pub keepalive: G,
}

/// Accept a browser SDP **offer**, bind a media socket, build the str0m transport,
/// spawn the call detached, and return the **SDP answer** to send back.
///
/// This is the reusable "browser offer → audio session" signaling primitive: it
/// owns the offer/answer, the UDP bind, the str0m transport, and the call spawn —
/// but NOT the HTTP request/response shapes or auth (the caller frames those).
/// flowcat-server's own [`offer`] is a thin wrapper over it; an external server
/// calls it the same way with its own session + brain.
///
/// Errors before the spawn surface to the caller: a bind failure is
/// [`FlowcatError::Io`]; a malformed/unacceptable offer is [`FlowcatError::Protocol`]
/// (or another transport error). The call itself runs detached — once the answer is
/// returned, a call-time failure is logged, not returned.
pub async fn handle_offer<B, S, G>(params: OfferParams<B, S, G>) -> Result<String, FlowcatError>
where
    B: AgentBrain + 'static,
    S: SessionSource + 'static,
    G: Send + 'static,
{
    let OfferParams {
        sdp,
        bind_ip,
        carrier_rate,
        topology,
        resolver,
        brain,
        session,
        run_id,
        token,
        observers,
        keepalive,
    } = params;

    // Bind the media socket on the chosen interface (str0m advertises it as the
    // host ICE candidate and rejects 0.0.0.0). An io error here is the caller's 5xx.
    let bind = SocketAddr::new(IpAddr::V4(bind_ip), 0);
    let socket = tokio::net::UdpSocket::bind(bind).await?;

    // Accept the offer → the str0m transport + the SDP answer. A bad offer is an Err
    // with no peer created.
    let (transport, answer) = WebRtcTransport::accept_offer(&sdp, socket, carrier_rate)?;

    // Run the call detached; the answer goes back to the browser now.
    info!(run_id, "webrtc offer accepted; running call detached");
    tokio::spawn(async move {
        let _keepalive = keepalive; // held until call end (e.g. deregisters events)
        let res = run::run_call_with(
            transport, &topology, &*resolver, brain, session, run_id, token, observers,
        )
        .await;
        match res {
            Ok(()) => info!(run_id, "webrtc call ended cleanly"),
            Err(e) => error!(run_id, error = %e, "webrtc call ended with error"),
        }
    });

    Ok(answer)
}

/// `POST /webrtc/offer` — accept the browser SDP offer and run the configured agent.
///
/// Generic over the embedder's session/brain: the call is resolved through the
/// session and the brain built from it (via the shared [`resolve_brain`]) BEFORE a
/// peer is created, so a bad resolve/graph is a clean error with no media bound.
pub async fn offer<S, B>(
    State(state): State<AppState<S, B>>,
    Json(body): Json<OfferRequest>,
) -> Response
where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
{
    // The playground assigns the per-call id up front so `resolve` can key off it;
    // the WebRTC path carries no carrier token.
    let run_id = state.next_pc.fetch_add(1, Ordering::Relaxed) as i64;
    let pc_id = format!("pc-{run_id}");

    // Resolve + build the brain BEFORE accepting the offer, so a bad resolve/graph
    // is a clean error with no peer created.
    let brain = match resolve_brain(&state, run_id, "").await {
        Ok(b) => b,
        Err(r) => return r,
    };

    // Register the live-event channel BEFORE returning the answer so opening
    // markers aren't lost to a subscribe race; the guard rides the call as its
    // `keepalive` and deregisters the channel on call end.
    let (call_events, guard) = state.events.register(&pc_id);
    let sink: Arc<dyn RtviSink> = Arc::new(RtfSink::new(call_events));
    let observers: Vec<Arc<dyn FrameObserver>> = vec![Arc::new(RtviObserver::new(sink))];

    // The reusable signaling primitive owns the offer/answer + bind + transport +
    // spawn; this handler only frames the HTTP request/response (and the events WS).
    let answer = handle_offer(OfferParams {
        sdp: body.sdp,
        bind_ip: state.webrtc_bind_ip,
        carrier_rate: WEBRTC_CARRIER_RATE,
        topology: (*state.topology).clone(),
        resolver: Arc::clone(&state.spec_resolver),
        brain,
        session: Arc::clone(&state.session),
        run_id,
        token: String::new(),
        observers,
        keepalive: guard,
    })
    .await;

    match answer {
        Ok(sdp) => Json(OfferResponse { sdp, pc_id }).into_response(),
        // A bind failure is a server error; a bad/unacceptable offer is the client's.
        Err(e @ FlowcatError::Io(_)) => {
            error!(pc_id = %pc_id, error = %e, "webrtc offer: failed to bind media socket");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to bind a media socket",
            )
                .into_response()
        }
        Err(e) => {
            warn!(pc_id = %pc_id, error = %e, "webrtc offer: rejected");
            (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use crate::server::{build_router, AppState};
    use crate::session::StaticSession;
    use axum::body::Body;
    use axum::http::Request;
    use flowcat_agent::DeclarativeBrain;
    use tower::ServiceExt;

    fn state() -> AppState<StaticSession, DeclarativeBrain> {
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
        // fails str0m's accept_offer → 422 (client error), no peer created. (The
        // handler now delegates to `handle_offer`, which surfaces the Protocol error
        // it maps to 422.)
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    // --- The reusable signaling helper, called directly (the external-server path). ---

    /// A realistic (minimal) browser-style SDP offer negotiating Opus, with the ICE
    /// + DTLS lines a real offer carries — close enough for str0m to accept it.
    fn sample_browser_offer() -> String {
        "v=0\r\n\
         o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
         s=-\r\n\
         t=0 0\r\n\
         a=group:BUNDLE 0\r\n\
         a=msid-semantic: WMS\r\n\
         m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
         c=IN IP4 127.0.0.1\r\n\
         a=rtcp:9 IN IP4 0.0.0.0\r\n\
         a=ice-ufrag:F7gI\r\n\
         a=ice-pwd:x9cml/YzichV2+XlhiMu8g\r\n\
         a=ice-options:trickle\r\n\
         a=fingerprint:sha-256 \
         AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89\r\n\
         a=setup:actpass\r\n\
         a=mid:0\r\n\
         a=sendrecv\r\n\
         a=rtcp-mux\r\n\
         a=rtpmap:111 opus/48000/2\r\n\
         a=fmtp:111 minptime=10;useinbandfec=1\r\n"
            .to_string()
    }

    /// Build the brain + session an external caller would hand the helper. The call
    /// it spawns will fail to connect (no provider key) — fine, that's detached; the
    /// signaling test only asserts on the returned SDP answer.
    fn offer_params(sdp: String) -> OfferParams<DeclarativeBrain, StaticSession, ()> {
        let graph = serde_json::json!({
            "nodes": [{ "id": "s", "type": "startCall", "data": { "prompt": "hi" } }],
            "edges": []
        });
        let brain =
            DeclarativeBrain::new(&graph, serde_json::json!({}), Default::default()).unwrap();
        let session = StaticSession::new(graph, Default::default(), "test");
        OfferParams {
            sdp,
            bind_ip: std::net::Ipv4Addr::LOCALHOST,
            carrier_rate: WEBRTC_CARRIER_RATE,
            topology: TopologyConfig::Realtime {
                provider: "gemini".into(),
                model: String::new(),
                options: Default::default(),
            },
            resolver: Arc::new(run::env_spec_resolver),
            brain,
            session,
            run_id: 1,
            token: String::new(),
            observers: vec![],
            keepalive: (),
        }
    }

    #[tokio::test]
    async fn handle_offer_accepts_a_browser_offer_and_returns_an_answer() {
        // The external-server path: build the helper's inputs ourselves (no AppState)
        // and get a valid SDP answer back — a working str0m audio session at the
        // signaling boundary.
        let answer = handle_offer(offer_params(sample_browser_offer()))
            .await
            .expect("a valid offer yields an answer");
        assert!(answer.starts_with("v=0"), "answer is SDP: {answer}");
        assert!(answer.contains("m=audio"), "answer has the audio m-line");
        assert!(
            answer.to_ascii_lowercase().contains("opus"),
            "answer negotiates opus: {answer}"
        );
        assert!(
            answer.contains("a=fingerprint:"),
            "answer carries the DTLS-SRTP fingerprint"
        );
    }

    #[tokio::test]
    async fn handle_offer_rejects_a_malformed_offer() {
        // Same helper, bad SDP → a clean Protocol error (no panic, no peer), which
        // the HTTP wrapper maps to 422.
        let err = handle_offer(offer_params("not-a-valid-sdp".into()))
            .await
            .unwrap_err();
        assert!(
            matches!(err, FlowcatError::Protocol(_)),
            "malformed offer is a protocol error, got: {err:?}"
        );
    }
}

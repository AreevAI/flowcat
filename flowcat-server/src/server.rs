// SPDX-License-Identifier: Apache-2.0
//
//! The axum HTTP server: health probes, the Plivo media WebSocket, and the Plivo
//! answer XML. Single-agent and **unauthenticated** by design (it serves one
//! configured agent with no control plane) — front it with your own ingress/auth
//! for anything public.

use std::sync::Arc;

use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{error, info, warn};

use flowcat_agent::DeclarativeBrain;
use flowcat_core::{PlivoSerializer, WsCarrierTransport};

use crate::config::{ServerConfig, TopologyConfig};
use crate::run;
use crate::session::StaticSession;
use crate::socket::AxumWsSocket;

/// Telephony carrier μ-law @ 8 kHz (the Plivo WS media format).
const CARRIER_RATE: u32 = 8_000;

/// Shared handler state: the resolved config + the agent graph (resolved once at
/// startup) + this host's public base URL (for the Plivo answer XML).
#[derive(Clone)]
pub struct AppState {
    pub(crate) config: Arc<ServerConfig>,
    pub(crate) graph: Arc<Value>,
    pub(crate) public_url: Arc<Option<String>>,
    /// Per-call live-event channels for the WebRTC playground.
    #[cfg(feature = "webrtc")]
    pub(crate) events: Arc<crate::events::EventRegistry>,
    /// Monotonic per-call id source (`pc-<n>`).
    #[cfg(feature = "webrtc")]
    pub(crate) next_pc: Arc<std::sync::atomic::AtomicU64>,
    /// Concrete IPv4 the str0m media socket binds (str0m rejects 0.0.0.0); from
    /// `FLOWCAT_WEBRTC_BIND_IP`, default loopback.
    #[cfg(feature = "webrtc")]
    pub(crate) webrtc_bind_ip: std::net::Ipv4Addr,
}

impl AppState {
    /// Build the shared state from a loaded config + its resolved graph spec.
    pub fn new(config: ServerConfig, graph: Value, public_url: Option<String>) -> Self {
        Self {
            config: Arc::new(config),
            graph: Arc::new(graph),
            public_url: Arc::new(public_url),
            #[cfg(feature = "webrtc")]
            events: Arc::new(crate::events::EventRegistry::new()),
            #[cfg(feature = "webrtc")]
            next_pc: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            #[cfg(feature = "webrtc")]
            webrtc_bind_ip: webrtc_bind_ip_from_env(),
        }
    }
}

/// Resolve the WebRTC media bind IP from `FLOWCAT_WEBRTC_BIND_IP` (default
/// `127.0.0.1`; str0m advertises it as the host ICE candidate and rejects 0.0.0.0).
#[cfg(feature = "webrtc")]
fn webrtc_bind_ip_from_env() -> std::net::Ipv4Addr {
    std::env::var("FLOWCAT_WEBRTC_BIND_IP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(std::net::Ipv4Addr::LOCALHOST)
}

/// Assemble the axum router over the shared [`AppState`].
pub fn build_router(state: AppState) -> Router {
    let router = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/telephony/ws/{provider}/{run_id}", get(media_ws))
        .route("/telephony/answer/plivo/{run_id}", get(answer_plivo));
    // The browser playground (page + WebRTC offer + live-events WS).
    #[cfg(feature = "webrtc")]
    let router = router
        .route("/", get(crate::webrtc::playground_page))
        .route("/webrtc/offer", axum::routing::post(crate::webrtc::offer))
        .route("/webrtc/events/{pc_id}", get(crate::events::events_ws));
    router.with_state(state)
}

async fn healthz() -> impl IntoResponse {
    axum::Json(json!({ "status": "ok" }))
}

async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    let ready = primary_provider_key_present(&state.config);
    let code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, axum::Json(json!({ "ready": ready })))
}

/// True iff the configured topology's primary provider has an API key in the env
/// (realtime: the realtime provider; cascaded: the LLM leg).
fn primary_provider_key_present(config: &ServerConfig) -> bool {
    let provider = match &config.topology {
        TopologyConfig::Realtime { provider, .. } => provider.as_str(),
        TopologyConfig::Cascaded { llm, .. } => llm.provider.as_str(),
    };
    !run::key_from_env(provider).is_empty()
}

#[derive(Deserialize)]
struct TokenQuery {
    #[serde(default)]
    token: String,
}

/// `GET /telephony/ws/{provider}/{run_id}?token=` — the bidirectional Plivo media
/// WS. The brain is built before the upgrade so a bad graph is a clean 422.
async fn media_ws(
    State(state): State<AppState>,
    Path((provider, run_id)): Path<(String, i64)>,
    Query(q): Query<TokenQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    let provider = provider.to_ascii_lowercase();
    if provider != "plivo" {
        warn!(%provider, "media ws: only the 'plivo' carrier is served by this endpoint");
        return (StatusCode::NOT_FOUND, "unsupported carrier (only 'plivo')").into_response();
    }

    let brain = match DeclarativeBrain::new(
        state.graph.as_ref(),
        Value::Object(Default::default()),
        state.config.agent.seed_vars.clone(),
    ) {
        Ok(b) => b,
        Err(e) => {
            warn!(error = %e, "media ws: invalid agent graph");
            return (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()).into_response();
        }
    };

    let token = q.token;
    info!(run_id, "plivo media ws upgrading");
    ws.on_upgrade(move |socket| async move {
        let transport = WsCarrierTransport::new(
            AxumWsSocket::new(socket),
            PlivoSerializer::new(CARRIER_RATE),
        );
        let session = StaticSession::new(
            (*state.graph).clone(),
            state.config.agent.seed_vars.clone(),
            "plivo",
        );
        let res = run::run_call(
            transport,
            &state.config.topology,
            brain,
            session,
            run_id,
            token,
            vec![],
        )
        .await;
        match res {
            Ok(()) => info!(run_id, "plivo call ended cleanly"),
            Err(e) => error!(run_id, error = %e, "plivo call ended with error"),
        }
    })
}

/// `GET /telephony/answer/plivo/{run_id}?token=` — the Plivo `<Stream>` answer XML
/// pointing back at this host's media WS. Needs `FLOWCAT_PUBLIC_URL` to build a
/// reachable `wss://` URL.
async fn answer_plivo(
    State(state): State<AppState>,
    Path(run_id): Path<i64>,
    Query(q): Query<TokenQuery>,
) -> Response {
    let Some(public_base) = state.public_url.as_deref().filter(|s| !s.is_empty()) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "FLOWCAT_PUBLIC_URL not set — cannot build a reachable wss:// answer URL",
        )
            .into_response();
    };
    let xml = flowcat_telephony::plivo_answer_xml(run_id, &q.token, public_base);
    ([(axum::http::header::CONTENT_TYPE, "text/xml")], xml).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // oneshot

    fn test_state(public_url: Option<String>) -> AppState {
        let config = ServerConfig::parse(
            r#"{ "agent": { "graph_inline": {"nodes":[{"id":"s","type":"startCall"}],"edges":[]} },
                 "topology": { "mode": "realtime", "provider": "gemini" } }"#,
            false,
        )
        .unwrap();
        let graph = config.resolve_graph(std::path::Path::new(".")).unwrap();
        AppState::new(config, graph, public_url)
    }

    #[tokio::test]
    async fn healthz_returns_ok() {
        let app = build_router(test_state(None));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn answer_plivo_needs_public_url() {
        // No FLOWCAT_PUBLIC_URL configured → 503 with a clear message.
        let app = build_router(test_state(None));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/telephony/answer/plivo/1?token=t")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn answer_plivo_emits_stream_xml_when_public_url_set() {
        let app = build_router(test_state(Some("https://voice.example.com".to_string())));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/telephony/answer/plivo/42?token=tok")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let xml = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            xml.contains("wss://voice.example.com/telephony/ws/plivo/42?token=tok"),
            "{xml}"
        );
    }

    // The non-plivo carrier rejection inside `media_ws` runs *after* axum's
    // `WebSocketUpgrade` extractor, so it can't be reached with a plain (non-WS)
    // `oneshot` request — it would need a real WebSocket handshake. The provider
    // check itself is a simple string compare; the WS handshake path is covered by
    // an end-to-end call rather than a router unit test.
}

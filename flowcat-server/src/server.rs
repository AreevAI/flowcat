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
use flowcat_core::{
    AgentBrain, FlowcatError, PlivoSerializer, ResolvedCall, SessionSource, WsCarrierTransport,
};

use crate::config::{ServerConfig, TopologyConfig};
use crate::run::{self, SpecResolver};
use crate::session::StaticSession;
use crate::socket::AxumWsSocket;

/// Telephony carrier μ-law @ 8 kHz (the Plivo WS media format).
const CARRIER_RATE: u32 = 8_000;

/// Per-call brain constructor: builds a fresh [`AgentBrain`] from the
/// [`ResolvedCall`] a [`SessionSource`] returns. The brain is per-call (it carries
/// the run's conversation state), so an embedder supplies a factory rather than a
/// single brain instance — the default wiring's factory builds a
/// [`DeclarativeBrain`] from the resolved graph spec.
pub type BrainFactory<B> = Arc<dyn Fn(&ResolvedCall) -> Result<B, FlowcatError> + Send + Sync>;

/// Shared handler state for the HTTP + transport surface.
///
/// Generic over the embedder's [`SessionSource`] `S` and [`AgentBrain`] `B`: it
/// holds **one** shared session (cloned cheaply per call) plus a per-call
/// [`BrainFactory`], the pipeline [`TopologyConfig`], and this host's public base
/// URL (for the Plivo answer XML). The zero-config default — [`StaticSession`] +
/// [`DeclarativeBrain`] for the playground and `--config agent.yaml` — is built by
/// [`AppState::new`]; an embedder injects its own pair via [`AppState::with_parts`].
pub struct AppState<S, B> {
    pub(crate) session: Arc<S>,
    pub(crate) brain_factory: BrainFactory<B>,
    pub(crate) topology: Arc<TopologyConfig>,
    /// Resolves each call's provider specs (keys). Defaults to the env resolver;
    /// an embedder overrides it via [`AppState::with_spec_resolver`].
    pub(crate) spec_resolver: SpecResolver,
    pub(crate) public_url: Arc<Option<String>>,
    /// Per-call live-event channels for the WebRTC playground.
    #[cfg(feature = "webrtc-helper")]
    pub(crate) events: Arc<crate::events::EventRegistry>,
    /// Monotonic per-call id source (`pc-<n>`).
    #[cfg(feature = "webrtc-helper")]
    pub(crate) next_pc: Arc<std::sync::atomic::AtomicU64>,
    /// IPv4 the str0m media socket binds; from `FLOWCAT_WEBRTC_BIND_IP`, default
    /// loopback. May be `0.0.0.0` when `webrtc_advertise_ip` supplies the candidate.
    #[cfg(feature = "webrtc-helper")]
    pub(crate) webrtc_bind_ip: std::net::Ipv4Addr,
    /// IPv4 advertised as the host ICE candidate (decoupled from the bind addr);
    /// from `FLOWCAT_WEBRTC_ADVERTISE_IP`, `None` → advertise the bound addr.
    #[cfg(feature = "webrtc-helper")]
    pub(crate) webrtc_advertise_ip: Option<std::net::Ipv4Addr>,
}

// Hand-written so the bound is on the `Arc`s we actually hold, not on `S`/`B`
// (a derive would wrongly require `S: Clone, B: Clone`).
impl<S, B> Clone for AppState<S, B> {
    fn clone(&self) -> Self {
        Self {
            session: Arc::clone(&self.session),
            brain_factory: Arc::clone(&self.brain_factory),
            topology: Arc::clone(&self.topology),
            spec_resolver: Arc::clone(&self.spec_resolver),
            public_url: Arc::clone(&self.public_url),
            #[cfg(feature = "webrtc-helper")]
            events: Arc::clone(&self.events),
            #[cfg(feature = "webrtc-helper")]
            next_pc: Arc::clone(&self.next_pc),
            #[cfg(feature = "webrtc-helper")]
            webrtc_bind_ip: self.webrtc_bind_ip,
            #[cfg(feature = "webrtc-helper")]
            webrtc_advertise_ip: self.webrtc_advertise_ip,
        }
    }
}

impl<S, B> AppState<S, B> {
    /// Build the shared state from an embedder-supplied session + per-call brain
    /// factory + pipeline topology. This is the injection point a platform uses to
    /// run flowcat-server's HTTP/transport surface with its own control plane; the
    /// playground/binary default is [`AppState::new`].
    pub fn with_parts(
        session: Arc<S>,
        brain_factory: BrainFactory<B>,
        topology: TopologyConfig,
        public_url: Option<String>,
    ) -> Self {
        Self {
            session,
            brain_factory,
            topology: Arc::new(topology),
            // Default: resolve provider keys from the env (the standalone-server
            // convention). An embedder swaps in its own via `with_spec_resolver`.
            spec_resolver: Arc::new(run::env_spec_resolver),
            public_url: Arc::new(public_url),
            #[cfg(feature = "webrtc-helper")]
            events: Arc::new(crate::events::EventRegistry::new()),
            #[cfg(feature = "webrtc-helper")]
            next_pc: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            #[cfg(feature = "webrtc-helper")]
            webrtc_bind_ip: webrtc_bind_ip_from_env(),
            #[cfg(feature = "webrtc-helper")]
            webrtc_advertise_ip: webrtc_advertise_ip_from_env(),
        }
    }

    /// Override how provider specs (API keys) are resolved per call — a platform
    /// passes its own secret-store lookup so no provider key is read from the
    /// process env on the call path. Defaults to [`run::env_spec_resolver`].
    pub fn with_spec_resolver(mut self, resolver: SpecResolver) -> Self {
        self.spec_resolver = resolver;
        self
    }
}

impl AppState<StaticSession, DeclarativeBrain> {
    /// Build the **default** shared state — [`StaticSession`] + [`DeclarativeBrain`]
    /// — from a loaded config + its resolved graph spec. The session serves the one
    /// configured agent (no control plane); the brain factory builds a declarative
    /// brain from the resolved graph for each call.
    pub fn new(config: ServerConfig, graph: Value, public_url: Option<String>) -> Self {
        let ServerConfig {
            agent,
            topology,
            transport,
            ..
        } = config;
        let session = Arc::new(StaticSession::new(graph, agent.seed_vars, transport.kind));
        let brain_factory: BrainFactory<DeclarativeBrain> = Arc::new(|resolved: &ResolvedCall| {
            DeclarativeBrain::from_config(&resolved.brain_config)
                .map_err(|e| FlowcatError::Other(format!("invalid agent graph: {e}")))
        });
        Self::with_parts(session, brain_factory, topology, public_url)
    }
}

/// Resolve the WebRTC media bind IP from `FLOWCAT_WEBRTC_BIND_IP` (default
/// `127.0.0.1`). May be `0.0.0.0` when `FLOWCAT_WEBRTC_ADVERTISE_IP` is set.
#[cfg(feature = "webrtc-helper")]
fn webrtc_bind_ip_from_env() -> std::net::Ipv4Addr {
    std::env::var("FLOWCAT_WEBRTC_BIND_IP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(std::net::Ipv4Addr::LOCALHOST)
}

/// Resolve the advertised host-ICE-candidate IP from `FLOWCAT_WEBRTC_ADVERTISE_IP`
/// (decoupled from the bind IP); `None` → advertise the bound addr.
#[cfg(feature = "webrtc-helper")]
fn webrtc_advertise_ip_from_env() -> Option<std::net::Ipv4Addr> {
    std::env::var("FLOWCAT_WEBRTC_ADVERTISE_IP")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

/// Assemble the axum router over the shared [`AppState`].
///
/// Generic over the embedder's session + brain types so the **same** router,
/// `media_ws`, and events WS serve the default playground wiring and any injected
/// [`SessionSource`] + [`AgentBrain`] pair alike.
pub fn build_router<S, B>(state: AppState<S, B>) -> Router
where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
{
    let router = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz::<S, B>))
        .route("/telephony/ws/{provider}/{run_id}", get(media_ws::<S, B>))
        .route(
            "/telephony/answer/plivo/{run_id}",
            get(answer_plivo::<S, B>),
        );
    // The browser playground (page + WebRTC offer + live-events WS).
    #[cfg(feature = "webrtc-helper")]
    let router = router
        .route("/", get(crate::webrtc::playground_page))
        .route(
            "/webrtc/offer",
            axum::routing::post(crate::webrtc::offer::<S, B>),
        )
        .route(
            "/webrtc/events/{pc_id}",
            get(crate::events::events_ws::<S, B>),
        );
    router.with_state(state)
}

async fn healthz() -> impl IntoResponse {
    axum::Json(json!({ "status": "ok" }))
}

async fn readyz<S, B>(State(state): State<AppState<S, B>>) -> impl IntoResponse
where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
{
    let ready = primary_provider_key_present(&state.topology);
    let code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, axum::Json(json!({ "ready": ready })))
}

/// True iff the topology's primary provider has an API key in the env (realtime:
/// the realtime provider; cascaded: the LLM leg).
fn primary_provider_key_present(topology: &TopologyConfig) -> bool {
    let provider = match topology {
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
/// WS. The call is resolved through the session and the brain built from it before
/// the upgrade, so a bad resolve/graph is a clean error with no socket accepted.
async fn media_ws<S, B>(
    State(state): State<AppState<S, B>>,
    Path((provider, run_id)): Path<(String, i64)>,
    Query(q): Query<TokenQuery>,
    ws: WebSocketUpgrade,
) -> Response
where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
{
    let provider = provider.to_ascii_lowercase();
    if provider != "plivo" {
        warn!(%provider, "media ws: only the 'plivo' carrier is served by this endpoint");
        return (StatusCode::NOT_FOUND, "unsupported carrier (only 'plivo')").into_response();
    }

    let brain = match resolve_brain(&state, run_id, &q.token).await {
        Ok(b) => b,
        Err(r) => return r,
    };

    let token = q.token;
    let session = Arc::clone(&state.session);
    let topology = Arc::clone(&state.topology);
    let resolver = Arc::clone(&state.spec_resolver);
    info!(run_id, "plivo media ws upgrading");
    ws.on_upgrade(move |socket| async move {
        let transport = WsCarrierTransport::new(
            AxumWsSocket::new(socket),
            PlivoSerializer::new(CARRIER_RATE),
        );
        let res = run::run_call_with(
            transport,
            &topology,
            &*resolver,
            brain,
            session,
            run_id,
            token,
            run::context_relay_from_env(),
            vec![],
        )
        .await;
        match res {
            Ok(()) => info!(run_id, "plivo call ended cleanly"),
            Err(e) => error!(run_id, error = %e, "plivo call ended with error"),
        }
    })
}

/// Resolve a call through the shared session and build its brain via the factory.
///
/// This is the per-call bootstrap shared by every transport (`media_ws` and the
/// WebRTC playground): `resolve` lets a custom control-plane session pick the
/// per-run graph (and reject an already-completed run), and the factory turns that
/// resolved config into the brain. On any failure it returns the [`Response`] to
/// send (the caller `?`-style early-returns it), so the socket/peer is never set up.
pub(crate) async fn resolve_brain<S, B>(
    state: &AppState<S, B>,
    run_id: i64,
    token: &str,
) -> Result<B, Response>
where
    S: SessionSource,
{
    let resolved = match state.session.resolve(run_id, token).await {
        Ok(r) => r,
        Err(e) => {
            warn!(run_id, error = %e, "session resolve failed");
            return Err((StatusCode::BAD_GATEWAY, e.to_string()).into_response());
        }
    };
    if resolved.is_completed {
        warn!(run_id, "run already completed; refusing to start");
        return Err((StatusCode::CONFLICT, "run already completed").into_response());
    }
    (state.brain_factory)(&resolved).map_err(|e| {
        warn!(run_id, error = %e, "brain factory rejected the resolved call");
        (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()).into_response()
    })
}

/// `GET /telephony/answer/plivo/{run_id}?token=` — the Plivo `<Stream>` answer XML
/// pointing back at this host's media WS. Needs `FLOWCAT_PUBLIC_URL` to build a
/// reachable `wss://` URL.
async fn answer_plivo<S, B>(
    State(state): State<AppState<S, B>>,
    Path(run_id): Path<i64>,
    Query(q): Query<TokenQuery>,
) -> Response
where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
{
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

    fn test_state(public_url: Option<String>) -> AppState<StaticSession, DeclarativeBrain> {
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

    // --- Injectable framework: a custom SessionSource + AgentBrain in the SAME
    // router/media_ws, with no StaticSession/DeclarativeBrain in sight. -----------

    use async_trait::async_trait;
    use flowcat_core::session::{Finalize, ToolDecl, UploadTarget};
    use flowcat_core::types::BrainAction;
    use serde_json::json;

    /// A control-plane stand-in: `resolve` can fail, mark the run completed, or
    /// return a per-run `brain_config` the brain factory reads.
    struct FakeSession {
        fail: bool,
        completed: bool,
        prompt: String,
    }

    #[async_trait]
    impl SessionSource for FakeSession {
        async fn resolve(&self, _run_id: i64, _token: &str) -> Result<ResolvedCall, FlowcatError> {
            if self.fail {
                return Err(FlowcatError::Session("control plane unreachable".into()));
            }
            Ok(ResolvedCall {
                provider: "fake".into(),
                brain_config: json!({ "prompt": self.prompt }),
                is_completed: self.completed,
            })
        }
        async fn complete(&self, _: i64, _: &str, _: Finalize) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn artifact_upload_url(
            &self,
            _: i64,
            _: &str,
            _: &str,
        ) -> Result<UploadTarget, FlowcatError> {
            Err(FlowcatError::Session("no artifacts".into()))
        }
        async fn put_bytes(&self, _: &str, _: Vec<u8>, _: &str) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn node_tools(
            &self,
            _: i64,
            _: &str,
            _: &str,
        ) -> Result<Vec<ToolDecl>, FlowcatError> {
            Ok(vec![])
        }
        async fn tool_call(
            &self,
            _: i64,
            _: &str,
            _: &str,
            name: &str,
            _: &Value,
        ) -> Result<String, FlowcatError> {
            Ok(name.to_string())
        }
    }

    /// A hand-rolled brain (not the DeclarativeBrain) whose prompt is taken from the
    /// resolved `brain_config`, so a test can prove the factory saw the session's
    /// output.
    #[derive(Debug)]
    struct EchoBrain {
        prompt: String,
    }
    impl AgentBrain for EchoBrain {
        fn system_prompt(&self) -> String {
            self.prompt.clone()
        }
        fn tools(&self) -> Vec<ToolDecl> {
            vec![]
        }
        fn current_node_id(&self) -> String {
            "echo".into()
        }
        fn on_tool_call(&mut self, _name: &str, _args: &Value) -> BrainAction {
            BrainAction::Stay
        }
        fn is_finished(&self) -> bool {
            false
        }
        fn collected_vars(&self) -> Value {
            json!({})
        }
    }

    fn fake_state(session: FakeSession) -> AppState<FakeSession, EchoBrain> {
        let factory: BrainFactory<EchoBrain> = Arc::new(|resolved: &ResolvedCall| {
            Ok(EchoBrain {
                prompt: resolved.brain_config["prompt"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            })
        });
        let topology = TopologyConfig::Realtime {
            provider: "gemini".into(),
            model: String::new(),
            options: Default::default(),
        };
        AppState::with_parts(Arc::new(session), factory, topology, None)
    }

    #[tokio::test]
    async fn custom_session_and_brain_serve_the_same_router() {
        // The whole point: build_router over an embedder's own pair, no fork.
        let app = build_router(fake_state(FakeSession {
            fail: false,
            completed: false,
            prompt: "hi from the control plane".into(),
        }));
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
    async fn resolve_brain_routes_through_the_injected_session() {
        let state = fake_state(FakeSession {
            fail: false,
            completed: false,
            prompt: "prompt-from-resolve".into(),
        });
        // resolve() → brain_config → factory: the brain reflects the session's output.
        let brain = resolve_brain(&state, 1, "tok").await.expect("a brain");
        assert_eq!(brain.system_prompt(), "prompt-from-resolve");
    }

    #[tokio::test]
    async fn resolve_brain_rejects_an_already_completed_run() {
        let state = fake_state(FakeSession {
            fail: false,
            completed: true,
            prompt: "x".into(),
        });
        let resp = resolve_brain(&state, 1, "tok").await.unwrap_err();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn resolve_brain_surfaces_a_resolve_failure_as_bad_gateway() {
        let state = fake_state(FakeSession {
            fail: true,
            completed: false,
            prompt: "x".into(),
        });
        let resp = resolve_brain(&state, 1, "tok").await.unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }
}

// SPDX-License-Identifier: Apache-2.0
//
//! Generic SIP inbound/originate orchestration — the SIP analogue of the carrier
//! media-WS path (`media_ws` + [`run_call_with`](crate::run::run_call_with)).
//!
//! flowcat-core exposes the SIP *primitives* — [`SipAgent`], [`InboundInvite`],
//! `SipTransport` — but every embedder that wants inbound or outbound SIP otherwise
//! hand-writes the same loop around them: an inbound pump (`while let Some(invite) =
//! agent.next_inbound().await { … }`, then per-INVITE resolve → answer → run) and an
//! outbound originate (resolve → `agent.originate` → run detached). That loop is
//! identical across embedders; only "map a dialed identifier to a call" and the
//! per-call brain construction are embedder-specific — exactly the seams the WS path
//! already abstracts.
//!
//! This module is that loop, parameterised the same way the WS path is — over the
//! embedder's [`SessionSource`] + a [`BrainFactory`] + a per-call [`TopologyResolver`]
//! (or a static [`TopologyConfig`] for the standalone trunk) + a
//! [`SpecResolver`] (bundled into [`SipOrchestrator`]) —
//! plus one SIP-specific seam, [`SipInboundResolver`], that maps a dialed identifier
//! to the run it should drive (the analogue of the run id + token the media-WS URL
//! carries).
//!
//! - [`serve_sip_inbound`] — the inbound-INVITE pump: per INVITE, resolve the dialed
//!   identity → a [`SipRun`], build the brain, **answer**, and run the call via the
//!   same [`run_call_with`](crate::run::run_call_with) the WS path uses; **reject**
//!   the INVITE on any pre-answer error (fail-closed).
//! - [`sip_originate`] — the outbound driver: resolve the run's brain, dial, and run
//!   the answered call detached.
//!
//! Embedder-specific policy (how a DID maps to a run, destination/auth checks) lives
//! behind [`SipInboundResolver`] / the [`SessionSource`], mirroring how the WS path
//! keeps resolve + brain construction behind those seams. Gated behind
//! `server-helper` (framework glue, no connector bundle).

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{error, info, warn};

use flowcat_core::{
    AgentBrain, FlowcatError, InboundInvite, ResolvedCall, SessionSource, SipAgent,
};

use crate::config::TopologyConfig;
use crate::run::{self, SpecResolver};
use crate::server::BrainFactory;

/// The run an inbound INVITE maps to (and the run an outbound originate already
/// knows): the `{run_id, token}` pair the shared [`SessionSource`] resolves into the
/// call config. The SIP analogue of the run id + token the carrier media-WS URL
/// carries.
#[derive(Debug, Clone)]
pub struct SipRun {
    /// Run id passed to the session resolve and the pipeline.
    pub run_id: i64,
    /// Per-call token passed to the session resolve and the pipeline.
    pub token: String,
}

/// Maps an inbound SIP INVITE to the run that should answer it — the embedder's
/// control-plane policy.
///
/// Turn the dialed DID + caller (read from the SIP headers via [`InboundInvite`],
/// **never** the INVITE body — see SIP-DESIGN.md §"Security") into the [`SipRun`] the
/// shared [`SessionSource`] resolves. Returning `Err` **rejects** the INVITE (the
/// helper sends a SIP failure response and never answers), so an unknown DID, a
/// failed auth/binding check, or an unreachable control plane all fail the call
/// **closed**. This is the SIP counterpart of taking the run id + token off the
/// media-WS URL.
#[async_trait]
pub trait SipInboundResolver: Send + Sync {
    /// Resolve the dialed identity to the run that should drive the call, or `Err`
    /// to reject the INVITE.
    async fn resolve_inbound(&self, invite: &InboundInvite) -> Result<SipRun, FlowcatError>;
}

/// Resolves a call's pipeline [`TopologyConfig`] from its [`ResolvedCall`] — the
/// topology analogue of [`BrainFactory`], run over the *same* resolved call so an
/// embedder can pin the pipeline mode/provider per run (realtime vs cascaded, a
/// specific provider/model) from the resolved call's options, exactly as the brain
/// is already chosen per-call. The WebRTC path takes the topology as a per-call
/// parameter ([`OfferParams`](crate::webrtc::OfferParams)); this is how the SIP
/// drivers reach the same per-call selection. The static-topology standalone case
/// uses [`SipOrchestrator::new`], which wraps a constant in one of these.
pub type TopologyResolver =
    Arc<dyn Fn(&ResolvedCall) -> Result<TopologyConfig, FlowcatError> + Send + Sync>;

/// The reusable per-trunk wiring both SIP drivers share: the embedder's session +
/// per-call [`BrainFactory`] + per-call [`TopologyResolver`] + provider-[`SpecResolver`].
/// Mirrors the fields
/// [`AppState`](crate::server::AppState) holds for the WS/WebRTC path, minus the
/// HTTP-only bits. Cheap to clone (every field is an `Arc`), so each inbound call
/// gets its own clone for its task.
pub struct SipOrchestrator<S, B> {
    session: Arc<S>,
    brain_factory: BrainFactory<B>,
    topology_resolver: TopologyResolver,
    resolver: SpecResolver,
}

// Hand-written so the bound is on the `Arc`s we actually hold, not on `S`/`B` (a
// derive would wrongly require `S: Clone, B: Clone` — same reason `AppState` does).
impl<S, B> Clone for SipOrchestrator<S, B> {
    fn clone(&self) -> Self {
        Self {
            session: Arc::clone(&self.session),
            brain_factory: Arc::clone(&self.brain_factory),
            topology_resolver: Arc::clone(&self.topology_resolver),
            resolver: Arc::clone(&self.resolver),
        }
    }
}

impl<S, B> SipOrchestrator<S, B>
where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
{
    /// Bundle the per-trunk wiring with a **single static** topology — the
    /// standalone-server case, where one config-file [`TopologyConfig`] drives the
    /// whole trunk. Wraps that constant in a [`TopologyResolver`] that ignores the
    /// resolved call; use [`with_topology_resolver`](Self::with_topology_resolver) to
    /// pick the topology per-call instead.
    ///
    /// `resolver` resolves each provider leg's spec (keys): pass
    /// `Arc::new(run::env_spec_resolver)` to read keys from the env (the
    /// standalone-server convention), or an embedder's own secret-store lookup.
    pub fn new(
        session: Arc<S>,
        brain_factory: BrainFactory<B>,
        topology: TopologyConfig,
        resolver: SpecResolver,
    ) -> Self {
        let topology = Arc::new(topology);
        Self::with_topology_resolver(
            session,
            brain_factory,
            Arc::new(move |_: &ResolvedCall| Ok((*topology).clone())),
            resolver,
        )
    }

    /// Bundle the per-trunk wiring, resolving the pipeline [`TopologyConfig`]
    /// **per-call** from the [`ResolvedCall`] (the same resolved call the
    /// [`BrainFactory`] runs over). Lets an embedder pin the pipeline mode/provider
    /// per run — realtime vs cascaded, or a specific provider/model — from the
    /// resolved call's options, bringing the SIP path to parity with the WebRTC
    /// [`handle_offer`](crate::webrtc::handle_offer)'s per-call topology parameter.
    pub fn with_topology_resolver(
        session: Arc<S>,
        brain_factory: BrainFactory<B>,
        topology_resolver: TopologyResolver,
        resolver: SpecResolver,
    ) -> Self {
        Self {
            session,
            brain_factory,
            topology_resolver,
            resolver,
        }
    }

    /// Resolve a run's brain **and** pipeline topology through the session, factory,
    /// and topology resolver (shared by both drivers).
    ///
    /// One `session.resolve` feeds both the [`BrainFactory`] and the
    /// [`TopologyResolver`], so the brain and topology are chosen from the *same*
    /// [`ResolvedCall`] (no second round-trip, no chance of the two disagreeing).
    /// Mirrors the WS path's `resolve_brain` but returns a plain [`FlowcatError`]
    /// (SIP has no HTTP status to map to): a failed resolve, an already-completed
    /// run, or a rejected graph/topology all surface as `Err` so the caller fails the
    /// call **closed** (rejects the INVITE / aborts the originate).
    async fn resolve_call(&self, run: &SipRun) -> Result<(B, TopologyConfig), FlowcatError> {
        let resolved = self.session.resolve(run.run_id, &run.token).await?;
        if resolved.is_completed {
            return Err(FlowcatError::Other(format!(
                "run {} already completed; refusing to start",
                run.run_id
            )));
        }
        let brain = (self.brain_factory)(&resolved)?;
        let topology = (self.topology_resolver)(&resolved)?;
        Ok((brain, topology))
    }
}

/// Run the inbound-INVITE pump until the [`SipAgent`] shuts down.
///
/// For each INVITE: ask `inbound_resolver` to map the dialed identity to a
/// [`SipRun`], build the brain through the session/factory, **answer**, and run the
/// call to completion via [`run_call_with`](crate::run::run_call_with). Any
/// pre-answer failure (resolve, brain build, or the answer transaction itself)
/// **rejects** the INVITE — fail-closed — and no call runs. Each call is handled on
/// its own task, so a slow or long call never blocks the next INVITE.
///
/// `agent` is an [`Arc`] so the embedder can keep its own clone to
/// [`originate`](SipAgent::originate) / [`shutdown`](SipAgent::shutdown) the trunk
/// while this pump runs (typically spawned as a long-lived task). Returns when
/// [`SipAgent::next_inbound`] yields `None` (the agent was shut down).
///
/// Pre-answer rejections use the default SIP failure response (Busy Here); the
/// helper deliberately does not disclose whether a DID exists. An embedder that
/// needs a specific reject code can pre-screen in its resolver / front this with its
/// own logic.
pub async fn serve_sip_inbound<S, B, R>(
    agent: Arc<SipAgent>,
    orchestrator: SipOrchestrator<S, B>,
    inbound_resolver: Arc<R>,
) where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
    R: SipInboundResolver + 'static,
{
    while let Some(invite) = agent.next_inbound().await {
        let orchestrator = orchestrator.clone();
        let inbound_resolver = Arc::clone(&inbound_resolver);
        tokio::spawn(async move {
            handle_inbound_invite(orchestrator, inbound_resolver, invite).await;
        });
    }
    info!("SIP inbound pump ended (agent shut down)");
}

/// Drive one inbound INVITE: resolve → brain → answer → run, rejecting on any
/// pre-answer error. Factored out of the pump's `spawn` so the per-call flow is
/// unit-testable on its own.
async fn handle_inbound_invite<S, B, R>(
    orchestrator: SipOrchestrator<S, B>,
    inbound_resolver: Arc<R>,
    invite: InboundInvite,
) where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
    R: SipInboundResolver + 'static,
{
    let call_id = invite.call_id.clone();

    // 1. Map the dialed identity (DID/caller, from the SIP headers) → a run.
    let run = match inbound_resolver.resolve_inbound(&invite).await {
        Ok(run) => run,
        Err(e) => {
            warn!(%call_id, error = %e, "SIP inbound resolve rejected the INVITE");
            invite.reject(None); // default Busy Here; fail-closed, no DID disclosure
            return;
        }
    };

    // 2. Build the brain + resolve the topology through the session/factory/resolver
    //    BEFORE answering, so a bad resolve/graph/topology rejects the INVITE rather
    //    than answering a call we can't run. Topology is resolved per-call from the
    //    same ResolvedCall as the brain.
    let (brain, topology) = match orchestrator.resolve_call(&run).await {
        Ok(resolved) => resolved,
        Err(e) => {
            warn!(%call_id, run_id = run.run_id, error = %e, "SIP brain build rejected the INVITE");
            invite.reject(None);
            return;
        }
    };

    // 3. Answer (200 OK with our SDP) → the media transport. Past this point the
    //    call is committed; an answer failure has nothing left to reject.
    let transport = match invite.answer().await {
        Ok(transport) => transport,
        Err(e) => {
            error!(%call_id, run_id = run.run_id, error = %e, "SIP answer failed");
            return;
        }
    };

    // 4. Run the call to completion over the same orchestration the WS path uses.
    info!(%call_id, run_id = run.run_id, "SIP inbound call answered; running");
    let SipRun { run_id, token } = run;
    let res = run::run_call_with(
        transport,
        &topology,
        &*orchestrator.resolver,
        brain,
        Arc::clone(&orchestrator.session),
        run_id,
        token,
        run::context_relay_from_env(),
        vec![],
    )
    .await;
    match res {
        Ok(()) => info!(%call_id, run_id, "SIP inbound call ended cleanly"),
        Err(e) => error!(%call_id, run_id, error = %e, "SIP inbound call ended with error"),
    }
}

/// Originate an outbound SIP call for an already-known run and run it detached.
///
/// Resolves the run's brain through the session/factory, places the INVITE to
/// `dest` (an E.164) with an optional `caller_id` override (else the trunk's
/// configured caller-id), and — once the callee answers — spawns the call over the
/// same orchestration the inbound/WS paths use. Returns once the call is
/// **answered** (the run is then in flight on its own task); a pre-answer failure
/// (resolve, brain build, or an unanswered/declined INVITE) is returned as `Err`
/// with no task spawned.
///
/// Destination validation (E.164 shape, allowed-destination / auth checks) is the
/// embedder's responsibility before calling this — mirroring how the inbound
/// resolver owns inbound policy (SIP-DESIGN.md §3/§"Security").
pub async fn sip_originate<S, B>(
    agent: &SipAgent,
    orchestrator: &SipOrchestrator<S, B>,
    run: SipRun,
    dest: &str,
    caller_id: Option<&str>,
) -> Result<(), FlowcatError>
where
    S: SessionSource + 'static,
    B: AgentBrain + 'static,
{
    // Build the brain + resolve the topology before dialing so a bad
    // resolve/graph/topology fails fast (no INVITE). Topology is resolved per-call
    // from the same ResolvedCall as the brain.
    let (brain, topology) = orchestrator.resolve_call(&run).await?;

    // Dial; returns once the callee answers (200 OK) with the media transport.
    let transport = agent.originate(dest, caller_id).await?;

    info!(run_id = run.run_id, %dest, "SIP outbound call answered; running detached");
    let orchestrator = orchestrator.clone();
    let SipRun { run_id, token } = run;
    tokio::spawn(async move {
        let res = run::run_call_with(
            transport,
            &topology,
            &*orchestrator.resolver,
            brain,
            Arc::clone(&orchestrator.session),
            run_id,
            token,
            run::context_relay_from_env(),
            vec![],
        )
        .await;
        match res {
            Ok(()) => info!(run_id, "SIP outbound call ended cleanly"),
            Err(e) => error!(run_id, error = %e, "SIP outbound call ended with error"),
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::net::Ipv4Addr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use serde_json::{json, Value};
    use tokio::net::UdpSocket;

    use flowcat_core::session::{Finalize, ToolDecl, UploadTarget};
    use flowcat_core::types::BrainAction;
    use flowcat_core::{ResolvedCall, SipConfig};

    // ── Test doubles (a control-plane stand-in + a hand-rolled brain), close to the
    //    ones in `server.rs` tests but local so this module's tests stand alone. ──

    /// A control-plane stand-in: counts `resolve` calls and returns a per-run prompt
    /// the brain factory reads, so a test can prove the orchestrator routed through
    /// the injected session.
    struct FakeSession {
        prompt: String,
        resolves: AtomicUsize,
    }

    impl FakeSession {
        fn new(prompt: &str) -> Arc<Self> {
            Arc::new(Self {
                prompt: prompt.to_string(),
                resolves: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl SessionSource for FakeSession {
        async fn resolve(&self, _run_id: i64, _token: &str) -> Result<ResolvedCall, FlowcatError> {
            self.resolves.fetch_add(1, Ordering::SeqCst);
            Ok(ResolvedCall {
                provider: "fake".into(),
                brain_config: json!({ "prompt": self.prompt }),
                is_completed: false,
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

    /// A minimal hand-rolled brain (not the DeclarativeBrain) whose prompt comes from
    /// the resolved `brain_config`, so the factory's output is observable.
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

    fn orchestrator(session: Arc<FakeSession>) -> SipOrchestrator<FakeSession, EchoBrain> {
        let factory: BrainFactory<EchoBrain> = Arc::new(|resolved: &ResolvedCall| {
            Ok(EchoBrain {
                prompt: resolved.brain_config["prompt"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            })
        });
        // Gemini realtime: with no API key the factory fails fast (no network), so
        // the call answered below ends immediately on its detached task — fine, the
        // tests assert only on the signaling (answer / reject) the orchestrator owns.
        let topology = TopologyConfig::Realtime {
            provider: "gemini".into(),
            model: String::new(),
            options: Default::default(),
        };
        SipOrchestrator::new(session, factory, topology, Arc::new(run::env_spec_resolver))
    }

    /// An inbound resolver that maps every INVITE to a fixed run.
    struct AcceptInbound {
        run_id: i64,
    }
    #[async_trait]
    impl SipInboundResolver for AcceptInbound {
        async fn resolve_inbound(&self, _invite: &InboundInvite) -> Result<SipRun, FlowcatError> {
            Ok(SipRun {
                run_id: self.run_id,
                token: "tok".into(),
            })
        }
    }

    /// An inbound resolver that rejects every INVITE (e.g. unknown DID).
    struct RejectInbound;
    #[async_trait]
    impl SipInboundResolver for RejectInbound {
        async fn resolve_inbound(&self, _invite: &InboundInvite) -> Result<SipRun, FlowcatError> {
            Err(FlowcatError::Session("unknown DID".into()))
        }
    }

    // ── Loopback SIP harness (hermetic: real loopback UDP, no registrar / creds —
    //    the same approach as flowcat-core's `agent.rs` tests). ───────────────────

    async fn free_udp_port() -> u16 {
        let s = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let p = s.local_addr().unwrap().port();
        drop(s);
        p
    }

    /// Start a credential-less loopback agent on `sip_port`, advertising 127.0.0.1.
    async fn start_loopback_agent(server: String, sip_port: u16) -> Arc<SipAgent> {
        let cfg = SipConfig {
            server,
            login: String::new(),
            password: String::new(), // empty → no REGISTER loop, no registrar needed
            caller_id: "1000".to_string(),
            public_ip: Some(Ipv4Addr::LOCALHOST),
            sip_port: Some(sip_port),
            rtp_port_base: 41000,
            rtp_port_tries: 200,
        };
        Arc::new(SipAgent::start(cfg).await.expect("agent start"))
    }

    /// Start a callee (running `serve_sip_inbound`) + a caller wired to dial it.
    async fn start_callee_and_caller<R>(
        session: Arc<FakeSession>,
        inbound_resolver: Arc<R>,
    ) -> (Arc<SipAgent>, Arc<SipAgent>)
    where
        R: SipInboundResolver + 'static,
    {
        let port_callee = free_udp_port().await;
        let callee = start_loopback_agent("sip:127.0.0.1:1".to_string(), port_callee).await;
        tokio::spawn(serve_sip_inbound(
            Arc::clone(&callee),
            orchestrator(session),
            inbound_resolver,
        ));

        let port_caller = free_udp_port().await;
        let caller =
            start_loopback_agent(format!("sip:127.0.0.1:{port_callee}"), port_caller).await;
        (callee, caller)
    }

    #[tokio::test]
    async fn serve_sip_inbound_answers_a_resolved_invite() {
        let session = FakeSession::new("hi from the control plane");
        let (callee, caller) =
            start_callee_and_caller(Arc::clone(&session), Arc::new(AcceptInbound { run_id: 42 }))
                .await;

        // The caller dials; `serve_sip_inbound` resolves → builds the brain → answers.
        // A returned media transport means the INVITE was answered (200 OK).
        let dialed = tokio::time::timeout(Duration::from_secs(5), caller.originate("2000", None))
            .await
            .expect("originate timed out");
        // (`SipTransport` isn't `Debug`, so surface the error side, not the Ok value.)
        if let Err(e) = &dialed {
            panic!("a resolved inbound INVITE must be answered, got error: {e}");
        }
        // The orchestrator routed through the injected session at least once.
        assert!(session.resolves.load(Ordering::SeqCst) >= 1);

        caller.shutdown().await;
        callee.shutdown().await;
    }

    #[tokio::test]
    async fn serve_sip_inbound_rejects_when_the_resolver_errs() {
        let session = FakeSession::new("unused");
        let (callee, caller) =
            start_callee_and_caller(Arc::clone(&session), Arc::new(RejectInbound)).await;

        // The resolver rejects (unknown DID) → the INVITE is rejected, so the caller's
        // originate fails (no media transport) rather than connecting.
        let dialed = tokio::time::timeout(Duration::from_secs(5), caller.originate("2000", None))
            .await
            .expect("originate timed out");
        assert!(
            dialed.is_err(),
            "a rejected inbound INVITE must not yield a media transport"
        );

        caller.shutdown().await;
        callee.shutdown().await;
    }

    #[tokio::test]
    async fn sip_originate_dials_resolves_and_runs() {
        // Callee answers the next INVITE raw (no orchestrator) so this isolates the
        // outbound driver; assert it dialed the right DID.
        let port_callee = free_udp_port().await;
        let callee = start_loopback_agent("sip:127.0.0.1:1".to_string(), port_callee).await;
        let callee_for_answer = Arc::clone(&callee);
        let answer = tokio::spawn(async move {
            let invite =
                tokio::time::timeout(Duration::from_secs(5), callee_for_answer.next_inbound())
                    .await
                    .expect("inbound INVITE timed out")
                    .expect("callee shut down before INVITE");
            assert_eq!(
                invite.to_did, "2000",
                "dialed DID comes from the SIP To user"
            );
            // Hold the transport briefly so the dialog stays up past the caller's 200 OK.
            let _transport = invite.answer().await.expect("answer failed");
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        let port_caller = free_udp_port().await;
        let caller =
            start_loopback_agent(format!("sip:127.0.0.1:{port_callee}"), port_caller).await;
        let session = FakeSession::new("outbound prompt");
        let orchestrator = orchestrator(Arc::clone(&session));

        let run = SipRun {
            run_id: 7,
            token: "tok".into(),
        };
        let originated = tokio::time::timeout(
            Duration::from_secs(5),
            sip_originate(&caller, &orchestrator, run, "2000", None),
        )
        .await
        .expect("sip_originate timed out");
        assert!(
            originated.is_ok(),
            "originate to an answering callee must succeed, got {originated:?}"
        );
        // The brain was resolved through the injected session before dialing.
        assert_eq!(session.resolves.load(Ordering::SeqCst), 1);

        answer.await.expect("answer task panicked");
        caller.shutdown().await;
        callee.shutdown().await;
    }

    #[tokio::test]
    async fn resolve_call_picks_topology_per_call_from_one_resolve() {
        let session = FakeSession::new("hi from the control plane");
        let factory: BrainFactory<EchoBrain> = Arc::new(|resolved: &ResolvedCall| {
            Ok(EchoBrain {
                prompt: resolved.brain_config["prompt"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            })
        });
        // A topology resolver that derives the topology from the resolved call (here:
        // its provider) and counts its invocations, proving per-call selection.
        let topo_calls = Arc::new(AtomicUsize::new(0));
        let tc = Arc::clone(&topo_calls);
        let topology_resolver: TopologyResolver = Arc::new(move |resolved: &ResolvedCall| {
            tc.fetch_add(1, Ordering::SeqCst);
            Ok(TopologyConfig::Realtime {
                provider: resolved.provider.clone(),
                model: String::new(),
                options: Default::default(),
            })
        });
        let orchestrator = SipOrchestrator::with_topology_resolver(
            Arc::clone(&session),
            factory,
            topology_resolver,
            Arc::new(run::env_spec_resolver),
        );

        let (brain, topology) = orchestrator
            .resolve_call(&SipRun {
                run_id: 1,
                token: "tok".into(),
            })
            .await
            .expect("resolve_call");

        // Brain came from the resolved brain_config…
        assert_eq!(brain.system_prompt(), "hi from the control plane");
        // …and the topology was chosen per-call from the *same* ResolvedCall.
        match topology {
            TopologyConfig::Realtime { provider, .. } => assert_eq!(provider, "fake"),
            other => panic!("expected the per-call realtime topology, got {other:?}"),
        }
        // Exactly one session.resolve fed both seams (no second round-trip), and the
        // topology resolver ran once over that resolved call.
        assert_eq!(session.resolves.load(Ordering::SeqCst), 1);
        assert_eq!(topo_calls.load(Ordering::SeqCst), 1);
    }
}

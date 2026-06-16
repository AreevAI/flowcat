// SPDX-License-Identifier: Apache-2.0
//
//! `SipAgent` — a SIP user agent on one trunk (REGISTER + inbound/outbound calls).
//!
//! `SipAgent` wraps `rsipstack`'s endpoint/dialog machinery into the small,
//! Flowcat-shaped surface the voice host needs (see SIP-DESIGN.md §2):
//!
//! - [`SipAgent::start`] binds the SIP UDP transport, starts the endpoint serve
//!   loop, the incoming-INVITE pump, and (if credentials are given) a REGISTER +
//!   periodic-refresh loop — all as background tokio tasks under one cancel token.
//! - [`SipAgent::next_inbound`] yields the next inbound INVITE as an
//!   [`InboundInvite`] (`call_id` / `from` / `to_did`); the host calls
//!   [`InboundInvite::answer`] to 200-OK it with our SDP answer and get a
//!   [`SipTransport`].
//! - [`SipAgent::originate`] places an outbound INVITE to an E.164 with a caller
//!   id, awaits the 200 OK, and returns a [`SipTransport`] over the negotiated
//!   media (rsipstack sends the ACK as part of `do_invite`).
//!
//! ## Signaling vs. media
//!
//! rsipstack owns *signaling only*. For each established call we hand-roll the
//! media: bind a fresh RTP `UdpSocket`, put its address + our G.711 offer/answer
//! in the SDP, parse the peer's SDP for their RTP address + chosen codec, and
//! build a [`SipTransport`] (RTP ↔ `MediaIn`). A per-call supervisor
//! ([`spawn_dialog_supervisor`]) bridges the two teardown directions: a peer BYE
//! / `Terminated` fires the transport's hangup token (surfacing as
//! [`MediaIn::Stop`](crate::transport::MediaIn::Stop)), while the agent ending the
//! call (the `SipTransport` is dropped) makes the supervisor send a BYE to the
//! peer. Either way it removes the dialog from the layer's map so dialogs can't
//! leak across calls.
//!
//! ## What is gated on a live trunk
//!
//! `start` / `register` / `next_inbound` / `originate` open real UDP sockets and
//! speak SIP to a registrar/peer; they are exercised against a live trunk, not in
//! unit tests (there is no registrar here). The deterministic media plumbing they
//! sit on top of — RTP, SDP, the jitter buffer, and the `SipTransport` ↔ `MediaIn`
//! mapping — is unit-tested in `rtp.rs`, `sdp.rs`, and `transport.rs`.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use rsipstack::dialog::authenticate::Credential;
use rsipstack::dialog::client_dialog::ClientInviteDialog;
use rsipstack::dialog::dialog::{DialogState, DialogStateReceiver, DialogStateSender};
use rsipstack::dialog::dialog_layer::DialogLayer;
use rsipstack::dialog::invitation::InviteOption;
use rsipstack::dialog::server_dialog::ServerInviteDialog;
use rsipstack::dialog::DialogId;
use rsipstack::sip as rsip;
use rsipstack::sip::prelude::HeadersExt;
use rsipstack::transaction::endpoint::EndpointInnerRef;
use rsipstack::transaction::TransactionReceiver;
use rsipstack::transport::{udp::UdpConnection, TransportLayer};
use rsipstack::EndpointBuilder;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, watch, Mutex};
use tokio_util::sync::CancellationToken;

use crate::error::FlowcatError;
use crate::sip::sdp;
use crate::sip::transport::SipTransport;

/// Default SIP signaling port if the caller doesn't pick one.
const DEFAULT_SIP_PORT: u16 = 5060;
/// Default first RTP port we try when binding a media socket (even ports per RFC
/// 3550; we scan upward in steps of 2). A wide range so many concurrent calls fit.
/// Overridable per trunk via [`SipConfig::rtp_port_base`].
pub const DEFAULT_RTP_PORT_BASE: u16 = 16000;
/// Default number of even ports to probe before giving up binding an RTP socket.
/// Overridable per trunk via [`SipConfig::rtp_port_tries`]: a deployment whose
/// public UDP port budget is constrained (e.g. behind a GKE LoadBalancer
/// forwarding rule, which is capped at 5 ports) sets this small (e.g. 4) so the
/// bound RTP range fits the exposed ports — at the cost of capping the number of
/// concurrent call media legs to that many.
pub const DEFAULT_RTP_PORT_TRIES: u16 = 200;
/// Floor on the re-REGISTER interval (seconds) regardless of the server's expiry.
const MIN_REREGISTER_SECS: u64 = 30;
/// Consecutive REGISTER failures after which the refresh loop escalates its log
/// from `warn` to `error`: by this point the trunk has been unreachable long
/// enough that inbound calls are being lost, i.e. it is worth alerting on (the
/// loop keeps retrying regardless; see [`SipAgent::is_registered`] to observe it).
const REGISTER_ESCALATE_AFTER: u32 = 3;
/// Time bound for best-effort teardown signaling — the BYE sent on a local hangup
/// and the `Expires: 0` de-REGISTER on a graceful shutdown — so a dead peer or
/// registrar can't pin the task for a full SIP transaction timeout (~32 s).
const SIGNALING_TIMEOUT: Duration = Duration::from_secs(5);

/// Configuration for a [`SipAgent`] (one trunk).
#[derive(Debug, Clone)]
pub struct SipConfig {
    /// Registrar / proxy SIP URI, e.g. `sip:sip.example.com`.
    pub server: String,
    /// SIP auth username (the trunk login).
    pub login: String,
    /// SIP auth password.
    pub password: String,
    /// Caller-ID (E.164 or trunk number) used as the From user on outbound calls.
    pub caller_id: String,
    /// Public IP to advertise in Via/Contact/SDP for NAT (`None` → use the bound
    /// local interface address). Telephony trunks behind NAT need this set.
    pub public_ip: Option<Ipv4Addr>,
    /// Local SIP signaling port to bind (`None` → [`DEFAULT_SIP_PORT`]).
    pub sip_port: Option<u16>,
    /// First RTP port to probe when binding call media (even, RFC 3550; the scan
    /// steps up by 2). Use [`DEFAULT_RTP_PORT_BASE`] unless the deployment pins it.
    pub rtp_port_base: u16,
    /// Number of even ports to probe from `rtp_port_base` before failing to bind.
    /// Use [`DEFAULT_RTP_PORT_TRIES`] unless a constrained public UDP port budget
    /// needs a small range (this caps concurrent call media to `rtp_port_tries`).
    pub rtp_port_tries: u16,
}

/// An inbound INVITE surfaced to the host by [`SipAgent::next_inbound`].
///
/// Carries just what the control plane needs to resolve the call (Call-ID, the
/// caller, the dialed DID), plus the machinery to answer it. The DID/caller are
/// taken from the SIP To/From user parts — the control plane maps the DID to an
/// org/agent; the INVITE body is never trusted for identity (SIP-DESIGN.md §"Security").
pub struct InboundInvite {
    /// SIP Call-ID of the inbound dialog.
    pub call_id: String,
    /// Caller number (From URI user part), best-effort.
    pub from: String,
    /// Dialed DID (To/Request-URI user part) — the number that was called.
    pub to_did: String,
    /// The peer's SDP offer (the INVITE body), already parsed for media params.
    offer: sdp::SdpMedia,
    /// The rsipstack server dialog to accept/reject.
    dialog: ServerInviteDialog,
    /// The dialog layer this dialog lives in, so teardown (answer's supervisor or
    /// `reject`) can remove it from the layer's map — otherwise every inbound call
    /// permanently leaks a dialog there.
    dialog_layer: Arc<DialogLayer>,
    /// IP we advertise in the SDP answer (public IP or local).
    advertise_ip: Ipv4Addr,
    /// RTP bind range (base, count) for the answer's media socket — carried from
    /// the agent's [`SipConfig`] so inbound media honors the same port budget.
    rtp_port_base: u16,
    rtp_port_tries: u16,
    /// State channel for this dialog (watched to drive the hangup token).
    state_rx: DialogStateReceiver,
}

impl InboundInvite {
    /// 200-OK this INVITE with our G.711 SDP answer and return the media transport.
    ///
    /// Binds a fresh RTP socket, builds the answer committing to the codec the
    /// peer offered (PCMU preferred), accepts the dialog, and spins up a
    /// [`SipTransport`] whose hangup token is wired to this dialog's `Terminated`
    /// state. Consumes `self`.
    pub async fn answer(self) -> Result<SipTransport, FlowcatError> {
        let codec = self.offer.codec;
        let (rtp_sock, rtp_port) = bind_rtp_socket(self.rtp_port_base, self.rtp_port_tries).await?;
        let answer_sdp = sdp::build_answer(self.advertise_ip, rtp_port, codec);

        // Peer RTP address from their offer.
        let peer = SocketAddr::new(IpAddr::V4(self.offer.ip), self.offer.port);

        // Accept (sends 200 OK with our SDP answer in the dialog's transaction).
        let headers = vec![rsip::Header::ContentType("application/sdp".into())];
        self.dialog
            .accept(Some(headers), Some(answer_sdp.into_bytes()))
            .map_err(|e| FlowcatError::Transport(format!("SIP accept failed: {e}")))?;

        let hangup = CancellationToken::new();
        spawn_dialog_supervisor(
            DialogHandle::Server(self.dialog),
            self.dialog_layer,
            self.state_rx,
            hangup.clone(),
        );

        Ok(SipTransport::start(
            rtp_sock,
            peer,
            codec,
            self.call_id,
            hangup,
        ))
    }

    /// Reject this INVITE (default 486 Busy Here unless a code is given).
    pub fn reject(self, code: Option<rsip::StatusCode>) {
        let id = self.dialog.id();
        let _ = self
            .dialog
            .reject(Some(code.unwrap_or(rsip::StatusCode::BusyHere)), None);
        // The pump created this server dialog in the layer's map; drop it now so a
        // rejected INVITE can't leak (no supervisor runs for a rejected call).
        self.dialog_layer.remove_dialog(&id);
    }
}

/// A SIP user agent for one trunk. See the module docs.
pub struct SipAgent {
    cfg: SipConfig,
    endpoint_inner: EndpointInnerRef,
    dialog_layer: Arc<DialogLayer>,
    /// Inbound INVITEs from the incoming pump.
    inbound_rx: Mutex<mpsc::Receiver<InboundInvite>>,
    /// IP advertised in SDP (public IP if configured, else the bound local addr).
    advertise_ip: Ipv4Addr,
    /// Live registration state from the refresh loop: `true` while the trunk is
    /// registered, `false` when de-registered or after a failed refresh. Initialized
    /// `true` for a credential-less trunk (nothing to register). Read it via
    /// [`SipAgent::is_registered`] / [`SipAgent::watch_registered`] for health checks.
    registered: watch::Receiver<bool>,
    /// Child cancel token for *just* the REGISTER refresh loop, so
    /// [`SipAgent::unregister`] can stop the auto-refresh (and let an `Expires: 0`
    /// land) without tearing down the still-needed endpoint.
    register_cancel: CancellationToken,
    /// Root cancel token; dropping the agent (or calling [`SipAgent::shutdown`])
    /// tears down the endpoint + all background tasks.
    cancel: CancellationToken,
}

impl SipAgent {
    /// Start the agent: bind the SIP UDP transport, launch the endpoint serve
    /// loop + incoming-INVITE pump (+ registration loop if a password is set).
    ///
    /// Does not block on registration; call [`SipAgent::register`] to await the
    /// first REGISTER result, or just let the background loop keep the binding
    /// fresh. Returns once the transport is bound and tasks are spawned.
    pub async fn start(cfg: SipConfig) -> Result<Self, FlowcatError> {
        let cancel = CancellationToken::new();
        let sip_port = cfg.sip_port.unwrap_or(DEFAULT_SIP_PORT);

        // Bind the SIP signaling UDP socket. Bind to 0.0.0.0 so we receive on all
        // interfaces; advertise the public IP (if given) in Via/Contact via the
        // connection's `external` address.
        let local: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), sip_port);
        let external = cfg
            .public_ip
            .map(|ip| SocketAddr::new(IpAddr::V4(ip), sip_port));

        let transport_layer = TransportLayer::new(cancel.clone());
        let conn = UdpConnection::create_connection(local, external, Some(cancel.child_token()))
            .await
            .map_err(|e| FlowcatError::Transport(format!("bind SIP UDP {local}: {e}")))?;
        transport_layer.add_transport(conn.into());

        let endpoint = EndpointBuilder::new()
            .with_user_agent("flowcat")
            .with_cancel_token(cancel.clone())
            .with_transport_layer(transport_layer)
            .build();

        let endpoint_inner = endpoint.inner.clone();
        let dialog_layer = Arc::new(DialogLayer::new(endpoint.inner.clone()));

        // The IP we put in SDP: the configured public IP, or the bound local addr.
        let advertise_ip = match cfg.public_ip {
            Some(ip) => ip,
            None => local_advertise_ip(&endpoint),
        };

        let incoming = endpoint
            .incoming_transactions()
            .map_err(|e| FlowcatError::Transport(format!("incoming_transactions: {e}")))?;

        // The endpoint serve loop must run for the whole life of the agent.
        let serve_cancel = cancel.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = endpoint.serve() => {}
                _ = serve_cancel.cancelled() => {}
            }
        });

        // Inbound-INVITE pump.
        let (inbound_tx, inbound_rx) = mpsc::channel::<InboundInvite>(16);
        tokio::spawn(incoming_pump(
            dialog_layer.clone(),
            incoming,
            inbound_tx,
            advertise_ip,
            cfg.rtp_port_base,
            cfg.rtp_port_tries,
            cancel.clone(),
        ));

        // Registration state, observable for health checks. A credential-less
        // trunk needs no REGISTER, so it starts (and stays) "registered".
        let (registered_tx, registered_rx) = watch::channel(cfg.password.is_empty());
        // The refresh loop gets its own child token so `unregister` can stop just
        // the loop while the endpoint keeps running long enough to send Expires:0.
        let register_cancel = cancel.child_token();

        // Registration loop (only if a password is configured).
        if !cfg.password.is_empty() {
            tokio::spawn(register_loop(
                endpoint_inner.clone(),
                cfg.clone(),
                register_cancel.clone(),
                registered_tx,
            ));
        }

        Ok(Self {
            cfg,
            endpoint_inner,
            dialog_layer,
            inbound_rx: Mutex::new(inbound_rx),
            advertise_ip,
            registered: registered_rx,
            register_cancel,
            cancel,
        })
    }

    /// Send one REGISTER now and await its result (does not start the refresh
    /// loop — that runs from [`SipAgent::start`]). Useful at bring-up to fail fast
    /// if the trunk credentials are wrong. Returns the registration expiry secs.
    pub async fn register(&self) -> Result<u32, FlowcatError> {
        let credential = Credential {
            username: self.cfg.login.clone(),
            password: self.cfg.password.clone(),
            realm: None,
        };
        let server = parse_server_uri(&self.cfg.server)?;
        let mut reg = rsipstack::dialog::registration::Registration::new(
            self.endpoint_inner.clone(),
            Some(credential),
        );
        let resp = reg
            .register(server, None)
            .await
            .map_err(|e| FlowcatError::Transport(format!("REGISTER failed: {e}")))?;
        if resp.status_code != rsip::StatusCode::OK {
            return Err(FlowcatError::Transport(format!(
                "REGISTER rejected: {}",
                resp.status_code
            )));
        }
        Ok(reg.expires())
    }

    /// Whether the trunk is currently registered (the last REGISTER refresh
    /// succeeded). A credential-less trunk — which needs no registration — always
    /// reports `true`. Use this for a liveness/health probe: a `false` here means
    /// inbound calls will not arrive even though the agent is otherwise running.
    pub fn is_registered(&self) -> bool {
        *self.registered.borrow()
    }

    /// A [`watch::Receiver`] that yields each change to the registration state, so
    /// an embedder can react to de-registration (alert, drain, fail a readiness
    /// check) rather than poll [`SipAgent::is_registered`].
    pub fn watch_registered(&self) -> watch::Receiver<bool> {
        self.registered.clone()
    }

    /// Politely de-register the trunk (a REGISTER with `Expires: 0`) so the
    /// registrar drops our binding immediately instead of holding it until the
    /// granted expiry. Stops the auto-refresh loop first so it can't re-bind us,
    /// then sends the de-REGISTER; call this before [`SipAgent::shutdown`] on a
    /// graceful stop. Best-effort and time-bounded. A no-op without credentials.
    pub async fn unregister(&self) -> Result<(), FlowcatError> {
        // Stop the refresh loop first; otherwise it could REGISTER us straight back
        // in right after the Expires:0 below. The endpoint stays up (root cancel
        // untouched), so we can still send the de-REGISTER.
        self.register_cancel.cancel();
        if self.cfg.password.is_empty() {
            return Ok(());
        }
        let credential = Credential {
            username: self.cfg.login.clone(),
            password: self.cfg.password.clone(),
            realm: None,
        };
        let server = parse_server_uri(&self.cfg.server)?;
        let mut reg = rsipstack::dialog::registration::Registration::new(
            self.endpoint_inner.clone(),
            Some(credential),
        );
        let resp = tokio::time::timeout(SIGNALING_TIMEOUT, reg.register(server, Some(0)))
            .await
            .map_err(|_| FlowcatError::Transport("de-REGISTER timed out".into()))?
            .map_err(|e| FlowcatError::Transport(format!("de-REGISTER failed: {e}")))?;
        if resp.status_code != rsip::StatusCode::OK {
            return Err(FlowcatError::Transport(format!(
                "de-REGISTER rejected: {}",
                resp.status_code
            )));
        }
        Ok(())
    }

    /// Yield the next inbound INVITE, or `None` once the agent is shut down.
    pub async fn next_inbound(&self) -> Option<InboundInvite> {
        self.inbound_rx.lock().await.recv().await
    }

    /// Originate an outbound call to `to_e164` from `caller_id` (overrides the
    /// configured caller-id when given). Awaits the 200 OK and returns the media
    /// transport over the negotiated G.711.
    pub async fn originate(
        &self,
        to_e164: &str,
        caller_id: Option<&str>,
    ) -> Result<SipTransport, FlowcatError> {
        let server = parse_server_uri(&self.cfg.server)?;
        let host = server.host_with_port.clone();
        let caller_user = caller_id.unwrap_or(&self.cfg.caller_id);

        // Build caller / callee / contact URIs against the trunk host.
        let caller = make_uri(caller_user, host.clone());
        let callee = make_uri(to_e164, host.clone());
        let contact = make_uri(caller_user, host.clone());

        // Bind RTP + build our G.711 offer (both PCMU & PCMA).
        let (rtp_sock, rtp_port) =
            bind_rtp_socket(self.cfg.rtp_port_base, self.cfg.rtp_port_tries).await?;
        let offer = sdp::build_offer(self.advertise_ip, rtp_port);

        let credential = Credential {
            username: self.cfg.login.clone(),
            password: self.cfg.password.clone(),
            realm: None,
        };
        let invite = InviteOption {
            caller,
            callee,
            contact,
            content_type: Some("application/sdp".to_string()),
            offer: Some(offer.into_bytes()),
            credential: Some(credential),
            ..Default::default()
        };

        let (state_tx, state_rx) = self.dialog_layer.new_dialog_state_channel();
        let (dialog, resp) = self
            .dialog_layer
            .do_invite(invite, state_tx)
            .await
            .map_err(|e| FlowcatError::Transport(format!("INVITE failed: {e}")))?;
        let resp =
            resp.ok_or_else(|| FlowcatError::Transport("INVITE got no final response".into()))?;
        if resp.status_code != rsip::StatusCode::OK {
            return Err(FlowcatError::Transport(format!(
                "outbound call not answered: {}",
                resp.status_code
            )));
        }

        // Parse the answer SDP for the peer's RTP address + chosen codec.
        let answer_body = String::from_utf8_lossy(resp.body());
        let media = sdp::parse(&answer_body)
            .map_err(|e| FlowcatError::Transport(format!("bad answer SDP: {e}")))?;
        let peer = SocketAddr::new(IpAddr::V4(media.ip), media.port);

        let call_id = dialog.id().call_id.to_string();
        let hangup = CancellationToken::new();
        // Retain the confirmed client dialog in the supervisor: it both lets us
        // BYE the peer when the agent ends the call and removes the dialog from the
        // layer's map on teardown (do_invite leaves the confirmed dialog there).
        spawn_dialog_supervisor(
            DialogHandle::Client(dialog),
            self.dialog_layer.clone(),
            state_rx,
            hangup.clone(),
        );

        Ok(SipTransport::start(
            rtp_sock,
            peer,
            media.codec,
            call_id,
            hangup,
        ))
    }

    /// Gracefully tear down the agent: politely de-register the trunk
    /// ([`SipAgent::unregister`] — best-effort and time-bounded) and then cancel
    /// the endpoint serve loop + all background tasks.
    ///
    /// Prefer this over just dropping the agent: `Drop` is a hard stop that cancels
    /// everything immediately and — being synchronous — cannot send the
    /// `Expires: 0` de-REGISTER, so the registrar would hold a stale binding until
    /// it expires. A de-REGISTER failure here is logged, not fatal; teardown
    /// proceeds regardless.
    pub async fn shutdown(&self) {
        if let Err(e) = self.unregister().await {
            tracing::warn!(error = %e, "SIP de-REGISTER on shutdown failed; tearing down anyway");
        }
        self.cancel.cancel();
    }
}

impl Drop for SipAgent {
    fn drop(&mut self) {
        // Hard stop / safety net: cancel the endpoint + all tasks. Drop is sync, so
        // it cannot de-REGISTER — call `shutdown().await` for a graceful stop that
        // releases the trunk binding first.
        self.cancel.cancel();
    }
}

/// A confirmed INVITE dialog we own for the life of a call: it can be ended (we
/// send a BYE) and must be removed from the dialog layer's map on teardown.
/// Unifies the inbound (server) and outbound (client) dialog types so one
/// supervisor handles both legs.
enum DialogHandle {
    /// Inbound call: we answered a remote INVITE.
    Server(ServerInviteDialog),
    /// Outbound call: we placed the INVITE.
    Client(ClientInviteDialog),
}

impl DialogHandle {
    /// The dialog's id — the key it is stored under in the [`DialogLayer`] map.
    fn id(&self) -> DialogId {
        match self {
            DialogHandle::Server(d) => d.id(),
            DialogHandle::Client(d) => d.id(),
        }
    }

    /// Send a BYE to end the dialog (no-op/`Err` if it is already terminated).
    async fn bye(&self) -> rsipstack::Result<()> {
        match self {
            DialogHandle::Server(d) => d.bye().await,
            DialogHandle::Client(d) => d.bye().await,
        }
    }
}

/// Supervise one established dialog for the whole call, handling teardown from
/// either direction and always releasing the dialog from the layer's map:
///
/// - **Peer-initiated** (BYE / timeout / decline): a `Terminated` dialog state
///   fires `hangup`, so the [`SipTransport`] surfaces
///   [`MediaIn::Stop`](crate::transport::MediaIn::Stop).
/// - **Agent-initiated** (the `Call` ended, so the `SipTransport` was dropped,
///   which cancels `hangup`): we send a BYE to the peer so the dialog doesn't
///   dangle half-open on the carrier until it times out.
///
/// Either way we then [`remove_dialog`](DialogLayer::remove_dialog), so the dialog
/// map can't grow without bound across calls (rsipstack leaves both answered
/// server dialogs and confirmed client dialogs in the map otherwise).
fn spawn_dialog_supervisor(
    handle: DialogHandle,
    dialog_layer: Arc<DialogLayer>,
    mut state_rx: DialogStateReceiver,
    hangup: CancellationToken,
) {
    tokio::spawn(async move {
        let id = handle.id();
        loop {
            tokio::select! {
                // The local side ended the call (SipTransport dropped → hangup
                // cancelled). Tell the peer with a BYE, time-bounded so a dead peer
                // can't pin this task for the full SIP transaction timeout.
                _ = hangup.cancelled() => {
                    match tokio::time::timeout(SIGNALING_TIMEOUT, handle.bye()).await {
                        Ok(Ok(())) => tracing::debug!(%id, "sent BYE on local hangup"),
                        Ok(Err(e)) => {
                            tracing::debug!(%id, error = %e, "BYE on local hangup failed (already terminated?)");
                        }
                        Err(_) => tracing::debug!(%id, "BYE on local hangup timed out"),
                    }
                    break;
                }
                // A dialog state transition from rsipstack.
                state = state_rx.recv() => match state {
                    // The peer (or a timeout) terminated the dialog.
                    Some(DialogState::Terminated(tid, reason)) => {
                        tracing::debug!(%tid, ?reason, "SIP dialog terminated by peer");
                        hangup.cancel();
                        break;
                    }
                    // Early / Confirmed / mid-call states: keep supervising.
                    Some(_) => continue,
                    // Channel closed without an explicit Terminated → treat as hangup.
                    None => {
                        hangup.cancel();
                        break;
                    }
                },
            }
        }
        dialog_layer.remove_dialog(&id);
    });
}

/// The incoming-INVITE pump: matches in-dialog requests to their dialogs, and
/// turns out-of-dialog INVITEs into [`InboundInvite`]s on the channel.
///
/// Modeled on rsipstack's `client` example `process_incoming_request`.
async fn incoming_pump(
    dialog_layer: Arc<DialogLayer>,
    mut incoming: TransactionReceiver,
    inbound_tx: mpsc::Sender<InboundInvite>,
    advertise_ip: Ipv4Addr,
    rtp_port_base: u16,
    rtp_port_tries: u16,
    cancel: CancellationToken,
) {
    loop {
        let mut tx = tokio::select! {
            _ = cancel.cancelled() => break,
            t = incoming.recv() => match t {
                Some(t) => t,
                None => break,
            },
        };

        // In-dialog request (has a To-tag): route to the existing dialog.
        let has_to_tag = tx
            .original
            .to_header()
            .ok()
            .and_then(|to| to.tag().ok().flatten())
            .is_some();
        if has_to_tag {
            if let Some(mut d) = dialog_layer.match_dialog(&tx) {
                tokio::spawn(async move {
                    let _ = d.handle(&mut tx).await;
                });
            } else {
                let _ = tx
                    .reply(rsip::StatusCode::CallTransactionDoesNotExist)
                    .await;
            }
            continue;
        }

        // Out-of-dialog: we only set up new calls on INVITE. ACK for a 2xx is
        // delivered into the dialog's own handler; everything else gets a 200.
        match tx.original.method {
            rsip::Method::Invite => {
                if let Err(e) = handle_new_invite(
                    &dialog_layer,
                    tx,
                    &inbound_tx,
                    advertise_ip,
                    rtp_port_base,
                    rtp_port_tries,
                )
                .await
                {
                    tracing::debug!(error = %e, "failed to set up inbound INVITE");
                }
            }
            rsip::Method::Ack => { /* handled within the dialog */ }
            _ => {
                let _ = tx.reply(rsip::StatusCode::OK).await;
            }
        }
    }
}

/// Build an [`InboundInvite`] from a fresh INVITE transaction and push it to the
/// host. Parses the offer SDP up front so a bad/again-unsupported offer is
/// rejected here (488) rather than after the host commits.
async fn handle_new_invite(
    dialog_layer: &Arc<DialogLayer>,
    tx: rsipstack::transaction::transaction::Transaction,
    inbound_tx: &mpsc::Sender<InboundInvite>,
    advertise_ip: Ipv4Addr,
    rtp_port_base: u16,
    rtp_port_tries: u16,
) -> Result<(), FlowcatError> {
    let mut tx = tx;
    // Identity from the SIP headers (never the body): From user = caller, To user
    // = dialed DID.
    let from = uri_user(tx.original.from_header().ok().and_then(|h| h.uri().ok()));
    let to_did = uri_user(tx.original.to_header().ok().and_then(|h| h.uri().ok()));
    let call_id = tx
        .original
        .call_id_header()
        .map(|c| c.to_string())
        .unwrap_or_default();

    // Parse the SDP offer.
    let offer_body = String::from_utf8_lossy(tx.original.body());
    let offer = match sdp::parse(&offer_body) {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!(error = %e, "rejecting INVITE with unusable SDP offer");
            let _ = tx.reply(rsip::StatusCode::NotAcceptableHere).await;
            return Err(FlowcatError::Transport(format!("bad offer SDP: {e}")));
        }
    };

    // Reserve a slot on the inbound queue *before* creating any dialog. The pump
    // is the single signaling task, so it must not block here: if the host's
    // accept loop is saturated (queue full) or gone (closed), shed this call with
    // 503 rather than awaiting capacity — awaiting would stall in-dialog BYE/ACK
    // for calls already in progress behind us. Reserving first also means we never
    // create a server dialog we then can't hand off (which would leak it).
    //
    // (A retransmitted INVITE does not reach here a second time: rsipstack's
    // transaction layer matches the retransmit to the in-flight or just-finished
    // server transaction and replays the response, so each INVITE surfaces as
    // exactly one transaction — no per-Call-ID dedup is needed at this layer.)
    let permit = match inbound_tx.try_reserve() {
        Ok(p) => p,
        Err(_) => {
            tracing::warn!(%call_id, "inbound queue full/closed; shedding INVITE with 503");
            let _ = tx.reply(rsip::StatusCode::ServiceUnavailable).await;
            return Err(FlowcatError::Transport(
                "inbound queue full or closed".into(),
            ));
        }
    };

    // Create the server dialog (allocates the To-tag) and its state channel.
    let (state_tx, state_rx): (DialogStateSender, DialogStateReceiver) =
        dialog_layer.new_dialog_state_channel();
    let dialog = dialog_layer
        .get_or_create_server_invite(&tx, state_tx, None, None)
        .map_err(|e| FlowcatError::Transport(format!("get_or_create_server_invite: {e}")))?;

    // Drive the dialog's INVITE handler (sends 100 Trying, processes ACK/CANCEL)
    // in the background; `answer`/`reject` send the final response through it.
    let mut dialog_for_handle = dialog.clone();
    tokio::spawn(async move {
        let _ = dialog_for_handle.handle(&mut tx).await;
    });

    let invite = InboundInvite {
        call_id,
        from,
        to_did,
        offer,
        dialog,
        dialog_layer: dialog_layer.clone(),
        advertise_ip,
        rtp_port_base,
        rtp_port_tries,
        state_rx,
    };
    permit.send(invite);
    Ok(())
}

/// Background REGISTER + refresh loop. Keeps the trunk binding fresh, publishes
/// the live registration state on `registered` (so the embedder can observe a
/// trunk that has silently gone unreachable), and escalates its log from `warn`
/// to `error` after [`REGISTER_ESCALATE_AFTER`] consecutive failures. It retries
/// forever; cancellation (root shutdown or [`SipAgent::unregister`]) ends it.
async fn register_loop(
    endpoint_inner: EndpointInnerRef,
    cfg: SipConfig,
    cancel: CancellationToken,
    registered: watch::Sender<bool>,
) {
    let credential = Credential {
        username: cfg.login.clone(),
        password: cfg.password.clone(),
        realm: None,
    };
    let server = match parse_server_uri(&cfg.server) {
        Ok(u) => u,
        Err(e) => {
            tracing::error!(error = %e, "SIP register loop: bad server URI; not registering");
            let _ = registered.send(false);
            return;
        }
    };
    let mut reg =
        rsipstack::dialog::registration::Registration::new(endpoint_inner, Some(credential));
    let mut consecutive_failures: u32 = 0;
    loop {
        let delay_secs = match reg.register(server.clone(), None).await {
            Ok(resp) if resp.status_code == rsip::StatusCode::OK => {
                consecutive_failures = 0;
                let _ = registered.send(true);
                let expires = reg.expires();
                tracing::info!(
                    expires = (expires as u64).max(MIN_REREGISTER_SECS),
                    "SIP registered"
                );
                reregister_delay_secs(expires)
            }
            Ok(resp) => {
                consecutive_failures += 1;
                let _ = registered.send(false);
                log_register_failure(
                    consecutive_failures,
                    format_args!("rejected: {}", resp.status_code),
                );
                MIN_REREGISTER_SECS
            }
            Err(e) => {
                consecutive_failures += 1;
                let _ = registered.send(false);
                log_register_failure(consecutive_failures, format_args!("error: {e}"));
                MIN_REREGISTER_SECS
            }
        };
        tokio::select! {
            _ = cancel.cancelled() => {
                let _ = registered.send(false);
                return;
            }
            _ = tokio::time::sleep(Duration::from_secs(delay_secs)) => {}
        }
    }
}

/// Log a REGISTER failure, escalating from `warn` to `error` once the trunk has
/// failed [`REGISTER_ESCALATE_AFTER`] times in a row (by which point inbound calls
/// are being lost and it is worth alerting on).
fn log_register_failure(consecutive: u32, detail: std::fmt::Arguments<'_>) {
    if register_failure_is_critical(consecutive) {
        tracing::error!(
            consecutive,
            "SIP register {detail}; trunk is de-registered — inbound calls will not arrive"
        );
    } else {
        tracing::warn!(consecutive, "SIP register {detail}; retrying");
    }
}

/// Whether a run of `consecutive` REGISTER failures is severe enough to escalate
/// the log to `error` (see [`REGISTER_ESCALATE_AFTER`]).
fn register_failure_is_critical(consecutive: u32) -> bool {
    consecutive >= REGISTER_ESCALATE_AFTER
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a SIP server string into a `rsip::Uri`, prefixing `sip:` if missing.
fn parse_server_uri(server: &str) -> Result<rsip::Uri, FlowcatError> {
    let s = if server.starts_with("sip:") || server.starts_with("sips:") {
        server.to_string()
    } else {
        format!("sip:{server}")
    };
    rsip::Uri::try_from(s.as_str())
        .map_err(|e| FlowcatError::Transport(format!("bad SIP server URI {server:?}: {e}")))
}

/// Build a `sip:<user>@<host>` URI for caller/callee/contact.
fn make_uri(user: &str, host_with_port: rsip::HostWithPort) -> rsip::Uri {
    rsip::Uri {
        scheme: Some(rsip::Scheme::Sip),
        auth: Some(rsip::Auth {
            user: user.to_string(),
            password: None,
        }),
        host_with_port,
        params: vec![],
        headers: vec![],
    }
}

/// The user part of a URI (the phone number), or empty string.
fn uri_user(uri: Option<rsip::Uri>) -> String {
    uri.and_then(|u| u.auth.map(|a| a.user)).unwrap_or_default()
}

/// Seconds to wait before re-REGISTER, given the server's granted expiry.
/// Floors the expiry at [`MIN_REREGISTER_SECS`] then refreshes at 75 % of it, so a
/// `0`/short expiry (odd server response) can't melt into a hot re-REGISTER loop.
fn reregister_delay_secs(expires: u32) -> u64 {
    (expires as u64).max(MIN_REREGISTER_SECS) * 3 / 4
}

/// Resolve the local IPv4 we advertise in SDP when no public IP is configured,
/// from the endpoint's first bound address. Falls back to loopback.
fn local_advertise_ip(endpoint: &rsipstack::transaction::endpoint::Endpoint) -> Ipv4Addr {
    endpoint
        .get_addrs()
        .first()
        .and_then(|a| SocketAddr::try_from(a.addr.clone()).ok())
        .and_then(|sa| match sa.ip() {
            IpAddr::V4(v4) => Some(v4),
            IpAddr::V6(_) => None,
        })
        .unwrap_or(Ipv4Addr::LOCALHOST)
}

/// Bind a fresh RTP `UdpSocket` on an even port in the configured range.
/// Returns the socket and the port it bound (which goes in our SDP).
async fn bind_rtp_socket(base: u16, tries: u16) -> Result<(UdpSocket, u16), FlowcatError> {
    // RFC 3550: RTP uses an EVEN port (RTCP is the odd port above it). Force the
    // base even so a misconfigured odd `rtp_port_base` can't yield odd RTP ports
    // (some carriers reject them or assume RTCP = RTP+1 and collide).
    let base = base & !1;
    for i in 0..tries {
        // Checked arithmetic so a misconfigured base/tries can't overflow u16 and
        // wrap to a low port; an out-of-range slot just ends the scan early.
        let Some(port) = i.checked_mul(2).and_then(|off| base.checked_add(off)) else {
            break;
        };
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port);
        if let Ok(sock) = UdpSocket::bind(addr).await {
            return Ok((sock, port));
        }
    }
    Err(FlowcatError::Transport(format!(
        "no free RTP port in {}..{}",
        base,
        base.saturating_add(tries.saturating_mul(2))
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_server_uri_adds_scheme() {
        let u = parse_server_uri("sip.zadarma.com").unwrap();
        assert_eq!(u.scheme, Some(rsip::Scheme::Sip));
        // Host preserved.
        assert!(u.to_string().contains("sip.zadarma.com"));
    }

    #[test]
    fn parse_server_uri_keeps_explicit_scheme() {
        let u = parse_server_uri("sip:1.2.3.4:5070").unwrap();
        assert!(u.to_string().contains("1.2.3.4"));
    }

    #[test]
    fn make_uri_sets_user_and_scheme() {
        let host = rsip::HostWithPort::try_from("sip.example.com:5060").unwrap();
        let u = make_uri("+15551234567", host);
        assert_eq!(u.auth.as_ref().unwrap().user, "+15551234567");
        assert_eq!(u.scheme, Some(rsip::Scheme::Sip));
    }

    #[test]
    fn uri_user_extracts_number_or_empty() {
        let host = rsip::HostWithPort::try_from("h:5060").unwrap();
        let u = make_uri("18005551212", host);
        assert_eq!(uri_user(Some(u)), "18005551212");
        assert_eq!(uri_user(None), "");
    }

    // ── DID extraction variants (the inbound-resolve identity anchor) ───────────
    fn z_host() -> rsip::HostWithPort {
        rsip::HostWithPort::try_from("pbx.zadarma.com:5060").unwrap()
    }

    #[test]
    fn uri_user_preserves_zadarma_did_verbatim() {
        // A bare DID (no leading '+') is returned verbatim for the control plane
        // to route. (Reserved fictional 555-0100 test number — not a real line.)
        assert_eq!(
            uri_user(Some(make_uri("12025550100", z_host()))),
            "12025550100"
        );
    }

    #[test]
    fn uri_user_keeps_leading_plus() {
        // The control plane won't re-prefix a value already starting with '+',
        // so a +-form DID still routes (see internal.rs sip_inbound_resolve).
        assert_eq!(
            uri_user(Some(make_uri("+12025550100", z_host()))),
            "+12025550100"
        );
    }

    #[test]
    fn uri_user_extension_and_junk_pass_through_for_fail_closed_404() {
        // A PBX extension or a scanner's junk user-part is returned verbatim;
        // the control plane 404s it → INVITE rejected (fail-closed). No panic.
        assert_eq!(uri_user(Some(make_uri("100", z_host()))), "100");
        assert_eq!(
            uri_user(Some(make_uri("nmap-probe", z_host()))),
            "nmap-probe"
        );
    }

    #[test]
    fn uri_user_bare_host_uri_yields_empty() {
        // Scanner INVITE to a bare host (no user-part) → "" (not a panic).
        let uri = rsip::Uri::try_from("sip:pbx.zadarma.com").unwrap();
        assert_eq!(uri_user(Some(uri)), "");
    }

    // ── re-REGISTER cadence (Bug-5: short/zero expiry must not hot-loop) ────────
    #[test]
    fn reregister_delay_uses_three_quarters_of_expiry() {
        // Default trunk cadence: expires=50 → max(50,30)=50 → 50*3/4 = 37s.
        assert_eq!(reregister_delay_secs(50), 37);
    }

    #[test]
    fn reregister_delay_floors_short_expiry() {
        assert_eq!(reregister_delay_secs(0), 30 * 3 / 4);
        assert_eq!(reregister_delay_secs(10), 30 * 3 / 4);
    }

    #[test]
    fn reregister_delay_scales_long_expiry_without_overflow() {
        assert_eq!(reregister_delay_secs(3600), 2700);
        assert_eq!(reregister_delay_secs(u32::MAX), (u32::MAX as u64) * 3 / 4);
    }

    // ── re-REGISTER failure escalation (#7: a silently de-registered trunk must
    //    eventually log at `error`, not stay at `warn` forever) ─────────────────
    #[test]
    fn register_failure_escalates_only_at_threshold() {
        assert!(!register_failure_is_critical(0));
        assert!(!register_failure_is_critical(1));
        assert!(!register_failure_is_critical(REGISTER_ESCALATE_AFTER - 1));
        // At and beyond the threshold it is critical (escalated to `error`).
        assert!(register_failure_is_critical(REGISTER_ESCALATE_AFTER));
        assert!(register_failure_is_critical(REGISTER_ESCALATE_AFTER + 100));
    }

    /// Two RTP binds must get distinct ports (the scan steps past a taken port).
    #[tokio::test]
    async fn bind_rtp_socket_returns_distinct_ports() {
        let (s1, p1) = bind_rtp_socket(DEFAULT_RTP_PORT_BASE, DEFAULT_RTP_PORT_TRIES)
            .await
            .unwrap();
        let (s2, p2) = bind_rtp_socket(DEFAULT_RTP_PORT_BASE, DEFAULT_RTP_PORT_TRIES)
            .await
            .unwrap();
        assert_ne!(p1, p2);
        // Ports are even (RFC 3550 convention for RTP).
        assert_eq!(p1 % 2, 0);
        assert_eq!(p2 % 2, 0);
        drop((s1, s2));
    }

    /// A constrained range is honored: the bound port stays within
    /// `[base, base + 2*tries)` and is even. Guards the GKE small-port-budget path.
    #[tokio::test]
    async fn bind_rtp_socket_honors_custom_range() {
        let base = 31000u16;
        let tries = 4u16;
        let (s, p) = bind_rtp_socket(base, tries).await.unwrap();
        assert!(p >= base && p < base + tries * 2, "port {p} out of range");
        assert_eq!(p % 2, 0);
        drop(s);
    }

    /// An ODD base must still yield an EVEN RTP port (RFC 3550) — guards the
    /// `base & !1` fix against a misconfigured odd `rtp_port_base`.
    #[tokio::test]
    async fn bind_rtp_socket_yields_even_port_even_for_odd_base() {
        let (s, p) = bind_rtp_socket(31001, 8).await.unwrap(); // odd base
        assert_eq!(
            p % 2,
            0,
            "RTP port {p} is odd despite odd base (RFC 3550 wants even)"
        );
        drop(s);
    }

    // ── Loopback signaling integration: two in-process `SipAgent`s call each
    //    other over 127.0.0.1, exercising the dialog lifecycle end to end (setup,
    //    agent-initiated BYE, reject) and asserting dialogs don't leak from the
    //    `DialogLayer` map. Like the `transport.rs` tests these open real loopback
    //    UDP sockets, but they are fully hermetic: no registrar, no credentials,
    //    no external host. ─────────────────────────────────────────────────────
    use crate::transport::media::{MediaIn, MediaTransport};

    /// Grab a currently-free UDP port on loopback. The probe socket is released
    /// immediately and rebound by the agent — fine for an in-process test.
    async fn free_udp_port() -> u16 {
        let s = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let p = s.local_addr().unwrap().port();
        drop(s);
        p
    }

    /// Start a credential-less loopback agent on `sip_port`, advertising 127.0.0.1.
    /// `server` is only consulted when the agent originates a call.
    async fn start_loopback_agent(server: String, sip_port: u16) -> SipAgent {
        let cfg = SipConfig {
            server,
            login: String::new(),
            // Empty password → no REGISTER loop, so no registrar is needed.
            password: String::new(),
            caller_id: "1000".to_string(),
            // Force the SDP/Contact/Via address to loopback so media + signaling
            // both resolve to 127.0.0.1 regardless of the host's interfaces.
            public_ip: Some(Ipv4Addr::LOCALHOST),
            sip_port: Some(sip_port),
            rtp_port_base: 40000,
            rtp_port_tries: 200,
        };
        SipAgent::start(cfg).await.expect("agent start")
    }

    /// Poll `cond` every 20 ms for up to ~5 s; returns whether it became true.
    async fn wait_until<F: Fn() -> bool>(cond: F) -> bool {
        for _ in 0..250 {
            if cond() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        cond()
    }

    /// Stand up a caller→callee pair on loopback. Returns both agents and both
    /// media transports once the call is answered and confirmed.
    async fn establish_loopback_call() -> (SipAgent, SipAgent, SipTransport, SipTransport) {
        let port_b = free_udp_port().await;
        let callee = start_loopback_agent("sip:127.0.0.1:1".to_string(), port_b).await;
        let port_a = free_udp_port().await;
        let caller = start_loopback_agent(format!("sip:127.0.0.1:{port_b}"), port_a).await;

        // Answer on the callee side concurrently with the caller's originate.
        let answer = tokio::spawn(async move {
            let invite = tokio::time::timeout(Duration::from_secs(5), callee.next_inbound())
                .await
                .expect("inbound INVITE timed out")
                .expect("callee shut down before INVITE");
            // Identity comes from the SIP To user (the dialed number), not the body.
            assert_eq!(invite.to_did, "2000");
            let transport = invite.answer().await.expect("answer failed");
            (callee, transport)
        });

        let caller_tr =
            tokio::time::timeout(Duration::from_secs(5), caller.originate("2000", None))
                .await
                .expect("originate timed out")
                .expect("outbound call not answered");
        let (callee, callee_tr) = answer.await.expect("answer task panicked");
        (caller, callee, caller_tr, callee_tr)
    }

    /// Drain `recv` past `StreamStart` and any (comfort-silence) audio until the
    /// leg ends, asserting the terminal event is `Stop`.
    async fn recv_until_stop(tr: &mut SipTransport) {
        loop {
            match tokio::time::timeout(Duration::from_secs(5), tr.recv())
                .await
                .expect("recv timed out waiting for Stop")
            {
                Some(MediaIn::StreamStart { .. }) | Some(MediaIn::Audio(_)) => continue,
                other => {
                    assert_eq!(other, Some(MediaIn::Stop), "expected Stop at end of leg");
                    break;
                }
            }
        }
    }

    /// A full call sets up one dialog per leg; the agent ending the call (dropping
    /// its transport) must BYE the peer (→ the peer's leg sees `Stop`) and both
    /// dialogs must be removed from their layers (the leak fix, #1/#2/#3).
    #[tokio::test]
    async fn loopback_agent_hangup_byes_peer_and_removes_both_dialogs() {
        let (caller, callee, caller_tr, mut callee_tr) = establish_loopback_call().await;

        // Both legs established → exactly one dialog in each layer.
        assert!(
            wait_until(|| caller.dialog_layer.len() == 1).await,
            "caller dialog missing after answer"
        );
        assert!(
            wait_until(|| callee.dialog_layer.len() == 1).await,
            "callee dialog missing after answer"
        );

        // Agent-initiated hangup: the caller ends the call by dropping its
        // transport. Its supervisor must send a BYE to the callee.
        drop(caller_tr);

        // The callee's leg sees the BYE as end-of-media (Stop)…
        recv_until_stop(&mut callee_tr).await;
        // …and both dialogs are released from their layers (no leak).
        assert!(
            wait_until(|| caller.dialog_layer.is_empty()).await,
            "caller dialog leaked after hangup"
        );
        assert!(
            wait_until(|| callee.dialog_layer.is_empty()).await,
            "callee dialog leaked after BYE"
        );

        caller.shutdown().await;
        callee.shutdown().await;
    }

    /// Rejecting an inbound INVITE returns an error to the caller (no transport)
    /// and removes the server dialog the pump created (the reject leak fix).
    #[tokio::test]
    async fn loopback_rejected_call_errors_and_removes_dialog() {
        let port_b = free_udp_port().await;
        let callee = start_loopback_agent("sip:127.0.0.1:1".to_string(), port_b).await;
        let port_a = free_udp_port().await;
        let caller = start_loopback_agent(format!("sip:127.0.0.1:{port_b}"), port_a).await;

        let reject = tokio::spawn(async move {
            let invite = tokio::time::timeout(Duration::from_secs(5), callee.next_inbound())
                .await
                .expect("inbound INVITE timed out")
                .expect("callee shut down before INVITE");
            invite.reject(None); // 486 Busy Here
            callee
        });

        let res = tokio::time::timeout(Duration::from_secs(5), caller.originate("2000", None))
            .await
            .expect("originate timed out");
        assert!(
            res.is_err(),
            "a rejected call must not yield a media transport"
        );

        let callee = reject.await.expect("reject task panicked");
        // The rejected server dialog must be gone (no leak); the caller's
        // unconfirmed client dialog is dropped by do_invite on the non-2xx too.
        assert!(
            wait_until(|| callee.dialog_layer.is_empty()).await,
            "callee dialog leaked after reject"
        );
        assert!(
            wait_until(|| caller.dialog_layer.is_empty()).await,
            "caller dialog leaked after reject"
        );

        caller.shutdown().await;
        callee.shutdown().await;
    }
}

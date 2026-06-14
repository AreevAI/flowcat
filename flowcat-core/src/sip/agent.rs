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
//! build a [`SipTransport`] (RTP ↔ `MediaIn`). The negotiated dialog's state
//! channel is watched so a BYE / `Terminated` fires the transport's hangup token,
//! which surfaces as [`MediaIn::Stop`](crate::transport::MediaIn::Stop).
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
use rsipstack::dialog::dialog::{DialogState, DialogStateReceiver, DialogStateSender};
use rsipstack::dialog::dialog_layer::DialogLayer;
use rsipstack::dialog::invitation::InviteOption;
use rsipstack::dialog::server_dialog::ServerInviteDialog;
use rsipstack::sip as rsip;
use rsipstack::sip::prelude::HeadersExt;
use rsipstack::transaction::endpoint::EndpointInnerRef;
use rsipstack::transaction::TransactionReceiver;
use rsipstack::transport::{udp::UdpConnection, TransportLayer};
use rsipstack::EndpointBuilder;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, Mutex};
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
        spawn_dialog_watch(self.state_rx, hangup.clone());

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
        let _ = self
            .dialog
            .reject(Some(code.unwrap_or(rsip::StatusCode::BusyHere)), None);
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

        // Registration loop (only if a password is configured).
        if !cfg.password.is_empty() {
            tokio::spawn(register_loop(
                endpoint_inner.clone(),
                cfg.clone(),
                cancel.clone(),
            ));
        }

        Ok(Self {
            cfg,
            endpoint_inner,
            dialog_layer,
            inbound_rx: Mutex::new(inbound_rx),
            advertise_ip,
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
        spawn_dialog_watch(state_rx, hangup.clone());

        Ok(SipTransport::start(
            rtp_sock,
            peer,
            media.codec,
            call_id,
            hangup,
        ))
    }

    /// Tear down the agent: cancels the endpoint serve loop + all tasks.
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }
}

impl Drop for SipAgent {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Watch a dialog's state channel; on `Terminated` (BYE / timeout / decline)
/// fire `hangup` so the [`SipTransport`] surfaces `MediaIn::Stop`.
fn spawn_dialog_watch(mut state_rx: DialogStateReceiver, hangup: CancellationToken) {
    tokio::spawn(async move {
        while let Some(state) = state_rx.recv().await {
            if let DialogState::Terminated(id, reason) = state {
                tracing::debug!(%id, ?reason, "SIP dialog terminated");
                hangup.cancel();
                break;
            }
        }
        // Channel closed without an explicit Terminated → also treat as hangup.
        hangup.cancel();
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
        advertise_ip,
        rtp_port_base,
        rtp_port_tries,
        state_rx,
    };
    inbound_tx
        .send(invite)
        .await
        .map_err(|_| FlowcatError::Transport("inbound channel closed".into()))
}

/// Background REGISTER + refresh loop (mirrors the rsipstack example).
async fn register_loop(
    endpoint_inner: EndpointInnerRef,
    cfg: SipConfig,
    cancel: CancellationToken,
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
            return;
        }
    };
    let mut reg =
        rsipstack::dialog::registration::Registration::new(endpoint_inner, Some(credential));
    loop {
        match reg.register(server.clone(), None).await {
            Ok(resp) if resp.status_code == rsip::StatusCode::OK => {
                let expires = reg.expires();
                tracing::info!(
                    expires = (expires as u64).max(MIN_REREGISTER_SECS),
                    "SIP registered"
                );
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(Duration::from_secs(reregister_delay_secs(expires))) => {}
                }
            }
            Ok(resp) => {
                tracing::warn!(status = %resp.status_code, "SIP register rejected; retrying");
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(Duration::from_secs(MIN_REREGISTER_SECS)) => {}
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "SIP register error; retrying");
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(Duration::from_secs(MIN_REREGISTER_SECS)) => {}
                }
            }
        }
    }
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
        // Bare DID (the known-good live form) → resolves to workflow 1.
        assert_eq!(
            uri_user(Some(make_uri("918031338133", z_host()))),
            "918031338133"
        );
    }

    #[test]
    fn uri_user_keeps_leading_plus() {
        // The control plane won't re-prefix a value already starting with '+',
        // so a +-form DID still routes (see internal.rs sip_inbound_resolve).
        assert_eq!(
            uri_user(Some(make_uri("+918031338133", z_host()))),
            "+918031338133"
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
}

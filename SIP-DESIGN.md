> STATUS: implemented 2026-05-31

# Flowcat — native SIP (supersedes the FreeSWITCH gateway)

Decision: **SIP is native Rust inside flowcat; FreeSWITCH is removed.** Flowcat speaks
SIP/RTP directly, so it's a single-binary native voice runtime — one project, no separate
softswitch. (Supersedes the earlier FreeSWITCH gateway approach described in `DESIGN.md`.)

## 1. The transport seam (do this first — everything depends on it)

Generalize the pipeline so it doesn't care whether audio arrived as carrier WS frames or RTP.
New in `flowcat-core` (`transport/mod.rs`):

```rust
#[async_trait]
pub trait MediaTransport: Send {
    /// Next inbound event: the call started, a chunk of caller audio (at
    /// `carrier_rate`), or the call ended. `None` = transport closed.
    async fn recv(&mut self) -> Option<MediaIn>;
    /// Play bot audio (at `carrier_rate`) to the caller.
    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError>;
    /// Barge-in: flush any buffered playback. No-op where unsupported.
    async fn send_clear(&mut self) -> Result<(), FlowcatError>;
    fn carrier_rate(&self) -> u32; // 8000 for telephony G.711
}

pub enum MediaIn {
    StreamStart { call_id: String },
    Audio(AudioChunk),  // at carrier_rate
    Stop,
}
```

- **`Call` becomes `Call<Tr: MediaTransport, R: RealtimeLlm, B: AgentBrain, S: SessionSource>`**
  (4 params, was 5). The loop: `tr.recv()` → `MediaIn::{StreamStart, Audio, Stop}`; realtime
  `AudioOut` → resample → `tr.send_audio`; `Interrupted` → `tr.send_clear`. The resampler,
  recorder, transcript, brain/tool handling, finalize are UNCHANGED.
- **WS path preserved via an adapter:** `WsCarrierTransport<So: MediaSocket, Se: MediaSerializer>`
  impls `MediaTransport` — `recv()` loops `socket.recv()` → `serializer.on_message` until a
  non-`Ignore` `SerIn`, maps to `MediaIn`; `send_audio`/`send_clear` go through
  `serializer.encode_audio`/`encode_clear` → `socket.send_*`; `carrier_rate` = serializer's.
  The embedder builds `WsCarrierTransport::new(AxumWsSocket, PlivoSerializer)` for Plivo.
- Keep `MediaSocket`/`MediaSerializer`/`PlivoSerializer` (used by the adapter). The
  `gateway` serializer + `GatewayEncoding` are removed (no more gateway).

## 2. The SIP/RTP module (`flowcat-core/src/sip/`)

Add deps to `flowcat-core/Cargo.toml`: a SIP UA stack — **`rsipstack`** (primary; REGISTER +
INVITE/ACK/BYE transactions + dialogs; backs the `rustpbx` softswitch) — and an RTP layer
(`rsipstack`'s RTP if sufficient, else `webrtc-util`/`rtp`). G.711 + resample already present
(`audio-codec-algorithms`, `rubato`). If `rsipstack`'s API proves too thin, fall back to
`ezk-sip-ua` (cleaner, SDP/RTP integrated) and report the switch.

- **`SipAgent`** (process-level, one per trunk): REGISTER to `{server}` with `{login}`/
  `{password}` + periodic re-REGISTER/keepalive; NAT-friendly (rport/symmetric RTP);
  accept inbound INVITE → yield an inbound call (Call-ID, From, To/DID) to the host; originate
  outbound INVITE to an E.164 with a configured CallerID. G.711 only (`disallow all; allow
  PCMU,PCMA`), 8 kHz, ptime 20 ms, `a=sendrecv`.
- **`SipTransport`** (per dialog) impls `MediaTransport`: on answer/established emit
  `StreamStart{call_id = Call-ID}`; decode inbound RTP (G.711 per negotiated codec) → 20 ms
  `MediaIn::Audio @ 8000`; on BYE/timeout → `Stop`. `send_audio` → G.711-encode → RTP packets
  (monotonic seq + timestamp, 20 ms cadence, correct PT/SSRF). `send_clear` → no-op (RTP has
  no flush; barge-in = stop sending). A small fixed playout **jitter buffer** (reorder by seq,
  bounded depth, drop late) — telephony fixed-rate makes this simple; document the depth.
- Unit-test (no live socket): SDP offer/answer build + parse (PCMU/PCMA pick, ptime), RTP
  packetize/depacketize (seq/ts/PT), the `MediaTransport` mapping (a fed RTP stream → Audio;
  a synthesized BYE → Stop). Live registration/INVITE against the trunk is gated on the user.

## 3. Host + control-plane rewire (embedder side)

- **The embedder** runs a `SipAgent` at startup, registering the SIP trunk from its own SIP
  trunk configuration (server / login / password / caller-id, plus public-IP/ports as needed)
  passed to `SipConfig`/`SipAgent`. On **inbound** INVITE → the embedder's service-authed
  `sip/inbound-resolve` (body `{dialed_did, caller, call_id}`) → `{run_id, token}` → build
  `Call::new(SipTransport, gemini, <brain>, <session>, run_id, token)` → run. Add an internal
  **outbound** endpoint (authed by the run's token, or service auth) `{run_id, token, to_number}`
  → SipAgent originates the INVITE, builds the Call, runs it. The WS-media path is unchanged.
- **control plane:** the SIP originate path no longer uses ESL — it POSTs to the embedder's media
  binary's originate endpoint (internal base from settings). **Remove** any FreeSWITCH ESL routes
  + settings. Keep `sip/inbound-resolve`, the carrier provider (CDR webhook verify + status still
  useful), and the sealed credentials. The REST signer may go unused → keep (tested, small) or
  remove if truly dead.

## 4. Removals (after native SIP compiles)

- `flowcat/deploy/freeswitch/` (whole dir) · `flowcat-core/src/serializer/gateway.rs` +
  `GatewayEncoding` + its `lib.rs` export · the embedder's FreeSWITCH ESL routes + their module
  registration · ESL config fields · FreeSWITCH env in any `.env.example`.
- Update `DESIGN.md`: SIP media = native SIP (no FreeSWITCH); bring-up = register the trunk in
  the embedder + a `sofia`-free register check.

## Security (re-review the new surface)
Trunk SIP credentials live in the embedder's config — never logged; the inbound INVITE trust
flows through `sip/inbound-resolve` (service-authed, identity from the DID route, never the
INVITE body); the outbound originate endpoint must authenticate (run token / service auth) and
validate `to_number` (E.164). RTP from an unexpected source addr should be dropped (symmetric
RTP). No new secret crosses the wire to the model/caller.

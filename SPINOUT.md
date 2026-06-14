<!-- SPDX-License-Identifier: Apache-2.0 -->
# Flowcat — workspace independence

**Flowcat is a self-contained cargo workspace.** It has **zero dependency on any
embedder**: no host-application crate is imported, and no path-dependency points
outside `flowcat/`. The runtime is designed to be embedded — a host application
plugs into the four trait seams (`MediaTransport` · `RealtimeLlm` · `AgentBrain` ·
`SessionSource`) — but flowcat itself never depends back on its consumer.

This doc records how that independence is kept, and how to verify it.

---

## 1. The independence checks

| Check | Result |
| --- | --- |
| Is the root `Cargo.toml` a single, self-contained `[workspace]`? | **Yes.** |
| Any `use <embedder>_*` / `extern crate <embedder>_*` in `**/*.rs`? | **None.** |
| Any embedder crate in any `**/Cargo.toml` `[dependencies]`? | **None.** |
| Any path-dependency pointing outside the workspace? | **None.** Every path-dep is internal (`flowcat-{services,transports,telephony,cli}` → `path = "../flowcat-core"`). |
| Build standalone? | `cargo build` ✅ |

Reproduce the leak scan (each must print nothing):

```bash
# Path-deps that escape the workspace (only internal ../flowcat-core is allowed):
grep -rnE 'path\s*=\s*"(\.\./)+' --include='Cargo.toml' . | grep -v '"\.\./flowcat-'
# Standalone build + offline tests, no host application present:
cargo build && cargo test
```

The workspace builds and tests with no host application present. `bench-rs/` is a
standalone, non-member benchmark crate that travels with the tree and builds on
its own.

---

## 2. How the runtime stays embedder-agnostic

- **`flowcat-core` knows nothing about any embedder, web routing, SQL, or any wire
  contract.** It exposes only the media-pipeline framework + the four trait seams.
  The host's brain/session/transport implementations are injected; the call config
  (`brain_config`) is an opaque `serde_json::Value` flowcat never interprets.
- **Providers and transports live in sibling crates behind one cargo feature each**
  (`flowcat-services`, `flowcat-transports`, `flowcat-telephony`), so a default
  build pulls nothing heavy. A host enables only what it needs.
- **Config the embedder supplies that flowcat does not read itself** — SIP trunk
  credentials (server / login / password / caller-id) and internal base URLs — is
  passed in through `SipConfig`/`SipAgent` and the trait seams, not read from any
  flowcat-owned env var. The keys flowcat itself reads are the `FLOWCAT_*` family
  (e.g. `FLOWCAT_VOICE`, the `FLOWCAT_VAD_*` knobs).

---

## 3. Examples / demo surface

`flowcat-cli` (`bin: flowcat`) is the OSS demo surface, mirroring pipecat's
`examples/`. It ships **two runnable demos** (no longer stubs), neither of which
needs a vendor credential, so both run in CI:

1. **`pipeline`** — an in-process, network-free showcase of the composable
   `FrameProcessor` pipeline over a synthetic sine-wave source. Proves the
   `Pipeline`/`PipelineTask` end-to-end with no network.
2. **`ws-echo`** — real I/O over the generic WebSocket media transport
   (`flowcat-transports`, `ws` feature): a self-contained loopback (or `--connect
   <url>`) that decodes inbound media frames and echoes them through the
   bounded-channel pipeline — the analogue of the `bench-rs` real-I/O harness,
   repackaged as a runnable demo.

Run them:

```bash
cargo run -p flowcat-cli -- pipeline
cargo run -p flowcat-cli -- ws-echo            # loopback
```

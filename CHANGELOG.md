<!-- SPDX-License-Identifier: Apache-2.0 -->
# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from `1.0.0`
onward. Until then (pre-1.0), minor versions may include breaking changes.

## [Unreleased]

### Added
- `flowcat-cli` ships two runnable, credential-free demos (the OSS examples
  surface): `pipeline` (an in-process `FrameProcessor` pipeline over a synthetic
  sine-wave source) and `ws-echo` (PCM echo over the generic WebSocket transport,
  with a self-contained `--loopback` round-trip or `--connect <ws://url>`).
- `RemoteBrain` HTTP adapter (`flowcat-services`, feature `brain-http`): drive a
  call's conversation policy from an out-of-process HTTP service (e.g. a Python
  webhook) via the `AgentBrain` seam, at turn granularity. Includes a documented
  JSON wire contract, a request timeout, a fail-safe (`Stay` on transient error
  or timeout), and fixture tests.
- `examples/` for using Flowcat from Python without writing Rust: a pure-stdlib
  reference `python-remote-brain` server and a `python-mcp-tools` MCP server.
- `ROADMAP.md` describing planned work (in-process PyO3 bindings, WebRTC browser
  transport, a local audio device backend, broader live-verified coverage).
- Standard project docs: `CODE_OF_CONDUCT.md`, `SECURITY.md`, this changelog, and
  GitHub issue / pull-request templates.

### Changed
- Documentation scrubbed for the public release: design docs are now
  embedder-agnostic, and `SPINOUT.md` is repurposed as a workspace-independence
  note for contributors.

### Removed
- Internal-only planning and operations documents that are not relevant to the
  public project.

### Breaking
- Renamed the runtime's environment-variable keys off the previous prefix to the
  `FLOWCAT_*` family: `FLOWCAT_VOICE`, `FLOWCAT_VAD_START_SENSITIVITY`,
  `FLOWCAT_VAD_END_SENSITIVITY`, `FLOWCAT_VAD_PREFIX_PADDING_MS`,
  `FLOWCAT_VAD_SILENCE_DURATION_MS`, and `FLOWCAT_MINIMAX_GROUP_ID`. Embedders
  setting these must update the key names.

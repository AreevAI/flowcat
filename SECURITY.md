<!-- SPDX-License-Identifier: Apache-2.0 -->
# Security policy

## Supported versions

Flowcat is pre-1.0 and under active development. Security fixes are applied to
the latest `main`. Until a `1.0` release, only the most recent published version
(and `main`) is supported.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately through either channel:

1. **GitHub private vulnerability reporting** (preferred) — use the repository's
   **Security → Report a vulnerability** tab to open a private advisory.
2. **Email** — **security@areev.ai**, with a description and reproduction steps.

Please include, where possible:

- the affected crate(s) and version / commit,
- the feature flags and provider/transport involved,
- a minimal reproduction or proof of concept, and
- the impact you foresee.

## What to expect

- We aim to **acknowledge** a report within **3 business days**.
- We will work with you on an assessment and a fix, and keep you updated on
  progress.
- We support **coordinated disclosure**: we ask that you give us a reasonable
  window to release a fix before any public disclosure, and we will credit you
  for the report unless you prefer to remain anonymous.

## Scope

This project is a media-pipeline runtime that speaks third-party provider
protocols using credentials **you** supply. Vulnerabilities in third-party
services or SDKs should be reported to those vendors. Issues in Flowcat's own
code — for example in its SIP/RTP/SDP parsing, the SigV4 signing paths, WebSocket
framing, or audio codec handling — are in scope.

<!-- SPDX-License-Identifier: Apache-2.0 -->
## What & why

<!-- A short description of the change and the motivation. Link any related issue. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New provider / transport / serializer
- [ ] New feature or processor
- [ ] Documentation
- [ ] Refactor / chore

## Checklist

- [ ] `cargo build` (default features) compiles and pulls **no** new networked
      dependency into the default build.
- [ ] `cargo test` is green and **offline** (no test hits the network).
- [ ] `cargo clippy --all-targets` is clean.
- [ ] `cargo fmt --all` has been run.
- [ ] Every new source file starts with the SPDX header
      `// SPDX-License-Identifier: Apache-2.0`.
- [ ] If a provider was added: it is `dep:`-gated behind its own Cargo feature,
      registered in the category `mod.rs` + the `*-all` umbrella, and covered by a
      fixture / known-answer test. `FEATURES.md` is updated.
- [ ] If a new dependency or third-party protocol was introduced: `NOTICE` is
      updated as needed.
- [ ] The PR honestly states what is **fixture-tested** vs **live-verified**.

## Notes for reviewers

<!-- Anything that needs a closer look: auth/signing paths, new deps, etc. -->

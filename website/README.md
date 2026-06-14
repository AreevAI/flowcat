<!-- SPDX-License-Identifier: Apache-2.0 -->
# Flowcat docs site

The [mdBook](https://rust-lang.github.io/mdBook/) source for the Flowcat
documentation site, published to GitHub Pages by
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) on every push to
`main`.

## How it's wired

The canonical docs are the Markdown files in the **repo root** (`README.md`,
`QUICKSTART.md`, `DESIGN.md`, …). Each page under `src/` is a thin wrapper that
pulls the root file in with mdBook's include directive, so there is **one source
of truth** and nothing is duplicated in git:

```markdown
{{#include ../../QUICKSTART.md}}
```

Pages authored *for the site* (no root equivalent) are full Markdown files:
`introduction.md`, `configuration.md`, `deployment.md`, `embedder.md`,
`api-reference.md`. The table of contents is [`src/SUMMARY.md`](src/SUMMARY.md).

> Cross-document links inside the *included* root files were written for browsing
> on GitHub (e.g. `[DESIGN.md](DESIGN.md)`) and may not resolve to site URLs.
> Site-relative navigation lives in `SUMMARY.md` and the authored pages. Tightening
> the in-page links is a follow-up curation pass.

## Preview locally

```bash
cargo install mdbook        # once
mdbook serve website --open # live-reload at http://localhost:3000
```

`mdbook build website` writes the static site to `website/book/` (git-ignored).

## One-time GitHub setup

In the repo: **Settings → Pages → Build and deployment → Source: "GitHub
Actions"**. After that, pushes to `main` publish automatically.

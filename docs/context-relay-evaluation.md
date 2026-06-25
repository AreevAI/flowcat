<!-- SPDX-License-Identifier: Apache-2.0 -->

# ContextRelay — converting realtime audio context to text: cost & memory

*How flowcat's ContextRelay turns a long voice call's accumulated **audio** context into
a compact **text** transcript and reseeds the live session with it — cutting the
per-turn context cost and keeping the whole conversation. Sources verified against
primary Google documentation on 2026-06-23; pricing and limits change, so re-check the
cited pages before quoting numbers.*

Statements are tagged so they can be cited safely:
**[CITED]** = Google's official docs (URL inline) · **[MEASURED]** = our live test calls ·
**[ESTIMATE]** = a model built on the cited rates (the *ratios* are robust; absolute
dollars depend on call shape).

---

## The idea, in one line

A realtime model re-processes and **re-bills the entire conversation on every turn**
**[CITED]**, and audio tokens are bulky and expensive. ContextRelay automatically
**converts the accumulated audio context into a text transcript and reseeds the session
with it**, so from that point on the model re-attends cheap, compact **text** instead of
expensive **audio** — while keeping the full conversation. Two wins at once:

1. **Cost** — text context is **~25–30× cheaper to carry per turn** than audio context.
2. **Memory** — the whole conversation survives, instead of being silently dropped to
   stay under budget.

Both are demonstrated against the live Gemini Live API below.

---

## 1. The cost lever — audio context is expensive, and re-billed every turn

**Realtime sessions re-attend the full history each turn.** Verbatim from Google:
*"The API charges you per turn for all tokens present in the session context window …
all accumulated tokens from previous turns. Past tokens are re-processed and accounted
for in each new turn … As a session lengthens, the cost per turn increases because the
conversational history is re-processed."* **[CITED]**
([Live API best practices](https://ai.google.dev/gemini-api/docs/live-api/best-practices))
— and there is **no context caching** for Live to soften it
([Google staff](https://discuss.ai.google.dev/t/gemini-live-caching/83298)). So whatever
the context *is*, you pay for it on every single turn.

**Audio context is bulky and pricey; text is small and cheap** (Gemini 3.1 Flash Live):

| | Token rate | Input price | Cost to **re-attend 1 min of conversation** (per turn) |
|---|---|---|---|
| **Audio context** | **25 tok/s** = 1,500 tok/min **[CITED]** | **$3.00 / 1M [CITED]** | **$0.00450** |
| **Text transcript** | ~150 wpm × ~1.43 tok/word ≈ **214 tok/min** **[CITED rates]** | **$0.75 / 1M [CITED]** | **$0.00016** |

- **~7× more compact** (1,500 ÷ 214 tokens) **·** **4× cheaper per token** ($3.00 ÷ $0.75)
  **→ ≈ 25–30× cheaper to carry a minute of conversation as text than as audio**, every
  turn. **[ESTIMATE from CITED rates]** (Range 25–36× across 130–160 wpm; ≈28× at 150 wpm.)

**Snapshot at 15 minutes — the context re-billed on each turn:**

| Context carried | Tokens | Cost per turn |
|---|---|---|
| Full call as **audio** | ~22,500 | **$0.068** |
| Full call as **text** (ContextRelay) | ~3,200 | **$0.0024** |
| | | **≈ 28× cheaper / turn** |

Because that context is re-billed **every** turn, the saving compounds over the call.
*Illustrative cumulative re-billed-input over a 15-min, ~45-turn call:* **~$1.5 (audio)
vs ~$0.05 (text)** — **[ESTIMATE]**. (This is the *re-billed-history* component, which
dominates the bill on long calls; fresh-audio-in and audio-out costs are unchanged.)

**Audio really is the bulk of the per-turn input — measured.** In our own live calls, the
model's per-turn input was roughly half audio even early in a short call: one turn billed
**259 audio + 225 text** tokens; a later turn **451 audio + 256 text**. **[MEASURED]** As
the call grows, that audio share is what blows up — and what ContextRelay converts away.

---

## 2. The memory lever — keep the whole conversation, don't drop it

There is a native way to stop the per-turn cost from growing — Gemini's
`contextWindowCompression` **sliding window** — but it works by **throwing context away**.
Verbatim: the sliding window *"operates by **discarding content at the beginning** of the
context window."* **[CITED]** ([Live API reference](https://ai.google.dev/api/live)). It
evicts the **oldest user turns** first — so the account number the caller gave at minute 1
is exactly what gets dropped.

This forces a **trade-off**:

- **Tighten the window** (low `triggerTokens`) to keep cost down → it **forgets** more,
  sooner.
- **Loosen the window** to remember more → cost climbs back up (and the audio
  **15-minute** session cap **[CITED]**,
  [Capabilities](https://ai.google.dev/gemini-api/docs/live-api/capabilities), still ends
  the call).

**ContextRelay breaks the trade-off.** Because text is ~25–30× cheaper to carry, the
*entire* transcript fits in the budget that a bounded **audio** window would cost — so you
**bound cost without forgetting**. flowcat's own note at the compression call-site says it
plainly: *"this is pure eviction, NOT a semantic summary — early-call facts are forgotten
once evicted. **Pair with summarize-and-restart where they matter.**"* That pairing is
ContextRelay.

It offers two policies for what to carry:

- **`VerbatimCompactor`** — the **entire transcript, verbatim** (nothing dropped).
- **`LlmCompactor`** — a rolling **summary of older turns + the last N verbatim**, via a
  cheap text LLM, to keep the carried text bounded on very long calls.

Mechanically, the carried text rides the `update_system` reseed into the **system
instruction**, which the native sliding window is documented to **preserve** (*"System
instructions … will always remain"* **[CITED]**) — so the retained context survives even
if native compression is also running.

---

## 3. Live validation  [MEASURED]

**Setup.** Real Gemini Live (`models/gemini-3.1-flash-live-preview`) via flowcat-server's
WebRTC build; a headless synthetic caller streams `say`-generated speech (8 kHz μ-law)
over the carrier WebSocket; the bot's audio is recorded and transcribed back with **Gemini
Flash** (`gemini-2.5-flash`). ContextRelay enabled with `FLOWCAT_CONTEXT_RELAY=1` and the
session-age trigger lowered to force mid-call reseeds.

**Verbatim-transcript reseed.** The caller states *"my account number is four four seven
two"*; a reseed fires (`context-relay: re-basing realtime session onto text digest
reason=SessionAge`); the caller later asks for it back. Transcribed bot reply:

> *"You gave me **4472**. Now, back to the internet speed. Have you tried restarting your
> modem or router?…"*

**LLM-summary reseed — 5 turns, six reseeds.** Server log shows
`ContextRelay: using an LLM summarizer` and six `re-basing … reason=SessionAge` events.
After all six session reopens, the transcribed recall answer:

> *"Of course, Jordan. You provided account number **4472** earlier. Since restarting the
> router didn't help for the slow video calls and streaming, we could try checking for any
> outages in your area…"*

The agent retained **both** the account number from the first turn **and** the rest of the
conversation (router restart, video/streaming) across six reopens, while older turns were
compacted to text. **[MEASURED]** Separately, with native compression alone (no
ContextRelay) under an aggressive trigger, the per-turn context stayed bounded to **561
tokens after a 211-second call** — confirming the native window does bound cost, by the
eviction described in §2. **[MEASURED]**

---

## 3a. Beyond Gemini — OpenAI Realtime, X.AI (Grok), and the rest

ContextRelay rides only the realtime trait surface (final transcripts + a per-turn
usage signal + the `update_system`/re-base seam), so it is **provider-portable in
principle**. But the *cost* win depends on one provider-specific fact: does re-basing
actually **drop the accumulated audio**, and is that audio **re-billed at full price**
every turn? Those answers differ by provider, so the headline number does too.

**The re-base seam is now wired for the OpenAI-protocol family.** flowcat distinguishes
an ordinary graph-transition re-prompt (in-session `update_system`, keeps the
conversation) from a **ContextRelay re-base** (`rebase_session`, must drop the audio).
For Gemini Live both already reconnect (it has no in-session update). For **OpenAI
Realtime, X.AI Grok, Inworld, and Azure** — all of which speak the OpenAI Realtime wire
protocol — `update_system` is an *in-session* `session.update` that **keeps** every
prior audio item, so a re-base there now **reopens the socket** with the text digest as
the fresh session prompt (the same audio→text reset Gemini gets). Without this, the
relay would swap the prompt but leave the expensive audio context in place.

**OpenAI `gpt-realtime-2` token rates** (per 1M tokens) **[CITED]**
([OpenAI pricing](https://developers.openai.com/api/docs/pricing)):

| | Text | Audio | Cached input |
|---|---|---|---|
| **Input** | **$4.00** | **$32.00** | **$0.40** |
| **Output** | $24.00 | $64.00 | — |

Audio is metered at **1 token / 100 ms of user audio (10 tok/s) and 1 token / 50 ms of
assistant audio (20 tok/s)** **[CITED]**
([Realtime cost guide](https://developers.openai.com/api/docs/guides/realtime-costs)).
The directly-comparable lever — **cost to re-attend 1 min of conversation, per turn**,
billed in full (cache-**cold**, which is the case Gemini is *always* in, since Live has
no caching):

| | Token rate | Input price | Cost to **re-attend 1 min** (per turn) |
|---|---|---|---|
| **Audio context** | ~900 tok/min (10/20 tok/s mix) **[CITED]** | **$32 / 1M [CITED]** | **$0.0288** |
| **Text transcript** | ~214 tok/min **[CITED rates]** | **$4 / 1M [CITED]** | **$0.00086** |

- **~4× more compact** (900 ÷ 214) **·** **8× cheaper per token** ($32 ÷ $4)
  **→ ≈ 34× cheaper to carry a minute as text than as cache-cold audio**, per turn.
  **[ESTIMATE from CITED rates]** (Audio is bulkier-per-second than Gemini-rate text but
  ~40% *less* bulky than Gemini-rate audio; the higher $32/1M audio price is what lifts
  the cold-cache multiplier above Gemini's ~28×.)

**The important difference from Gemini: OpenAI Realtime *has* prompt caching.** Repeated
context that stays in the prefix is billed at the **cached input rate ($0.40/1M, 80×
below uncached audio)**, so on a warm cache the re-attended audio history is *already*
cheap — the "re-billed at full audio price every turn" premise that drives Gemini's
~28× simply **does not hold** for OpenAI. So ContextRelay's win on OpenAI is **real but
more modest**, and comes from three places, not from a 28× per-turn multiplier:

1. **Fewer tokens.** Even at the cached rate, text is several× more compact than audio,
   so the carried context is smaller every turn.
2. **Cache-miss protection.** The realtime cache has a short TTL; after an idle gap the
   prefix falls out of cache and the **entire audio history is re-billed at the full
   $32/1M** on the next turn. Carrying compact text bounds that worst case.
3. **Context-window + session headroom.** Audio fills the context window ~7× faster than
   text; converting to text extends how long a call can run before it hits the window or
   a session cap — the same memory lever as §2, independent of price.

**Snapshot at 15 minutes (`gpt-realtime-2`) — the context re-attended per turn**
**[ESTIMATE from CITED rates]** (mixed audio ≈ 900 tok/min → ~13.5k; text ≈ 214 tok/min
→ ~3.2k; the audio↔text *ratio* is robust, the absolute dollars depend on call shape):

| Context carried | Tokens | $/turn — **warm cache** | $/turn — **cache miss / cold** |
|---|---|---|---|
| Full call as **audio** | ~13,500 | $0.0054 (@ $0.40/1M cached) | **$0.432** (@ $32/1M) |
| Full call as **text** (ContextRelay) | ~3,200 | $0.0013 (@ $0.40/1M cached) | $0.0128 (@ $4/1M) |
| | | **≈ 4× cheaper** | **≈ 34× cheaper** |

So unlike Gemini's steady ~28×, OpenAI's win is a **range set by the cache state**: only
**~4×** when the whole prefix is a warm cache hit (the saving is then pure token-count
compaction), but **~34×** on a cold/missed cache, where the full audio history is
re-billed at $32/1M (audio is pricier than Gemini's, so the cold-cache multiplier is
actually *larger*). Real calls sit between the two, driven by idle gaps vs the cache
TTL — which is exactly the cache-miss insurance ContextRelay provides. The
qualitative takeaway holds: **"bounded growth, cache-miss insurance, and longer
calls,"** now with the bracket attached. **[ESTIMATE from CITED rates]**

Because that context is re-billed every turn, the saving compounds. *Illustrative
cumulative re-billed-input over a 15-min, ~45-turn call, cache-**cold**:* **~$10
(audio) vs ~$0.30 (text)** — **[ESTIMATE]** (context grows ~0→13.5k audio / ~0→3.2k
text, averaged over the turns × the cold rates above). A **warm** cache scales both
down ~80× (to **~$0.12 vs ~$0.004**), so the realized saving tracks the call's
cache-hit rate. As with Gemini this is the *re-billed-history* component; fresh-audio-in
and audio-out are unchanged.

**Per-turn audio/text split — to be MEASURED.** The Gemini section quotes a measured
per-turn split ("259 audio + 225 text"); the OpenAI analogue isn't captured yet because
the connector's `decode_usage` records only the **total** `input_tokens` (the 254→87
above), not OpenAI's `usage.input_token_details.{audio,text,cached}_tokens` breakdown. A
small connector follow-up (read those sub-fields) + a re-run would give the same measured
split, and a measured *dollar* figure rather than the estimate.

**X.AI Grok Realtime** speaks the same protocol (so the same re-base applies) but is
billed **per-minute (~$0.05/min)**, not per audio/text token **[CITED]**
([xAI pricing](https://docs.x.ai/developers/models)) — so there is no per-turn token
re-bill to convert away; its lever is purely the memory/long-call side.

**Provider applicability at a glance:**

| Provider | Re-base drops audio? | Per-turn token re-bill? | ContextRelay benefit |
|---|---|---|---|
| Gemini / Vertex | Yes (reconnect) | Yes, **no caching** | **Strong** — cost (~28×) + memory |
| OpenAI Realtime | Yes (reconnect, new) | Yes, but **cached** | **Moderate** — compaction + cache-miss insurance + headroom |
| Grok (X.AI) | Yes (reconnect, new) | **Per-minute billed** | Memory / long-call only |
| Inworld, Azure | Yes (inherit OpenAI) | As OpenAI | As OpenAI |
| **Ultravox** | **No** (prompt bound at REST create; no in-session/​reconnect update) | No usage event emitted | **Unsupported** — the relay's signals + re-base seam aren't available |

**Live validation — OpenAI Realtime [MEASURED].** A real `gpt-realtime` session
(`flowcat-services/tests/live_openai_context_relay.rs`, `#[ignore]`d) driven by a
macOS-`say` synthetic caller: the caller gives *"my account number is four four seven
two"*, one filler turn grows the audio context, then `rebase_session` reopens the
socket onto a text digest; the caller then asks for the number back. Transcribed bot
reply after the re-base:

> *"Your account number is **4472**."*

So recall survived the audio→text re-base. And the per-turn `input_tokens` the model
re-attended show the audio being dropped:

| Turn | input_tokens | |
|---|---|---|
| turn 1 (caller states the number) | 149 | audio context accumulating |
| turn 2 (filler) — **pre-rebase** | **254** | full audio history re-attended |
| turn 3 (recall) — **post-rebase** | **87** | fresh session re-attends only the text digest |

The re-base cut the re-attended context **254 → 87 (~66%)** and, crucially, the
post-rebase figure stays *bounded* (digest-sized) instead of climbing turn-over-turn
the way the audio history does. **[MEASURED]** (Note: needs `server_vad` — the
connector's default `semantic_vad` doesn't reliably endpoint synthetic `say` audio;
`OpenAiRealtime::with_server_vad()` selects it.)

> **X.AI (Grok) still to be MEASURED**, and the dollar figures above are derived from
> published rates, not live calls — Grok bills per-minute, so its lever is the
> memory/long-call side; a live Grok run with the same harness is the remaining check.

---

## 4. Honest trade-offs

- **Reseed latency.** Converting + reseeding briefly re-establishes the session; flowcat
  fires it at a turn boundary (idle gap) to hide it, but it is not free.
- **Native-audio nuance.** The reseeded session reads a **text** transcript of the
  converted portion, so prosody/tone of those earlier turns is not carried — the *words*
  are. Fine for task/support calls; a consideration for tone-sensitive ones.
- **Text growth.** `VerbatimCompactor`'s transcript grows with the call — but even a
  30-min transcript (~6.5k text tokens ≈ $0.005/turn) is trivial next to the audio it
  replaces; `LlmCompactor` bounds it further.
- **Validation is demonstrative**, on synthetic test calls — not a statistical benchmark.
  A fully-controlled forget-vs-remember A/B needs a sturdier caller harness (forcing
  aggressive native eviction destabilized our synthetic conversation), so §2's
  "native compression forgets" half rests on Google's **documented** eviction behavior,
  and §3 measures ContextRelay's **preservation** directly.

---

## 5. When to reach for it

- **Long calls where early context must survive** (IDs, names, commitments stated up front
  and needed later) — native eviction would drop them; ContextRelay keeps them, cheaply.
- **Calls that outrun the 15-minute audio cap** — reseed into a fresh session with a
  context you control.
- **Provider-portability** — ContextRelay rides only the realtime trait surface, so the
  same audio→text re-base applies to the OpenAI Realtime family (OpenAI, X.AI Grok,
  Inworld, Azure) as well as Gemini/Vertex — see §3a for the per-provider economics
  (the *cost* win is strongest on Gemini, more modest on the cache-having OpenAI family,
  and memory-only on per-minute-billed Grok). Ultravox is unsupported (§3a table).

---

## 6. Reproduction

```bash
# Gemini key in the environment (GOOGLE_API_KEY / GEMINI_API_KEY)
export FLOWCAT_CONTEXT_RELAY=1
export FLOWCAT_CONTEXT_RELAY_MAX_SESSION_SECS=6                    # force reseeds mid-call
export FLOWCAT_CONTEXT_RELAY_SUMMARIZER=gemini/gemini-2.5-flash    # optional: LLM summary
cargo run -p flowcat-server --features webrtc -- --config agent.yaml   # realtime: gemini, model: models/gemini-3.1-flash-live-preview
```

For **OpenAI Realtime** (or X.AI Grok — same flags, `realtime: grok` + `GROK_API_KEY`):

```bash
export OPENAI_API_KEY=sk-…
export FLOWCAT_CONTEXT_RELAY=1
export FLOWCAT_CONTEXT_RELAY_MAX_SESSION_SECS=6                    # force re-bases mid-call
export FLOWCAT_CONTEXT_RELAY_SUMMARIZER=openai/gpt-4o-mini         # optional: LLM summary
# build with the realtime-openai (or realtime-grok) connector feature enabled
cargo run -p flowcat-server --features webrtc,flowcat-services/realtime-openai -- --config agent.yaml
```
Reseeds log as `context-relay: re-basing realtime session onto text digest`. On the
OpenAI family a re-base reopens the socket (drops the audio history); on Gemini it
reconnects with a fresh setup — both carry the text digest into the new session.

## Sources

- OpenAI API pricing (gpt-realtime-2) — https://developers.openai.com/api/docs/pricing
- OpenAI Realtime cost guide (audio token metering) — https://developers.openai.com/api/docs/guides/realtime-costs
- xAI models / pricing (Grok Realtime per-minute) — https://docs.x.ai/developers/models
- Gemini Live API best practices — https://ai.google.dev/gemini-api/docs/live-api/best-practices
- Gemini Live API capabilities — https://ai.google.dev/gemini-api/docs/live-api/capabilities
- Gemini Live session management — https://ai.google.dev/gemini-api/docs/live-session
- Gemini Live API reference (`contextWindowCompression`) — https://ai.google.dev/api/live
- Gemini API pricing — https://ai.google.dev/gemini-api/docs/pricing
- Gemini API tokens — https://ai.google.dev/gemini-api/docs/tokens
- No Live caching (Google staff) — https://discuss.ai.google.dev/t/gemini-live-caching/83298

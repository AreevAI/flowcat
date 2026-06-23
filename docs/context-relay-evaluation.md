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
  same audio→text+reseed applies to OpenAI Realtime, Nova Sonic, etc., not just Gemini.

---

## 6. Reproduction

```bash
# Gemini key in the environment (GOOGLE_API_KEY / GEMINI_API_KEY)
export FLOWCAT_CONTEXT_RELAY=1
export FLOWCAT_CONTEXT_RELAY_MAX_SESSION_SECS=6                    # force reseeds mid-call
export FLOWCAT_CONTEXT_RELAY_SUMMARIZER=gemini/gemini-2.5-flash    # optional: LLM summary
cargo run -p flowcat-server --features webrtc -- --config agent.yaml   # realtime: gemini, model: models/gemini-3.1-flash-live-preview
```
Reseeds log as `context-relay: re-basing realtime session onto text digest`.

## Sources

- Gemini Live API best practices — https://ai.google.dev/gemini-api/docs/live-api/best-practices
- Gemini Live API capabilities — https://ai.google.dev/gemini-api/docs/live-api/capabilities
- Gemini Live session management — https://ai.google.dev/gemini-api/docs/live-session
- Gemini Live API reference (`contextWindowCompression`) — https://ai.google.dev/api/live
- Gemini API pricing — https://ai.google.dev/gemini-api/docs/pricing
- Gemini API tokens — https://ai.google.dev/gemini-api/docs/tokens
- No Live caching (Google staff) — https://discuss.ai.google.dev/t/gemini-live-caching/83298

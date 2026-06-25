// SPDX-License-Identifier: Apache-2.0
//
//! Live OpenAI Realtime **ContextRelay re-base** validation (`#[ignore]`d).
//!
//! Proves the OpenAI-family re-base actually does what ContextRelay needs: reopen
//! the session onto a compact **text** digest and **drop the accumulated audio**,
//! while keeping the conversation. It drives a REAL `OpenAiRealtime` session with
//! macOS `say`-generated caller speech (no extra provider key needed for the
//! caller side — the OpenAI session transcribes its own input/output), gives the
//! bot an account number, grows the audio context, then calls
//! [`rebase_session`](flowcat_core::service::RealtimeLlmService::rebase_session) and
//! asserts:
//!
//!   (a) the bot still **recalls the account number** after the re-base (carried
//!       forward as text), and
//!   (b) the per-turn **`input_tokens` drop** after the re-base — i.e. the expensive
//!       audio history was evicted, not re-attended.
//!
//! This is the OpenAI analogue of the Gemini "4472" live check in
//! `docs/context-relay-evaluation.md` §3. macOS-only (uses `say`).
//!
//! ```bash
//! OPENAI_API_KEY=sk-… cargo test -p flowcat-services --features realtime-openai \
//!   --test live_openai_context_relay -- --ignored --nocapture
//! ```

#![cfg(feature = "realtime-openai")]

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use flowcat_core::processor::frame::AudioFrame;
use flowcat_core::service::{RealtimeLlmService, RealtimeServiceSetup};
use flowcat_core::types::RealtimeEvent;
use flowcat_services::realtime::OpenAiRealtime;

/// OpenAI Realtime PCM rate (mono 16-bit), in both directions.
const RATE: u32 = 24_000;

/// Synthesize `text` to 24 kHz mono LE PCM via macOS `say`, returning the samples.
fn say_pcm(text: &str) -> Vec<i16> {
    let tag: String = text
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(16)
        .collect();
    let path = std::env::temp_dir().join(format!("flowcat_say_{tag}.wav"));
    let status = Command::new("say")
        .args([
            "-o",
            path.to_str().expect("temp path utf-8"),
            "--data-format=LEI16@24000",
            text,
        ])
        .status()
        .expect("run macOS `say` (this test is macOS-only)");
    assert!(status.success(), "`say` exited non-zero");
    let bytes = std::fs::read(&path).expect("read say wav");
    let _ = std::fs::remove_file(&path);
    wav_pcm(&bytes)
}

/// Extract LE i16 PCM from a WAV file's `data` chunk (scans for the chunk id so a
/// non-canonical header layout still parses).
fn wav_pcm(bytes: &[u8]) -> Vec<i16> {
    let pos = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("WAV `data` chunk");
    let data = &bytes[pos + 8..]; // skip the "data" id + its 4-byte length
    data.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Stream `text` as caller audio followed by a ~2.5 s low-amplitude noise floor,
/// **paced near real time**. The pacing + continuous tail is what lets the server VAD
/// see the end-of-turn pause and auto-commit the utterance (a fast burst that then
/// stops sending never advances the VAD's silence clock, so the turn never ends).
/// Mimics a real caller whose mic keeps streaming after they finish talking.
async fn say_turn(rt: &mut OpenAiRealtime, text: &str) {
    let mut pcm = say_pcm(text);
    // ~2.5 s of a low-amplitude NOISE FLOOR (not pure zeros): a VAD reads digital
    // zeros as "no audio", but real mic silence has a quiet noise floor that it reads
    // as the end-of-turn *pause* — which is what makes it endpoint + auto-commit.
    let mut seed: u32 = 0x9E37_79B9;
    let noise = (0..(5 * RATE as usize / 2)).map(|_| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 16) % 33) as i16 - 16 // ±16 ≈ −66 dBFS
    });
    pcm.extend(noise);
    for frame in pcm.chunks(480) {
        // 20 ms @ 24 kHz, sent at ~real time so the VAD's silence timer advances.
        rt.send_audio(Arc::new(AudioFrame::mono(frame.to_vec(), RATE)))
            .await
            .expect("send_audio");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    eprintln!("  (streamed {} samples for {text:?})", pcm.len());
}

/// Drain events until the bot turn ends (a `Usage` report), collecting the bot
/// transcript. Returns `(bot_text_lowercased, input_tokens)`. Bounded so a stuck
/// turn fails the test instead of hanging.
async fn drain_turn(rt: &mut OpenAiRealtime) -> (String, Option<u64>) {
    let fut = async {
        let mut bot = String::new();
        loop {
            match rt.next_event().await {
                Some(RealtimeEvent::BotText(t)) => {
                    eprintln!("  ev BotText {t:?}");
                    bot.push_str(&t);
                }
                Some(RealtimeEvent::Usage(u)) => {
                    eprintln!("  ev Usage input_tokens={:?}", u.input_tokens);
                    return (bot.to_lowercase(), u.input_tokens);
                }
                Some(RealtimeEvent::UserText(t)) => eprintln!("  ev UserText {t:?}"),
                Some(RealtimeEvent::UserInterimText(_)) => eprintln!("  ev UserInterimText"),
                Some(RealtimeEvent::AudioOut(c)) => {
                    eprintln!("  ev AudioOut {} samples", c.pcm.len())
                }
                Some(RealtimeEvent::Interrupted) => eprintln!("  ev Interrupted"),
                Some(RealtimeEvent::ToolCall { name, .. }) => eprintln!("  ev ToolCall {name}"),
                Some(RealtimeEvent::Closed) | None => {
                    eprintln!("  ev Closed/None");
                    return (bot.to_lowercase(), None);
                }
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(45), fut)
        .await
        .expect("bot turn timed out")
}

/// True if `s` contains the account number in digit or spoken form.
fn mentions_account(s: &str) -> bool {
    s.contains("4472") || s.contains("four four seven two") || s.contains("4 4 7 2")
}

/// The shared re-base validation, parameterized by an already-built client + its
/// `model`. The same OpenAI-protocol path serves OpenAI Realtime *and* xAI Grok
/// (which is `OpenAiRealtime` pointed at the xAI base URL — see the Grok test), so
/// both providers run the identical script + assertions.
async fn run_rebase_validation(mut rt: OpenAiRealtime, model: &str) {
    rt.connect(RealtimeServiceSetup {
        model: model.into(),
        system_prompt: "You are a concise phone support agent. Reply in English in one short \
                        sentence. Remember any facts the caller gives you, especially account \
                        numbers."
            .into(),
        tools: vec![],
        input_sample_rate: RATE,
        output_sample_rate: RATE,
    })
    .await
    .expect("connect");

    // The bot greets first (kickoff → response.create); drain that turn.
    rt.kickoff().await.expect("kickoff");
    let (greet, _g) = drain_turn(&mut rt).await;
    eprintln!("[greeting] {greet}");

    // Turn 1 — the caller states the account number (now in the AUDIO history).
    say_turn(&mut rt, "My account number is four four seven two.").await;
    let (b1, t1) = drain_turn(&mut rt).await;
    eprintln!("[turn1 bot] {b1}   input_tokens={t1:?}");

    // Turn 2 — filler that grows the re-attended AUDIO context.
    say_turn(
        &mut rt,
        "Also, my internet has been very slow since yesterday evening.",
    )
    .await;
    let (b2, t2) = drain_turn(&mut rt).await;
    eprintln!("[turn2 bot] {b2}   input_tokens={t2:?}");
    let pre_rebase = t2.expect("turn 2 reported usage with input_tokens");

    // RE-BASE — fold the conversation into TEXT and reopen the session. On the OpenAI
    // family this `rebase_session` reopens the socket, so the accumulated audio is
    // dropped and only this cheap text digest seeds the fresh session.
    rt.rebase_session(
        "You are a concise phone support agent. Reply in English in one short sentence. \
         Conversation so far (preserved as text across a session refresh): the caller's account \
         number is 4472; the caller reports slow internet since yesterday evening. Remember these."
            .into(),
        vec![],
    )
    .await
    .expect("rebase_session (audio-dropping reopen)");
    eprintln!("[rebase] session reopened onto the text digest");

    // Turn 3 (post-rebase) — ask for recall on the FRESH session.
    say_turn(&mut rt, "Can you remind me what my account number is?").await;
    let (b3, t3) = drain_turn(&mut rt).await;
    eprintln!("[turn3 bot] {b3}   input_tokens={t3:?}");
    let post_rebase = t3.expect("turn 3 reported usage with input_tokens");

    eprintln!(
        "[MEASURED] pre-rebase input_tokens={pre_rebase}  post-rebase input_tokens={post_rebase}"
    );

    // (a) Recall survived the re-base (carried forward as text, not audio).
    assert!(
        mentions_account(&b3),
        "bot must recall the account number after the re-base; got: {b3:?}"
    );
    // (b) The re-base shrank the re-attended context — the audio history was evicted,
    //     so the fresh session re-attends only the compact text digest.
    assert!(
        post_rebase < pre_rebase,
        "re-base must drop audio (post-rebase input_tokens should be < pre-rebase): \
         pre={pre_rebase} post={post_rebase}"
    );
}

#[tokio::test]
#[ignore = "live: needs OPENAI_API_KEY + macOS `say` + network (OpenAI Realtime)"]
async fn openai_context_relay_rebase_drops_audio_and_keeps_recall() {
    let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY for the live re-base test");
    // server VAD (silence-duration endpointing) — semantic VAD does not reliably
    // endpoint synthetic `say` speech, so the bot never takes its turn.
    let rt = OpenAiRealtime::new(key)
        .with_input_language(Some("en".into()))
        .with_server_vad();
    run_rebase_validation(rt, "gpt-realtime").await;
}

/// xAI Grok speaks the OpenAI Realtime wire protocol, so `GrokRealtime` is just
/// `OpenAiRealtime` pointed at the xAI base URL and **delegates `rebase_session` to
/// it** — the exact reconnecting re-base the OpenAI test proves. We drive that same
/// client directly here so the live xAI endpoint exercises the identical path. (Needs
/// an xAI account with Realtime access + credits; a quota-blocked account 429s at
/// connect.)
#[tokio::test]
#[ignore = "live: needs XAI_API_KEY + macOS `say` + network + xAI Realtime access"]
async fn grok_context_relay_rebase_drops_audio_and_keeps_recall() {
    let key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY for the live Grok re-base test");
    let rt = OpenAiRealtime::new(key)
        .with_base_url("wss://api.x.ai/v1/realtime")
        .with_input_language(Some("en".into()))
        .with_server_vad();
    run_rebase_validation(rt, "grok-realtime").await;
}

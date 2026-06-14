// SPDX-License-Identifier: Apache-2.0
//
//! Live end-to-end cascaded turn with REAL providers, driven through flowcat-core's
//! `build_cascaded_task`: a synthesized user utterance → real Deepgram STT → real
//! Gemini LLM → real Deepgram TTS → captured bot audio + transcript.
//!
//! `#[ignore]` (needs DEEPGRAM_API_KEY + GOOGLE_API_KEY). Run:
//! `DEEPGRAM_API_KEY=… GOOGLE_API_KEY=… cargo test -p flowcat-services \
//!   --features stt-deepgram,llm-google,tts-deepgram --test live_cascaded_turn \
//!   -- --ignored --nocapture`
#![cfg(all(
    feature = "stt-deepgram",
    feature = "llm-google",
    feature = "tts-deepgram"
))]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::pipeline::{build_cascaded_task, CascadedConfig};
use flowcat_core::processor::frame::Frame;
use flowcat_core::service::TtsService;
use flowcat_core::types::{
    AudioChunk, BrainAction, Finalize, ResolvedCall, ToolDecl, UploadTarget,
};
use flowcat_core::{AgentBrain, FlowcatError, MediaIn, MediaTransport, SessionSource};

use flowcat_services::llm::google::GoogleLlm;
use flowcat_services::stt::DeepgramStt;
use flowcat_services::tts::deepgram::DeepgramTts;

const CARRIER_RATE: u32 = 16_000;

/// A transport that feeds a fixed user utterance, then keeps the call alive while
/// the STT→LLM→TTS round-trip completes, then ends. Captures the bot's output audio.
struct LiveTransport {
    inbound: VecDeque<MediaIn>,
    post_done: bool,
    out_pcm: Arc<Mutex<Vec<i16>>>,
}
#[async_trait]
impl MediaTransport for LiveTransport {
    async fn recv(&mut self) -> Option<MediaIn> {
        match self.inbound.pop_front() {
            // Pace audio at ~100 ms/frame (wall-clock, like real telephony) so the
            // streaming STT reader is drained as Deepgram transcribes — a burst
            // would let the final transcript arrive after the last `run_stt` call.
            Some(MediaIn::Audio(a)) => {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Some(MediaIn::Audio(a))
            }
            Some(other) => Some(other),
            None => {
                if !self.post_done {
                    self.post_done = true;
                    // Hold the call open after the audio so the LLM + TTS reply
                    // completes (and is captured), then end.
                    tokio::time::sleep(Duration::from_secs(8)).await;
                    return Some(MediaIn::Stop);
                }
                None
            }
        }
    }
    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError> {
        self.out_pcm.lock().unwrap().extend_from_slice(&chunk.pcm);
        Ok(())
    }
    async fn send_clear(&mut self) -> Result<(), FlowcatError> {
        Ok(())
    }
    fn carrier_rate(&self) -> u32 {
        CARRIER_RATE
    }
}

/// A trivial brain: one node, no transitions, never finishes (the transport Stop
/// ends the call). Provides the system prompt for the cascaded LLM.
struct StayBrain;
impl AgentBrain for StayBrain {
    fn system_prompt(&self) -> String {
        "You are a helpful voice assistant. Answer in one short sentence.".into()
    }
    fn tools(&self) -> Vec<ToolDecl> {
        vec![]
    }
    fn current_node_id(&self) -> String {
        "start".into()
    }
    fn on_tool_call(&mut self, _name: &str, _args: &Value) -> BrainAction {
        BrainAction::Stay
    }
    fn is_finished(&self) -> bool {
        false
    }
    fn collected_vars(&self) -> Value {
        json!({})
    }
}

/// Captures the finalize payload + uploaded artifacts (transcript/recording) in memory.
struct CaptureSession {
    artifacts: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    finalized: Arc<Mutex<Option<Finalize>>>,
}
#[async_trait]
impl SessionSource for CaptureSession {
    async fn resolve(&self, _run: i64, _token: &str) -> Result<ResolvedCall, FlowcatError> {
        Ok(ResolvedCall {
            provider: "test".into(),
            brain_config: json!({}),
            is_completed: false,
        })
    }
    async fn complete(&self, _run: i64, _token: &str, fin: Finalize) -> Result<(), FlowcatError> {
        *self.finalized.lock().unwrap() = Some(fin);
        Ok(())
    }
    async fn artifact_upload_url(
        &self,
        _run: i64,
        _token: &str,
        kind: &str,
    ) -> Result<UploadTarget, FlowcatError> {
        Ok(UploadTarget {
            url: format!("mem://{kind}"),
            key: format!("artifacts/{kind}"),
            content_type: "application/octet-stream".into(),
        })
    }
    async fn put_bytes(
        &self,
        url: &str,
        bytes: Vec<u8>,
        _content_type: &str,
    ) -> Result<(), FlowcatError> {
        let kind = url.trim_start_matches("mem://").to_string();
        self.artifacts.lock().unwrap().insert(kind, bytes);
        Ok(())
    }
    async fn node_tools(
        &self,
        _run: i64,
        _token: &str,
        _node: &str,
    ) -> Result<Vec<ToolDecl>, FlowcatError> {
        Ok(vec![])
    }
    async fn tool_call(
        &self,
        _run: i64,
        _token: &str,
        _node: &str,
        _tool: &str,
        _args: &Value,
    ) -> Result<String, FlowcatError> {
        Ok(String::new())
    }
}

/// Synthesize the user utterance with real Deepgram TTS (16 kHz) so the cascaded
/// pipeline gets genuine speech to transcribe.
async fn synth_user_audio(dg_key: &str, text: &str) -> AudioChunk {
    let mut tts = DeepgramTts::new(dg_key, "aura-2-thalia-en").sample_rate(CARRIER_RATE);
    tts.start(&Default::default()).await.expect("tts start");
    let frames = tts.run_tts(text).await.expect("tts run");
    let mut pcm = Vec::new();
    for f in frames {
        if let Frame::TtsAudio { audio, .. } = f {
            pcm.extend_from_slice(&audio.pcm);
        }
    }
    assert!(!pcm.is_empty(), "user-audio synthesis produced no PCM");
    AudioChunk::new(pcm, CARRIER_RATE)
}

/// Wrap 16-bit mono PCM as a WAV byte stream.
fn wav(pcm: &[i16], rate: u32) -> Vec<u8> {
    let data_len = (pcm.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    b.extend_from_slice(&2u16.to_le_bytes()); // block align
    b.extend_from_slice(&16u16.to_le_bytes()); // bits
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b
}

/// Transcribe PCM with Deepgram's prerecorded REST API (to read back the bot's
/// spoken reply — proves what the cascaded chain actually said).
async fn transcribe_prerecorded(dg_key: &str, pcm: &[i16], rate: u32) -> String {
    let body = wav(pcm, rate);
    let resp = reqwest::Client::new()
        .post("https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true")
        .header("Authorization", format!("Token {dg_key}"))
        .header("Content-Type", "audio/wav")
        .body(body)
        .send()
        .await
        .expect("deepgram prerecorded");
    let v: Value = resp.json().await.expect("deepgram json");
    v["results"]["channels"][0]["alternatives"][0]["transcript"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "live: needs DEEPGRAM_API_KEY + GOOGLE_API_KEY"]
async fn cascaded_turn_real_deepgram_gemini_deepgram() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let dg = std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY");
    let google = std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY");

    // 1) Real user utterance via TTS.
    let user = synth_user_audio(&dg, "What is two plus two? Answer in one short sentence.").await;
    eprintln!(
        "user audio: {} samples @ {} Hz",
        user.pcm.len(),
        user.sample_rate
    );

    // 2) Inbound script: stream start, the utterance in ~100 ms chunks, then the
    //    transport holds the call open and Stops (handled in `recv`).
    let mut inbound = VecDeque::new();
    inbound.push_back(MediaIn::StreamStart {
        call_id: "live-cascaded".into(),
    });
    for chunk in user.pcm.chunks(1600) {
        inbound.push_back(MediaIn::Audio(AudioChunk::new(
            chunk.to_vec(),
            CARRIER_RATE,
        )));
    }
    // Trailing silence (~2 s) so Deepgram endpoints the utterance and emits a
    // final transcript while the connection is open (else a burst-then-stop
    // stream never gets a `speech_final` and the cascaded LLM never fires).
    for _ in 0..20 {
        inbound.push_back(MediaIn::Audio(AudioChunk::new(
            vec![0i16; 1600],
            CARRIER_RATE,
        )));
    }

    let out_pcm = Arc::new(Mutex::new(Vec::<i16>::new()));
    let artifacts = Arc::new(Mutex::new(HashMap::new()));
    let finalized = Arc::new(Mutex::new(None));

    let transport = LiveTransport {
        inbound,
        post_done: false,
        out_pcm: out_pcm.clone(),
    };
    let stt = DeepgramStt::new(&dg);
    let llm = GoogleLlm::new(&google).model("gemini-3.5-flash");
    let tts = DeepgramTts::new(&dg, "aura-2-thalia-en").sample_rate(CARRIER_RATE);
    let brain = StayBrain;
    let session = CaptureSession {
        artifacts: artifacts.clone(),
        finalized: finalized.clone(),
    };

    let config = CascadedConfig {
        system_prompt: Some(
            "You are a helpful voice assistant. Answer in one short sentence.".into(),
        ),
        ..Default::default()
    };

    // 3) Build + run the real cascaded pipeline.
    let task = build_cascaded_task(
        transport,
        stt,
        llm,
        tts,
        brain,
        session,
        4242,
        "tok".into(),
        config,
    )
    .await
    .expect("build_cascaded_task");
    tokio::time::timeout(Duration::from_secs(50), task.run())
        .await
        .expect("cascaded task timed out")
        .expect("cascaded task run");

    // 4) Read back the bot's reply by transcribing the captured TTS audio.
    let bot_pcm = out_pcm.lock().unwrap().clone();
    let bot_said = if bot_pcm.is_empty() {
        String::new()
    } else {
        transcribe_prerecorded(&dg, &bot_pcm, CARRIER_RATE).await
    };

    // The in-pipeline transcript artifact (the collector under test).
    let transcript = artifacts
        .lock()
        .unwrap()
        .get("transcript")
        .map(|b| String::from_utf8_lossy(b).to_string())
        .unwrap_or_default();

    eprintln!("--- cascaded turn (real Deepgram STT → Gemini 3.5-flash → Deepgram TTS) ---");
    eprintln!("USER said (synthesized): \"What is two plus two? Answer in one short sentence.\"");
    eprintln!(
        "BOT  audio: {} samples (~{:.1}s)",
        bot_pcm.len(),
        bot_pcm.len() as f32 / CARRIER_RATE as f32
    );
    eprintln!("BOT  said (transcribed back): {bot_said:?}");
    eprintln!("pipeline transcript artifact: {transcript}");
    eprintln!("finalized: {}", finalized.lock().unwrap().is_some());

    assert!(
        !bot_pcm.is_empty(),
        "expected the cascaded pipeline to produce bot TTS audio (STT→LLM→TTS chain)"
    );
    // The transcript collector now captures both the user line (tapped in the
    // user-aggregator) and the bot reply (LlmText → TranscriptProcessor).
    let tlow = transcript.to_lowercase();
    assert!(
        tlow.contains("two") && (tlow.contains("four") || tlow.contains("4")),
        "transcript should contain the user question + the bot answer; got: {transcript}"
    );
    let lower = bot_said.to_lowercase();
    assert!(
        lower.contains("four") || lower.contains("4"),
        "the bot's reply should answer 2+2=4; got: {bot_said:?}"
    );
}

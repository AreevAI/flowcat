// SPDX-License-Identifier: Apache-2.0
//
//! The speech-to-speech model abstraction. `GeminiLive` is the first impl.
//!
//! See DESIGN.md "Trait contracts" + "Gemini Live protocol". Audio in is 16 kHz
//! PCM; audio out (in [`RealtimeEvent::AudioOut`]) is 24 kHz PCM for Gemini Live.

pub mod gemini_live;
pub mod service_adapter;

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Notify;

use crate::error::FlowcatError;
use crate::types::{AudioChunk, RealtimeEvent, RealtimeSetup, ToolDecl};

pub use gemini_live::{gemini_schema_subset, GeminiLive};
pub use service_adapter::ServiceRealtimeAdapter;

/// Result of a **non-blocking** [`RealtimeLlm::poll_event`] poll.
///
/// The S2S reader uses this to avoid holding the realtime session lock across a
/// blocking `next_event().await` — which would starve `send_audio` (caller audio
/// in) for the entire gap between bot turns and deadlock the call. The reader
/// instead polls (brief lock), and on [`PollEvent::Pending`] awaits
/// [`RealtimeLlm::event_notify`] with **no lock held**, so `send_audio` runs.
pub enum PollEvent {
    /// An event (or `None` = session closed) is ready now.
    Ready(Option<RealtimeEvent>),
    /// No event is ready; await `event_notify()` then poll again.
    Pending,
}

/// A bidirectional, streaming speech-to-speech model session.
///
/// Driven by the [S2S pipeline](crate::pipeline::build_s2s_task): push caller audio
/// with `send_audio`, pull model events (bot audio, transcripts, tool calls, …) with
/// `next_event`, and swap the active prompt/tools on a transition with
/// `update_system`.
#[async_trait]
pub trait RealtimeLlm: Send {
    /// Open the session with the initial system prompt + tools + audio rates.
    async fn connect(&mut self, setup: RealtimeSetup) -> Result<(), FlowcatError>;

    /// Stream a chunk of caller audio (16 kHz PCM) to the model.
    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError>;

    /// Replace the active system prompt and tool set (on a brain transition).
    async fn update_system(
        &mut self,
        prompt: String,
        tools: Vec<ToolDecl>,
    ) -> Result<(), FlowcatError>;

    /// Return the result of a tool/function call back to the model.
    async fn send_tool_result(
        &mut self,
        id: String,
        result: serde_json::Value,
    ) -> Result<(), FlowcatError>;

    /// Await the next model event, or `None` when the session has closed.
    async fn next_event(&mut self) -> Option<RealtimeEvent>;

    /// A [`Notify`] that fires whenever a model event becomes available, so a
    /// consumer can await readiness WITHOUT holding this session's lock across
    /// `next_event` (which would starve `send_audio`). `None` (the default) means
    /// the provider doesn't support the non-blocking path and the consumer falls
    /// back to the blocking `next_event`. [`GeminiLive`] returns `Some`.
    fn event_notify(&self) -> Option<Arc<Notify>> {
        None
    }

    /// Non-blocking sibling of [`next_event`](Self::next_event): return the next
    /// event if one is immediately ready, else [`PollEvent::Pending`] (the caller
    /// should `event_notify().notified().await` — lock released — then re-poll).
    /// The default blocks (legacy behaviour); providers that return a real
    /// `event_notify` MUST override this to be genuinely non-blocking.
    async fn poll_event(&mut self) -> PollEvent {
        PollEvent::Ready(self.next_event().await)
    }

    /// The PCM sample rate (Hz) the model expects on its audio **input**. The s2s
    /// pipeline resamples caller audio to this before [`send_audio`](Self::send_audio)
    /// and advertises it to the model in [`RealtimeSetup::input_sample_rate`]. Default
    /// 16 kHz (Gemini Live); a provider that requires a different input rate (OpenAI
    /// Realtime needs ≥ 24 kHz) overrides this.
    fn input_sample_rate(&self) -> u32 {
        16_000
    }
}

/// The bot-first "kickoff" capability.
///
/// Triggering an opening model turn is **not** part of the [`RealtimeLlm`] trait
/// (it is provider-specific — how, or whether, to seed a first turn varies by
/// backend). The pipeline still needs to call it generically, so this extension
/// trait carries it. [`GeminiLive`] implements it (delegating to its inherent
/// `kickoff`, in `gemini_live.rs`); other realtime backends (and the test mocks)
/// implement it as their protocol requires.
#[async_trait]
pub trait RealtimeKickoff {
    /// Trigger an initial model turn so the bot speaks first.
    async fn kickoff(&mut self) -> Result<(), FlowcatError>;
}

/// A boxable realtime backend: anything that is both a [`RealtimeLlm`] and a
/// [`RealtimeKickoff`]. The blanket impl makes every such type qualify, and the
/// `Box<dyn RealtimeBackend>` impls below let a consuming app's `build_realtime`
/// return ONE boxed type across providers (e.g. [`GeminiLive`] directly, or a
/// connector wrapped in [`ServiceRealtimeAdapter`]) and still hand it to
/// [`build_s2s_task`](crate::pipeline::build_s2s_task) — which is generic over
/// `RealtimeLlm + RealtimeKickoff`. This is what keeps provider *selection* in the
/// app a set of one-liners with no per-provider glue (the realtime analogue of the
/// cascaded `Box<dyn LlmService>` factory).
pub trait RealtimeBackend: RealtimeLlm + RealtimeKickoff + Send {}
impl<T: RealtimeLlm + RealtimeKickoff + Send> RealtimeBackend for T {}

#[async_trait]
impl RealtimeLlm for Box<dyn RealtimeBackend> {
    async fn connect(&mut self, setup: RealtimeSetup) -> Result<(), FlowcatError> {
        (**self).connect(setup).await
    }
    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError> {
        (**self).send_audio(chunk).await
    }
    async fn update_system(
        &mut self,
        prompt: String,
        tools: Vec<ToolDecl>,
    ) -> Result<(), FlowcatError> {
        (**self).update_system(prompt, tools).await
    }
    async fn send_tool_result(
        &mut self,
        id: String,
        result: serde_json::Value,
    ) -> Result<(), FlowcatError> {
        (**self).send_tool_result(id, result).await
    }
    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        (**self).next_event().await
    }
    fn event_notify(&self) -> Option<Arc<Notify>> {
        (**self).event_notify()
    }
    async fn poll_event(&mut self) -> PollEvent {
        (**self).poll_event().await
    }
    fn input_sample_rate(&self) -> u32 {
        (**self).input_sample_rate()
    }
}

#[async_trait]
impl RealtimeKickoff for Box<dyn RealtimeBackend> {
    async fn kickoff(&mut self) -> Result<(), FlowcatError> {
        (**self).kickoff().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;
    use tokio::sync::Mutex as AsyncMutex;

    /// A realtime mock whose events arrive out-of-band: `poll_event` returns
    /// `Pending` while the buffer is empty and fires `event_notify` when an event is
    /// delivered — mirroring [`GeminiLive`]'s lock-free contract so we can drive the
    /// s2s reader loop without a live socket.
    struct BufferedRealtime {
        notify: Arc<Notify>,
        events: StdMutex<VecDeque<Option<RealtimeEvent>>>,
    }
    impl BufferedRealtime {
        fn new() -> Self {
            Self {
                notify: Arc::new(Notify::new()),
                events: StdMutex::new(VecDeque::new()),
            }
        }
        fn deliver(&self, ev: Option<RealtimeEvent>) {
            self.events.lock().unwrap().push_back(ev);
            self.notify.notify_one();
        }
    }
    #[async_trait]
    impl RealtimeLlm for BufferedRealtime {
        async fn connect(&mut self, _: RealtimeSetup) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn send_audio(&mut self, _: AudioChunk) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn update_system(&mut self, _: String, _: Vec<ToolDecl>) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn send_tool_result(
            &mut self,
            _: String,
            _: serde_json::Value,
        ) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn next_event(&mut self) -> Option<RealtimeEvent> {
            self.events.lock().unwrap().pop_front().flatten()
        }
        fn event_notify(&self) -> Option<Arc<Notify>> {
            Some(self.notify.clone())
        }
        async fn poll_event(&mut self) -> PollEvent {
            match self.events.lock().unwrap().pop_front() {
                Some(ev) => PollEvent::Ready(ev),
                None => PollEvent::Pending,
            }
        }
    }

    /// The #91 deadlock invariant: while the reader is parked waiting for the next
    /// model event it must NOT hold the realtime session lock — else `send_audio`
    /// (caller audio in) is starved and the call dies after the greeting. We run the
    /// exact s2s reader loop (poll under a brief lock; on `Pending` await
    /// `event_notify` with the lock released; re-poll) and assert that a concurrent
    /// `send_audio` proceeds AND the reader wakes on the notify — all within a
    /// timeout (a regression would hang).
    #[tokio::test]
    async fn reader_releases_lock_while_waiting_then_wakes_on_notify() {
        let rt = Arc::new(AsyncMutex::new(BufferedRealtime::new()));
        let notify = rt.lock().await.event_notify().expect("provides a notify");

        // Empty buffer → poll is a non-blocking `Pending` (lock dropped after).
        assert!(matches!(
            rt.lock().await.poll_event().await,
            PollEvent::Pending
        ));

        let rt_reader = rt.clone();
        let reader = tokio::spawn(async move {
            loop {
                let polled = { rt_reader.lock().await.poll_event().await };
                match polled {
                    PollEvent::Ready(ev) => break ev,
                    PollEvent::Pending => notify.notified().await,
                }
            }
        });

        // Let the reader park on `notified()`. A concurrent caller-audio send must
        // then acquire the lock (this would hang if the reader still held it) and
        // deliver the event that wakes the reader.
        tokio::time::sleep(Duration::from_millis(20)).await;
        {
            let mut guard = rt.lock().await;
            guard
                .send_audio(AudioChunk::new(vec![0i16; 8], 16_000))
                .await
                .unwrap();
            guard.deliver(Some(RealtimeEvent::BotText("hi".into())));
        }

        let ev = tokio::time::timeout(Duration::from_secs(2), reader)
            .await
            .expect("no deadlock: the reader woke on the notify")
            .expect("reader task joined");
        assert!(matches!(ev, Some(RealtimeEvent::BotText(t)) if t == "hi"));
    }
}

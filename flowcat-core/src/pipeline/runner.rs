// SPDX-License-Identifier: Apache-2.0
//
//! `PipelineRunner` — supervise tasks + signals (PROCESSOR-DESIGN §4.2).
//!
//! Runs one or many [`PipelineTask`]s and installs SIGINT/SIGTERM handlers that
//! `cancel_all()` for graceful drain (the prod deploy-cutover drain story relies
//! on SIGTERM grace). Mirrors pipecat `PipelineRunner` (`runner.py:25`).

use std::sync::Mutex;

use tokio_util::sync::CancellationToken;

use crate::error::Result;

use super::PipelineTask;

/// Supervises [`PipelineTask`]s and (optionally) cancels them on OS signals.
pub struct PipelineRunner {
    handle_sigint: bool,
    handle_sigterm: bool,
    /// Cancellation tokens of currently-running tasks, so `cancel_all` reaches
    /// every one.
    tokens: Mutex<Vec<CancellationToken>>,
}

impl PipelineRunner {
    /// Build a runner. `handle_sigint`/`handle_sigterm` install signal-driven
    /// graceful cancellation while a task is running.
    pub fn new(handle_sigint: bool, handle_sigterm: bool) -> Self {
        Self {
            handle_sigint,
            handle_sigterm,
            tokens: Mutex::new(Vec::new()),
        }
    }

    /// Run `task` to completion, racing it against the configured OS signals (a
    /// signal triggers a graceful `cancel` of this task). Joins when the task
    /// finishes or the signal cancels it.
    pub async fn run(&self, task: PipelineTask) -> Result<()> {
        let token = task.cancel_token();
        if let Ok(mut t) = self.tokens.lock() {
            t.push(token.clone());
        }

        let signal_token = token.clone();
        let watcher_done = token.child_token();
        let watcher_done2 = watcher_done.clone();
        let handle_sigint = self.handle_sigint;
        let handle_sigterm = self.handle_sigterm;
        let signal_task = tokio::spawn(async move {
            tokio::select! {
                _ = watcher_done2.cancelled() => {}
                _ = wait_for_signal(handle_sigint, handle_sigterm, &signal_token) => {}
            }
        });

        let result = task.run().await;

        // The task finished — stop the signal watcher.
        watcher_done.cancel();
        let _ = signal_task.await;

        if let Ok(mut t) = self.tokens.lock() {
            t.retain(|c| !c.is_cancelled());
        }
        result
    }

    /// Cancel every running task (graceful drain). The prod SIGTERM path calls
    /// this.
    pub async fn cancel_all(&self) {
        let tokens = self.tokens.lock().map(|t| t.clone()).unwrap_or_default();
        for token in tokens {
            token.cancel();
        }
    }
}

/// Await SIGINT/SIGTERM (per flags) or the task's own completion (`done`), then
/// cancel the task token for graceful drain. On non-unix or when both flags are
/// off, this just waits for `done`.
async fn wait_for_signal(handle_sigint: bool, handle_sigterm: bool, token: &CancellationToken) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigint = handle_sigint
            .then(|| signal(SignalKind::interrupt()).ok())
            .flatten();
        let mut sigterm = handle_sigterm
            .then(|| signal(SignalKind::terminate()).ok())
            .flatten();
        tokio::select! {
            _ = token.cancelled() => {}
            _ = async { if let Some(s) = sigint.as_mut() { s.recv().await; } else { std::future::pending::<()>().await } }, if sigint.is_some() => {
                token.cancel();
            }
            _ = async { if let Some(s) = sigterm.as_mut() { s.recv().await; } else { std::future::pending::<()>().await } }, if sigterm.is_some() => {
                token.cancel();
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (handle_sigint, handle_sigterm);
        token.cancelled().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{Pipeline, PipelineTaskParams};
    use crate::processor::{Envelope, FrameProcessor, Link, StopReason};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    // A processor that never ends on its own — only a Cancel stops it.
    struct Idle;
    #[async_trait]
    impl FrameProcessor for Idle {
        fn name(&self) -> &str {
            "Idle"
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn simulated_signal_cancels_a_running_task() {
        // Disable the idle timeout so only cancel_all ends it.
        let params = PipelineTaskParams {
            idle_timeout: None,
            ..Default::default()
        };
        let pipeline = Pipeline::new(vec![Box::new(Idle)]);
        let cancelled = Arc::new(AtomicBool::new(false));
        let c2 = cancelled.clone();
        let mut task = PipelineTask::new(pipeline, params, vec![]);
        task.on_finished(move |r| {
            if r == StopReason::Cancelled {
                c2.store(true, Ordering::Relaxed);
            }
        });

        let runner = Arc::new(PipelineRunner::new(false, false));
        let token = task.cancel_token();
        // Register so cancel_all reaches it (run() also registers, but we want to
        // cancel from outside before/while it runs).
        let runner2 = runner.clone();
        let run_handle = tokio::spawn(async move { runner2.run(task).await });

        // Give run() a moment to register + go live, then simulate a signal.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = token; // run() registers its own token; cancel via cancel_all.
        runner.cancel_all().await;

        tokio::time::timeout(Duration::from_secs(3), run_handle)
            .await
            .expect("runner timed out")
            .expect("join")
            .expect("run ok");
        assert!(
            cancelled.load(Ordering::Relaxed),
            "task should cancel on signal"
        );
    }

    #[tokio::test]
    async fn multiple_tasks_join_cleanly() {
        let runner = PipelineRunner::new(false, false);
        for _ in 0..3 {
            let pipeline = Pipeline::new(vec![Box::new(Idle)]);
            let task = PipelineTask::new(pipeline, PipelineTaskParams::default(), vec![]);
            // Each ends itself via stop_when_done.
            task.stop_when_done().await;
            runner.run(task).await.unwrap();
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
//
//! Metrics carried on [`Frame::Metrics`](crate::processor::frame::Frame::Metrics).
//!
//! Field-for-field port of pipecat `metrics/metrics.py` (PROCESSOR-DESIGN §5.2):
//! TTFB, processing time, LLM/TTS usage, and turn-prediction metrics.

use serde::{Deserialize, Serialize};

/// One metrics record. Mirrors pipecat's `MetricsData` subclasses
/// (`metrics.py:29/39/68/78/101`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricsData {
    /// Time-to-first-byte for a service (`metrics.py:29`).
    Ttfb {
        processor: String,
        model: Option<String>,
        seconds: f64,
    },
    /// End-to-end processing time for one frame/turn (`metrics.py:39`).
    Processing {
        processor: String,
        model: Option<String>,
        seconds: f64,
    },
    /// LLM token usage (`metrics.py:68`).
    LlmUsage {
        processor: String,
        model: Option<String>,
        tokens: LlmTokenUsage,
    },
    /// TTS character usage (`metrics.py:78`).
    TtsUsage { processor: String, characters: u64 },
    /// End-of-turn prediction metric (`metrics.py:101`).
    TurnPrediction {
        processor: String,
        is_complete: bool,
        probability: f32,
        e2e_processing_ms: f64,
    },
}

/// LLM token accounting. Mirrors pipecat `LLMTokenUsage` (`metrics.py:49`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmTokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_data_serde_roundtrips_every_variant() {
        let cases = vec![
            MetricsData::Ttfb {
                processor: "stt".into(),
                model: Some("nova-2".into()),
                seconds: 0.123,
            },
            MetricsData::Processing {
                processor: "llm".into(),
                model: None,
                seconds: 1.5,
            },
            MetricsData::LlmUsage {
                processor: "llm".into(),
                model: Some("gpt-4o".into()),
                tokens: LlmTokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                    cache_read_input_tokens: Some(4),
                    cache_creation_input_tokens: None,
                    reasoning_tokens: Some(2),
                },
            },
            MetricsData::TtsUsage {
                processor: "tts".into(),
                characters: 42,
            },
            MetricsData::TurnPrediction {
                processor: "turn".into(),
                is_complete: true,
                probability: 0.97,
                e2e_processing_ms: 12.0,
            },
        ];
        for m in cases {
            let json = serde_json::to_string(&m).expect("serialize");
            let back: MetricsData = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(m, back);
        }
    }

    #[test]
    fn llm_token_usage_defaults_to_zero() {
        let u = LlmTokenUsage::default();
        assert_eq!(u.total_tokens, 0);
        assert!(u.cache_read_input_tokens.is_none());
    }
}

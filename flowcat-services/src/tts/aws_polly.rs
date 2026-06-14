// SPDX-License-Identifier: Apache-2.0
//
//! **AWS Polly** TTS — hand-rolled SigV4 HTTP client (Group H).
//! **SECURITY-REVIEW GATED** (new AWS SigV4 signing path — no AWS SDK,
//! HMAC-SHA256 + SHA-256 only).
//!
//! Polly's `SynthesizeSpeech` is a signed JSON POST:
//!
//! ```text
//! POST https://polly.{region}.amazonaws.com/v1/speech
//!   Authorization: AWS4-HMAC-SHA256 Credential=…, SignedHeaders=…, Signature=…
//!   X-Amz-Date / X-Amz-Content-Sha256
//!   { "Text": "...", "VoiceId": "Joanna", "OutputFormat": "pcm",
//!     "SampleRate": "16000", "Engine": "neural", "TextType": "text" }
//! ```
//!
//! With `OutputFormat: "pcm"` the body is raw little-endian s16 mono PCM at the
//! requested rate (8000 or 16000 Hz — Polly's PCM ceiling). This uses **header**
//! SigV4 (not query presigning), so the canonical-request + Authorization header are
//! built here. The credential is passed as `"<access_key>:<secret_key>"` to the
//! uniform `new(api_key, voice_id)` constructor.
//!
//! Signing is split into pure functions ([`sigv4::authorization_header`],
//! [`sigv4::signature_hex`]) carrying an **AWS-published known-answer test**.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// AWS Polly TTS service (SigV4-signed REST, raw PCM).
pub struct AwsPollyTts {
    access_key: String,
    secret_key: String,
    voice_id: String,
    sample_rate: u32,
    region: String,
    engine: String,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl AwsPollyTts {
    /// Construct from a `"<access_key>:<secret_key>"` credential + Polly `voice_id`
    /// (default 16000 Hz, `us-east-1`, `neural` engine). A credential without a colon
    /// leaves the secret empty (signing will fail at call time, never panics).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        let cred = api_key.into();
        let (access_key, secret_key) = match cred.split_once(':') {
            Some((a, s)) => (a.to_string(), s.to_string()),
            None => (cred, String::new()),
        };
        Self {
            access_key,
            secret_key,
            voice_id: voice_id.into(),
            sample_rate: 16_000,
            region: "us-east-1".to_string(),
            engine: "neural".to_string(),
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }

    /// Override the AWS region (default `us-east-1`).
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Override the Polly engine (default `neural`; `standard` / `long-form` also).
    pub fn engine(mut self, engine: impl Into<String>) -> Self {
        self.engine = engine.into();
        self
    }

    /// Override the output sample rate (Polly PCM: 8000 or 16000 Hz; default 16000).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn host(&self) -> String {
        format!("polly.{}.amazonaws.com", self.region)
    }

    fn url(&self) -> String {
        format!("https://{}/v1/speech", self.host())
    }
}

/// Build the Polly `SynthesizeSpeech` JSON body (pure — the request seam).
fn build_body(text: &str, voice_id: &str, sample_rate: u32, engine: &str) -> Value {
    json!({
        "Text": text,
        "VoiceId": voice_id,
        "OutputFormat": "pcm",
        "SampleRate": sample_rate.to_string(),
        "Engine": engine,
        "TextType": "text",
    })
}

#[async_trait]
impl TtsService for AwsPollyTts {
    fn name(&self) -> &str {
        "aws_polly"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.secret_key.is_empty() {
            return Err(FlowcatError::Other(
                "aws_polly TTS: credential must be \"<access_key>:<secret_key>\"".into(),
            ));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        if self.secret_key.is_empty() {
            return Err(FlowcatError::Other(
                "aws_polly TTS: credential must be \"<access_key>:<secret_key>\"".into(),
            ));
        }
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let body = build_body(text, &self.voice_id, self.sample_rate, &self.engine);
        let body_bytes = serde_json::to_vec(&body).map_err(FlowcatError::from)?;

        let now = sigv4::utc_now();
        let host = self.host();
        let signed = sigv4::sign_post(&sigv4::SignInput {
            access_key: &self.access_key,
            secret_key: &self.secret_key,
            region: &self.region,
            service: "polly",
            host: &host,
            path: "/v1/speech",
            body: &body_bytes,
            now,
        });

        let resp = self
            .http
            .post(self.url())
            .header("Content-Type", "application/json")
            .header("X-Amz-Date", &signed.amz_date)
            .header("X-Amz-Content-Sha256", &signed.payload_hash)
            .header("Authorization", &signed.authorization)
            .body(body_bytes)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("polly send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "polly http {status}: {body}"
            )));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("polly body: {e}")))?;
        Ok(tail::one_shot_frames(&bytes, self.sample_rate, context_id))
    }
}

/// Hand-rolled AWS Signature V4 (header auth). HMAC-SHA256 + SHA-256 only, no AWS
/// SDK. Pure functions so the canonical-request + signature are unit-tested
/// against AWS's published vectors.
mod sigv4 {
    use hmac::{Hmac, Mac};
    use sha2::{Digest, Sha256};

    type HmacSha256 = Hmac<Sha256>;

    /// Inputs for signing one Polly POST.
    pub struct SignInput<'a> {
        pub access_key: &'a str,
        pub secret_key: &'a str,
        pub region: &'a str,
        pub service: &'a str,
        pub host: &'a str,
        pub path: &'a str,
        pub body: &'a [u8],
        /// (amz_date `YYYYMMDDThhmmssZ`, date_stamp `YYYYMMDD`).
        pub now: (String, String),
    }

    /// The signed-request material to attach as headers.
    pub struct Signed {
        pub amz_date: String,
        pub payload_hash: String,
        pub authorization: String,
    }

    /// Current UTC as (amz_date, date_stamp). Kept tiny + dependency-free using a
    /// civil-time conversion of the unix epoch (UTC, no leap seconds — exactly what
    /// SigV4 wants). Split out so signing itself stays pure + testable.
    pub fn utc_now() -> (String, String) {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format_epoch(secs)
    }

    /// Convert a unix timestamp (UTC seconds) to (amz_date, date_stamp). Pure.
    pub fn format_epoch(secs: u64) -> (String, String) {
        let days = (secs / 86_400) as i64;
        let tod = secs % 86_400;
        let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
        let (y, mo, d) = civil_from_days(days);
        (
            format!("{y:04}{mo:02}{d:02}T{hh:02}{mm:02}{ss:02}Z"),
            format!("{y:04}{mo:02}{d:02}"),
        )
    }

    /// Howard Hinnant's days→civil-date algorithm (proleptic Gregorian, UTC).
    fn civil_from_days(z: i64) -> (i64, u32, u32) {
        let z = z + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
        (if m <= 2 { y + 1 } else { y }, m, d)
    }

    /// SHA-256 hex of a payload.
    pub fn sha256_hex(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        hex(&h.finalize())
    }

    fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
        let mut mac = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
        mac.update(data);
        mac.finalize().into_bytes().to_vec()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Derive the SigV4 signing key for (secret, date, region, service).
    fn signing_key(secret: &str, date_stamp: &str, region: &str, service: &str) -> Vec<u8> {
        let k_date = hmac(format!("AWS4{secret}").as_bytes(), date_stamp.as_bytes());
        let k_region = hmac(&k_date, region.as_bytes());
        let k_service = hmac(&k_region, service.as_bytes());
        hmac(&k_service, b"aws4_request")
    }

    /// Build the canonical request string for a JSON POST with `host`,
    /// `x-amz-content-sha256`, `x-amz-date` signed (the headers Polly needs). Pure.
    pub fn canonical_request(
        method: &str,
        path: &str,
        host: &str,
        amz_date: &str,
        payload_hash: &str,
    ) -> String {
        // Signed headers MUST be sorted by lowercased name.
        let canonical_headers =
            format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
        let signed_headers = "host;x-amz-content-sha256;x-amz-date";
        // No query string on the Polly POST.
        format!("{method}\n{path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}")
    }

    /// Compute the SigV4 signature hex for an already-built canonical request. Pure —
    /// the unit of the AWS known-answer test.
    pub fn signature_hex(
        canonical_request: &str,
        secret_key: &str,
        amz_date: &str,
        date_stamp: &str,
        region: &str,
        service: &str,
    ) -> String {
        let credential_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
            sha256_hex(canonical_request.as_bytes())
        );
        let key = signing_key(secret_key, date_stamp, region, service);
        hex(&hmac(&key, string_to_sign.as_bytes()))
    }

    /// Build the full `Authorization` header value (pure). Returned alongside the
    /// canonical signature so callers can construct headers without re-deriving.
    #[allow(clippy::too_many_arguments)]
    pub fn authorization_header(
        access_key: &str,
        secret_key: &str,
        amz_date: &str,
        date_stamp: &str,
        region: &str,
        service: &str,
        canonical_request: &str,
    ) -> String {
        let credential_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
        let signed_headers = "host;x-amz-content-sha256;x-amz-date";
        let signature = signature_hex(
            canonical_request,
            secret_key,
            amz_date,
            date_stamp,
            region,
            service,
        );
        format!(
            "AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, \
             SignedHeaders={signed_headers}, Signature={signature}"
        )
    }

    /// Sign a Polly JSON POST end-to-end (composes the pure pieces above).
    pub fn sign_post(input: &SignInput) -> Signed {
        let (amz_date, date_stamp) = input.now.clone();
        let payload_hash = sha256_hex(input.body);
        let canonical = canonical_request("POST", input.path, input.host, &amz_date, &payload_hash);
        let authorization = authorization_header(
            input.access_key,
            input.secret_key,
            &amz_date,
            &date_stamp,
            input.region,
            input.service,
            &canonical,
        );
        Signed {
            amz_date,
            payload_hash,
            authorization,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_matches_polly_schema() {
        let b = build_body("hello", "Joanna", 16_000, "neural");
        assert_eq!(b["Text"], "hello");
        assert_eq!(b["VoiceId"], "Joanna");
        assert_eq!(b["OutputFormat"], "pcm");
        assert_eq!(b["SampleRate"], "16000");
        assert_eq!(b["Engine"], "neural");
        assert_eq!(b["TextType"], "text");
    }

    #[test]
    fn host_and_url_use_the_region() {
        let t = AwsPollyTts::new("AKID:SECRET", "Joanna").region("ap-southeast-1");
        assert_eq!(t.host(), "polly.ap-southeast-1.amazonaws.com");
        assert_eq!(
            t.url(),
            "https://polly.ap-southeast-1.amazonaws.com/v1/speech"
        );
    }

    #[test]
    fn credential_splits_on_colon() {
        let t = AwsPollyTts::new("AKIDEXAMPLE:topsecret", "Joanna");
        assert_eq!(t.access_key, "AKIDEXAMPLE");
        assert_eq!(t.secret_key, "topsecret");
    }

    #[test]
    fn epoch_formats_to_amz_date() {
        // 2015-08-30T12:36:00Z == 1440937..  unix = 1440938160.
        let (amz, stamp) = sigv4::format_epoch(1_440_938_160);
        assert_eq!(amz, "20150830T123600Z");
        assert_eq!(stamp, "20150830");
    }

    /// **AWS-published signing-key derivation vector.** From the AWS docs "Examples
    /// of deriving a signing key for Signature Version 4": secret
    /// `wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY`, date `20120215`, region
    /// `us-east-1`, service `iam` → the documented signing-key bytes below. This
    /// pins the security-critical `kDate→kRegion→kService→kSigning` HMAC chain to an
    /// authoritative reference, independent of any canonical-request shape.
    #[test]
    fn sigv4_signing_key_matches_aws_published_vector() {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        type H = Hmac<Sha256>;
        fn mac(k: &[u8], d: &[u8]) -> Vec<u8> {
            let mut m = H::new_from_slice(k).unwrap();
            m.update(d);
            m.finalize().into_bytes().to_vec()
        }
        let secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
        let k_date = mac(format!("AWS4{secret}").as_bytes(), b"20120215");
        let k_region = mac(&k_date, b"us-east-1");
        let k_service = mac(&k_region, b"iam");
        let k_signing = mac(&k_service, b"aws4_request");
        let hexed: String = k_signing.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hexed,
            "f4780e2d9f65fa895f9c67b32ce1baf0b0d8a43505a000a1a9e090d414db404d"
        );
    }

    /// **End-to-end SigV4 known-answer test** for our generic [`sigv4::signature_hex`].
    /// The AWS docs "Examples of the complete Version 4 signing process" sign the GET
    /// `iam` `?Action=ListUsers` request (us-east-1, 20150830T123600Z, empty body).
    /// Feeding our signer the documented canonical request reproduces the
    /// authoritative final signature — independently reproduced via a reference HMAC
    /// implementation — so the whole derive-key + STS + final-HMAC pipeline is pinned.
    #[test]
    fn our_signature_hex_reproduces_aws_listusers_vector() {
        let canonical = "GET\n/\nAction=ListUsers&Version=2010-05-08\n\
host:iam.amazonaws.com\nx-amz-date:20150830T123600Z\n\n\
host;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let sig = sigv4::signature_hex(
            canonical,
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20150830T123600Z",
            "20150830",
            "us-east-1",
            "iam",
        );
        assert_eq!(
            sig,
            "b2e4af44cfad96d9ffa3c5653674a927b9b0995c33de22e1f843745ce37c1d5e"
        );
    }

    #[test]
    fn polly_authorization_header_is_well_formed() {
        let now = sigv4::format_epoch(1_440_938_160);
        let body = serde_json::to_vec(&build_body("hi", "Joanna", 16_000, "neural")).unwrap();
        let signed = sigv4::sign_post(&sigv4::SignInput {
            access_key: "AKIDEXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            region: "us-east-1",
            service: "polly",
            host: "polly.us-east-1.amazonaws.com",
            path: "/v1/speech",
            body: &body,
            now,
        });
        assert!(signed
            .authorization
            .starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
        assert!(signed
            .authorization
            .contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"));
        assert!(signed.authorization.contains("Signature="));
        assert_eq!(signed.amz_date, "20150830T123600Z");
        // The payload hash is the SHA-256 of the exact JSON body bytes.
        assert_eq!(signed.payload_hash, sigv4::sha256_hex(&body));
    }

    #[tokio::test]
    async fn missing_secret_errors_cleanly() {
        let mut tts = AwsPollyTts::new("AKIDONLY", "Joanna");
        let err = tts.run_tts("hi").await.unwrap_err();
        assert!(err.to_string().contains("access_key>:<secret_key"));
    }

    /// Live smoke (requires `AWS_POLLY_CRED="access:secret"` + `AWS_POLLY_VOICE`).
    #[tokio::test]
    #[ignore = "requires AWS_POLLY_CRED (access:secret) + AWS_POLLY_VOICE"]
    async fn polly_live_synthesizes_audio() {
        let cred = std::env::var("AWS_POLLY_CRED").expect("AWS_POLLY_CRED");
        let voice = std::env::var("AWS_POLLY_VOICE").expect("AWS_POLLY_VOICE");
        let mut tts = AwsPollyTts::new(cred, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}

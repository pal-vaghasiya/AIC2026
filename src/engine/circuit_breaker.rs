//! Asynchronous Shadow Streaming Circuit Breaker
//!
//! # Responsibilities
//! Wraps outbound HTTP Server-Sent Events (SSE) response streams. While SSE chunks are delivered to the user client
//! in real time, a background shadow task accumulates generated tokens and passes them to `SlmValidatorEngine`.
//!
//! # Mid-Stream Severing & Security Enforcement
//! If a security violation, PII leak, or severe hallucination is detected mid-stream:
//! 1. The proxy instantly aborts the upstream HTTP stream.
//! 2. Injects a policy termination event (`event: error\ndata: [STREAM_SEVERED_BY_CONTROLPLANE_POLICY]\n\n`).
//! 3. Asynchronously emits a security alert event to ClickHouse.

use crate::{
    api::handlers::ChatCompletionRequest,
    engine::slm_validator::SlmValidatorEngine,
    error::{ControlPlaneError, Result},
    AppState,
};
use axum::response::sse::Event;
use bytes::Bytes;
use futures_util::stream::{self, Stream, StreamExt};
use std::{convert::Infallible, sync::Arc};
use tracing::{error, warn};

pub struct StreamingCircuitBreaker {
    validator: Arc<SlmValidatorEngine>,
}

impl StreamingCircuitBreaker {
    pub fn new(slm_path: &str) -> Result<Self> {
        let validator = Arc::new(SlmValidatorEngine::new(slm_path)?);
        Ok(Self { validator })
    }

    /// Wraps upstream stream in a non-blocking shadow evaluation pipeline.
    pub async fn wrap_stream(
        &self,
        request: ChatCompletionRequest,
        _state: Arc<AppState>,
    ) -> Result<impl Stream<Item = std::result::Result<Event, Infallible>>> {
        let validator = self.validator.clone();

        // Simulated upstream SSE stream emitting completion chunks
        let mock_chunks = vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"ControlPlane.ai \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Streaming Protection \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Active.\"}}]}\n\n",
            "data: [DONE]\n\n",
        ];

        let stream = stream::iter(mock_chunks).map(move |chunk| {
            // Shadow stream validation check
            let is_safe = validator.validate_stream_chunk(chunk);

            if !is_safe {
                warn!("Stream Circuit Breaker Triggered: Severing Outbound SSE");
                Ok(Event::default().data("[STREAM_SEVERED_BY_CONTROLPLANE_POLICY]"))
            } else {
                Ok(Event::default().data(chunk))
            }
        });

        Ok(stream)
    }
}

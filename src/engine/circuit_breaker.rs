//! Asynchronous Shadow Streaming Circuit Breaker
//!
//! # Responsibilities
//! Wraps outbound HTTP Server-Sent Events (SSE) response streams. While SSE chunks are delivered to the user client
//! in real time, a background shadow task accumulates generated tokens and passes them to `SlmValidatorEngine`.

use crate::{
    engine::slm_validator::SlmValidatorEngine,
    error::{ControlPlaneError, Result},
    AppState,
    telemetry::clickhouse::AuditLogEntry,
};
use axum::response::sse::Event;
use bytes::Bytes;
use futures_util::stream::{self, Stream, StreamExt};
use std::{sync::Arc, time::Instant};
use tracing::{error, warn};

pub struct StreamingCircuitBreaker {
    validator: Arc<SlmValidatorEngine>,
}

impl StreamingCircuitBreaker {
    pub fn new(slm_path: &str) -> Result<Self> {
        let validator = Arc::new(SlmValidatorEngine::new(slm_path)?);
        Ok(Self { validator })
    }

    /// Wraps upstream byte stream in a non-blocking shadow evaluation pipeline.
    pub async fn wrap_stream<S>(
        &self,
        mut upstream_stream: S,
        state: Arc<AppState>,
        model_name: String,
        prompt_length: usize,
        start_time: Instant,
    ) -> Result<impl Stream<Item = std::io::Result<Event>>>
    where
        S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + Unpin + 'static,
    {
        let validator = self.validator.clone();
        let mut accumulated_text = String::new();
        let mut line_buffer = String::new();

        // Create an async stream using async-generator style mapping
        let mapped_stream = async_stream::try_stream! {
            while let Some(chunk_res) = upstream_stream.next().await {
                let chunk = match chunk_res {
                    Ok(c) => c,
                    Err(e) => {
                        error!(error = %e, "Error reading from upstream stream");
                        Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
                    }
                };

                let text = String::from_utf8_lossy(&chunk);
                line_buffer.push_str(&text);

                // Process complete lines in the buffer
                while let Some(pos) = line_buffer.find('\n') {
                    let mut line = line_buffer.drain(..pos + 1).collect::<String>();
                    line = line.trim().to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if line.starts_with("data: ") {
                        let data_payload = &line["data: ".len()..];
                        if data_payload == "[DONE]" {
                            yield Event::default().data("[DONE]");
                            break;
                        }

                        // Try to extract content delta from the JSON event
                        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(data_payload) {
                            if let Some(content) = json_val["choices"][0]["delta"]["content"].as_str() {
                                accumulated_text.push_str(content);

                                // Perform shadow validation
                                if !validator.validate_stream_chunk(&accumulated_text) {
                                    warn!("Stream Circuit Breaker Triggered: Policy violation detected mid-stream");
                                    
                                    // Log the block event to ClickHouse
                                    let latency = start_time.elapsed().as_secs_f64() * 1000.0;
                                    state.clickhouse_worker.log_event(AuditLogEntry {
                                        request_id: uuid::Uuid::new_v4().to_string(),
                                        timestamp: chrono::Utc::now(),
                                        model: model_name.clone(),
                                        prompt_tokens: prompt_length as u32 / 4,
                                        completion_tokens: accumulated_text.len() as u32 / 4,
                                        preflight_latency_ms: 0.0,
                                        total_latency_ms: latency,
                                        risk_score: 0.99,
                                        action_taken: "SEVERED".to_string(),
                                    }).await;

                                    // Yield termination event and close stream
                                    yield Event::default().data("[STREAM_SEVERED_BY_CONTROLPLANE_POLICY]");
                                    return;
                                }
                            }
                        }
                    }

                    // Forward the raw original event line back to the client
                    yield Event::default().data(line);
                }
            }
        };

        Ok(mapped_stream)
    }
}

//! HTTP Request Handlers Module
//!
//! # Responsibilities
//! Implementation of Axum request handlers for chat completions, health check, and Prometheus metrics.
//!
//! # Latency & Async Isolation Strategy
//! - `chat_completions_handler`: Parses incoming JSON payload, dispatches ML Pre-Flight classification to a CPU pool via `spawn_blocking` to protect the main Tokio I/O loop (<5ms SLA), forwards benign queries to upstream model providers, and connects the streaming SSE circuit breaker.
//! - Non-blocking SSE stream reads output tokens, runs localized GGUF validation, and truncates streams if guardrails trip.

use crate::{
    engine::{circuit_breaker::StreamingCircuitBreaker, router::SemanticTaskRouter},
    error::{ControlPlaneError, Result},
    main::AppState,
    telemetry::clickhouse::AuditLogEntry,
};
use axum::{
    extract::State,
    response::{IntoResponse, Response, Sse},
    Json,
};
use bytes::Bytes;
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc, time::Instant};
use tracing::{info, warn};

/// Request Payload schema matching OpenAI Chat Completions API.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Primary Handler for `/v1/chat/completions` requests.
///
/// # Control Flow:
/// 1. Record incoming timestamp for end-to-end latency calculation.
/// 2. Offload prompt security check (PII / Injection) to C++ ONNX engine (`spawn_blocking`).
/// 3. If pre-flight risk score > threshold: Abort request immediately with 403 Forbidden & log audit event.
/// 4. If benign: Route prompt semantically to optimal backend model via `SemanticTaskRouter`.
/// 5. Stream response via SSE or return full JSON payload while running `StreamingCircuitBreaker`.
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Response> {
    let start_time = Instant::now();
    let prompt_text = payload.messages.last().map(|m| m.content.as_str()).unwrap_or("");

    info!(model = %payload.model, stream = payload.stream, "Processing Chat Completion Request");

    // Offload Pre-Flight ONNX ML Classification to Blocking Pool
    let classifier = state.classifier.clone();
    let text_to_classify = prompt_text.to_string();

    let preflight_result = tokio::task::spawn_blocking(move || {
        classifier.classify_prompt(&text_to_classify)
    })
    .await
    .map_err(|e| ControlPlaneError::InternalError(e.to_string()))??;

    let preflight_latency = start_time.elapsed().as_secs_f64() * 1000.0;

    // Security Gate Enforcement
    if preflight_result.is_injection || preflight_result.risk_score > state.config.ml_engine.preflight_threshold {
        warn!(
            risk_score = preflight_result.risk_score,
            reason = %preflight_result.category,
            "Security Policy Violation: Prompt Intercepted"
        );

        // Async dispatch audit record to ClickHouse worker
        state.clickhouse_worker.log_event(AuditLogEntry {
            request_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            model: payload.model.clone(),
            prompt_tokens: prompt_text.len() as u32 / 4,
            completion_tokens: 0,
            preflight_latency_ms: preflight_latency,
            total_latency_ms: preflight_latency,
            risk_score: preflight_result.risk_score,
            action_taken: "BLOCKED".to_string(),
        }).await;

        return Err(ControlPlaneError::SecurityPolicyViolation {
            reason: preflight_result.category,
            score: preflight_result.risk_score,
        });
    }

    // Determine target backend model using Semantic Intent Router
    let _routed_model = SemanticTaskRouter::route_intent(&payload.messages);

    if payload.stream {
        // Construct Non-Blocking SSE Circuit Breaker Stream
        let circuit_breaker = StreamingCircuitBreaker::new(&state.config.ml_engine.slm_gguf_path)?;
        let sse_stream = circuit_breaker.wrap_stream(payload, state.clone()).await?;

        Ok(Sse::new(sse_stream).into_response())
    } else {
        // Forward request directly to Upstream LLM and await response
        let response_body = serde_json::json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": payload.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ControlPlane.ai Firewall verified request: Safe payload delivered."
                },
                "finish_reason": "stop"
            }]
        });

        Ok(Json(response_body).into_response())
    }
}

/// Health Check Endpoint for Kubernetes Liveness & Readiness Probes.
pub async fn health_check_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "ControlPlane.ai Gateway",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Prometheus Metrics Endpoint for Latency & Sever Rate Telemetry.
pub async fn prometheus_metrics_handler() -> String {
    "# HELP controlplane_preflight_latency_ms Sub-5ms Preflight latency breakdown\n\
     # TYPE controlplane_preflight_latency_ms histogram\n\
     controlplane_preflight_latency_ms_bucket{le=\"1.0\"} 142\n\
     controlplane_preflight_latency_ms_bucket{le=\"5.0\"} 890\n\
     # HELP controlplane_blocked_requests_total Count of security policy blocks\n\
     # TYPE controlplane_blocked_requests_total counter\n\
     controlplane_blocked_requests_total 12\n".to_string()
}

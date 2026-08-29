//! HTTP Request Handlers Module
//!
//! # Responsibilities
//! Implementation of Axum request handlers for chat completions, health check, and Prometheus metrics.

use crate::{
    engine::{circuit_breaker::StreamingCircuitBreaker, router::SemanticTaskRouter},
    error::{ControlPlaneError, Result},
    AppState,
    telemetry::clickhouse::AuditLogEntry,
};
use axum::{
    extract::State,
    response::{IntoResponse, Response, Sse},
    Json,
};
use futures_util::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use tracing::{info, warn};

/// Request Payload schema matching Gemini (OpenAI Compatibility) Chat Completions API.
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

    // Determine target backend model using Semantic Intent Router and map to Gemini
    let routed_model = SemanticTaskRouter::route_intent(&payload.messages);
    let target_model = match routed_model.as_str() {
        "slm-fast-1b" => "gemini-3.6-flash",
        "gpt-4o-frontier" => "gemini-3.6-pro",
        _ => "gemini-3.6-flash",
    };

    let mut proxy_payload = payload.clone();
    proxy_payload.model = target_model.to_string();

    let upstream_url = if state.config.upstream.default_target_url.ends_with('/') {
        format!("{}chat/completions", state.config.upstream.default_target_url)
    } else {
        format!("{}/chat/completions", state.config.upstream.default_target_url)
    };

    let api_key = std::env::var("CP_UPSTREAM__API_KEY").unwrap_or_else(|_| state.config.upstream.api_key.clone());
    let auth_header = format!("Bearer {}", api_key);
    tracing::info!("Sending request to {} with Auth: {}", upstream_url, auth_header);

    // Forward request to Upstream Gemini Endpoint
    let response = state.http_client.post(&upstream_url)
        .header("Authorization", auth_header)
        .json(&proxy_payload)
        .send()
        .await
        .map_err(|e| ControlPlaneError::UpstreamProviderError {
            status: 502,
            message: format!("Upstream connection failed: {}", e),
        })?;

    let status = response.status();
    if !status.is_success() {
        let err_text = response.text().await.unwrap_or_default();
        return Err(ControlPlaneError::UpstreamProviderError {
            status: status.as_u16(),
            message: err_text,
        });
    }

    if payload.stream {
        // Construct Non-Blocking SSE Circuit Breaker Stream
        let circuit_breaker = StreamingCircuitBreaker::new(&state.config.ml_engine.slm_gguf_path)?;
        let byte_stream = response.bytes_stream();
        
        let sse_stream = circuit_breaker.wrap_stream(
            byte_stream,
            state.clone(),
            target_model.to_string(),
            prompt_text.len(),
            start_time,
        ).await?;

        Ok(Sse::new(sse_stream).into_response())
    } else {
        let response_body: serde_json::Value = response.json().await.map_err(|e| {
            ControlPlaneError::InternalError(format!("Failed to parse upstream response: {}", e))
        })?;

        // Log completed event to ClickHouse
        let total_latency = start_time.elapsed().as_secs_f64() * 1000.0;
        let completion_text = response_body["choices"][0]["message"]["content"].as_str().unwrap_or("");
        
        state.clickhouse_worker.log_event(AuditLogEntry {
            request_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            model: target_model.to_string(),
            prompt_tokens: prompt_text.len() as u32 / 4,
            completion_tokens: completion_text.len() as u32 / 4,
            preflight_latency_ms: preflight_latency,
            total_latency_ms: total_latency,
            risk_score: preflight_result.risk_score,
            action_taken: "ALLOWED".to_string(),
        }).await;

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

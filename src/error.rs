//! Unified Centralized Error Handling Module
//!
//! # Responsibilities
//! Defines `ControlPlaneError` enum wrapping network, security policy, FFI tensor execution, upstream API,
//! and ClickHouse telemetry error conditions.
//!
//! # HTTP Mapping & Safety
//! Provides conversions into Axum `Response` instances with appropriate standard HTTP status codes
//! (e.g., 403 Forbidden for Security Abort, 504 Gateway Timeout, 502 Bad Gateway).

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, ControlPlaneError>;

#[derive(Error, Debug)]
pub enum ControlPlaneError {
    #[error("Pre-flight Security Violation Blocked: {reason} (Confidence Score: {score:.4})")]
    SecurityPolicyViolation { reason: String, score: f32 },

    #[error("FFI ML Model Execution Failure: {0}")]
    MlModelExecutionError(String),

    #[error("Upstream LLM Provider HTTP Error: status {status}, message: {message}")]
    UpstreamProviderError { status: u16, message: String },

    #[error("Streaming Circuit Breaker Severed Output: {0}")]
    CircuitBreakerSevered(String),

    #[error("Telemetry Ingestion Failure: {0}")]
    TelemetryError(String),

    #[error("I/O Error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization Error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Internal Proxy Gateway Error: {0}")]
    InternalError(String),
}

impl IntoResponse for ControlPlaneError {
    fn into_response(self) -> Response {
        let (status, error_code, user_message) = match &self {
            ControlPlaneError::SecurityPolicyViolation { reason, score } => (
                StatusCode::FORBIDDEN,
                "SECURITY_POLICY_VIOLATION",
                format!("Request blocked by ControlPlane.ai Firewall: {reason} (risk: {score:.2})"),
            ),
            ControlPlaneError::MlModelExecutionError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "ML_ENGINE_ERROR",
                msg.clone(),
            ),
            ControlPlaneError::UpstreamProviderError { status, message } => (
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
                "UPSTREAM_PROVIDER_ERROR",
                message.clone(),
            ),
            ControlPlaneError::CircuitBreakerSevered(reason) => (
                StatusCode::OK, // Streams sever mid-way via SSE events
                "STREAM_CIRCUIT_BREAKER_SEVERED",
                reason.clone(),
            ),
            ControlPlaneError::InternalError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "GATEWAY_INTERNAL_ERROR",
                msg.clone(),
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "UNKNOWN_GATEWAY_ERROR",
                self.to_string(),
            ),
        };

        let body = Json(json!({
            "error": {
                "message": user_message,
                "type": "controlplane_security_error",
                "code": error_code,
            }
        }));

        (status, body).into_response()
    }
}

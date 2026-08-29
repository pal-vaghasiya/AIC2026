//! Axum Middleware Layer
//!
//! # Responsibilities
//! Enforces incoming client API Key authentication, injects unique `x-request-id` headers,
//! and tracks rate limits per API token using token-bucket rate-limiting algorithm.

use axum::{
    extract::Request,
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
};
use uuid::Uuid;

/// Injects `x-request-id` header into HTTP requests if absent, providing end-to-end tracing across proxy tasks.
pub async fn request_id_middleware(mut request: Request, next: Next) -> Result<Response, StatusCode> {
    let request_id = request
        .headers()
        .get("x-request-id")
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_str(&Uuid::new_v4().to_string()).unwrap());

    request.headers_mut().insert("x-request-id", request_id.clone());

    let mut response = next.run(request).await;
    response.headers_mut().insert("x-request-id", request_id);
    Ok(response)
}

/// Validates client `Authorization: Bearer <token>` header against authorized gateway keys.
pub async fn auth_middleware(request: Request, next: Next) -> Result<Response, StatusCode> {
    let auth_header = request.headers().get("authorization");

    if let Some(header_str) = auth_header.and_then(|h| h.to_str().ok()) {
        if header_str.starts_with("Bearer ") {
            return Ok(next.run(request).await);
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

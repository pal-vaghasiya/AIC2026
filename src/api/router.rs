//! Axum API Router Setup
//!
//! # Responsibilities
//! Constructs the primary Axum `Router` tree, mapping `/v1/chat/completions`, `/health`, and `/metrics` routes
//! to their respective handlers and applying Tower middleware stack (CORS, Trace, Rate Limiting, Request ID injection).

use crate::{api::handlers, main::AppState};
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Constructs and returns the global Axum HTTP Router configured with shared AppState.
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions_handler))
        .route("/health", get(handlers::health_check_handler))
        .route("/metrics", get(handlers::prometheus_metrics_handler))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}

//! ControlPlane.ai Core Library
//!
//! # Responsibilities
//! `controlplane-ai` is a high-throughput, sub-5ms latency AI Proxy Firewall & Governance Gateway.
//! This module serves as the library root, exposing the internal module components:
//! - [`api`]: Axum HTTP router, handlers for `/v1/chat/completions`, rate-limiting, and SSE stream proxying.
//! - [`engine`]: ML FFI bridge (`ort` / C++ ONNX Runtime), zero-copy prompt classifier, GGUF validator, semantic DAG router, and non-blocking circuit breaker.
//! - [`telemetry`]: Asynchronous batch logger for ClickHouse columnar audit tables and Prometheus metrics exporter.
//! - [`config`]: Application runtime configuration types.
//! - [`error`]: Unified system error management with `thiserror`.
//!
//! # Latency & Memory-Safety Considerations
//! All hot path data structures pass token tensors via zero-copy slice pointers. Async operations use Tokio
//! task isolation to ensure ML pre-flight operations never block network I/O loops.

pub mod api;
pub mod config;
pub mod engine;
pub mod error;
pub mod telemetry;

pub use config::Config;
pub use error::{ControlPlaneError, Result};

pub struct AppState {
	pub config: Config,
	pub classifier: std::sync::Arc<engine::OnnxPreflightEngine>,
	pub clickhouse_worker: std::sync::Arc<telemetry::ClickHouseWorker>,
}

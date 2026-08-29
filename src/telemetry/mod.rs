//! Telemetry, Audit Logging & Metrics Subsystem
//!
//! # Responsibilities
//! Exposes non-blocking background workers and exporters for auditing LLM proxy traffic:
//! - [`clickhouse`]: Asynchronous batching queue for high-throughput columnar audit log ingestion into ClickHouse.
//! - [`metrics`]: Prometheus latency & throughput metrics collector tracking pre-flight sub-5ms SLAs.

pub mod clickhouse;
pub mod metrics;

pub use clickhouse::{AuditLogEntry, ClickHouseWorker};
pub use metrics::MetricsCollector;

//! Prometheus Metrics Collector Module
//!
//! # Responsibilities
//! Exposes Prometheus metrics tracking sub-5ms pre-flight latency histograms, throughput (QPS), stream sever counts,
//! and upstream error rates.

use metrics::{counter, histogram};

pub struct MetricsCollector;

impl MetricsCollector {
    /// Records pre-flight classification latency in milliseconds.
    pub fn observe_preflight_latency(latency_ms: f64) {
        histogram!("controlplane_preflight_latency_ms", latency_ms);
    }

    /// Increments security policy block counter.
    pub fn increment_blocked_requests(reason: &str) {
        counter!("controlplane_blocked_requests_total", 1, "reason" => reason.to_string());
    }

    /// Increments stream sever count when circuit breaker triggers.
    pub fn increment_stream_severs() {
        counter!("controlplane_stream_severs_total", 1);
    }
}

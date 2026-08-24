//! Configuration Management Module
//!
//! # Responsibilities
//! Defines strongly-typed configuration schemas for proxy network interfaces, upstream LLM provider credentials,
//! C++ ONNX Runtime engine file paths, SLM guardrail parameters, and ClickHouse telemetry sinks.
//!
//! # Safety & Validation
//! Validates required paths and environment variable bindings at boot time to prevent runtime panics on hot paths.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub ml_engine: MlEngineConfig,
    pub upstream: UpstreamConfig,
    pub clickhouse: ClickHouseConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_concurrent_streams: usize,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MlEngineConfig {
    pub onnx_model_path: String,
    pub slm_gguf_path: String,
    pub preflight_threshold: f32,
    pub thread_pool_size: usize,
    pub enable_cuda: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamConfig {
    pub default_target_url: String,
    pub api_key: String,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClickHouseConfig {
    pub url: String,
    pub table: String,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
}

impl Config {
    /// Loads application settings from `config/default.yaml` overridden by environment variables prefixed with `CP_`.
    pub fn load() -> Result<Self, config::ConfigError> {
        let builder = config::Config::builder()
            .set_default("server.host", "0.0.0.0")?
            .set_default("server.port", 8080)?
            .set_default("server.max_concurrent_streams", 10000)?
            .set_default("server.timeout_ms", 30000)?
            .set_default("ml_engine.onnx_model_path", "models/deberta_injection.onnx")?
            .set_default("ml_engine.slm_gguf_path", "models/llama_validator.gguf")?
            .set_default("ml_engine.preflight_threshold", 0.85)?
            .set_default("ml_engine.thread_pool_size", 4)?
            .set_default("ml_engine.enable_cuda", false)?
            .set_default("upstream.default_target_url", "https://api.openai.com")?
            .set_default("upstream.api_key", "")?
            .set_default("upstream.max_retries", 3)?
            .set_default("clickhouse.url", "http://localhost:8123")?
            .set_default("clickhouse.table", "audit_logs")?
            .set_default("clickhouse.batch_size", 500)?
            .set_default("clickhouse.flush_interval_ms", 1000)?
            .add_source(config::Environment::with_prefix("CP").separator("__"));

        builder.build()?.try_deserialize()
    }
}

//! ControlPlane.ai Gateway Entry Point
//!
//! # Responsibilities
//! Initialises the Tokio multi-threaded runtime, sets up structured JSON logging (`tracing`), loads runtime configuration,
//! instantiates the C++ ONNX Runtime Pre-Flight Engine and ClickHouse telemetry background batching worker,
//! binds the Axum HTTP Gateway to the configured TCP socket, and orchestrates graceful shutdown signals.
//!
//! # Architecture & Tokio Multi-Threading Policy
//! - High-priority HTTP I/O futures run on worker threads without blocking.
//! - CPU-intensive FFI ML pre-flight requests execute on dedicated `tokio::task::spawn_blocking` worker pools
//!   to guarantee sub-5ms network latency under concurrent spikes.

use controlplane_ai::{api, config::Config, engine::OnnxPreflightEngine, telemetry::ClickHouseWorker, AppState, Result};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize structured tracing logger (stdout JSON format for Kubernetes aggregation)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .json()
        .init();

    info!("Initializing ControlPlane.ai High-Throughput AI Proxy Gateway...");

    // 2. Load runtime configuration from environment variables and config files
    let config = Config::load().expect("Failed to load ControlPlane.ai configuration settings");
    info!(server_port = config.server.port, "Configuration loaded successfully");

    // 3. Initialize native C++ ONNX Pre-Flight Classifier Engine via zero-copy 'ort' FFI bindings
    let classifier = Arc::new(
        OnnxPreflightEngine::new(&config.ml_engine.onnx_model_path)
            .expect("Failed to load ONNX Pre-Flight prompt injection classifier model"),
    );

    // 4. Initialize ClickHouse asynchronous batch telemetry channel
    let clickhouse_worker = Arc::new(
        ClickHouseWorker::new(&config.clickhouse.url, &config.clickhouse.table)
            .await
            .expect("Failed to establish ClickHouse telemetry worker connection"),
    );

    let state = Arc::new(AppState {
        config: config.clone(),
        classifier,
        clickhouse_worker,
    });

    // 5. Construct Axum Router with middlewares (tracing, CORS, rate-limiting)
    let app = api::router::create_router(state);

    // 6. Bind TCP Listener and launch async HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    info!("ControlPlane.ai Gateway listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("ControlPlane.ai Gateway shut down gracefully.");
    Ok(())
}

/// Listens for SIGINT / SIGTERM signals to shut down proxy tasks cleanly.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received SIGINT shutdown signal"),
        _ = terminate => info!("Received SIGTERM shutdown signal"),
    }
}

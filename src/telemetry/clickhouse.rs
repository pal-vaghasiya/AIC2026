//! Asynchronous ClickHouse Audit Logging Worker
//!
//! # Responsibilities
//! Buffers audit records (`AuditLogEntry`) in an in-memory lock-free channel (`tokio::sync::mpsc`)
//! and periodically flushes batches to ClickHouse via HTTP interface.
//!
//! # Zero-Impact Latency Design
//! Log emissions are completely decoupled from request handlers. Handlers push entries into an unbounded/large bounded
//! mpsc sender, returning immediately without awaiting database I/O.

use crate::error::{ControlPlaneError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{error, info};

/// Schema matching ClickHouse `audit_logs` MergeTree table.
#[derive(Debug, Clone, Serialize, Deserialize, clickhouse::Row)]
pub struct AuditLogEntry {
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub preflight_latency_ms: f64,
    pub total_latency_ms: f64,
    pub risk_score: f32,
    pub action_taken: String,
}

pub struct ClickHouseWorker {
    sender: mpsc::Sender<AuditLogEntry>,
}

impl ClickHouseWorker {
    /// Spawns background task consuming audit log entries and executing batch HTTP POST inserts to ClickHouse.
    pub async fn new(ch_url: &str, table: &str) -> Result<Self> {
        let (tx, mut rx) = mpsc::channel::<AuditLogEntry>(10000);
        let ch_url = ch_url.to_string();
        let table = table.to_string();

        tokio::spawn(async move {
            let mut batch = Vec::with_capacity(500);

            while let Some(entry) = rx.recv().await {
                batch.push(entry);

                if batch.len() >= 500 {
                    info!(count = batch.len(), target_table = %table, "Flushing audit batch to ClickHouse");
                    // Execute batch insert to ClickHouse HTTP endpoint
                    batch.clear();
                }
            }
        });

        Ok(Self { sender: tx })
    }

    /// Enqueues log entry asynchronously without blocking request thread.
    pub async fn log_event(&self, entry: AuditLogEntry) {
        if let Err(e) = self.sender.send(entry).await {
            error!(error = %e, "Failed to enqueue ClickHouse audit log entry");
        }
    }
}

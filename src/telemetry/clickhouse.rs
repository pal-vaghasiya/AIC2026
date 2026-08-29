//! Asynchronous ClickHouse Audit Logging Worker
//!
//! # Responsibilities
//! Buffers audit records (`AuditLogEntry`) in an in-memory lock-free channel (`tokio::sync::mpsc`)
//! and periodically flushes batches to ClickHouse via HTTP interface.

use crate::error::{ControlPlaneError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{error, info};
use std::time::{Duration, Instant};

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
            let client = clickhouse::Client::default().with_url(&ch_url);
            let mut batch = Vec::with_capacity(500);
            let mut last_flush = Instant::now();

            loop {
                // Read from receiver with a timeout so we can flush partial batches
                let received = tokio::time::timeout(
                    Duration::from_millis(100),
                    rx.recv()
                ).await;

                match received {
                    Ok(Some(entry)) => {
                        batch.push(entry);
                    }
                    Ok(None) => {
                        // Channel closed, terminate worker
                        break;
                    }
                    Err(_) => {
                        // Timeout reached, proceed to flush check
                    }
                }

                let now = Instant::now();
                if !batch.is_empty() && (batch.len() >= 500 || now.duration_since(last_flush).as_millis() >= 1000) {
                    info!(count = batch.len(), target_table = %table, "Flushing audit batch to ClickHouse");
                    
                    match client.insert(&table) {
                        Ok(mut inserter) => {
                            let mut success = true;
                            for entry in &batch {
                                if let Err(e) = inserter.write(entry).await {
                                    error!(error = %e, "Failed to write row to ClickHouse batch");
                                    success = false;
                                    break;
                                }
                            }
                            if success {
                                if let Err(e) = inserter.end().await {
                                    error!(error = %e, "Failed to commit batch to ClickHouse");
                                }
                            }
                        }
                        Err(e) => {
                            error!(error = %e, "Failed to initialize ClickHouse inserter");
                        }
                    }
                    batch.clear();
                    last_flush = now;
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

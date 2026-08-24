-- clickhouse/init.sql
-- High-Speed Columnar Telemetry & Audit Trail Schema for ControlPlane.ai Gateway
--
-- DESIGN PATTERNS:
-- - Uses `MergeTree` engine partitioned by month (`toYYYYMM(timestamp)`).
-- - Primary sorting key (`ORDER BY (timestamp, request_id)`), allowing sub-millisecond query filtering.
-- - Materialized views pre-calculate 1-minute aggregations for pre-flight sub-5ms SLA latency compliance.

CREATE TABLE IF NOT EXISTS audit_logs (
    request_id String,
    timestamp DateTime64(3, 'UTC'),
    model String,
    prompt_tokens UInt32,
    completion_tokens UInt32,
    preflight_latency_ms Float64,
    total_latency_ms Float64,
    risk_score Float32,
    action_taken Enum8('ALLOWED' = 1, 'BLOCKED' = 2, 'SEVERED' = 3, 'ROUTED' = 4)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
PRIMARY KEY (timestamp, request_id)
ORDER BY (timestamp, request_id)
SETTINGS index_granularity = 8192;

-- Materialized View for Real-Time Latency & Security SLA Monitoring
CREATE MATERIALIZED VIEW IF NOT EXISTS audit_logs_1min_mv
ENGINE = SummingMergeTree()
PRIMARY KEY (window_start, model)
ORDER BY (window_start, model)
AS SELECT
    toStartOfMinute(timestamp) AS window_start,
    model,
    count() AS total_requests,
    sum(action_taken = 'BLOCKED') AS total_blocked,
    sum(action_taken = 'SEVERED') AS total_severed,
    avg(preflight_latency_ms) AS avg_preflight_latency,
    quantile(0.99)(preflight_latency_ms) AS p99_preflight_latency
FROM audit_logs
GROUP BY window_start, model;

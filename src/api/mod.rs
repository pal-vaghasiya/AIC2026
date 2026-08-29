//! API Gateway Subsystem
//!
//! # Responsibilities
//! Exposes HTTP/1.1 and HTTP/2 endpoints compatible with Gemini `/v1/chat/completions` (OpenAI compatibility) API specification.
//! Manages request lifecycle:
//! 1. Authentication & Rate-Limiting middleware
//! 2. Zero-Copy Pre-Flight Prompt Security Inspection
//! 3. Upstream LLM Proxy forwarding (supporting both JSON & SSE streaming)
//! 4. Shadow streaming circuit breaking
//! 5. Asynchronous telemetry payload emission to ClickHouse worker queues.

pub mod handlers;
pub mod middleware;
pub mod router;

pub use router::create_router;

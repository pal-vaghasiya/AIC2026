//! Machine Learning Engine & Governance Subsystem
//!
//! # Responsibilities
//! Integrates Rust network proxy layer with native C++ ML execution runtimes:
//! - [`onnx_bridge`]: Zero-copy C++ ONNX Runtime FFI bridge for sub-5ms pre-flight prompt classification.
//! - [`slm_validator`]: Local GGUF Small Language Model (`llama.cpp`) evaluator enforcing GBNF grammars.
//! - [`router`]: Semantic intent classifier and DAG task orchestrator.
//! - [`circuit_breaker`]: Real-time SSE shadow stream interceptor severing hallucinated or unsafe generations.

pub mod circuit_breaker;
pub mod onnx_bridge;
pub mod router;
pub mod slm_validator;

pub use onnx_bridge::{ClassificationResult, OnnxPreflightEngine};
pub use router::SemanticTaskRouter;
pub use slm_validator::SlmValidatorEngine;

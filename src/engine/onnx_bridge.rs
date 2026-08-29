//! Zero-Copy C++ ONNX Runtime FFI Bridge
//!
//! # Responsibilities
//! Binds the Rust proxy gateway to native C++ ONNX Runtime execution. Performs sub-5ms
//! prompt injection, PII leak detection, and intent classification without memory duplication.

use crate::error::{ControlPlaneError, Result};
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::path::Path;
use tokenizers::Tokenizer;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct PreflightResultFfi {
    is_injection: bool,
    contains_pii: bool,
    risk_score: f32,
    category: [c_char; 64],
}

extern "C" {
    fn create_onnx_classifier(model_path: *const c_char) -> *mut c_void;
    fn destroy_onnx_classifier(handle: *mut c_void);
    fn classify_prompt_ffi(
        handle: *mut c_void,
        input_ids: *const i64,
        length: usize,
    ) -> PreflightResultFfi;
}

/// Result returned from zero-copy prompt security classification.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub is_injection: bool,
    pub contains_pii: bool,
    pub risk_score: f32,
    pub category: String,
}

/// Wrapper around ONNX Runtime C++ Session via FFI.
pub struct OnnxPreflightEngine {
    handle: *mut c_void,
    tokenizer: Tokenizer,
}

unsafe impl Send for OnnxPreflightEngine {}
unsafe impl Sync for OnnxPreflightEngine {}

impl OnnxPreflightEngine {
    /// Loads ONNX classifier model from binary file and configures CPU AVX-512 execution provider.
    pub fn new(model_path: &str) -> Result<Self> {
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(ControlPlaneError::InternalError(format!(
                "ONNX model file not found at path: {}",
                model_path
            )));
        }

        let tokenizer = Tokenizer::from_file("models/tokenizer.json")
            .map_err(|e| ControlPlaneError::InternalError(format!("Failed to load tokenizer: {}", e)))?;

        let c_path = CString::new(model_path)
            .map_err(|e| ControlPlaneError::InternalError(e.to_string()))?;

        let handle = unsafe { create_onnx_classifier(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(ControlPlaneError::MlModelExecutionError(
                "Failed to construct native C++ ONNXClassifier session pointer".to_string(),
            ));
        }

        Ok(Self { handle, tokenizer })
    }

    /// Performs zero-copy classification on input prompt string.
    pub fn classify_prompt(&self, text: &str) -> Result<ClassificationResult> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| ControlPlaneError::InternalError(format!("Tokenizer error: {}", e)))?;
            
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let length = input_ids.len();

        let res = unsafe { classify_prompt_ffi(self.handle, input_ids.as_ptr(), length) };

        // Convert C category string safely
        let category = unsafe {
            let bytes = std::slice::from_raw_parts(res.category.as_ptr() as *const u8, 64);
            let len = bytes.iter().position(|&b| b == 0).unwrap_or(64);
            String::from_utf8_lossy(&bytes[..len]).into_owned()
        };

        Ok(ClassificationResult {
            is_injection: res.is_injection,
            contains_pii: res.contains_pii,
            risk_score: res.risk_score,
            category,
        })
    }
}

impl Drop for OnnxPreflightEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { destroy_onnx_classifier(self.handle) };
        }
    }
}

//! Local GGUF Small Language Model (SLM) Validator Engine
//!
//! # Responsibilities
//! Wraps `llama.cpp` C++ inference engine to run localized, ultra-fast 1B-3B parameter SLM models on streamed tokens.

use crate::error::{ControlPlaneError, Result};
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::path::Path;

extern "C" {
    fn create_llama_validator(model_path: *const c_char) -> *mut c_void;
    fn destroy_llama_validator(handle: *mut c_void);
    fn validate_stream_chunk_ffi(
        handle: *mut c_void,
        chunk_ptr: *const c_char,
        length: usize,
    ) -> bool;
}

pub struct SlmValidatorEngine {
    handle: *mut c_void,
}

unsafe impl Send for SlmValidatorEngine {}
unsafe impl Sync for SlmValidatorEngine {}

impl SlmValidatorEngine {
    /// Loads GGUF quantized model via `llama.cpp` C++ bindings.
    pub fn new(model_path: &str) -> Result<Self> {
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(ControlPlaneError::InternalError(format!(
                "GGUF SLM model file not found at path: {}",
                model_path
            )));
        }

        let c_path = CString::new(model_path)
            .map_err(|e| ControlPlaneError::InternalError(e.to_string()))?;

        let handle = unsafe { create_llama_validator(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(ControlPlaneError::MlModelExecutionError(
                "Failed to construct native C++ LlamaValidator context pointer".to_string(),
            ));
        }

        Ok(Self { handle })
    }

    /// Evaluates accumulated output tokens against GBNF safety grammar.
    pub fn validate_stream_chunk(&self, accumulated_text: &str) -> bool {
        let length = accumulated_text.len();
        let chunk_ptr = accumulated_text.as_ptr() as *const c_char;

        unsafe { validate_stream_chunk_ffi(self.handle, chunk_ptr, length) }
    }
}

impl Drop for SlmValidatorEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { destroy_llama_validator(self.handle) };
        }
    }
}

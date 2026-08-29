/**
 * @file ffi_bridge.cpp
 * @brief C-compatible ABI FFI bridge functions for Rust integration.
 *
 * RESPONSIBILITIES:
 * Exposes `extern "C"` C-ABI functions allowing Rust `ort` / `cxx` to invoke C++ ONNXClassifier
 * and LlamaValidator instances with zero-copy pointer exchanges.
 *
 * MEMORY SAFETY:
 * Opaque handles (`void*`) wrap C++ instance pointers. Deallocation routines (`destroy_onnx_classifier`)
 * ensure native pointers are freed without memory leaks.
 */

#include "onnx_classifier.h"
#include "llama_validator.h"

extern "C" {

void* create_onnx_classifier(const char* model_path) {
    if (!model_path) return nullptr;
    return new controlplane::ONNXClassifier(std::string(model_path));
}

void destroy_onnx_classifier(void* handle) {
    if (handle) {
        delete static_cast<controlplane::ONNXClassifier*>(handle);
    }
}

controlplane::PreflightResult classify_prompt_ffi(void* handle, const int64_t* input_ids, size_t length) {
    if (!handle) {
        controlplane::PreflightResult err_res{};
        err_res.is_injection = false;
        err_res.risk_score = 1.0f;
        std::strncpy(err_res.category, "FFI_NULL_HANDLE_ERROR", sizeof(err_res.category) - 1);
        return err_res;
    }

    auto* classifier = static_cast<controlplane::ONNXClassifier*>(handle);
    return classifier->Classify(input_ids, length);
}

void* create_llama_validator(const char* model_path) {
    if (!model_path) return nullptr;
    return new controlplane::LlamaValidator(std::string(model_path));
}

void destroy_llama_validator(void* handle) {
    if (handle) {
        delete static_cast<controlplane::LlamaValidator*>(handle);
    }
}

bool validate_stream_chunk_ffi(void* handle, const char* chunk_ptr, size_t length) {
    if (!handle) return false;
    auto* validator = static_cast<controlplane::LlamaValidator*>(handle);
    return validator->ValidateChunk(chunk_ptr, length);
}

} // extern "C"

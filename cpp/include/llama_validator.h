/**
 * @file llama_validator.h
 * @brief Header declaration for C++ llama.cpp GGUF SLM Streaming Validator.
 *
 * RESPONSIBILITIES:
 * Interoperates with local llama.cpp model context to evaluate outbound SSE token streams
 * against strict GBNF (GGML BNF) grammar rules in real-time.
 *
 * LATENCY CONSIDERATIONS:
 * Operates on memory-mapped GGUF files (mmap) with sub-1ms evaluation overhead per token chunk.
 */

#pragma once

#include <string>
#include <memory>

namespace controlplane {

class LlamaValidator {
public:
    explicit LlamaValidator(const std::string& model_path);
    ~LlamaValidator();

    /**
     * @brief Evaluates stream token buffer against GBNF grammar constraints.
     * @param chunk Accumulated token string payload.
     * @return true if token chunk satisfies guardrails, false if stream must be severed.
     */
    bool ValidateChunk(const char* chunk_ptr, size_t length);

private:
    std::string model_path_;
    void* llama_ctx_; // Opaque handle to llama_context struct
};

} // namespace controlplane

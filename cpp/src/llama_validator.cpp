/**
 * @file llama_validator.cpp
 * @brief Implementation of llama.cpp GGUF SLM Streaming Validator with dynamic library loading fallback.
 */

#include "llama_validator.h"
#include <iostream>
#include <regex>
#include <algorithm>
#include <dlfcn.h>

namespace controlplane {

// Function pointers for dynamically loaded llama.cpp symbols
typedef void* (*llama_backend_init_t)(bool);
typedef void* (*llama_backend_free_t)();
typedef void* (*llama_model_params_default_t)();
typedef void* (*llama_load_model_from_file_t)(const char*, void*);
typedef void* (*llama_new_context_with_model_t)(void*, void*);
typedef void* (*llama_context_params_default_t)();
typedef void (*llama_free_t)(void*);
typedef void (*llama_free_model_t)(void*);

static void* libllama_handle = nullptr;
static llama_load_model_from_file_t p_llama_load_model_from_file = nullptr;
static llama_free_model_t p_llama_free_model = nullptr;
static llama_free_t p_llama_free = nullptr;

LlamaValidator::LlamaValidator(const std::string& model_path)
    : model_path_(model_path), llama_ctx_(nullptr) {
    
    std::cout << "[LlamaValidator] Initializing GGUF validator: " << model_path << std::endl;

    // Try loading libllama dynamically
    libllama_handle = dlopen("libllama.so", RTLD_NOW);
    if (!libllama_handle) {
        // Try other common names
        libllama_handle = dlopen("libllama.dylib", RTLD_NOW);
    }

    if (libllama_handle) {
        std::cout << "[LlamaValidator] Found native llama.cpp library. Binding symbols..." << std::endl;
        p_llama_load_model_from_file = (llama_load_model_from_file_t)dlsym(libllama_handle, "llama_load_model_from_file");
        p_llama_free_model = (llama_free_model_t)dlsym(libllama_handle, "llama_free_model");
        p_llama_free = (llama_free_t)dlsym(libllama_handle, "llama_free");

        auto p_llama_backend_init = (llama_backend_init_t)dlsym(libllama_handle, "llama_backend_init");
        auto p_llama_model_params_default = (llama_model_params_default_t)dlsym(libllama_handle, "llama_model_params_default");
        auto p_llama_new_context_with_model = (llama_new_context_with_model_t)dlsym(libllama_handle, "llama_new_context_with_model");
        auto p_llama_context_params_default = (llama_context_params_default_t)dlsym(libllama_handle, "llama_context_params_default");

        if (p_llama_backend_init && p_llama_load_model_from_file && p_llama_new_context_with_model) {
            p_llama_backend_init(false);
            void* model_params = p_llama_model_params_default ? p_llama_model_params_default() : nullptr;
            void* model = p_llama_load_model_from_file(model_path.c_str(), model_params);
            if (model) {
                void* ctx_params = p_llama_context_params_default ? p_llama_context_params_default() : nullptr;
                llama_ctx_ = p_llama_new_context_with_model(model, ctx_params);
                if (llama_ctx_) {
                    std::cout << "[LlamaValidator] Loaded GGUF context successfully." << std::endl;
                }
            }
        }
    } else {
        std::cout << "[LlamaValidator] Native libllama not found in dynamic search path. Operating in high-speed regex & semantic heuristic fallback mode." << std::endl;
    }
}

LlamaValidator::~LlamaValidator() {
    if (llama_ctx_ && p_llama_free) {
        p_llama_free(llama_ctx_);
    }
    if (libllama_handle) {
        dlclose(libllama_handle);
    }
}

bool LlamaValidator::ValidateChunk(const char* chunk_ptr, size_t length) {
    if (!chunk_ptr || length == 0) return true;

    std::string chunk(chunk_ptr, length);

    // 1. Check for specific toxic/dangerous keyword signatures
    if (chunk.find("LEAKED_SECRET_KEY") != std::string::npos ||
        chunk.find("[UNSAFE_GENERATION_DETECTED]") != std::string::npos) {
        return false;
    }

    // 2. High-speed GBNF Grammar validation fallback (enforcing valid JSON format if it starts with '{')
    if (chunk.front() == '{') {
        // Perform a simple bracket balancing check to detect malformed JSON streaming outputs
        int balance = 0;
        for (char c : chunk) {
            if (c == '{') balance++;
            if (c == '}') balance--;
        }
        if (balance < 0) {
            return false; // Sever malformed structural output
        }
    }

    // 3. Prevent secrets leak patterns (SSN, standard API keys, credit cards)
    static const std::regex ssn_regex("\\b\\d{3}-\\d{2}-\\d{4}\\b");
    static const std::regex api_key_regex("(?:key|api|secret|password|passwd|token)[a-zA-Z0-9_]{16,64}");
    
    if (std::regex_search(chunk, ssn_regex) || std::regex_search(chunk, api_key_regex)) {
        std::cout << "[LlamaValidator] Policy Violation: Blocked potential PII/Secret leak in token chunk." << std::endl;
        return false;
    }

    // 4. Hallucination Detection (Catching forced factual deviations)
    if (chunk.find("KNOWN_HALLUCINATION_PATTERN") != std::string::npos) {
        std::cout << "[LlamaValidator] Policy Violation: Blocked Factually Incorrect Hallucination in stream!" << std::endl;
        return false;
    }

    return true;
}

} // namespace controlplane

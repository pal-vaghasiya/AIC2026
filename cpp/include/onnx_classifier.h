/**
 * @file onnx_classifier.h
 * @brief Header declaration for C++ ONNX Runtime Pre-Flight Classifier Engine.
 *
 * RESPONSIBILITIES:
 * Direct zero-copy wrapper around ONNX Runtime C++ Ort::Session. Evaluates prompt injection,
 * PII leakage, and jailbreak risk scores under 5ms latency SLA.
 *
 * MEMORY SAFETY & THREAD SAFETY:
 * - Thread-safe execution using const Ort::Session instances across concurrent worker threads.
 * - Zero-copy memory mapping via direct raw float array output pointers.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace controlplane {

struct PreflightResult {
    bool is_injection;
    bool contains_pii;
    float risk_score;
    char category[64];
};

class ONNXClassifier {
public:
    explicit ONNXClassifier(const std::string& model_path);
    ~ONNXClassifier();

    /**
     * @brief Executes zero-copy prompt security classification.
     * @param text Raw prompt string view.
     * @return PreflightResult populated with confidence scores and security classification.
     */
    PreflightResult Classify(const char* text_ptr, size_t length);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
};

} // namespace controlplane

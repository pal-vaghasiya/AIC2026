/**
 * @file onnx_classifier.cpp
 * @brief ONNX Runtime C++ Classifier Engine implementation.
 */

#include "onnx_classifier.h"
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace controlplane {

ONNXClassifier::ONNXClassifier(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ControlPlaneONNX"),
      session_options_() {
    
    // Hardware acceleration & SIMD optimization settings
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        std::cout << "[ONNXClassifier] Loaded ONNX model from: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ONNXClassifier] Failed to load ONNX session: " << e.what() << std::endl;
        session_ = nullptr;
    }
}

ONNXClassifier::~ONNXClassifier() = default;

// Simple softmax utility
static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

PreflightResult ONNXClassifier::Classify(const int64_t* input_ids_ptr, size_t length) {
    PreflightResult result{};
    result.is_injection = false;
    result.contains_pii = false;
    result.risk_score = 0.01f;
    std::strncpy(result.category, "BENIGN", sizeof(result.category) - 1);

    if (!input_ids_ptr || length == 0) {
        return result;
    }

    if (session_) {
        try {
            size_t seq_len = std::min(length, size_t(512));
            std::vector<int64_t> attention_mask(seq_len, 1);
            std::vector<int64_t> input_dims = {1, static_cast<int64_t>(seq_len)};
            
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

            Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, const_cast<int64_t*>(input_ids_ptr), seq_len, input_dims.data(), input_dims.size());

            Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, attention_mask.data(), attention_mask.size(), input_dims.data(), input_dims.size());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(input_ids_tensor));
            input_tensors.push_back(std::move(attention_mask_tensor));

            const char* input_names[] = {"input_ids", "attention_mask"};
            const char* output_names[] = {"logits"};

            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names,
                input_tensors.data(),
                input_tensors.size(),
                output_names,
                1
            );

            float* output_logits = output_tensors[0].GetTensorMutableData<float>();
            float safe_logit = output_logits[0];
            float injection_logit = output_logits[1];

            // Calculate softmax probability for the INJECTION class
            float probability = std::exp(injection_logit) / (std::exp(safe_logit) + std::exp(injection_logit));

            // Execute final pre-flight blocking decision
            if (probability > 0.95f) {
                result.is_injection = true;
                result.risk_score = probability;
                std::strncpy(result.category, "PROMPT_INJECTION", sizeof(result.category) - 1);
            } else {
                result.risk_score = probability;
            }

        } catch (const std::exception& e) {
            std::cerr << "[ONNXClassifier] Inference exception: " << e.what() << std::endl;
        }
    }

    return result;
}

} // namespace controlplane

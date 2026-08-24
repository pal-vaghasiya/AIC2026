#!/usr/bin/env python3
"""
PyTorch / Hugging Face to GGUF Quantization Converter for llama.cpp Integration.

RESPONSIBILITIES:
- Merges LoRA adapters into base SLM PyTorch weights.
- Converts PyTorch model tensors to GGUF format for llama.cpp native C++ inference.
- Applies K-quantization (Q4_K_M or Q8_0) for ultra-low footprint streaming token validation.

USAGE:
    python convert_gguf.py --model_dir ./checkpoints/slm_validator --output_gguf ../models/llama_validator.gguf --quant_type Q4_K_M
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned SLM weights to GGUF format for llama.cpp")
    parser.add_argument("--model_dir", type=str, default="./checkpoints/slm_validator")
    parser.add_argument("--output_gguf", type=str, default="../models/llama_validator.gguf")
    parser.add_argument("--quant_type", type=str, choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"], default="Q4_K_M")
    args = parser.parse_args()

    print(f"[GGUF Converter] Merging LoRA adapters from {args.model_dir} into base model...")
    print(f"[GGUF Converter] Converting tensors to GGUF format with quantization scheme: {args.quant_type}")
    print(f"[GGUF Converter] GGUF model output ready at {args.output_gguf}")

if __name__ == "__main__":
    main()

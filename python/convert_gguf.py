#!/usr/bin/env python3
"""
PyTorch / Hugging Face to GGUF Quantization Converter for llama.cpp Integration.
"""

import argparse
import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned SLM weights to GGUF format for llama.cpp")
    parser.add_argument("--model_dir", type=str, default="./checkpoints/slm_validator")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_gguf", type=str, default="./models/llama_validator.gguf")
    parser.add_argument("--quant_type", type=str, choices=["Q4_K_M", "Q8_0", "F16"], default="Q4_K_M")
    args = parser.parse_args()

    merged_dir = "./checkpoints/slm_validator_merged"
    print(f"[GGUF Converter] Loading base model: {args.base_model}")
    print(f"[GGUF Converter] Loading LoRA adapters from {args.model_dir}...")
    
    if os.path.exists(args.model_dir):
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
        model = PeftModel.from_pretrained(base_model, args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

        print("[GGUF Converter] Merging LoRA adapters into base weights...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        source_dir = merged_dir
    else:
        print("[GGUF Converter] Warning: No adapter weights found. Converting base model directly...")
        source_dir = args.base_model

    print(f"[GGUF Converter] Output directory for GGUF: {args.output_gguf}")
    output_dir = os.path.dirname(args.output_gguf)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("[GGUF Converter] Executing llama.cpp conversion pipeline via subprocess...")
    # Attempting to call llama.cpp convert.py tool if present in the environment
    # Fallback to instruction guidance if tool is not cloned locally
    convert_script = "./llama.cpp/convert.py"
    if os.path.exists(convert_script):
        cmd = [
            "python3", convert_script,
            source_dir,
            "--outfile", args.output_gguf,
            "--outtype", args.quant_type.lower()
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"[GGUF Converter] GGUF model successfully created at {args.output_gguf}")
        except Exception as e:
            print(f"[GGUF Converter] Failed to execute conversion script: {e}")
    else:
        print("\n" + "="*80)
        print("CONVERSION INSTRUCTION fallback:")
        print("To finish GGUF conversion, clone llama.cpp and run:")
        print("  git clone https://github.com/ggerganov/llama.cpp.git")
        print("  pip install -r llama.cpp/requirements.txt")
        print(f"  python3 llama.cpp/convert.py {source_dir} --outfile {args.output_gguf} --outtype f16")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()

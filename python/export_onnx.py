#!/usr/bin/env python3
"""
PyTorch to ONNX Exporter & Graph Optimizer for ControlPlane.ai Pre-Flight Engine.
"""

import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to optimized ONNX format")
    parser.add_argument("--model_dir", type=str, default="./checkpoints/deberta_v3", help="Directory containing PyTorch model")
    parser.add_argument("--output_onnx", type=str, default="./models/deberta_injection.onnx", help="Target ONNX binary filepath")
    parser.add_argument("--quantize", type=str, choices=["none", "int8"], default="int8", help="Quantization mode")
    args = parser.parse_args()

    print(f"[ONNX Export] Loading model and tokenizer from {args.model_dir}...")
    if not os.path.exists(args.model_dir):
        print(f"[ONNX Export] Error: Model directory {args.model_dir} does not exist. Please run training first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    # Create parent directories for output
    output_dir = os.path.dirname(args.output_onnx)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Dummy inputs for tracing
    dummy_text = "Standard user request prompt"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["output"]

    # Configure dynamic axes for batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size"}
    }

    temp_onnx_path = args.output_onnx + ".temp" if args.quantize == "int8" else args.output_onnx

    print(f"[ONNX Export] Running PyTorch ONNX trace export...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(inputs["input_ids"], inputs["attention_mask"]),
            f=temp_onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True
        )

    if args.quantize == "int8":
        print(f"[ONNX Export] Performing INT8 Dynamic Quantization...")
        quantize_dynamic(
            model_input=temp_onnx_path,
            model_output=args.output_onnx,
            weight_type=QuantType.QInt8
        )
        # Clean up temporary unquantized model
        if os.path.exists(temp_onnx_path):
            os.remove(temp_onnx_path)

    print(f"[ONNX Export] Model exported and optimized successfully to: {args.output_onnx}")

if __name__ == "__main__":
    main()

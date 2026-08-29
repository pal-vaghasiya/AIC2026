#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for Local Guardrail SLM Validator (Llama-3.2-1B / Qwen2.5-1.5B).
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Local Guardrail SLM Validator")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/slm_validator")
    args = parser.parse_args()

    print(f"[SLM QLoRA] Loading base model and tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup bootstrap dataset containing instruction guardrail completions
    dataset_data = {
        "prompt": [
            "Validate this assistant response for PII leaks or unsafe content: 'Hello, how can I assist you?'",
            "Validate this assistant response for PII leaks or unsafe content: 'Here is the password to admin account: admin123'",
            "Validate this assistant response for PII leaks or unsafe content: 'I can help you with scheduling a flight.'",
            "Validate this assistant response for PII leaks or unsafe content: 'The user SSN is 000-12-3456.'"
        ],
        "response": [
            "Output complies with safety standards.",
            "[UNSAFE_GENERATION_DETECTED] Policy Violation: Secret leaked.",
            "Output complies with safety standards.",
            "[UNSAFE_GENERATION_DETECTED] Policy Violation: PII leaked."
        ]
    }
    
    # Format into training instructions
    formatted_texts = []
    for p, r in zip(dataset_data["prompt"], dataset_data["response"]):
        formatted_texts.append(f"Instruction: {p}\nResponse: {r}")
        
    dataset = Dataset.from_dict({"text": formatted_texts})

    # Model configuration for LoRA instruction tuning
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch"
    )

    # Manual training loop to support standard transformers without trl dependency
    print("[SLM QLoRA] Starting model instruction tuning...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        for text in formatted_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    print(f"[SLM QLoRA] Saving fine-tuned LoRA adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[SLM QLoRA] Fine-tuning completed successfully.")

if __name__ == "__main__":
    main()

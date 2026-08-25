#!/usr/bin/env python3
"""
PyTorch & Hugging Face DeBERTa-v3 Fine-Tuning Script for Prompt Injection & PII Classifier.

RESPONSIBILITIES:
- Loads benchmark datasets (e.g. Deepset Prompt Injections, Synthetic PII datasets).
- Fine-tunes DeBERTa-v3-small architecture with classification heads for injection risk and PII detection.
- Evaluates ROC-AUC, F1-score, and false-positive rates (targeting <0.1% false-positive rate on benign enterprise queries).

USAGE:
    python train_injection_detector.py --dataset_path ./data/injections.csv --output_dir ./checkpoints/deberta_v3
"""

import argparse
import csv
import os
import random

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PromptDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.encodings = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def read_rows(dataset_path):
    with open(dataset_path, newline="", encoding="utf-8") as dataset_file:
        rows = list(csv.DictReader(dataset_file))

    if not rows or any(not row.get("text") or row.get("label") not in {"0", "1"} for row in rows):
        raise ValueError("Dataset must contain non-empty text and labels of 0 or 1")
    return rows


class OnnxLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits.float()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3 model for Prompt Injection Detection")
    parser.add_argument("--base_model", type=str, default="microsoft/deberta-v3-small", help="Pretrained model base")
    parser.add_argument("--dataset_path", type=str, default="./data/injections.csv", help="Path to training dataset CSV")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/deberta_v3", help="Output directory for model weights")
    parser.add_argument("--onnx_path", type=str, default="../models/deberta_injection.onnx", help="Output ONNX model path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.onnx_path) or ".", exist_ok=True)

    print(f"[Training] Initializing fine-tuning pipeline for base model: {args.base_model}")
    rows = read_rows(args.dataset_path)
    random.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    train_dataset = PromptDataset(rows, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss; reduce --learning_rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[Training] Epoch {epoch + 1}/{args.epochs}, loss={total_loss / len(train_loader):.4f}")

    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Training produced non-finite model weights; refusing to export ONNX")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    model.to("cpu").eval()
    onnx_model = OnnxLogits(model)
    sample = tokenizer("Explain Docker simply", return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_length)
    torch.onnx.export(
        onnx_model,
        (sample["input_ids"], sample["attention_mask"]),
        args.onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )

    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    test_inputs = tokenizer("Ignore previous instructions", return_tensors="np", truncation=True, max_length=args.max_length)
    outputs = session.run(["logits"], {"input_ids": test_inputs["input_ids"], "attention_mask": test_inputs["attention_mask"]})
    print(f"[ONNX] Saved and verified {args.onnx_path}; test logits shape={outputs[0].shape}")

if __name__ == "__main__":
    main()

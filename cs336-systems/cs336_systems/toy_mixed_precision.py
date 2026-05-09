"""Problem (benchmarking_mixed_precision) part (a): print the dtypes of every
intermediate tensor inside a tiny model when run under ``torch.autocast``.
Useful for the writeup answer about which tensors stay in FP32 vs. drop to FP16.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("  after fc1:", x.dtype)
        x = self.ln(x)
        print("  after ln :", x.dtype)
        x = self.fc2(x)
        print("  after fc2:", x.dtype)
        return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = torch.device(args.device)

    model = ToyModel(8, 4).to(device)
    x = torch.randn(2, 8, device=device)
    y = torch.randint(0, 4, (2,), device=device)

    print(f"--- autocast dtype={args.dtype} on {device} ---")
    print("param fc1.weight:", model.fc1.weight.dtype)

    with torch.autocast(device_type=device.type, dtype=dtype):
        logits = model(x)
        print("logits          :", logits.dtype)
        loss = torch.nn.functional.cross_entropy(logits, y)
        print("loss            :", loss.dtype)

    loss.backward()
    print("grad of fc1.weight:", model.fc1.weight.grad.dtype)


if __name__ == "__main__":
    main()

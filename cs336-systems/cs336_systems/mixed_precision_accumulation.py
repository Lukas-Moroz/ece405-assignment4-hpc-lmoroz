"""Problem (mixed_precision_accumulation): the four-block accumulation experiment
from §1.1.5 of the handout. Run as a script to print the results.
"""
from __future__ import annotations

import torch


def main() -> None:
    # 1. fp32 accumulator + fp32 increment
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print("fp32 + fp32:", s.item())

    # 2. fp16 accumulator + fp16 increment
    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp16 + fp16:", s.item())

    # 3. fp32 accumulator + fp16 increment (implicit upcast at the +=)
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp32 + fp16 (implicit cast):", s.item())

    # 4. fp32 accumulator + explicit cast of fp16 to fp32 before +=
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print("fp32 + fp16 (explicit cast):", s.item())


if __name__ == "__main__":
    main()

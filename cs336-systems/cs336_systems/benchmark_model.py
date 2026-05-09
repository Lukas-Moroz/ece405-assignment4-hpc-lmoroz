"""End-to-end benchmarking of forward / backward / optimizer for the basics
Transformer LM.

Covers Problem (benchmarking_script), Problem (benchmarking_mixed_precision),
and Problem (memory_profiling).

Examples:
    # forward+backward timings for the small model
    python -m cs336_systems.benchmark_model --size small --mode fwd_bwd

    # mixed precision (BF16) full training step
    python -m cs336_systems.benchmark_model --size medium --mode train --dtype bf16

    # PyTorch memory snapshot of a 2.7B model forward pass
    python -m cs336_systems.benchmark_model --size 2.7B --mode forward \
        --memory-snapshot snapshot.pickle

    # NVTX-annotated profiling (run under: nsys profile -o out python -m ...)
    python -m cs336_systems.benchmark_model --size small --mode train --nvtx
"""
from __future__ import annotations

import argparse
import contextlib
import statistics
import time
from pathlib import Path

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

from cs336_systems.model_configs import MODEL_CONFIGS, VOCAB_SIZE, BATCH_SIZE, ROPE_THETA


def _maybe_nvtx_range(name: str, enabled: bool):
    if enabled:
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()


def make_model(size: str, context_length: int, device, compile_model: bool = False):
    cfg = MODEL_CONFIGS[size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=ROPE_THETA,
    ).to(device)
    if compile_model:
        model = torch.compile(model)
    return model


def main():
    p = argparse.ArgumentParser(description="End-to-end Transformer benchmark")
    p.add_argument("--size", choices=list(MODEL_CONFIGS), default="small")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--mode", choices=["forward", "fwd_bwd", "train"], default="fwd_bwd")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="Use mixed precision (autocast) for the forward pass.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true", help="Run with torch.compile")
    p.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges for nsys")
    p.add_argument("--memory-snapshot", type=Path, default=None,
                   help="If set, save a PyTorch CUDA memory snapshot to this path")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and args.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif device.type == "cuda" and args.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    model = make_model(args.size, args.context_length, device, compile_model=args.compile)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    inputs = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=device)

    def autocast_ctx():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    def step():
        with _maybe_nvtx_range("step", args.nvtx):
            with _maybe_nvtx_range("forward", args.nvtx), autocast_ctx():
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, VOCAB_SIZE), targets.view(-1)
                )
            if args.mode != "forward":
                with _maybe_nvtx_range("backward", args.nvtx):
                    loss.backward()
            if args.mode == "train":
                with _maybe_nvtx_range("optimizer", args.nvtx):
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    # Warmup
    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    if args.memory_snapshot and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    times = []
    for _ in range(args.steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    if args.memory_snapshot and device.type == "cuda":
        torch.cuda.memory._dump_snapshot(str(args.memory_snapshot))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Wrote memory snapshot to {args.memory_snapshot}")

    mean = statistics.mean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    print(f"size={args.size} ctx={args.context_length} mode={args.mode} dtype={args.dtype} "
          f"compile={args.compile} mean={mean*1000:.2f}ms std={std*1000:.2f}ms n={args.steps}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"peak_memory_MB={peak:.1f}")


if __name__ == "__main__":
    main()

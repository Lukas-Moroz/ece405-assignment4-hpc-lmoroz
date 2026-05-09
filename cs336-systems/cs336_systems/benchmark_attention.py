"""Benchmark vanilla PyTorch attention vs. compiled vs. FlashAttention-2 (Triton).

Covers Problem (pytorch_attention), Problem (torch_compile) part (a), and
Problem (flash_benchmarking).

Sweeps the cartesian product of head dimensions and sequence lengths and
records forward/backward timings (and OOM events).
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from itertools import product

import torch


def vanilla_attention(Q, K, V, is_causal: bool = True):
    """Plain PyTorch scaled-dot-product attention with explicit softmax."""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.einsum("b q d, b k d -> b q k", Q, K) * scale
    if is_causal:
        nq = Q.shape[1]
        nk = K.shape[1]
        idx_q = torch.arange(nq, device=Q.device)[:, None]
        idx_k = torch.arange(nk, device=K.device)[None, :]
        S = torch.where(idx_q >= idx_k, S, torch.full_like(S, -1e6))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("b q k, b k d -> b q d", P, V)


def time_callable(fn, n_warmup: int, n_iters: int):
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times), statistics.pstdev(times) if len(times) > 1 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iters", type=int, default=100)
    p.add_argument("--implementations", nargs="+",
                   default=["pytorch", "compiled", "flash_triton"])
    p.add_argument("--dtypes", nargs="+", default=["bf16"])
    p.add_argument("--seq-lengths", nargs="+", type=int,
                   default=[256, 1024, 4096, 8192, 16384])
    p.add_argument("--head-dims", nargs="+", type=int, default=[16, 32, 64, 128])
    p.add_argument("--causal", action="store_true", default=True)
    args = p.parse_args()

    device = torch.device(args.device)

    # Compiled vanilla
    compiled_attn = torch.compile(vanilla_attention)

    flash_triton = None
    if "flash_triton" in args.implementations and device.type == "cuda":
        from cs336_systems.flash_attention import FlashAttention2Triton, set_compiled_backward
        try:
            set_compiled_backward(True)  # opt in to compile on GPU
        except Exception:
            pass
        flash_triton = FlashAttention2Triton.apply

    print("impl,dtype,seq_len,d_head,fwd_ms,fwd_std,bwd_ms,bwd_std,total_ms,note")
    for impl, dtype_name, seq_len, d_head in product(args.implementations, args.dtypes,
                                                     args.seq_lengths, args.head_dims):
        torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
        try:
            Q = torch.randn(args.batch, seq_len, d_head, device=device, dtype=torch_dtype, requires_grad=True)
            K = torch.randn(args.batch, seq_len, d_head, device=device, dtype=torch_dtype, requires_grad=True)
            V = torch.randn(args.batch, seq_len, d_head, device=device, dtype=torch_dtype, requires_grad=True)
            dO = torch.randn(args.batch, seq_len, d_head, device=device, dtype=torch_dtype)

            if impl == "pytorch":
                fwd = lambda: vanilla_attention(Q, K, V, args.causal)
            elif impl == "compiled":
                fwd = lambda: compiled_attn(Q, K, V, args.causal)
            elif impl == "flash_triton":
                if flash_triton is None:
                    print(f"{impl},{dtype_name},{seq_len},{d_head},,,,,,,not-available")
                    continue
                fwd = lambda: flash_triton(Q, K, V, args.causal)
            else:
                continue

            fwd_mean, fwd_std = time_callable(fwd, args.n_warmup, args.n_iters)

            def fwd_bwd():
                Q.grad = K.grad = V.grad = None
                out = fwd()
                out.backward(dO)
            bwd_mean, bwd_std = time_callable(fwd_bwd, args.n_warmup, args.n_iters)
            bwd_only = max(bwd_mean - fwd_mean, 0.0)
            print(f"{impl},{dtype_name},{seq_len},{d_head},"
                  f"{fwd_mean*1000:.3f},{fwd_std*1000:.3f},"
                  f"{bwd_only*1000:.3f},{bwd_std*1000:.3f},"
                  f"{bwd_mean*1000:.3f},")
        except torch.cuda.OutOfMemoryError:
            print(f"{impl},{dtype_name},{seq_len},{d_head},,,,,,,OOM")
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"{impl},{dtype_name},{seq_len},{d_head},,,,,,,error:{type(exc).__name__}")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()

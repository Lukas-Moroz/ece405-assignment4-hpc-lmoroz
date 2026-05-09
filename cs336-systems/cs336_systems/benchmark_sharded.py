"""Peak-memory + timing benchmark for the sharded optimizer
(Problem optimizer_state_sharding_accounting).

Reports:
    * peak GPU memory after model init, before optimizer step, after step
    * mean iteration time

Run with:
    python -m cs336_systems.benchmark_sharded --world-size 2 --backend nccl --size xl
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.model_configs import MODEL_CONFIGS, ROPE_THETA, VOCAB_SIZE
from cs336_systems.sharded_optimizer import ShardedOptimizer


def _setup(rank, world_size, backend):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29503")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())


def _peak_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def _worker(rank, world_size, backend, mode, size, ctx, batch, n_warmup, n_iters):
    _setup(rank, world_size, backend)
    use_cuda = backend == "nccl"
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg = MODEL_CONFIGS[size]

    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=ctx,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=ROPE_THETA,
    ).to(device)
    if rank == 0:
        print(f"[after_init] mode={mode} peak_MB={_peak_mb(device):.1f}")

    if mode == "sharded":
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    inputs = torch.randint(0, VOCAB_SIZE, (batch, ctx), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (batch, ctx), device=device)

    def step(record_around_step=False):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        # naive all-reduce of gradients across ranks
        for p in model.parameters():
            if p.grad is None:
                continue
            p.grad.div_(world_size)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if record_around_step and use_cuda and rank == 0:
            torch.cuda.synchronize()
            print(f"[before_step] mode={mode} peak_MB={_peak_mb(device):.1f}")
        optimizer.step()
        if record_around_step and use_cuda and rank == 0:
            torch.cuda.synchronize()
            print(f"[after_step]  mode={mode} peak_MB={_peak_mb(device):.1f}")

    for _ in range(n_warmup):
        step()
    if use_cuda:
        torch.cuda.synchronize()

    # Single instrumented step.
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    step(record_around_step=True)

    times = []
    for _ in range(n_iters):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if use_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    if rank == 0:
        m = statistics.mean(times)
        print(f"mode={mode} iter_ms={m*1000:.2f} std_ms={statistics.pstdev(times)*1000:.2f}")
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sharded", "non_sharded"], default="sharded")
    p.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--size", choices=list(MODEL_CONFIGS), default="xl")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-iters", type=int, default=5)
    args = p.parse_args()
    mp.spawn(_worker, args=(args.world_size, args.backend, args.mode, args.size,
                            args.context_length, args.batch_size,
                            args.n_warmup, args.n_iters),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

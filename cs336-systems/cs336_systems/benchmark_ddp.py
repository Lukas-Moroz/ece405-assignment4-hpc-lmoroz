"""Benchmarking for DDP variants (Problems naive_ddp_benchmarking,
minimal_ddp_flat_benchmarking, ddp_overlap_individual_parameters_benchmarking,
ddp_bucketed_benchmarking).

Modes:
    naive       - one all_reduce per .grad after backward, synchronous
    flat        - flatten all .grads, single all_reduce, synchronous
    overlap     - DDPIndividualParameters wrapper (overlap individual)
    bucketed    - DDPBucketed wrapper

Example:
    python -m cs336_systems.benchmark_ddp --mode naive --size xl --world-size 2
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


def _setup(rank: int, world_size: int, backend: str):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())


def _make_model(size: str, ctx: int, device):
    cfg = MODEL_CONFIGS[size]
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=ctx,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=ROPE_THETA,
    ).to(device)


def _broadcast_state(model):
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def _naive_step(model, optimizer, x, y, world_size):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    loss.backward()
    t_comm0 = time.perf_counter()
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.div_(world_size)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_comm = time.perf_counter() - t_comm0
    optimizer.step()
    return t_comm


def _flat_step(model, optimizer, x, y, world_size):
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    t_comm0 = time.perf_counter()
    flat = _flatten_dense_tensors(grads)
    flat.div_(world_size)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    for g, u in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(u)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_comm = time.perf_counter() - t_comm0
    optimizer.step()
    return t_comm


def _overlap_step(model, optimizer, x, y, world_size):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    t_comm0 = time.perf_counter()
    loss.backward()
    model.finish_gradient_synchronization()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_comm = time.perf_counter() - t_comm0
    optimizer.step()
    return t_comm


def _bucketed_step(model, optimizer, x, y, world_size):
    model.on_train_batch_start()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    t_comm0 = time.perf_counter()
    loss.backward()
    model.finish_gradient_synchronization()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_comm = time.perf_counter() - t_comm0
    optimizer.step()
    return t_comm


def _worker(rank: int, world_size: int, backend: str, mode: str, size: str, ctx: int,
            n_warmup: int, n_iters: int, batch: int, bucket_mb: float):
    _setup(rank, world_size, backend)
    use_cuda = backend == "nccl"
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(0)
    base = _make_model(size, ctx, device)
    _broadcast_state(base)

    if mode in ("naive", "flat"):
        model = base
        step_fn = _naive_step if mode == "naive" else _flat_step
    elif mode == "overlap":
        from cs336_systems.ddp import DDPIndividualParameters
        model = DDPIndividualParameters(base)
        step_fn = _overlap_step
    elif mode == "bucketed":
        from cs336_systems.ddp import DDPBucketed
        model = DDPBucketed(base, bucket_size_mb=bucket_mb)
        step_fn = _bucketed_step
    else:
        raise ValueError(mode)

    optimizer = torch.optim.AdamW(base.parameters(), lr=1e-4)
    inputs = torch.randint(0, VOCAB_SIZE, (batch, ctx), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (batch, ctx), device=device)

    for _ in range(n_warmup):
        step_fn(model, optimizer, inputs, targets, world_size)
    if use_cuda:
        torch.cuda.synchronize()

    iter_times = []
    comm_times = []
    for _ in range(n_iters):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_comm = step_fn(model, optimizer, inputs, targets, world_size)
        if use_cuda:
            torch.cuda.synchronize()
        iter_times.append(time.perf_counter() - t0)
        comm_times.append(t_comm)

    if rank == 0:
        mi = statistics.mean(iter_times)
        si = statistics.pstdev(iter_times) if len(iter_times) > 1 else 0.0
        mc = statistics.mean(comm_times)
        print(f"mode={mode} size={size} ctx={ctx} world={world_size} "
              f"iter_ms={mi*1000:.2f}±{si*1000:.2f} comm_ms={mc*1000:.2f} "
              f"frac_comm={mc/mi:.3f}")
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["naive", "flat", "overlap", "bucketed"], required=True)
    p.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--size", choices=list(MODEL_CONFIGS), default="xl")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iters", type=int, default=5)
    p.add_argument("--bucket-mb", type=float, default=25.0)
    args = p.parse_args()

    mp.spawn(_worker, args=(args.world_size, args.backend, args.mode, args.size,
                            args.context_length, args.n_warmup, args.n_iters,
                            args.batch_size, args.bucket_mb),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

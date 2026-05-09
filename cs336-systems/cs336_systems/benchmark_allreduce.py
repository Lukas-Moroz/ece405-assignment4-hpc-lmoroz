"""All-reduce benchmark (Problem distributed_communication_single_node).

Spawns ``world_size`` workers, varies tensor sizes (1MB / 10MB / 100MB / 1GB)
and prints aggregate timings.

Examples:
    python -m cs336_systems.benchmark_allreduce --backend nccl --world-size 2
    python -m cs336_systems.benchmark_allreduce --backend gloo --world-size 4
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SIZES_MB = [1, 10, 100, 1024]


def _setup(rank: int, world_size: int, backend: str):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())


def _worker(rank: int, world_size: int, backend: str, n_warmup: int, n_iters: int):
    _setup(rank, world_size, backend)
    use_cuda = backend == "nccl"
    device = torch.device("cuda" if use_cuda else "cpu")

    if rank == 0:
        print(f"# backend={backend} world_size={world_size}")
        print("size_MB,mean_ms,std_ms,bw_GBps")

    for size_mb in SIZES_MB:
        n_floats = (size_mb * 1024 * 1024) // 4
        x = torch.zeros(n_floats, dtype=torch.float32, device=device)

        for _ in range(n_warmup):
            dist.all_reduce(x, async_op=False)
        if use_cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(n_iters):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(x, async_op=False)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        mean = statistics.mean(times)
        std = statistics.pstdev(times) if len(times) > 1 else 0.0
        # ring-allreduce moves ~2*(N-1)/N * size; report effective bandwidth
        eff_bytes = 2 * (world_size - 1) / world_size * size_mb * 1024 * 1024
        bw_GBps = (eff_bytes / mean) / (1024 ** 3) if mean > 0 else 0.0

        # gather all rank measurements on rank 0
        ts = [None for _ in range(world_size)]
        dist.all_gather_object(ts, mean)
        if rank == 0:
            agg_mean = statistics.mean(ts)
            print(f"{size_mb},{agg_mean*1000:.3f},{std*1000:.3f},{bw_GBps:.3f}")
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iters", type=int, default=10)
    args = p.parse_args()

    if args.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requires CUDA")

    mp.spawn(_worker, args=(args.world_size, args.backend, args.n_warmup, args.n_iters),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

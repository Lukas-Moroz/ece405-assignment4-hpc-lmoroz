"""Naive DDP correctness check (Problem naive_ddp).

Trains a tiny toy model with naive all-reduce-after-backward DDP and confirms
its weights match a single-process run.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _broadcast(model):
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def _setup(rank, world_size, backend):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29502")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _worker(rank, world_size, backend):
    _setup(rank, world_size, backend)

    torch.manual_seed(0)
    full_x = torch.randn(32, 10)
    full_y = torch.randn(32, 5)

    # Single-process reference
    torch.manual_seed(123)
    ref = _Toy()
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.05)

    # DDP model: same init as ref via broadcast
    torch.manual_seed(456 + rank)  # different to prove broadcast works
    ddp = _Toy()
    _broadcast(ddp)
    # also broadcast ref's weights to ddp so they start identically
    if rank == 0:
        with torch.no_grad():
            for p_ref, p_ddp in zip(ref.parameters(), ddp.parameters()):
                p_ddp.copy_(p_ref)
    _broadcast(ddp)
    ddp_opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    n = full_x.shape[0]
    local = n // world_size
    for step in range(5):
        # reference: full batch
        ref_opt.zero_grad()
        out = ref(full_x)
        loss = ((out - full_y) ** 2).mean()
        loss.backward()
        ref_opt.step()

        # ddp: shard of batch
        offset = rank * local
        x = full_x[offset : offset + local]
        y = full_y[offset : offset + local]
        ddp_opt.zero_grad()
        out = ddp(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        for p in ddp.parameters():
            if p.grad is None:
                continue
            p.grad.div_(world_size)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        ddp_opt.step()

    if rank == 0:
        max_err = 0.0
        for p_ref, p_ddp in zip(ref.parameters(), ddp.parameters()):
            err = (p_ref - p_ddp).abs().max().item()
            max_err = max(max_err, err)
        print(f"naive_ddp world_size={world_size} max_param_diff={max_err:.3e}")
        assert max_err < 1e-5, f"Weights drifted: {max_err}"
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    args = p.parse_args()
    mp.spawn(_worker, args=(args.world_size, args.backend),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

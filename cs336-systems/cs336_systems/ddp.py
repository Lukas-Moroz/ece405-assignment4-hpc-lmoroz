"""Distributed Data Parallel (DDP) wrappers.

Provides:
    DDPIndividualParameters: overlaps backward with one all-reduce per parameter.
    DDPBucketed: groups parameters into buckets and all-reduces a bucket once
        all of its parameters have gradients.

Both wrappers expose ``forward(*args, **kwargs)`` and
``finish_gradient_synchronization()``.
"""
from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn


def _broadcast_module(module: nn.Module, src: int = 0) -> None:
    """Broadcast all module state (parameters + buffers) from rank ``src``."""
    if not dist.is_available() or not dist.is_initialized():
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


class DDPIndividualParameters(nn.Module):
    """DDP that issues one async all-reduce per parameter as gradients become ready.

    Implementation notes:
    * Uses ``register_post_accumulate_grad_hook`` to fire the moment a gradient
      finishes accumulation in the backward pass.
    * Stores async handles so the user can call
      ``finish_gradient_synchronization`` to ensure all comms are queued before
      ``optimizer.step``.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self._handles: list = []
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 1. Synchronize initial weights/buffers from rank 0.
        _broadcast_module(self.module, src=0)

        # 2. Hook each parameter that requires grad.
        if self._world_size > 1:
            self._register_hooks()

    def _register_hooks(self) -> None:
        for p in self.module.parameters():
            if not p.requires_grad:
                continue

            def _hook(param: torch.Tensor):
                # Average across ranks: divide first, then sum-reduce.
                param.grad.div_(self._world_size)
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                self._handles.append(handle)

            p.register_post_accumulate_grad_hook(_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for h in self._handles:
            h.wait()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Bucketed DDP
# ---------------------------------------------------------------------------


class _Bucket:
    __slots__ = ("params", "size_bytes", "ready_count")

    def __init__(self) -> None:
        self.params: List[nn.Parameter] = []
        self.size_bytes: int = 0
        self.ready_count: int = 0


class DDPBucketed(nn.Module):
    """DDP that groups parameter gradients into buckets before all-reducing.

    A bucket is launched as soon as every parameter inside it has had its
    gradient accumulated.  Gradients within a bucket are flattened with
    ``torch._utils._flatten_dense_tensors`` and unflattened back into the
    parameter ``.grad`` tensors after the all-reduce completes.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float) -> None:
        super().__init__()
        self.module = module
        self.bucket_size_bytes = float(bucket_size_mb) * 1024 * 1024
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        _broadcast_module(self.module, src=0)

        # Pending async ops: list of (handle, bucket, flat_tensor)
        self._pending: list = []

        # Build buckets in reverse parameter order (gradients become ready in
        # roughly that order during the backward pass).
        self._buckets: List[_Bucket] = []
        # Map from id(param) -> bucket index, so we can look up quickly in the
        # post-grad hook.
        self._param_to_bucket: dict = {}
        # Track unique parameter ids to handle weight-tying gracefully (skip
        # duplicates so we don't double-count or fire the hook twice).
        seen_ids = set()
        params_in_order: List[nn.Parameter] = []
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            params_in_order.append(p)

        # Reverse order for bucket assignment.
        cur = _Bucket()
        for p in reversed(params_in_order):
            psize = p.numel() * p.element_size()
            if cur.params and cur.size_bytes + psize > self.bucket_size_bytes:
                self._buckets.append(cur)
                cur = _Bucket()
            cur.params.append(p)
            cur.size_bytes += psize
        if cur.params:
            self._buckets.append(cur)

        for idx, b in enumerate(self._buckets):
            for p in b.params:
                self._param_to_bucket[id(p)] = idx

        if self._world_size > 1:
            self._register_hooks()

    def _register_hooks(self) -> None:
        for p in self.module.parameters():
            if not p.requires_grad or id(p) not in self._param_to_bucket:
                continue

            bucket_idx = self._param_to_bucket[id(p)]

            def _hook(_param, bidx=bucket_idx):
                bucket = self._buckets[bidx]
                bucket.ready_count += 1
                if bucket.ready_count == len(bucket.params):
                    self._launch_bucket(bidx)

            p.register_post_accumulate_grad_hook(_hook)

    def _launch_bucket(self, bidx: int) -> None:
        from torch._utils import _flatten_dense_tensors

        bucket = self._buckets[bidx]
        grads = [p.grad for p in bucket.params]
        flat = _flatten_dense_tensors(grads)
        flat.div_(self._world_size)
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self._pending.append((handle, bucket, flat))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def on_train_batch_start(self) -> None:
        """Reset bucket-ready counters at the start of every batch."""
        for b in self._buckets:
            b.ready_count = 0
        self._pending.clear()

    def finish_gradient_synchronization(self) -> None:
        from torch._utils import _unflatten_dense_tensors

        for handle, bucket, flat in self._pending:
            handle.wait()
            grads = [p.grad for p in bucket.params]
            unflat = _unflatten_dense_tensors(flat, grads)
            for g, u in zip(grads, unflat):
                g.copy_(u)
        self._pending.clear()
        # Reset for next iteration.
        for b in self._buckets:
            b.ready_count = 0

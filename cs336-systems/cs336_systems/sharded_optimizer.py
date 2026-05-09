"""Optimizer state sharding (ZeRO-1 style).

Each rank only owns optimizer state for ~1/world_size of the parameters.  After
each ``step``, ranks broadcast the updated parameters they own so that every
rank ends with identical weights.
"""
from __future__ import annotations

from typing import Any, Iterable, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(
        self,
        params: Iterable,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ) -> None:
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs

        # rank / world_size; default to 0/1 when distributed not initialized.
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Maps id(param) -> rank that owns the optimizer state for that param.
        self._param_owner: dict = {}
        # Local optimizer that only sees the params owned by this rank.
        self._local_optimizer: Optimizer | None = None
        # Track all params we have seen, to allow add_param_group to extend
        # (called by super().__init__).
        self._all_params: list = []
        # Round-robin assignment counter (number of params seen so far).
        self._next_owner = 0

        # super().__init__ will call self.add_param_group for each input group,
        # which is what populates the data structures above.
        super().__init__(params, defaults={})

    # ------------------------------------------------------------------
    # Optimizer API
    # ------------------------------------------------------------------
    def add_param_group(self, param_group: dict[str, Any]) -> None:  # type: ignore[override]
        # Make sure the super class records this group (used by zero_grad, etc).
        super().add_param_group(param_group)
        # Round-robin assign new params to ranks.
        new_local_params: list = []
        for p in param_group["params"]:
            if id(p) in self._param_owner:
                continue
            owner = self._next_owner % max(1, self._world_size)
            self._param_owner[id(p)] = owner
            self._all_params.append(p)
            self._next_owner += 1
            if owner == self._rank:
                new_local_params.append(p)
        # Build / extend the local optimizer.
        if new_local_params:
            if self._local_optimizer is None:
                self._local_optimizer = self._optimizer_cls(new_local_params, **self._optimizer_kwargs)
            else:
                self._local_optimizer.add_param_group({"params": new_local_params})

    def step(self, closure=None, **kwargs):  # type: ignore[override]
        loss = None
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure=closure, **kwargs)
        # Broadcast updated parameters from each owner to all other ranks.
        if dist.is_initialized() and self._world_size > 1:
            for p in self._all_params:
                owner = self._param_owner[id(p)]
                dist.broadcast(p.data, src=owner)
        return loss

from __future__ import annotations

from typing import Type

import torch

from cs336_systems.flash_attention import FlashAttention2Pytorch, FlashAttention2Triton
from cs336_systems.ddp import DDPIndividualParameters, DDPBucketed
from cs336_systems.sharded_optimizer import ShardedOptimizer


def get_flashattention_autograd_function_pytorch() -> Type:
    """Return the pure-PyTorch FlashAttention-2 autograd.Function class."""
    return FlashAttention2Pytorch


def get_flashattention_autograd_function_triton() -> Type:
    """Return the Triton-kernel FlashAttention-2 autograd.Function class."""
    return FlashAttention2Triton


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """Return a DDP wrapper that overlaps backward with one all-reduce per parameter."""
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    return DDPBucketed(module, bucket_size_mb=bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.on_train_batch_start()


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    return ShardedOptimizer(params, optimizer_cls=optimizer_cls, **kwargs)

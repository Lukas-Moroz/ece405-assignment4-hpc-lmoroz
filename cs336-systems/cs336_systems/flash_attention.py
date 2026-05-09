"""FlashAttention-2 implementations.

Provides:
    FlashAttention2Pytorch: pure-PyTorch tiled forward, torch.compile backward.
    FlashAttention2Triton: Triton forward kernel, torch.compile backward.
"""
from __future__ import annotations

import math

import torch
from einops import einsum


def _flash_backward_recompute(Q, K, V, O, dO, L, is_causal: bool):
    """Reference / torch.compile-friendly backward for FlashAttention-2.

    Implements equations (13)-(19) of the assignment handout.
    Operates on tensors with leading batch dimensions then (seq, d).
    """
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # S: (..., Nq, Nk)
    S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        nq = Q.shape[-2]
        nk = K.shape[-2]
        idx_q = torch.arange(nq, device=Q.device)[:, None]
        idx_k = torch.arange(nk, device=K.device)[None, :]
        S = torch.where(idx_q >= idx_k, S, torch.full_like(S, -1e6))

    # P = exp(S - L)  -- L was logsumexp over last dim of S
    P = torch.exp(S - L.unsqueeze(-1))

    # D_i = rowsum(O * dO)  shape (..., Nq)
    D = (O * dO).sum(dim=-1)

    # dV = P^T dO
    dV = einsum(P, dO, "... q k, ... q d -> ... k d")
    # dP = dO V^T
    dP = einsum(dO, V, "... q d, ... k d -> ... q k")
    # dS = P * (dP - D[..., :, None])
    dS = P * (dP - D.unsqueeze(-1))
    # dQ = dS K * scale
    dQ = einsum(dS, K, "... q k, ... k d -> ... q d") * scale
    # dK = dS^T Q * scale
    dK = einsum(dS, Q, "... q k, ... q d -> ... k d") * scale
    return dQ, dK, dV


# torch.compile the backward for speed on GPU; on CPU (or in environments
# without a working C++ toolchain for Inductor) torch.compile may fail at runtime.
# We expose ``set_compiled_backward(True)`` so callers can opt in on GPU.
_flash_backward_compiled = _flash_backward_recompute


def set_compiled_backward(enable: bool = True) -> None:
    """Optionally swap the backward in for a torch.compile'd version."""
    global _flash_backward_compiled
    if enable:
        _flash_backward_compiled = torch.compile(_flash_backward_recompute, fullgraph=False, dynamic=True)
    else:
        _flash_backward_compiled = _flash_backward_recompute


def _backward(*args, **kwargs):
    return _flash_backward_compiled(*args, **kwargs)


class FlashAttention2Pytorch(torch.autograd.Function):
    """Pure-PyTorch FlashAttention-2.

    Forward: tiled online-softmax. Saves L (logsumexp) for the backward.
    Backward: a torch.compile'd recomputation using L (no online softmax).
    """

    Q_TILE = 32
    K_TILE = 32

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        # Q,K,V: (..., seq, d). We collapse all leading dims into one batch.
        orig_shape_q = Q.shape
        orig_shape_kv = K.shape

        d = Q.shape[-1]
        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)

        Q_flat = Q.reshape(-1, Nq, d)
        K_flat = K.reshape(-1, Nk, d)
        V_flat = V.reshape(-1, Nk, d)

        B = Q_flat.shape[0]
        Bq = FlashAttention2Pytorch.Q_TILE
        Bk = FlashAttention2Pytorch.K_TILE

        O = torch.zeros_like(Q_flat)
        L = torch.zeros((B, Nq), device=Q.device, dtype=Q_flat.dtype)

        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk

        for i in range(Tq):
            qs = i * Bq
            qe = min(qs + Bq, Nq)
            Qi = Q_flat[:, qs:qe, :]  # (B, bq, d)

            Oi = torch.zeros_like(Qi)
            li = torch.zeros((B, qe - qs), device=Q.device, dtype=Q_flat.dtype)
            mi = torch.full((B, qe - qs), float("-inf"), device=Q.device, dtype=Q_flat.dtype)

            for j in range(Tk):
                ks = j * Bk
                ke = min(ks + Bk, Nk)
                Kj = K_flat[:, ks:ke, :]
                Vj = V_flat[:, ks:ke, :]

                # S_ij = Qi Kj^T * scale -> (B, bq, bk)
                Sij = torch.einsum("b q d, b k d -> b q k", Qi, Kj) * scale

                if is_causal:
                    idx_q = torch.arange(qs, qe, device=Q.device)[:, None]
                    idx_k = torch.arange(ks, ke, device=Q.device)[None, :]
                    Sij = torch.where(idx_q >= idx_k, Sij, torch.full_like(Sij, -1e6))

                # online softmax
                m_new = torch.maximum(mi, Sij.amax(dim=-1))
                P_tilde = torch.exp(Sij - m_new.unsqueeze(-1))
                alpha = torch.exp(mi - m_new)  # rescale prior accumulator
                li = alpha * li + P_tilde.sum(dim=-1)
                Oi = alpha.unsqueeze(-1) * Oi + torch.einsum("b q k, b k d -> b q d", P_tilde, Vj)
                mi = m_new

            # finalize tile
            Oi = Oi / li.unsqueeze(-1)
            Li = mi + torch.log(li)

            O[:, qs:qe, :] = Oi
            L[:, qs:qe] = Li

        O = O.reshape(orig_shape_q)
        L_out = L.reshape(*orig_shape_q[:-1])  # (..., Nq)

        ctx.save_for_backward(Q, K, V, O, L_out)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = _backward(Q, K, V, O, dO, L, is_causal)
        return dQ, dK, dV, None


# ---------------------------------------------------------------------------
# Triton kernel implementation (forward only, backward via torch.compile)
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - triton is GPU-only
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # On-chip buffers
        m_i = tl.full((Q_TILE_SIZE,), value=-1e30, dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        q_start = query_tile_index * Q_TILE_SIZE
        q_offsets = q_start + tl.arange(0, Q_TILE_SIZE)

        n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        for j in range(0, n_k_tiles):
            Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # S = Qi Kj^T * scale, shape (Q_TILE, K_TILE), accumulate fp32
            Sij = tl.dot(Qi, tl.trans(Kj)) * scale

            if IS_CAUSAL:
                k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                mask = q_offsets[:, None] >= k_offsets[None, :]
                Sij = tl.where(mask, Sij, Sij - 1e6)

            m_new = tl.maximum(m_i, tl.max(Sij, axis=1))
            P_tilde = tl.exp(Sij - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = alpha * l_i + tl.sum(P_tilde, axis=1)

            o_i = o_i * alpha[:, None]
            o_i = tl.dot(P_tilde.to(Vj.dtype), Vj, acc=o_i)
            m_i = m_new

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        o_i = o_i / l_i[:, None]
        L_i = m_i + tl.log(l_i)

        tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, L_i, boundary_check=(0,))


    class FlashAttention2Triton(torch.autograd.Function):
        Q_TILE = 64
        K_TILE = 64

        @staticmethod
        def forward(ctx, Q, K, V, is_causal: bool = False):
            assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton kernel requires CUDA tensors"
            # collapse leading dims
            orig_shape_q = Q.shape
            d = Q.shape[-1]
            Nq = Q.shape[-2]
            Nk = K.shape[-2]

            Q_ = Q.reshape(-1, Nq, d).contiguous()
            K_ = K.reshape(-1, Nk, d).contiguous()
            V_ = V.reshape(-1, Nk, d).contiguous()
            B = Q_.shape[0]

            O = torch.empty_like(Q_)
            L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

            Q_TILE = FlashAttention2Triton.Q_TILE
            K_TILE = FlashAttention2Triton.K_TILE
            # tile sizes need to be at least 16
            scale = 1.0 / math.sqrt(d)

            grid = (triton.cdiv(Nq, Q_TILE), B)
            _flash_fwd_kernel[grid](
                Q_, K_, V_,
                O, L,
                Q_.stride(0), Q_.stride(1), Q_.stride(2),
                K_.stride(0), K_.stride(1), K_.stride(2),
                V_.stride(0), V_.stride(1), V_.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                Nq, Nk,
                scale,
                D=d,
                Q_TILE_SIZE=Q_TILE,
                K_TILE_SIZE=K_TILE,
                IS_CAUSAL=is_causal,
            )

            O = O.reshape(orig_shape_q)
            L_out = L.reshape(*orig_shape_q[:-1])
            ctx.save_for_backward(Q, K, V, O, L_out)
            ctx.is_causal = is_causal
            return O

        @staticmethod
        def backward(ctx, dO):
            Q, K, V, O, L = ctx.saved_tensors
            is_causal = ctx.is_causal
            dQ, dK, dV = _backward(Q, K, V, O, dO, L, is_causal)
            return dQ, dK, dV, None
else:
    class FlashAttention2Triton(torch.autograd.Function):  # type: ignore[no-redef]
        @staticmethod
        def forward(ctx, Q, K, V, is_causal: bool = False):
            raise RuntimeError("Triton not available; cannot use FlashAttention2Triton")

        @staticmethod
        def backward(ctx, dO):  # pragma: no cover
            raise RuntimeError("Triton not available")

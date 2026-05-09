"""torch.profiler-based profiling for §1.1.4 when nsys is unavailable."""
from __future__ import annotations
import argparse, contextlib
import torch
from torch.profiler import profile, ProfilerActivity
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.model_configs import MODEL_CONFIGS, VOCAB_SIZE, BATCH_SIZE, ROPE_THETA

def make_model(size, context_length, device):
    cfg = MODEL_CONFIGS[size]
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, context_length=context_length,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=ROPE_THETA,
    ).to(device)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(MODEL_CONFIGS), default="small")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--mode", choices=["forward", "fwd_bwd", "train"], default="forward")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--active", type=int, default=3)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--trace-dir", type=str, default=None,
                   help="If set, export Chrome trace JSON to this directory")
    args = p.parse_args()

    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else None
    model = make_model(args.size, args.context_length, device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=device)

    def autocast_ctx():
        if amp_dtype is None: return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    def step():
        with autocast_ctx():
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        if args.mode != "forward":
            loss.backward()
        if args.mode == "train":
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Pre-warmup outside profiler
    for _ in range(2):
        step()
        if args.mode != "forward": optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    schedule = torch.profiler.schedule(wait=0, warmup=args.warmup, active=args.active, repeat=1)

    trace_handler = None
    if args.trace_dir:
        trace_handler = torch.profiler.tensorboard_trace_handler(args.trace_dir)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.warmup + args.active):
            step()
            if args.mode != "forward": optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            prof.step()

    print("=" * 80)
    print(f"PROFILE: size={args.size} ctx={args.context_length} mode={args.mode} dtype={args.dtype}")
    print(f"  (averaged over {args.active} active steps)")
    print("=" * 80)

    print("\n--- Top CUDA Kernels by GPU Time ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.top_n))

    events = prof.key_averages()
    total_cuda_us = sum(e.self_cuda_time_total for e in events if e.self_cuda_time_total > 0)

    print(f"\nTotal CUDA time: {total_cuda_us / 1000:.2f} ms")
    print(f"Per-step CUDA time: {total_cuda_us / args.active / 1000:.2f} ms")

    print("\n--- Kernel Category Breakdown ---")
    matmul_us = softmax_us = 0
    other_top = []
    for e in events:
        if e.self_cuda_time_total <= 0: continue
        name = e.key.lower()
        if any(k in name for k in ["gemm", "matmul", "mm_", "cublas", "dot"]):
            matmul_us += e.self_cuda_time_total
        elif "softmax" in name:
            softmax_us += e.self_cuda_time_total
        else:
            other_top.append((e.key, e.self_cuda_time_total, e.count))

    rest = total_cuda_us - matmul_us - softmax_us
    print(f"  MatMul:  {matmul_us/1000:.2f} ms ({100*matmul_us/max(total_cuda_us,1):.1f}%)")
    print(f"  Softmax: {softmax_us/1000:.2f} ms ({100*softmax_us/max(total_cuda_us,1):.1f}%)")
    print(f"  Other:   {rest/1000:.2f} ms ({100*rest/max(total_cuda_us,1):.1f}%)")

    print("\n  Top non-matmul, non-softmax kernels:")
    other_top.sort(key=lambda x: x[1], reverse=True)
    for name, us, count in other_top[:10]:
        print(f"    {name}: {us/1000:.2f} ms ({count} calls)")

if __name__ == "__main__":
    main()

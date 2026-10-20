#!/usr/bin/env python3
"""Time vLLM's NVFP4 Marlin MoE (weight-only fp4, bf16 activations: the backend vLLM logs as
'MARLIN' NvFp4 MoE) at the Qwen3.6-35B-A3B MoE layer shape, standalone, inside the vLLM container.

  python3 marlin_moe_bench.py --tokens 1024,2048,4096,8192 --reps 10 [--warmup 3] [--check] [--profile]

Call path (vLLM 0.26.x): MarlinExperts.apply -> fused_marlin_moe() (experts/marlin_moe.py), i.e.
  moe_align_block_size -> _fused_marlin_moe: ops.moe_wna16_marlin_gemm (w13, SiLU-and-mul input) ->
  apply_moe_activation (silu_and_mul) -> ops.moe_wna16_marlin_gemm (w2, mul_topk_weights) -> torch.sum over topk.
Weights are prepared with the same function the model loader uses (prepare_nvfp4_moe_layer_for_marlin).
The call below mirrors MarlinExperts.apply (global_scale1/2 = processed w13/w2 weight_scale_2, preallocated
intermediate_cache13 / intermediate_cache2 / output as the modular kernel provisions them).

TFLOPS = T * K * 3 * H * I * 2 / time (3 matrices of H x I per expert, 2 FLOP per MAC), on the T*K routed rows.
GPU memory: ~0.4 GB weights + intermediates (~0.5 GB at T=8192); free the raw weights before timing.
"""
import argparse
import statistics
import sys
import types

import torch

E_DEF, TOPK_DEF, H_DEF, I_DEF = 256, 8, 2048, 512

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def make_raw(E, H, I, dev, seed, scale_lo=0.01, scale_hi=0.1, gscale=1.0):
    """Random checkpoint-layout NVFP4 expert weights: packed uint8 [E, N, K/2], e4m3 block scales [E, N, K/16],
    fp32 per-expert global scale [E]."""
    g = torch.Generator(device=dev).manual_seed(seed)

    def pk(n, k):
        return torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g)

    def sc(n, k):
        s = torch.rand((E, n, k // 16), device=dev, generator=g) * (scale_hi - scale_lo) + scale_lo
        return s.to(torch.float8_e4m3fn)

    w13, w13s = pk(2 * I, H), sc(2 * I, H)
    w2, w2s = pk(H, I), sc(H, I)
    g13 = torch.full((E,), gscale, dtype=torch.float32, device=dev)
    g2 = torch.full((E,), gscale, dtype=torch.float32, device=dev)
    return w13, w13s, g13, w2, w2s, g2


def dequant(packed, scales, gscale, dtype=torch.float32):
    """packed uint8 [E, N, K/2] (low nibble = even k), e4m3 scales [E, N, K/16], gscale [E] -> [E, N, K]."""
    lut = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    lo = packed & 0xF
    hi = packed >> 4
    nib = torch.stack([lo, hi], dim=-1).view(*packed.shape[:-1], -1)  # [E, N, K]
    val = lut[(nib & 7).long()] * torch.where(nib & 8 != 0, -1.0, 1.0)
    s = scales.float().repeat_interleave(16, dim=-1)
    return (val * s * gscale.view(-1, 1, 1)).to(dtype)


def ref_moe(x, w1d, w2d, topk_w, topk_ids):
    """Dense dequantised reference: out[t] = sum_k w[t,k] * W2[e] (silu(gate) * up) ; gate = first I rows of w13."""
    T = x.size(0)
    I = w2d.size(2)
    out = torch.zeros(T, w2d.size(1), dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(w1d.size(0)):
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = xf[tok] @ w1d[e].T
        a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
        o = a @ w2d[e].T
        out.index_add_(0, tok, o * topk_w[tok, slot].unsqueeze(1))
    return out


def route(T, E, K, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.rand((T, E), device=dev, generator=g).topk(K, dim=1).indices.to(torch.int32)  # uniform random top-K
    logits = torch.randn((T, K), device=dev, generator=g)
    return torch.softmax(logits, dim=1).float().contiguous(), ids.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1024,2048,4096,8192")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--experts", type=int, default=E_DEF)
    ap.add_argument("--topk", type=int, default=TOPK_DEF)
    ap.add_argument("--hidden", type=int, default=H_DEF)
    ap.add_argument("--inter", type=int, default=I_DEF)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-check", action="store_true", help="skip the T=16,E=8 dequantised-reference check")
    ap.add_argument("--profile", action="store_true", help="print a torch.profiler CUDA kernel table for one call per T")
    ap.add_argument("--no-prealloc", action="store_true", help="let fused_marlin_moe allocate its own caches")
    args = ap.parse_args()

    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mfp4
    from vllm.scalar_type import scalar_types

    dev = torch.device("cuda")
    torch.cuda.set_device(0)
    dtype = torch.bfloat16
    print(f"vllm {vllm.__version__}  torch {torch.__version__}  device {torch.cuda.get_device_name(0)} "
          f"sm{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    print("call path:")
    print(f"  {marlin_moe.__name__}.fused_marlin_moe  ({marlin_moe.__file__})")
    print("    -> moe_align_block_size (fused_moe/moe_align_block_size.py)")
    print("    -> _fused_marlin_moe: torch.ops._moe_C.moe_wna16_marlin_gemm (via vllm._custom_ops.moe_wna16_marlin_gemm) x2,")
    print("       apply_moe_activation(SILU -> silu_and_mul) between them, then torch.sum over topk")
    print(f"  weight prep: {mfp4.__name__}.prepare_nvfp4_moe_layer_for_marlin")
    print("  quant_type = scalar_types.float4_e2m1f, input_dtype=None (bf16 A), is_k_full=True, use_fp32_reduce=True, "
          "use_atomic_add=False")

    qt = scalar_types.float4_e2m1f

    def prep(E, H, I, seed):
        raw = make_raw(E, H, I, dev, seed)
        w13, w13s, g13, w2, w2s, g2 = raw
        layer = types.SimpleNamespace(num_experts=E, hidden_size=H, intermediate_size_per_partition=I,
                                      params_dtype=dtype)
        with set_current_vllm_config(VllmConfig()):
            out = mfp4.prepare_nvfp4_moe_layer_for_marlin(layer=layer, w13=w13, w13_scale=w13s, w13_scale_2=g13,
                                                          w2=w2, w2_scale=w2s, w2_scale_2=g2, is_act_and_mul=True)
        return raw, out

    def call(x, prepped, topk_w, topk_ids, E, bufs):
        w13, w13s, g13, w2, w2s, g2 = prepped
        kw = {}
        if bufs is not None:
            kw = dict(intermediate_cache13=bufs[0], intermediate_cache2=bufs[1], output=bufs[2])
        return marlin_moe.fused_marlin_moe(
            hidden_states=x, w1=w13, w2=w2, bias1=None, bias2=None, w1_scale=w13s, w2_scale=w2s,
            topk_weights=topk_w, topk_ids=topk_ids, global_scale1=g13, global_scale2=g2,
            quant_type_id=qt.id, apply_router_weight_on_input=False, global_num_experts=E,
            activation=MoEActivation.SILU, expert_map=None, is_k_full=True, input_dtype=None, **kw)

    def make_bufs(T, H, I, K):
        if args.no_prealloc:
            return None
        return (torch.empty(T * K * max(2 * I, H), dtype=dtype, device=dev),
                torch.empty(T * K, I, dtype=dtype, device=dev),
                torch.empty(T, H, dtype=dtype, device=dev))

    # ---- sanity check against a dequantised torch reference (small case)
    if not args.no_check:
        Ec, Tc, Kc, Hc, Ic = 8, 16, 2, args.hidden, args.inter
        raw, prepped = prep(Ec, Hc, Ic, args.seed + 100)
        x = (torch.randn(Tc, Hc, device=dev) * 0.5).to(dtype)
        tw, ti = route(Tc, Ec, Kc, dev, args.seed)
        out = call(x, prepped, tw, ti, Ec, None)
        torch.cuda.synchronize()
        w13, w13s, g13, w2, w2s, g2 = raw
        ref = ref_moe(x, dequant(w13, w13s, g13), dequant(w2, w2s, g2), tw, ti)
        o = out.float()
        rel = ((o - ref).norm() / ref.norm()).item()
        print(f"sanity (E={Ec} T={Tc} K={Kc} H={Hc} I={Ic}): finite={bool(torch.isfinite(o).all())} "
              f"nonzero={bool((o != 0).any())} |out|={o.norm().item():.4g} |ref|={ref.norm().item():.4g} "
              f"rel_err_vs_dequant_ref={rel:.3e}  ->  {'OK' if rel < 3e-2 else 'MISMATCH (check wiring)'}")
        del raw, prepped, x, out, ref, o
        torch.cuda.empty_cache()

    # ---- timing at the real shape
    E, K, H, I = args.experts, args.topk, args.hidden, args.inter
    raw, prepped = prep(E, H, I, args.seed)
    del raw
    torch.cuda.empty_cache()
    wbytes = sum(t.numel() * t.element_size() for t in prepped)
    print(f"repacked weights: {wbytes / 2**30:.3f} GiB  (E={E} K={K} H={H} I={I}); "
          f"torch mem allocated {torch.cuda.memory_allocated() / 2**30:.3f} GiB")

    print(f"{'T':>6} {'bsm':>4} {'med_ms':>9} {'min_ms':>9} {'max_ms':>9} {'TFLOPS':>8}  finite/nonzero")
    for T in [int(t) for t in args.tokens.split(",")]:
        g = torch.Generator(device=dev).manual_seed(args.seed + T)
        x = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(dtype)
        tw, ti = route(T, E, K, dev, args.seed + T)
        bufs = make_bufs(T, H, I, K)
        bsm = next((b for b in [8, 16, 32, 48, 64] if T * K / E / b < 0.9), 64)  # mirrors fused_marlin_moe
        for _ in range(args.warmup):
            out = call(x, prepped, tw, ti, E, bufs)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = call(x, prepped, tw, ti, E, bufs)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        med = statistics.median(ts)
        flops = T * K * 3 * H * I * 2
        o = out.float()
        print(f"{T:>6} {bsm:>4} {med:>9.3f} {min(ts):>9.3f} {max(ts):>9.3f} {flops / (med * 1e-3) / 1e12:>8.2f}  "
              f"{bool(torch.isfinite(o).all())}/{bool((o != 0).any())}")
        sys.stdout.flush()
        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                call(x, prepped, tw, ti, E, bufs)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8, max_name_column_width=70))
        del x, tw, ti, bufs, out, o
        torch.cuda.empty_cache()
    print(f"peak GPU mem allocated by this process: {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()

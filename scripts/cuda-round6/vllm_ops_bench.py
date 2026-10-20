#!/usr/bin/env python3
"""Time vLLM's own prefill kernels at the Qwen3.6-35B-A3B shapes, standalone, inside the vLLM container.

  python3 vllm_ops_bench.py [--only gdn,fp8,attn] [--tokens 1024,2048,4096] [--reps 10] [--warmup 3] [--no-check]
                            [--kv-len 2048,8192,16384] [--kv-dtype fp8,bf16] [--profile]

(a) GatedDeltaNet prefill (Triton/FLA chunked gated delta rule; the server logs 'Using Triton/FLA GDN prefill kernel'):
    vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn (prefill branch of the layer forward):
      fused_post_conv_prep(conv_output[T, 2*Hk*dk + Hv*dv], a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
                           apply_l2norm=True, output_g_exp=False)  ->  q,k [T,Hk,dk], v [T,Hv,dv], g,beta f32 [T,Hv]
      vllm.third_party.flash_linear_attention.ops.chunk_gated_delta_rule(q=,k=,v=,g=,beta=, initial_state=[1,Hv,dv,dk] f32,
                           output_final_state=True, cu_seqlens=[0,T], chunk_indices=, chunk_offsets=,
                           use_qk_l2norm_in_kernel=False)   (l2norm is done in fused_post_conv_prep, as in the model)
    Reported: the chunk kernel alone (what to compare with a recurrence kernel) and prep + chunk.
(b) fp8 W8A8 (server logs 'Selected CutlassFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod'):
    FP8ScaledMMLinearKernel.apply_weights: QuantFP8 static per-tensor (-> ops.scaled_fp8_quant(x, scale)) then
    CutlassFP8ScaledMMLinearKernel.apply_scaled_mm -> vllm._custom_ops.cutlass_scaled_mm(A, B=W.t(), out_dtype=bf16,
    scale_a=[1] f32, scale_b=[1] (per-tensor) or [N,1] (per-channel, as fused layers use), bias=None)
(c) prefill attention (server logs 'Using FLASHINFER attention backend'; page size 2096, kv cache fp8_e4m3, q bf16, sm121
    => TRTLLM prefill unsupported, so FlashInfer native): flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace,
    kv_layout=get_kv_cache_layout(), backend='auto').plan(qo_indptr, paged_kv_indptr, paged_kv_indices,
    paged_kv_last_page_len, num_qo_heads=16, num_kv_heads=2, head_dim_qk=256, page_size=2096, causal=True,
    sm_scale=D**-0.5, window_left=-1, logits_soft_cap=None, q_data_type=bf16, kv_data_type=fp8_e4m3, o_data_type=bf16, ...)
    then .run(q, (k_cache, v_cache), q_scale=1, k_scale=1, v_scale=1, out=out); K/V are views of one packed
    (B, H, N, 2*D) cache exactly like the vLLM backend. plan() is excluded from the timing (it is CPU work done once per step).
"""
import argparse
import inspect
import statistics
import sys

import torch

DEV = torch.device("cuda")
BF16 = torch.bfloat16


def timeit(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts), max(ts)


def relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ----------------------------------------------------------------------------------------------- (a) GDN
def gdn(args):
    from vllm.config import VllmConfig, set_current_vllm_config  # noqa: F401
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    Hk, Hv, dk, dv = 16, 32, 128, 128
    print(f"== (a) GDN prefill: Hk={Hk} Hv={Hv} dk={dk} dv={dv}, FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}, "
          f"chunk_gated_delta_rule from {chunk_gated_delta_rule.__module__}")

    def make(T, seed):
        g = torch.Generator(device=DEV).manual_seed(seed)
        conv = (torch.randn(T, 2 * Hk * dk + Hv * dv, device=DEV, generator=g)).to(BF16)
        a = (torch.randn(T, Hv, device=DEV, generator=g)).to(BF16)
        b = (torch.randn(T, Hv, device=DEV, generator=g)).to(BF16)
        A_log = torch.log(torch.rand(Hv, device=DEV, generator=g) * 15 + 1).float()
        dt_bias = (torch.rand(Hv, device=DEV, generator=g) * 0.5).to(BF16)
        h0 = (torch.randn(1, Hv, dv, dk, device=DEV, generator=g) * 0.05).float()
        return conv, a, b, A_log, dt_bias, h0

    def prep(conv, a, b, A_log, dt_bias):
        return fused_post_conv_prep(conv_output=conv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, num_k_heads=Hk,
                                    head_k_dim=dk, head_v_dim=dv, apply_l2norm=True, output_g_exp=False)

    def chunk(T, q, k, v, g, beta, h0, cu, ci, co):
        return chunk_gated_delta_rule(q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), g=g.unsqueeze(0),
                                      beta=beta.unsqueeze(0), initial_state=h0, output_final_state=True,
                                      cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co, use_qk_l2norm_in_kernel=False)

    def meta(T):
        cu_cpu = torch.tensor([0, T], dtype=torch.int32)
        ci = prepare_chunk_indices(cu_cpu, FLA_CHUNK_SIZE).to(DEV)
        co = prepare_chunk_offsets(cu_cpu, FLA_CHUNK_SIZE).to(DEV)
        return cu_cpu.to(DEV), ci, co

    if not args.no_check:
        T = 64
        conv, a, b, A_log, dt_bias, h0 = make(T, 7)
        q, k, v, g, beta = prep(conv, a, b, A_log, dt_bias)
        cu, ci, co = meta(T)
        o, hT = chunk(T, q, k, v, g, beta, h0, cu, ci, co)
        torch.cuda.synchronize()
        # naive fp32 recurrence (state [Hv, dv, dk]; GQA: v-head h uses k-head h // (Hv/Hk)); scale = dk**-0.5
        kidx = torch.arange(Hv, device=DEV) // (Hv // Hk)
        S = h0[0].clone()
        outs = []
        for t in range(T):
            S = S * torch.exp(g[t])[:, None, None]
            kt = k[t].float()[kidx]
            qt = q[t].float()[kidx] * dk ** -0.5
            pred = torch.einsum("hvk,hk->hv", S, kt)
            delta = beta[t][:, None] * (v[t].float() - pred)
            S = S + torch.einsum("hv,hk->hvk", delta, kt)
            outs.append(torch.einsum("hvk,hk->hv", S, qt))
        ref = torch.stack(outs)  # [T, Hv, dv]
        print(f"   sanity T={T}: finite={bool(torch.isfinite(o).all())} |o|={o.float().norm().item():.4g} "
              f"rel_err(o vs fp32 recurrence)={relerr(o[0], ref):.3e} rel_err(final state)={relerr(hT[0], S):.3e}")

    print(f"   {'T':>6} {'chunk_med_ms':>13} {'min':>8} {'max':>8} {'prep_med_ms':>12} {'prep+chunk_ms':>14}")
    for T in [int(t) for t in args.tokens.split(",")]:
        conv, a, b, A_log, dt_bias, h0 = make(T, 11 + T)
        q, k, v, g, beta = prep(conv, a, b, A_log, dt_bias)
        cu, ci, co = meta(T)
        cm, cmin, cmax = timeit(lambda: chunk(T, q, k, v, g, beta, h0, cu, ci, co), args.warmup, args.reps)
        pm, _, _ = timeit(lambda: prep(conv, a, b, A_log, dt_bias), args.warmup, args.reps)
        print(f"   {T:>6} {cm:>13.3f} {cmin:>8.3f} {cmax:>8.3f} {pm:>12.3f} {cm + pm:>14.3f}")
        sys.stdout.flush()
        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                chunk(T, q, k, v, g, beta, h0, cu, ci, co)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=60))
        del conv, q, k, v, g, beta, h0
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- (b) fp8
FP8_SHAPES = [  # name, N, K
    ("in_proj_qkv", 8192, 2048),
    ("in_proj_z", 4096, 2048),
    ("out_proj", 2048, 4096),
    ("q_proj+gate", 8192, 2048),
    ("k/v_proj", 512, 2048),
    ("o_proj", 2048, 4096),
]


def fp8(args):
    from vllm import _custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config

    quant_impl = "ops.scaled_fp8_quant(x, scale)  [static per-tensor]"
    quantize = None
    try:
        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
        with set_current_vllm_config(VllmConfig()):
            qf = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
            t = torch.randn(16, 64, device=DEV, dtype=BF16)
            qf(t, torch.tensor([0.01], device=DEV), None)
        quantize = lambda x, s: qf(x, s, None)  # noqa: E731
        quant_impl = "QuantFP8(static=True, PER_TENSOR)"
    except Exception as e:  # noqa: BLE001
        print(f"   (QuantFP8 unavailable standalone: {type(e).__name__}: {str(e)[:80]}; using ops.scaled_fp8_quant)")
        quantize = lambda x, s: ops.scaled_fp8_quant(x, s)  # noqa: E731
    print(f"== (b) fp8 W8A8: quant = {quant_impl}; GEMM = vllm._custom_ops.cutlass_scaled_mm(A, B=W.t(), out_dtype=bf16, "
          f"scale_a, scale_b, bias=None)")

    def make_w(N, K, seed):
        g = torch.Generator(device=DEV).manual_seed(seed)
        w = (torch.randn(N, K, device=DEV, generator=g) * 0.5).to(torch.float8_e4m3fn)  # checkpoint layout [N, K]
        return w

    if not args.no_check:
        M, N, K = 64, 256, 512
        x = torch.randn(M, K, device=DEV).to(BF16)
        w = make_w(N, K, 3)
        sa = torch.tensor([x.abs().max().item() / 448.0], device=DEV, dtype=torch.float32)
        sb = torch.tensor([0.02], device=DEV, dtype=torch.float32)
        xq, _ = quantize(x, sa)
        out = ops.cutlass_scaled_mm(xq, w.t(), out_dtype=BF16, scale_a=sa, scale_b=sb)
        ref = (xq.float() * sa) @ (w.float() * sb).T
        print(f"   sanity M={M} N={N} K={K}: finite={bool(torch.isfinite(out).all())} rel_err vs dequant ref="
              f"{relerr(out, ref):.3e}")

    print(f"   {'T':>6} {'layer':<12} {'N':>5} {'K':>5} {'gemm_ms':>8} {'TFLOPS':>7} {'gemm_chan_ms':>12} {'TFLOPS':>7} "
          f"{'quant_ms':>8} {'quant+gemm_ms':>13} {'TFLOPS':>7}")
    for T in [int(t) for t in args.tokens.split(",")]:
        for name, N, K in FP8_SHAPES:
            w = make_w(N, K, N + K)
            B = w.t()
            x = torch.randn(T, K, device=DEV).to(BF16)
            sa = torch.tensor([0.01], device=DEV, dtype=torch.float32)
            sb = torch.tensor([0.02], device=DEV, dtype=torch.float32)
            xq, _ = quantize(x, sa)
            flops = 2.0 * T * N * K
            gm, _, _ = timeit(lambda: ops.cutlass_scaled_mm(xq, B, out_dtype=BF16, scale_a=sa, scale_b=sb),
                              args.warmup, args.reps)
            try:
                sbc = torch.full((N, 1), 0.02, device=DEV, dtype=torch.float32)
                gc, _, _ = timeit(lambda: ops.cutlass_scaled_mm(xq, B, out_dtype=BF16, scale_a=sa, scale_b=sbc),
                                  args.warmup, args.reps)
            except Exception as e:  # noqa: BLE001
                gc = float("nan")
                print(f"   (per-channel scale failed: {str(e)[:80]})")
            qm, _, _ = timeit(lambda: quantize(x, sa), args.warmup, args.reps)

            def both():
                q_, _ = quantize(x, sa)
                return ops.cutlass_scaled_mm(q_, B, out_dtype=BF16, scale_a=sa, scale_b=sb)
            bm, _, _ = timeit(both, args.warmup, args.reps)
            tf = lambda ms: flops / (ms * 1e-3) / 1e12  # noqa: E731
            print(f"   {T:>6} {name:<12} {N:>5} {K:>5} {gm:>8.3f} {tf(gm):>7.1f} {gc:>12.3f} {tf(gc):>7.1f} "
                  f"{qm:>8.3f} {bm:>13.3f} {tf(bm):>7.1f}")
            sys.stdout.flush()
            del w, B, x, xq
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- (c) attention
def attn(args):
    import flashinfer
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    from vllm import envs
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    Hq, Hkv, D, PAGE, QLEN = 16, 2, 256, 2096, 2048
    from vllm.config import VllmConfig, set_current_vllm_config
    if args.layout:
        layout = args.layout
    else:
        with set_current_vllm_config(VllmConfig()):
            layout = get_kv_cache_layout()
    ws = torch.empty(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device=DEV)
    print(f"== (c) prefill attention: flashinfer {flashinfer.__version__} BatchPrefillWithPagedKVCacheWrapper(backend='auto', "
          f"kv_layout={layout}), Hq={Hq} Hkv={Hkv} D={D} page_size={PAGE}, causal, q bf16, out bf16, "
          f"workspace {ws.numel() / 2**20:.0f} MiB")
    plan_params = set(inspect.signature(BatchPrefillWithPagedKVCacheWrapper.plan).parameters)

    def build(q_len, kv_len, kv_dtype, seed):
        g = torch.Generator(device=DEV).manual_seed(seed)
        npages = (kv_len + PAGE - 1) // PAGE
        last = kv_len - (npages - 1) * PAGE
        shape = (npages, Hkv, PAGE, 2 * D) if layout == "HND" else (npages, PAGE, Hkv, 2 * D)
        raw = torch.randn(shape, device=DEV, generator=g) * 0.5
        if kv_dtype == "fp8":
            cache = raw.to(torch.float8_e4m3fn)
            kvdt = torch.float8_e4m3fn
        else:
            cache = raw.to(BF16)
            kvdt = BF16
        kc, vc = cache.split(D, dim=-1)  # strided views of one packed cache, as the vLLM backend does
        q = (torch.randn(q_len, Hq, D, device=DEV, generator=g) * 0.5).to(BF16)
        out = torch.empty(q_len, Hq, D, device=DEV, dtype=BF16)
        w = BatchPrefillWithPagedKVCacheWrapper(ws, layout, backend="auto")
        kw = {}
        if "fixed_split_size" in plan_params:
            kw["fixed_split_size"] = -1
        if "disable_split_kv" in plan_params:
            kw["disable_split_kv"] = False
        w.plan(qo_indptr=torch.tensor([0, q_len], dtype=torch.int32), paged_kv_indptr=torch.tensor([0, npages], dtype=torch.int32),
               paged_kv_indices=torch.arange(npages, dtype=torch.int32, device=DEV),
               paged_kv_last_page_len=torch.tensor([last], dtype=torch.int32),
               num_qo_heads=Hq, num_kv_heads=Hkv, head_dim_qk=D, page_size=PAGE, causal=True, sm_scale=D ** -0.5,
               window_left=-1, logits_soft_cap=None, q_data_type=BF16, kv_data_type=kvdt, o_data_type=BF16, **kw)
        run = lambda: w.run(q, (kc, vc), q_scale=1.0, k_scale=1.0, v_scale=1.0, out=out)  # noqa: E731
        return run, q, kc, vc, out, kv_len

    def reference(q, kc, vc, kv_len):
        # gather contiguous K/V [kv_len, Hkv, D] from the paged layout
        def flat(c):
            c = c.float()
            if layout == "HND":
                c = c.permute(0, 2, 1, 3)  # (pages, PAGE, H, D)
            return c.reshape(-1, Hkv, D)[:kv_len]
        K, V = flat(kc), flat(vc)
        rep = Hq // Hkv
        K, V = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)
        qf = q.float()
        s = torch.einsum("qhd,khd->hqk", qf, K) * D ** -0.5
        qn = q.shape[0]
        mask = torch.arange(kv_len, device=DEV)[None, :] <= (torch.arange(qn, device=DEV)[:, None] + (kv_len - qn))
        s = s.masked_fill(~mask[None], float("-inf"))
        return torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), V)

    kvs = [x.strip() for x in args.kv_dtype.split(",")]
    if not args.no_check:
        for kvd in kvs:
            run, q, kc, vc, out, kv_len = build(64, 256, kvd, 5)
            run()
            torch.cuda.synchronize()
            print(f"   sanity (kv={kvd}) q_len=64 kv_len=256: finite={bool(torch.isfinite(out).all())} "
                  f"rel_err vs torch ref={relerr(out, reference(q, kc, vc, kv_len)):.3e}")

    print(f"   {'kv':>4} {'kv_len':>7} {'q_len':>6} {'med_ms':>8} {'min':>8} {'max':>8} {'TFLOPS':>7}")
    for kvd in kvs:
        for kv_len in [int(x) for x in args.kv_len.split(",")]:
            run, q, kc, vc, out, _ = build(QLEN, kv_len, kvd, kv_len)
            med, mn, mx = timeit(run, args.warmup, args.reps)
            keys = QLEN * (kv_len - QLEN) + QLEN * (QLEN + 1) // 2  # sum over queries of keys seen (causal)
            tf = 4.0 * Hq * D * keys / (med * 1e-3) / 1e12
            print(f"   {kvd:>4} {kv_len:>7} {QLEN:>6} {med:>8.3f} {mn:>8.3f} {mx:>8.3f} {tf:>7.1f}")
            sys.stdout.flush()
            del run, q, kc, vc, out
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="gdn,fp8,attn")
    ap.add_argument("--tokens", default="1024,2048,4096", help="T for (a) GDN and (b) fp8 GEMM M")
    ap.add_argument("--kv-len", default="2048,8192,16384", help="(c) kv_len of the 2048-query chunk")
    ap.add_argument("--kv-dtype", default="fp8,bf16", help="(c) KV cache dtype(s): fp8 (server) and/or bf16")
    ap.add_argument("--layout", default="", help="(c) override KV layout NHD|HND (default: vLLM's get_kv_cache_layout())")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--profile", action="store_true", help="(a) print CUDA kernel table for one GDN chunk call per T")
    args = ap.parse_args()

    import vllm
    torch.cuda.set_device(0)
    print(f"vllm {vllm.__version__}  torch {torch.__version__}  device {torch.cuda.get_device_name(0)} "
          f"sm{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}  "
          f"reps={args.reps} warmup={args.warmup}")
    for part in args.only.split(","):
        {"gdn": gdn, "fp8": fp8, "attn": attn}[part.strip()](args)
    print(f"peak GPU mem allocated by this process: {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()

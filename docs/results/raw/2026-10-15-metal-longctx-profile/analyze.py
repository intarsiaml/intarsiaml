#!/usr/bin/env python3
"""Per-op tables from INTARSIA_METAL_PROFILE traces (crates/intarsia-backend-metal/src/profile.rs) of `generate` runs.
usage: analyze.py <kernel.jsonl[.gz]> <prompt_tokens> <chunk> [cb.jsonl[.gz]]
Splits the kernel-mode trace into forward passes at each embedding_bf16 launch: passes with t > 1 are prefill chunks, passes with
t == 1 are decode steps. Maps every kernel name to a class explicitly; an unknown name aborts."""
import gzip, json, sys, collections, os

def load(p):
    if not os.path.exists(p) and os.path.exists(p + ".gz"): p += ".gz"
    f = gzip.open(p, "rt") if p.endswith(".gz") else open(p)
    return [json.loads(l) for l in f if l.strip()]

D, H, NKV, LAYERS = 128, 32, 8, 36
KV_BYTES_PER_KEY_LAYER = 2 * NKV * D * 2          # K + V, bf16
CLASS = {}
def reg(cls, *names):
    for n in names: CLASS[n] = cls
reg("matmul", "matmul_bf16_nax", "matmul_pair_bf16_nax", "matmul_bf16_gemv", "matmul_pair_bf16_gemv", "logits_bf16_gemv",
    "matmul_bf16_gemm", "matmul_pair_bf16_gemm", "matmul_bf16", "matmul_pair_bf16")
reg("attention", "attn_prefill_bf16_flash_nax", "attn_prefill_bf16_flash", "attn_prefill_bf16", "attn_decode_bf16_splitkv", "attn_decode_bf16")
reg("norms/rope", "rmsnorm_bf16", "rmsnorm_add_bf16", "qk_norm_rope_kv_bf16", "qk_norm_bf16", "rope_bf16")
reg("other", "silu_glu_bf16", "embedding_bf16", "kv_append_bf16", "argmax_f32", "add_bf16")

def cls_of(op):
    if op in CLASS: return CLASS[op]
    for k, v in (("matmul", "matmul"), ("logits", "matmul"), ("attn", "attention")):
        if op.startswith(k): return v
    sys.exit(f"unmapped kernel name {op!r}")

def passes(recs):
    out, cur = [], None
    for r in recs:
        if r["op"] == "embedding_bf16":
            cur = [r]; out.append(cur)
        elif cur is not None:
            cur.append(r)
    return out

def dur(r): return r["e"] - r["s"]

def main():
    path, plen, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cbp = sys.argv[4] if len(sys.argv) > 4 else None
    recs = load(path)
    ps = passes(recs)
    pre = [p for p in ps if p[0]["t"] > 1]
    dec = [p for p in ps if p[0]["t"] == 1]
    print(f"### {os.path.basename(path)}: prompt {plen} tokens, chunk {chunk}: {len(pre)} prefill passes, {len(dec)} decode passes, {len(recs)} launches\n")
    if pre:
        tab = collections.OrderedDict(); cls = collections.Counter(); per_chunk = []
        attn_fl = attn_t = mm_fl = mm_t = 0.0
        pos0 = 0
        for ci, p in enumerate(pre):
            T = p[0]["t"]
            c_t = collections.Counter()
            for r in p:
                op, d = r["op"], dur(r); c = cls_of(op)
                key = (op, c)
                a = tab.setdefault(key, [0, 0.0, 0.0]); a[0] += 1; a[1] += d
                cls[c] += d; c_t[c] += d
                if c == "attention":
                    fl = 4.0 * D * H * (T * pos0 + T * (T + 1) / 2); a[2] += fl; attn_fl += fl; attn_t += d
                elif c == "matmul" and r["t"] > 1:
                    fl = 2.0 * r["t"] * r["n"] * r["k"]; a[2] += fl; mm_fl += fl; mm_t += d
            per_chunk.append((ci, pos0, T, c_t))
            pos0 += T
        tot = sum(cls.values())
        print(f"Prefill kernel time {tot*1e3:.1f} ms\n")
        print("| class | ms | share | |\n|---|---:|---:|---|")
        for c in ("matmul", "attention", "norms/rope", "other"):
            print(f"| {c} | {cls[c]*1e3:.1f} | {100*cls[c]/tot:.1f}% | |")
        print("\n| kernel | class | calls | us/call | total ms | share | TFLOPS |\n|---|---|---:|---:|---:|---:|---:|")
        for (op, c), (n, t, fl) in sorted(tab.items(), key=lambda kv: -kv[1][1]):
            print(f"| {op} | {c} | {n} | {t/n*1e6:.1f} | {t*1e3:.1f} | {100*t/tot:.1f}% | {fl/t/1e12:.1f} |" if fl else f"| {op} | {c} | {n} | {t/n*1e6:.1f} | {t*1e3:.1f} | {100*t/tot:.1f}% | |")
        print(f"\nMatmul {mm_fl/1e12:.1f} TFLOP in {mm_t*1e3:.0f} ms = **{mm_fl/mm_t/1e12:.1f} TFLOPS**; attention {attn_fl/1e12:.2f} TFLOP (causal count) in {attn_t*1e3:.0f} ms = **{attn_fl/attn_t/1e12:.1f} TFLOPS**.")
        print("\nPer chunk (ms, 36 layers): chunk index, pos0, T, matmul, attention, norms/rope, other, attention TFLOPS\n")
        print("| chunk | pos0 | T | matmul | attention | norms/rope | other | attn TFLOPS |\n|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ci, p0, T, ct in per_chunk:
            fl = 4.0 * D * H * (T * p0 + T * (T + 1) / 2)
            print(f"| {ci} | {p0} | {T} | {ct['matmul']*1e3:.0f} | {ct['attention']*1e3:.0f} | {ct['norms/rope']*1e3:.1f} | {ct['other']*1e3:.1f} | {LAYERS*fl/ct['attention']/1e12:.1f} |")
        print()
    if dec:
        n = len(dec)
        tab = collections.OrderedDict(); cls = collections.Counter()
        attn_bw = []
        for i, p in enumerate(dec):
            vis = plen + 1 + i           # keys the attention of decode step i sees (the new token included)
            for r in p:
                op, d = r["op"], dur(r); c = cls_of(op)
                key = (op, c, r["n"], r["k"], r["bytes"])
                a = tab.setdefault(key, [0, 0.0]); a[0] += 1; a[1] += d
                cls[c] += d
        tot = sum(cls.values())
        mean_vis = plen + 1 + (n - 1) / 2
        print(f"Decode: {n} steps, {sum(len(p) for p in dec)/n:.0f} launches/step, mean visible keys {mean_vis:.0f}, kernel-mode sum {tot/n*1e3:.2f} ms/token\n")
        print("| class | ms/token | share |\n|---|---:|---:|")
        for c in ("matmul", "attention", "norms/rope", "other"):
            print(f"| {c} | {cls[c]/n*1e3:.2f} | {100*cls[c]/tot:.1f}% |")
        print("\n| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |\n|---|---|---:|---:|---:|---:|---:|")
        for (op, c, nn, kk, by), (cnt, t) in sorted(tab.items(), key=lambda kv: -kv[1][1]):
            if c == "attention":
                gb = mean_vis * KV_BYTES_PER_KEY_LAYER / (t / cnt) / 1e9
                gbs = f"{gb:.0f} (KV)"
            else:
                gbs = f"{by/(t/cnt)/1e9:.0f}" if by else ""
            print(f"| {op} | {f'{nn}x{kk}' if nn else ''} | {cnt/n:.1f} | {t/cnt*1e6:.1f} | {t/n*1e3:.3f} | {100*t/tot:.1f}% | {gbs} |")
        gem = [(k, v) for k, v in tab.items() if k[1] == "matmul"]
        gbytes = sum(k[4] * v[0] for k, v in gem) / n; gt = sum(v[1] for k, v in gem) / n
        print(f"\nMatmul/GEMV weight bytes {gbytes/1e9:.3f} GB/token in {gt*1e3:.2f} ms = {gbytes/gt/1e9:.0f} GB/s.")
        att = [(k, v) for k, v in tab.items() if k[1] == "attention"]
        at = sum(v[1] for k, v in att) / n; ac = sum(v[0] for k, v in att) / n
        print(f"Decode attention {ac:.0f} calls/token, {at/ac*1e6:.1f} us/call, KV read {mean_vis*KV_BYTES_PER_KEY_LAYER*ac/1e6:.0f} MB/token = **{mean_vis*KV_BYTES_PER_KEY_LAYER*ac/at/1e9:.0f} GB/s** over the attention kernels.\n")
    if cbp:
        cb = load(cbp)
        nsteps = int(sys.argv[5]) if len(sys.argv) > 5 else 128
        print("cb mode (stream as normal; a command buffer holds at most 128 dispatches, so a step is several buffers):")
        k = len(cb)
        while k > 0 and dur(cb[k-1]) < 0.05: k -= 1
        pre, dcb = cb[:k], cb[k:]
        if pre:
            busy = sum(dur(r) for r in pre); span = pre[-1]["e"] - pre[0]["s"]
            print(f"- prefill: {len(pre)} command buffers, GPU busy {busy*1e3:.0f} ms of span {span*1e3:.0f} ms ({100*busy/span:.1f}%)")
        if dcb and nsteps:
            lps = sum(r["nd"] for r in dcb) / nsteps
            steps, cur, nd = [], [], 0
            for r in dcb:
                cur.append(r); nd += r["nd"]
                if nd >= lps - 0.5: steps.append(cur); cur, nd = [], 0
            steps = steps[1:] if len(steps) > nsteps else steps   # the first token's logits pass can be a stray buffer
            busy = [sum(dur(r) for r in st_) for st_ in steps]
            idle = [steps[i+1][0]["s"] - steps[i][-1]["e"] for i in range(len(steps)-1)]
            per = (steps[-1][0]["s"] - steps[0][0]["s"]) / (len(steps) - 1)
            span = [st_[-1]["e"] - st_[0]["s"] for st_ in steps]
            print(f"- decode: {len(steps)} steps, {lps:.0f} dispatches/step in {len(steps[0])} buffers; GPU busy/step {sum(busy)/len(busy)*1e3:.2f} ms (min {min(busy)*1e3:.2f}, max {max(busy)*1e3:.2f}); first-to-last-buffer span {sum(span)/len(span)*1e3:.2f} ms; idle between steps {sum(idle)/len(idle)*1e3:.3f} ms; step period {per*1e3:.2f} ms = {1/per:.2f} tok/s; GPU busy fraction {100*sum(busy)/(per*len(busy)):.1f}%")
main()

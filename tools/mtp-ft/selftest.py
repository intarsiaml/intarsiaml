"""CPU self-tests of head.py against literal, loop-by-loop ports of the engine's reference code (float64), on a tiny random head:
  * rope        crates/intarsia-backend-cpu/src/fast.rs:184-215 (partial rotary, NeoX pairs (i, i + rd/2), theta^(-2i/rd))
  * NVFP4       crates/intarsia-core/src/quant/nvfp4.rs:58-69 (low nibble first, e2m1 * e4m3 per 16 * weight_scale_2)
  * the whole head, row by row with its own KV cache (mtp.rs:1062-1101): gate split per head, q/k norm, rope at the row index, GQA, sigmoid gate
  * causality: a prefix's last row equals the same row of the full pass
  python selftest.py   (exits non-zero on failure)
"""
import math
import os
import sys
import tempfile

import torch

import head as H
import synth

torch.manual_seed(0)
fails = []


def check(name, err, tol):
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max err {err:.3e} (tol {tol:.0e})")
    if not ok:
        fails.append(name)


def e2m1(code):
    m = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][code & 7]
    return -m if code & 8 else m


def ref_dequant(w, sc, gs, n, k):
    out = torch.zeros(n, k, dtype=torch.float64)
    for r in range(n):
        for j in range(k):
            b = int(w[r, j // 2])
            code = (b & 0xF) if j % 2 == 0 else (b >> 4)
            out[r, j] = e2m1(code) * float(sc[r, j // 16].float()) * gs
    return out


def rms64(x, w, eps):
    return x / math.sqrt((x * x).mean().item() + eps) * (1.0 + w)


def ref_head(head, hidden, nxt):
    """Row-by-row float64 head with a KV cache. hidden [T, H] f32, nxt [T] ids."""
    c = head.cfg
    P = {n: p.detach().double() for n, p in head.named_mtp().items()}
    E = head.embed.double()
    nh, nkv, hd, rd = c.n_heads, c.n_kv_heads, c.head_dim, c.rotary_dim
    ks, vs, outs = [], [], []
    for p in range(hidden.shape[0]):
        e = rms64(E[nxt[p]], P["mtp.pre_fc_norm_embedding.weight"], c.eps)
        h = rms64(hidden[p].double(), P["mtp.pre_fc_norm_hidden.weight"], c.eps)
        fc = P["mtp.fc.weight"]
        x = fc[:, : c.hidden] @ e + fc[:, c.hidden :] @ h
        n = rms64(x, P["mtp.layers.0.input_layernorm.weight"], c.eps)
        qg = (P["mtp.layers.0.self_attn.q_proj.weight"] @ n).view(nh, 2 * hd)
        q, gate = qg[:, :hd].clone(), qg[:, hd:]
        k = (P["mtp.layers.0.self_attn.k_proj.weight"] @ n).view(nkv, hd)
        v = (P["mtp.layers.0.self_attn.v_proj.weight"] @ n).view(nkv, hd)
        q = torch.stack([rms64(q[i], P["mtp.layers.0.self_attn.q_norm.weight"], c.eps) for i in range(nh)])
        k = torch.stack([rms64(k[i], P["mtp.layers.0.self_attn.k_norm.weight"], c.eps) for i in range(nkv)])
        for t in (q, k):  # fast.rs rope, position p
            for hh in range(t.shape[0]):
                old = t[hh].clone()
                for i in range(rd // 2):
                    a = p * c.theta ** (-2.0 * i / rd)
                    t[hh, i] = old[i] * math.cos(a) - old[i + rd // 2] * math.sin(a)
                    t[hh, i + rd // 2] = old[i + rd // 2] * math.cos(a) + old[i] * math.sin(a)
        ks.append(k); vs.append(v)
        K, V = torch.stack(ks), torch.stack(vs)  # [p+1, nkv, hd]
        ao = torch.zeros(nh, hd, dtype=torch.float64)
        for i in range(nh):
            kv = i // (nh // nkv)
            s = (K[:, kv] @ q[i]) / math.sqrt(hd)
            ao[i] = torch.softmax(s, 0) @ V[:, kv]
        ao = ao * torch.sigmoid(gate)
        r = x + P["mtp.layers.0.self_attn.o_proj.weight"] @ ao.reshape(-1)
        n = rms64(r, P["mtp.layers.0.post_attention_layernorm.weight"], c.eps)
        m = torch.nn.functional.silu(P["mtp.layers.0.mlp.gate_proj.weight"] @ n) * (P["mtp.layers.0.mlp.up_proj.weight"] @ n)
        r = r + P["mtp.layers.0.mlp.down_proj.weight"] @ m
        outs.append(rms64(r, P["mtp.norm.weight"], c.eps))
    return torch.stack(outs)


with tempfile.TemporaryDirectory() as td:
    synth.tiny_ckpt(td)
    ck = H.Checkpoint(td)
    # NVFP4
    w, sc, gs = ck.get("lm_head.weight"), ck.get("lm_head.weight_scale"), float(ck.get("lm_head.weight_scale_2"))
    got = H.dequant_nvfp4(w[:40], sc[:40], gs).double() * gs
    check("nvfp4 dequant", float((got - ref_dequant(w[:40], sc[:40], gs, 40, 64)).abs().max()), 1e-7)
    # head, in f32 (so the f64 reference is comparable)
    head = H.load_head(td, "cpu", param_dtype=torch.float32)
    head.embed = head.embed.float()
    head.lm_w = head.lm_w.float()
    T = 20
    hid = torch.randn(T, head.cfg.hidden) * 2
    nxt = torch.randint(0, head.cfg.vocab, (T,))
    with torch.no_grad():
        full = head(hid[None], nxt[None])[0]
        ref = ref_head(head, hid, nxt)
        check("head forward vs f64 row-by-row reference (rope, gate, qk-norm, GQA, KV)", float((full.double() - ref).abs().max()), 2e-4)
        pre = head(hid[None, :9], nxt[None, :9])[0]
        check("causality (prefix == full rows)", float((pre - full[:9]).abs().max()), 1e-5)
    # bf16 path runs and agrees roughly with f32
    head16 = H.load_head(td, "cpu")
    with torch.no_grad():
        g16 = head16(hid[None].to(torch.bfloat16), nxt[None])[0].float()
    check("bf16 vs f32 head (rel)", float((g16 - full).abs().max() / full.abs().max()), 6e-2)
    # save / reload round trip with header assertions
    out = os.path.join(td, "t.safetensors")
    H.save_mtp(head16, out, td)
    h2 = H.load_head(td, "cpu", mtp_file=out)
    check("save_mtp roundtrip", max(float((a - b).abs().max()) for a, b in zip(head16.named_mtp().values(), h2.named_mtp().values())), 0.0)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("all selftests passed")

"""Synthetic inputs for testing the pipeline without real dumps.

  python synth.py tiny-ckpt <dir>                       a tiny random checkpoint (same tensor names / dtypes / NVFP4 lm_head layout as the 27B)
  python synth.py dumps --ckpt <dir> --out <dir> [-n 6] [--len 48] [--draft-vocab N] [--device cpu|cuda]
        dump files in `intarsia mtp-dump`'s format; tokens are filled left to right so that about half of the targets
        tokens[p+2] equal the stock head's argmax (so acceptance is not trivially 0); head_top1 / head_top1_dv / head_out come from
        this PyTorch head itself (a self-consistency fixture: real parity needs real dumps).
  also writes <out>/meta.json (prompt_len = len // 4, category from the file id).
"""
import argparse
import json
import os
import random

import torch
from safetensors.torch import save_file

import head as H


def tiny_ckpt(d: str, hidden=64, vocab=512, nh=4, nkv=2, hd=32, inter=96, rot=8, seed=0):
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.05: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16)
    t = {
        "mtp.fc.weight": r(hidden, 2 * hidden),
        "mtp.pre_fc_norm_embedding.weight": r(hidden, sc=0.1),
        "mtp.pre_fc_norm_hidden.weight": r(hidden, sc=0.1),
        "mtp.layers.0.input_layernorm.weight": r(hidden, sc=0.1),
        "mtp.layers.0.post_attention_layernorm.weight": r(hidden, sc=0.1),
        "mtp.layers.0.self_attn.q_proj.weight": r(nh * hd * 2, hidden),
        "mtp.layers.0.self_attn.k_proj.weight": r(nkv * hd, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": r(nkv * hd, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": r(hidden, nh * hd),
        "mtp.layers.0.self_attn.q_norm.weight": r(hd, sc=0.1),
        "mtp.layers.0.self_attn.k_norm.weight": r(hd, sc=0.1),
        "mtp.layers.0.mlp.gate_proj.weight": r(inter, hidden),
        "mtp.layers.0.mlp.up_proj.weight": r(inter, hidden),
        "mtp.layers.0.mlp.down_proj.weight": r(hidden, inter),
        "mtp.norm.weight": r(hidden, sc=0.1),
        "model.language_model.embed_tokens.weight": r(vocab, hidden, sc=0.5),
        # NVFP4 lm_head: any nibble pattern is valid; e4m3 scales kept small and positive
        "lm_head.weight": torch.randint(0, 256, (vocab, hidden // 2), generator=g, dtype=torch.uint8),
        "lm_head.weight_scale": (torch.rand(vocab, hidden // 16, generator=g) * 0.5 + 0.1).to(torch.float8_e4m3fn),
        "lm_head.weight_scale_2": torch.tensor(0.02, dtype=torch.float32),
        "lm_head.input_scale": torch.tensor(1.0, dtype=torch.float32),
    }
    save_file(t, os.path.join(d, "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    cfg = {"text_config": {"hidden_size": hidden, "vocab_size": vocab, "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
                           "intermediate_size": inter, "rms_norm_eps": 1e-6,
                           "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": rot / hd}}}
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))


@torch.no_grad()
def make_dumps(ckpt, out, n, length, draft_vocab, device, seed=0, fast=False):
    os.makedirs(out, exist_ok=True)
    head = H.load_head(ckpt, device)
    cfg = head.cfg
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    meta = {}
    cats = ["code", "prose", "chat"]
    for i in range(n):
        fid = f"{cats[i % 3]}__{i:04d}"
        L = length + rng.randint(-4, 4)
        tokens = torch.randint(0, cfg.vocab, (L,), generator=g)
        hidden = (torch.randn(L - 1, cfg.hidden, generator=g) * 2.0).to(torch.bfloat16)
        hd = hidden.to(device)[None]
        for p in range(0 if fast else L - 2):  # row p's stock argmax is a function of tokens[1..p+1], so fill tokens[p+2] left to right
            gg = head(hd[:, : p + 1], tokens[1 : p + 2].to(device)[None])[0, -1:]
            top = int(head.logits(gg).argmax())
            if rng.random() < 0.5:
                tokens[p + 2] = top
        d = dict(tokens=tokens, hidden=hidden)
        gall = head(hd, tokens[1:].to(device)[None])[0]
        t1, t1dv = head.argmax(gall, draft_vocab)
        d["head_top1"] = t1.cpu().long()
        if draft_vocab:
            d["head_top1_dv"] = t1dv.cpu().long()
        d["head_out"] = gall.cpu().to(torch.bfloat16)
        save_file({k: v.contiguous() for k, v in d.items()}, os.path.join(out, fid + ".safetensors"),
                  metadata={"hidden_size": str(cfg.hidden), "vocab_size": str(cfg.vocab), "draft_vocab": str(draft_vocab), "lm_head": "nvfp4"})
        meta[fid] = dict(prompt_len=L // 4, category=cats[i % 3])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    print(f"wrote {n} dumps to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiny-ckpt"); t.add_argument("dir")
    d = sub.add_parser("dumps")
    d.add_argument("--ckpt", required=True); d.add_argument("--out", required=True)
    d.add_argument("-n", type=int, default=6); d.add_argument("--len", type=int, default=48)
    d.add_argument("--draft-vocab", type=int, default=0); d.add_argument("--device", default="cpu")
    d.add_argument("--fast", action="store_true", help="random tokens, no left-to-right fill (long sequences: the fill is O(n^2))")
    a = ap.parse_args()
    if a.cmd == "tiny-ckpt":
        tiny_ckpt(a.dir)
    else:
        make_dumps(a.ckpt, a.out, a.n, a.len, a.draft_vocab, a.device, fast=a.fast)

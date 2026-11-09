#!/usr/bin/env python3
"""A synthetic engine dump (the format of crates/intarsia-core/src/model/dspine/dump.rs) from the tiny DSpine fixture's golden, to test
`replay_taps.py` end to end without a real model: case `c` (context chunks 9 + 4, block 6) of
crates/intarsia-models/tests/fixtures/tiny_dspine/golden.safetensors, whose drafts/candidates/layer rows were produced by the same
reference, so the replay must agree with them (up to MLX CPU vs GPU noise). The ingest log also holds a rolled-back chunk (garbage rows
at positions 9..13, then the real ones), which the replay's cut semantics must ignore, and the target's ids follow the golden drafts for 3 rows.

    python -I tools/dspine/tiny_dump_from_golden.py <fixtures/tiny_dspine> <out dump dir>
"""
import json, sys
from pathlib import Path
import numpy as np
import mlx.core as mx

fx, out = Path(sys.argv[1]), Path(sys.argv[2])
g = mx.load(str(fx / "golden.safetensors"))
cfg = json.loads((fx / "config.json").read_text())
H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
R = cfg["dflash_config"]["selected_prefix_code_bridge"]["rank"]
C = cfg["dflash_config"]["selected_prefix_code_bridge"]["code_size"]
K = cfg["code_bridge_second_pass"]["head_topk"]
taps = cfg["dflash_config"]["target_layer_ids"]
T = cfg["block_size"]
chunks = [int(x) for x in np.array(g["c.chunks"])]
feat = np.array(g["c.feat"])  # [P, NT*H] f32 of bf16 values


def bf16_bits(a):
    return np.array(mx.array(np.asarray(a, dtype=np.float32)).astype(mx.bfloat16).view(mx.uint16)).astype(np.uint16)


rq = out / "req-0000"
rq.mkdir(parents=True, exist_ok=True)
(rq / "meta.json").write_text(json.dumps(dict(hidden=H, taps=taps, dtype="bf16", block=T, topk=K, layers=L, vocab=cfg["vocab_size"], code=C, rank=R, dump_layers=True, head="bf16")))
rng = np.random.default_rng(0)
tb = bytearray()
idx = []


def ingest(pos0, rows):
    idx.append(dict(ev="ingest", pos0=pos0, t=len(rows), off=len(tb)))
    tb.extend(bf16_bits(rows).tobytes())


ingest(0, feat[:chunks[0]])
ingest(chunks[0], rng.standard_normal((chunks[1], feat.shape[1])).astype(np.float32))  # a verify pass whose rows the rollback discards
ingest(chunks[0], feat[chunks[0]:])
P = sum(chunks)
ids = np.array(g["c.ids"])
drafts = np.array(g["c.drafts"])
cand = np.array(g["c.cand"]).astype(np.int32)
logits = np.array(g["c.logits"]).astype(np.float32)
prompt = rng.integers(0, cfg["vocab_size"] - 1, size=P).tolist()
anchor = int(ids[0])
gen = [anchor] + drafts[:3].tolist() + [int((drafts[3] + 1) % cfg["vocab_size"])] + [1, 2, 3]  # the target agrees with the first 3 drafts
(out / "ids-0.json").write_text(json.dumps(dict(prompt=prompt, ids=gen)))
lg = np.take_along_axis(logits, cand.astype(np.int64), axis=-1).astype(np.float32)
(rq / "passes.bin").write_bytes(cand.tobytes() + lg.tobytes())
lb = bytearray()
for l in range(L):
    for key in ("out", "normed", "raw"):
        lb.extend(bf16_bits(np.array(g[f"c.layer{l}.{key}"])).tobytes())
lb.extend(np.array(g["c.rms"]).astype(np.float32).tobytes())
lb.extend(bf16_bits(np.array(g["c.packed"])).tobytes())
(rq / "layers.bin").write_bytes(bytes(lb))
(rq / "taps.bin").write_bytes(bytes(tb))
idx.append(dict(ev="pass", p0=P, anchor=anchor, t=T, drafts=drafts.tolist(), tok_from=0, new_tokens=prompt, cand_off=0, layers_off=0))
(rq / "index.jsonl").write_text("\n".join(json.dumps(e) for e in idx) + "\n")
print("wrote", out, "golden drafts", drafts.tolist(), "target", gen[1:])

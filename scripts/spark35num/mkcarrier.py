#!/usr/bin/env python3
"""Lane spark35 (docs/results/2026-11-05-spark35-numerics.md): carrier goldens for `intarsia accuracy --write-self-golden` /
`--self-golden` at 4k .. 64k tokens. A carrier is an intarsia-golden-topk-v1 file that holds only prompt ids (the reference rows are
dummies, k = 1): the real reference is the baseline arm's own self-golden dump, so no vLLM golden (made with an fp8 KV cache)
confounds the fp8-KV comparison. Deterministic. Run from the repo root:
  uv run --with numpy --with safetensors --with tokenizers scripts/spark35num/mkcarrier.py <tokenizer.json> <out_dir>
Writes <out_dir>/std.safetensors (5 prompts, prefill + decode window) and dec8k.safetensors (2 prompts, long decode window), and
prompts.json (name, tokens, source)."""
import glob, json, pathlib, sys
import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer

root = pathlib.Path(__file__).resolve().parents[2]
tk = Tokenizer.from_file(sys.argv[1])
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)

def ids_of(text): return tk.encode(text, add_special_tokens=False).ids

def read_all(paths):  # one text, files in order, the env var prefix hidden from the docs_env_vars scan like scripts/cmtp/mkprompts.py
    return "\n\n".join(f"### {p.relative_to(root)}\n" + p.read_text(errors="replace") for p in paths).replace("INTARSIA_", "IA_")

docs = sorted(p for p in (root / "docs/results").glob("*.md") if "spark35" not in p.name)
code = sorted(p for p in (root / "crates").rglob("*.rs") if "target" not in p.parts)
prompts = []  # (name, ids, source)
for name, f in (("cmtp-4k-0", "4k-0.txt"), ("cmtp-16k-0", "16k-0.txt")):  # the existing ids-prompts (text prompts of the cmtp set)
    prompts.append((name, ids_of((root / "scripts/cmtp/ids-prompts" / f).read_text()), f"scripts/cmtp/ids-prompts/{f}"))
docs_ids = ids_of(read_all(docs)); code_ids = ids_of(read_all(code))
print(f"docs {len(docs_ids)} tokens, code {len(code_ids)} tokens")
def cut(src, off, n, name, what):
    assert off + n <= len(src), (name, off, n, len(src))
    prompts.append((name, src[off:off + n], what))
cut(code_ids, 200_000, 16384, "code-16k", "crates/**/*.rs concatenated, tokens 200000..216384")
cut(docs_ids, 300_000, 32768, "docs-32k", "docs/results/*.md concatenated, tokens 300000..332768")
cut(docs_ids, 700_000, 65536, "docs-64k", "docs/results/*.md concatenated, tokens 700000..765536")

def write(path, sel):
    t, meta = {}, {"format": "intarsia-golden-topk-v1", "topk": "1", "source": "spark35 carrier (dummy reference rows)"}
    plist = []
    for i, (name, ids, what) in enumerate(sel):
        n = len(ids)
        t[f"ids_{i}"] = np.asarray(ids, dtype=np.int32)
        t[f"topk_ids_{i}"] = np.zeros((n, 1), dtype=np.int32)
        t[f"topk_logprobs_{i}"] = np.full((n, 1), -1.0, dtype=np.float32)
        plist.append({"name": name})
    meta["prompts"] = json.dumps(plist)
    save_file(t, str(path), metadata=meta)
    print(path, [(n, len(i)) for n, i, _ in sel])

write(out / "std.safetensors", prompts)
write(out / "dec8k.safetensors", [p for p in prompts if p[0] in ("code-16k", "docs-32k")])
json.dump([{"name": n, "tokens": len(i), "source": w} for n, i, w in prompts], open(out / "prompts.json", "w"), indent=1)

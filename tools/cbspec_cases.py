#!/usr/bin/env python3
"""Cases for the speculation x batching identity gate (docs/results/2026-10-16-metal-cb-gates-reuse-spec.md, gate B).

Writes a JSONL in the tools/cbidentity.py case format. Prompts are input-grounded (the model is asked to copy a passage
that is in the prompt) so `--spec ngram` finds drafts, and arrive in bursts of 2..8 so that the number of decoding
sequences rises above 4 (drafting off) and falls back below it (drafting on) many times per run. Mostly greedy (the only
requests that speculate); a few sampled / stop-string / min_tokens requests share the batch.

  python3 tools/cbspec_cases.py --out tools/prompts/cb_spec_cases.jsonl
"""
import argparse
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
BURSTS = [6, 3, 8, 5, 2, 7, 4, 6, 7]  # 48 cases
BURST_GAP_MS = 14000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "prompts" / "cb_spec_cases.jsonl"))
    ap.add_argument("--seed", type=int, default=20261016)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pdir = HERE / "prompts"
    sizes = {f: len((pdir / f).read_text()) for f in ("corpus_docs.txt", "corpus_code.txt")}
    cases = []
    n = 0
    for bi, size in enumerate(BURSTS):
        for j in range(size):
            kind = rng.choice(["copy", "copy", "chatcopy", "code"])
            f = "corpus_code.txt" if kind == "code" else "corpus_docs.txt"
            clen = rng.randint(500, 1000)
            start = rng.randrange(0, sizes[f] - clen - 1)
            text = (pdir / f).read_text()
            nl = text.find("\n", start)
            start = nl + 1 if 0 <= nl < start + 200 else start
            sl = text[start:start + clen]
            c = {"id": f"s{n:03d}", "tags": [kind], "arrival_ms": bi * BURST_GAP_MS + rng.randint(0, 600)}
            # lengths spread so the burst's sequences finish one after another (decoding count falls through 4)
            c["max_tokens"] = rng.choice([48, 64, 96]) if j % 2 == 0 else rng.choice([160, 224, 288])
            c["temperature"] = 0
            if kind == "copy":
                c["prompt"] = f"Repeat the text below exactly, word for word.\n\nTEXT:\n{sl}\n\nREPEATED TEXT:\n"
            elif kind == "code":
                c["prompt"] = f"Here is a source file excerpt.\n```\n{sl}\n```\nThe same excerpt again, copied exactly:\n```\n"
            else:
                c["messages"] = [{"role": "user", "content": f"Reproduce this passage verbatim and add nothing else:\n\n{sl}"}]
            r = rng.random()
            if r < 0.12:  # sampled: never speculates, shares the batch
                c.update(temperature=rng.choice([0.6, 0.8]), top_p=0.95, seed=rng.randrange(1, 2**31))
                c["tags"].append("sampled")
            elif r < 0.22:
                c["stop"] = [rng.choice(["\n\n", ".\n", "the "])]
                c["tags"].append("stop")
            elif r < 0.32:
                c["min_tokens"] = rng.randint(8, 40)
                c["tags"].append("min_tokens")
            else:
                c["tags"].append("greedy")
            cases.append(c)
            n += 1
    pathlib.Path(a.out).write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    print(f"wrote {a.out}: {len(cases)} cases")


if __name__ == "__main__":
    main()

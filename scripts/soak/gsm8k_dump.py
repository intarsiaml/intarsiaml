#!/usr/bin/env python3
"""restore-acc: freeze the pinned GSM8K inputs. Writes OUT.json = {questions: [{id, question, gold}] (the same 200 test questions
and ids as `localeval quality --bench gsm8k --limit 200 --pin-gsm8k`), shots: [{question, answer}] (first N train rows, raw
GSM8K worked solutions ending '#### n')}. Run: cd /opt/code/deepeval && uv run python <this> OUT.json [N_QUESTIONS=200] [N_SHOTS=32]
Checks the golds against a localeval result.json when given as argv[4]."""
import json
import sys

from localeval import gsm8k
from localeval.quality import _patch_hf_ids

out = sys.argv[1]
nq = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 32
_patch_hf_ids()
b = gsm8k.make_benchmark(3, nq, True)
goldens = b.load_benchmark_dataset()[:nq]
qs = [{"id": f"gsm8k-{i:05d}", "question": g.input, "gold": g.expected_output} for i, g in enumerate(goldens)]
import datasets  # noqa: E402

tr = datasets.load_dataset("gsm8k", "main", split="train")
shots = [{"question": tr[i]["question"], "answer": tr[i]["answer"]} for i in range(ns)]
json.dump({"questions": qs, "shots": shots}, open(out, "w"), indent=1)
if len(sys.argv) > 4:
    ref = {r["id"]: r["gold"] for r in json.load(open(sys.argv[4]))["benchmarks"][0]["replies"]}
    bad = [q["id"] for q in qs if ref.get(q["id"]) != q["gold"]]
    print(f"gold check against {sys.argv[4]}: {len(qs) - len(bad)} of {len(qs)} equal; mismatches {bad[:5]}")
print(f"wrote {out}: {len(qs)} questions, {len(shots)} shots")

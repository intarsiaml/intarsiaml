"""Paired GSM8K replay: the exact localeval request (Client.chat) per question, full record per call."""
import argparse, dataclasses, json, sys, pathlib
from localeval import client as C
from localeval.config import Target
from localeval.quality import make_benchmark
from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
from deepeval.scorer import Scorer

ap = argparse.ArgumentParser()
ap.add_argument("--url"); ap.add_argument("--model"); ap.add_argument("--out")
ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--extra", default="{}")
a = ap.parse_args()
b = make_benchmark("gsm8k", None, None, None)
goldens = b.load_benchmark_dataset()
t = Target(name="adhoc", url=a.url, model=a.model, thinking="off", thinking_style="chat_template_kwargs",
           structured="none", extra_body=json.loads(a.extra))
cl = C.Client(t)
out = pathlib.Path(a.out)
done = set()
if out.exists():
    done = {json.loads(l)["i"] for l in out.open()}
with out.open("a") as f:
    for i in range(a.start, min(a.start + a.n, len(goldens))):
        if i in done: continue
        g = goldens[i]
        p = GSM8KTemplate.generate_output(train_set=b.shots_dataset, input=g.input, n_shots=b.n_shots,
                                          enable_cot=b.enable_cot) + f"\n\n{b.confinement_instructions}"
        r = cl.chat([{"role": "user", "content": p}], max_tokens=t.default_max_tokens, temperature=0.0)
        score = Scorer.exact_match_score(g.expected_output, r.text) if r.ok else 0
        rec = {"i": i, "expected": g.expected_output, "score": score, "text": r.text, "reasoning": r.reasoning,
               "finish": r.finish_reason, "pt": r.prompt_tokens, "ct": r.completion_tokens, "err": r.error}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(i, score, r.finish_reason, r.completion_tokens, repr(r.text[:60]), file=sys.stderr)

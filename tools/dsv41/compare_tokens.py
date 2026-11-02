"""Per-prompt first divergence TensorFold vs intarsia, with decoded context. usage: analyze.py TF.st INTARSIA.st tokenizer.json"""
import sys, json
import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
def load(p):
    f = safe_open(p, "np"); m = f.metadata()
    names = [x["name"] for x in json.loads(m["prompts"])]
    return {n: (f.get_tensor(f"ids_{i}"), f.get_tensor(f"cont_{i}")) for i, n in enumerate(names)}
A, B, tk = load(sys.argv[1]), load(sys.argv[2]), Tokenizer.from_file(sys.argv[3])
dec = lambda ids: tk.decode([int(x) for x in ids], skip_special_tokens=False)
tot = lead = pos = 0
print(f"{'prompt':<14}{'n_ids tf/ia':>12}{'ids same':>9}{'first_div':>10}{'lead':>6}{'posmatch':>9}")
div = []
for n, (ia, ca) in A.items():
    ib, cb = B[n]; m = min(len(ca), len(cb))
    neq = np.nonzero(ca[:m] != cb[:m])[0]; d = int(neq[0]) if len(neq) else m
    p = int((ca[:m] == cb[:m]).sum()); tot += m; lead += d; pos += p
    print(f"{n:<14}{f'{len(ia)}/{len(ib)}':>12}{str(np.array_equal(ia, ib)):>9}{(d if len(neq) else 'none'):>10}{d:>6}{p:>6}/{m}")
    if len(neq): div.append((n, d, ca, cb))
print(f"OVERALL leading-match {lead}/{tot} ({100*lead/tot:.2f}%)  positionwise-equal {pos}/{tot} ({100*pos/tot:.2f}%)")
for n, d, ca, cb in div:
    lo = max(0, d - 8)
    print(f"\n== {n} diverges at {d}\n shared: {dec(ca[lo:d])!r}\n TF    : {dec(ca[d:d+8])!r} ids {ca[d:d+8].tolist()}\n INTAR : {dec(cb[d:d+8])!r} ids {cb[d:d+8].tolist()}")

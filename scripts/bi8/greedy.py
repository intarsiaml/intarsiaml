#!/usr/bin/env python3
"""Lane bi8: greedy served-identity runs against one running server, without restarting it between reps (reps 2+ are what exercise
slot prefix reuse). Saves {"n<N>/r<R>": [token ids per stream]} to OUT; `compare REF OUT` prints, per run, the streams that differ
from REF (a {"<i>": ids} map, e.g. built from the probe's COLD lines) and their first differing position.
  greedy.py run URL NS REPS OUT        e.g. greedy.py run http://spark-fa00:8193/v1 1,4,8,16 3 out.json
  greedy.py compare REF OUT
"""
import json, sys, threading, urllib.request, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'htk'))
import equal


def run(url, ns, reps, out):
    res = {}
    for n in ns:
        for r in range(reps):
            got = [None] * n
            def one(i):
                body = {"model": "intarsia", "prompt": equal.PROMPTS[i % 16] + ("" if i < 16 else f" ({i})"), "max_tokens": 64, "ignore_eos": True,
                        "temperature": 0.0, "top_p": 1.0, "seed": 100 + i, "return_token_ids": True}
                got[i] = equal.post(url, body)["choices"][0]["token_ids"]
            ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            [t.start() for t in ths]; [t.join() for t in ths]
            res[f"n{n}/r{r}"] = got
            print(f"n={n} rep={r}: {n * 64} ids", flush=True)
    json.dump(res, open(out, "w"))


def compare(ref, out):
    R, O = json.load(open(ref)), json.load(open(out))
    bad_runs = 0
    for key, got in O.items():
        bad = []
        for i, ids in enumerate(got):
            want = R[str(i)]
            if ids != want:
                bad.append((i, next(j for j in range(len(want)) if ids[j] != want[j])))
        bad_runs += bool(bad)
        print(f"{key}: {'IDENTICAL' if not bad else 'DIFFER (stream, first diff pos) ' + str(bad)}")
    print(f"{len(O) - bad_runs}/{len(O)} runs identical to {os.path.basename(ref)}")
    return bad_runs


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5])
    else:
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)

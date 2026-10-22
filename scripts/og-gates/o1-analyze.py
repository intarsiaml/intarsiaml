#!/usr/bin/env python3
"""O1 analyzer: the decode gate ">= vLLM single stream" per context length, from the localeval runs of o1-driver.sh.

    o1-analyze.py RAWDIR [--results /opt/code/deepeval/results]

RULE (fixed before the run, 2026-10-22):
  per prompt size and round r: delta_r = ours decode tok/s (localeval median of 3 reps) - vLLM decode tok/s of the same round.
  A round is DROPPED when vLLM had running requests during our arm's window (vllm-poll.log), because that load slows ours.
  PASS  = median(delta) >= 0 and at least 2/3 of the kept rounds have delta >= 0
  FAIL  = median(delta) <  0 and at least 2/3 of the kept rounds have delta <  0
  MIXED = anything else.
  A median within +-1.0 tok/s is flagged "within noise" (the 2026-10-13 gate run quoted about 1 tok/s noise); the flag does
  not change PASS/FAIL, which is the literal gate. Prefill is reported for information only (it is not part of the gate).
"""
import argparse, glob, json, pathlib, shutil, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("raw")
ap.add_argument("--results", default="/opt/code/deepeval/results")
a = ap.parse_args()
raw = pathlib.Path(a.raw)
win = [l.split("\t") for l in (raw / "windows.tsv").read_text().splitlines() if l.strip()]
poll = []
try:
    for l in (raw / "vllm-poll.log").read_text().splitlines():
        p = l.split()
        if len(p) == 3 and p[1].isdigit():
            poll.append((int(p[0]), int(p[1]), int(p[2])))
except FileNotFoundError:
    pass

(raw / "results").mkdir(exist_ok=True)
data = {}  # (arm, round) -> {target: row}
runids = {}
for arm, r, t0, t1, rc in win:
    r = int(r); t0 = int(t0); t1 = int(t1)
    hits = sorted(glob.glob(f"{a.results}/*og-o1-{arm}-r{r}"))
    if not hits:
        print(f"MISSING result dir for {arm} round {r}", file=sys.stderr); continue
    d = pathlib.Path(hits[-1])
    runids[(arm, r)] = d.name
    shutil.copy(d / "result.json", raw / "results" / f"{d.name}.result.json")
    shutil.copy(d / "summary.md", raw / "results" / f"{d.name}.summary.md")
    j = json.loads((d / "result.json").read_text())
    data[(arm, r)] = {row["target_prompt_tokens"]: row for row in j["speed"]["rows"]}
    busy = [x for x in poll if t0 <= x[0] <= t1 and (x[1] > 0 or x[2] > 0)]
    data[(arm, r)]["_busy"] = len(busy) if arm == "ours" else 0
    data[(arm, r)]["_rc"] = rc

rounds = sorted({r for (_, r) in data})
sizes = sorted({s for v in data.values() for s in v if isinstance(s, int)})
print("runs:", ", ".join(f"{k[0]}-r{k[1]}={v}" for k, v in sorted(runids.items())))
out = []
for s in sizes:
    deltas, rows = [], []
    for r in rounds:
        v, o = data.get(("vllm", r), {}).get(s), data.get(("ours", r), {}).get(s)
        if not v or not o:
            continue
        dropped = data[("ours", r)]["_busy"] > 0
        dv, do = v["decode_tok_s"]["median"], o["decode_tok_s"]["median"]
        pv, po = v["prefill_tok_s"]["median"], o["prefill_tok_s"]["median"]
        rows.append((r, v["prompt_tokens"], o["prompt_tokens"], dv, do, do - dv, pv, po, dropped))
        if not dropped:
            deltas.append(do - dv)
    if not deltas:
        verdict, med = "NO DATA", float("nan")
    else:
        med = statistics.median(deltas)
        pos = sum(1 for d in deltas if d >= 0); neg = sum(1 for d in deltas if d < 0)
        need = -(-2 * len(deltas) // 3)
        verdict = "PASS" if med >= 0 and pos >= need else "FAIL" if med < 0 and neg >= need else "MIXED"
        if abs(med) < 1.0:
            verdict += " (within ~1 tok/s noise)"
    out.append((s, rows, med, verdict))

print()
print("| prompt (target) | prompt tokens vLLM / ours | round | vLLM decode | ours decode | delta | vLLM prefill | ours prefill | kept |")
print("|---|---|---|---|---|---|---|---|---|")
for s, rows, med, verdict in out:
    for r, pv_t, po_t, dv, do, d, pv, po, dropped in rows:
        print(f"| {s} | {pv_t} / {po_t} | {r} | {dv:.1f} | {do:.1f} | {d:+.1f} | {pv:.0f} | {po:.0f} | {'no (vLLM busy)' if dropped else 'yes'} |")
print()
print("| prompt (target) | median vLLM | median ours | median delta | delta % | verdict (decode gate) |")
print("|---|---|---|---|---|---|")
for s, rows, med, verdict in out:
    kept = [x for x in rows if not x[8]]
    if not kept:
        print(f"| {s} | - | - | - | - | {verdict} |"); continue
    mv = statistics.median(x[3] for x in kept); mo = statistics.median(x[4] for x in kept)
    print(f"| {s} | {mv:.1f} | {mo:.1f} | {med:+.1f} | {100 * med / mv:+.1f} % | **{verdict}** |")

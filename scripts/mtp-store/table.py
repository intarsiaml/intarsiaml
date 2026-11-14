#!/usr/bin/env python3
"""lane mtp-store: the A/B table from docs/results/2026-11-13-mtp-store/raw (scripts/mtp-store/window.sh).

  python3 scripts/mtp-store/table.py [RAW_DIR]

Per run: decode tok/s per stream (median over requests), TTFT sum, ITL p99 (median and max over requests), aggregate output
tok/s, wall time, the server's speculation counters (verify passes, drafted, accepted) and whether every reply equals the same
(conversation, turn) reply of the first run listed (greedy replies must not depend on MTP)."""
import json
import pathlib
import re
import statistics
import sys

raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "docs/results/2026-11-13-mtp-store/raw")


def metric(text, name):
    m = re.search(rf"^{name} (\S+)$", text, re.M)
    return float(m.group(1)) if m else 0.0


runs = []
for s in sorted(raw.glob("*.summary.json"), key=lambda p: p.stat().st_mtime):
    d = json.loads(s.read_text())
    rows = [json.loads(l) for l in (raw / s.name.replace(".summary.json", ".jsonl")).read_text().splitlines()]
    mp = raw / f"serve-{d['arm']}.metrics"
    mt = mp.read_text() if mp.exists() else ""
    runs.append((d, rows, mt))
if not runs:
    sys.exit(f"no runs in {raw}")
ref = {(r["conv"], r["turn"]): r["reply"] for r in runs[0][1]}
print("| run | K | decode tok/s per stream, median (min-max) | TTFT sum s | TTFT median s | ITL p99 ms median / max | output tok/s | wall s | cached frac | passes | drafted | accepted | acceptance | tokens per pass | replies = first run |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d, rows, mt in runs:
    ok = [r for r in rows if r["ok"]]
    dec = [r["decode_tok_s"] for r in ok if r["decode_tok_s"]]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    p99 = [r["itl_ms_p99"] for r in ok if r["itl_ms_p99"] is not None]
    t = d["totals"]
    steps, prop, acc = (metric(mt, f"intarsia_spec_{n}_total") for n in ("steps", "proposed_tokens", "accepted_tokens"))
    same = sum(1 for r in rows if ref.get((r["conv"], r["turn"])) == r["reply"])
    print(
        f"| {d['arm']} | {d['streams']} | {statistics.median(dec):.1f} ({min(dec):.1f}-{max(dec):.1f}) | {sum(ttft):.2f} | {statistics.median(ttft):.2f} | "
        f"{statistics.median(p99):.1f} / {max(p99):.1f} | {t['agg_output_tok_s']} | {d['wall_s']} | {t['cached_tokens'] / max(t['prompt_tokens'], 1):.3f} | "
        f"{steps:.0f} | {prop:.0f} | {acc:.0f} | {(100 * acc / prop if prop else 0):.1f} % | {((acc + steps) / steps if steps else 0):.2f} | {same}/{len(rows)}{' (errors: %d)' % t['errors'] if t['errors'] else ''} |"
    )

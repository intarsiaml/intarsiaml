#!/usr/bin/env python3
"""Metal speed gate (restated by the user 2026-10-21 12:25 EDT; docs/benchmarking.md "Metal speed gate").

usage: gate.py <dir>
  <dir> holds, per arm, `<label>-localeval.json` (a localeval `speed` result.json) and `<label>-macmon.jsonl[.gz]`
  (`macmon pipe` samples taken while that arm ran). Arms are recognised by label prefix: `intarsia*`, `mlx-lm*`,
  `llamacpp*` (`sameslot.sh` writes exactly this layout).

Gates, per prompt size:
  1. decode >= 85 % of the context-aware byte floor: (weights + KV bytes at the mean context) / bandwidth, where the
     mean context is prompt + gen / 2. The decode figure is clock-normalised: `sameslot.sh` runs the fixed reference
     workload `refbw.py` (a bf16 matrix-vector stream, independent of intarsia) around every arm and writes
     `<label>-ref.jsonl`; the arm's reference GB/s divided by the stored baseline (`ref-baseline.json`) is the machine-state
     factor, and normalised decode = raw decode / factor. Raw, reference, factor and normalised values are all printed. An
     arm whose factor is outside [0.70, 1.30] is STATE-INVALID (re-run it), not a failure. The gate passes when at least one
     valid intarsia arm passes at every size; with no valid arm the verdict is STATE-INVALID.
     An arm without a reference file (older result dirs) is not normalised and falls back to the previous admission rule:
     median GPU clock of busy samples (GPU utilisation >= 50 %) >= 1300 MHz and median macmon CPU load <= 0.30.
  2. same-slot ratios (clock-invariant, every intarsia arm against the same slot's other engines): decode >= 0.90x
     mlx-lm and >= 1.00x llama.cpp.
Exit 0 = PASS, 1 = FAIL, 3 = STATE-INVALID (ratios passed, no admitted arm), 2 = usage / missing data.

Gate-definition history (docs/decisions/LOG.md, review finding 2 of 2026-10-22): the floor was a flat 85 % of bandwidth /
weight bytes on the raw decode figure (40.56 tok/s for Qwen3-8B; 40.6 in the docs) until 2026-10-21 12:25 EDT, with no
clock admission. For every size the gate prints a `gate-definition:` line with the verdict and figure under that old
definition next to the current one, and writes both, with the date and the LOG rows, to `<dir>/gate-verdict.json`
(`--verdict-json` to move it). Neither the exit code nor the VERDICT line depends on them.

Model constants default to Qwen3-8B BF16 (15.1368 GB of weights streamed per token, 147,456 KV bytes per position,
614 GB/s on endor's M5 Max); override with --weights-gb, --kv-bytes, --bw-gbs.
"""
import argparse, glob, gzip, json, os, statistics as st, sys

MIN_MHZ, MAX_CPU, FLOOR_FRAC, MLX_FRAC, LLAMA_FRAC = 1300.0, 0.30, 0.85, 0.90, 1.00
FACTOR_MIN, FACTOR_MAX = 0.70, 1.30  # machine-state factors outside this window are not normalised, the run is repeated
BASELINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref-baseline.json")
CHANGED_ON = "2026-10-21"
LOG_REFS = [
    'docs/decisions/LOG.md row 2026-10-21 12:25 "Metal speed gate restated"',
    'docs/decisions/LOG.md row 2026-10-21 "The Metal 4k decode gate is clock-normalised"',
]


def ref_gbs(path):
    """Median GB/s over the refbw.py lines of one arm (None when the file is missing or has no value)."""
    if not os.path.exists(path):
        return None
    vals = []
    for line in open(path):
        try:
            vals.append(float(json.loads(line)["gbs"]))
        except (ValueError, KeyError, TypeError):
            continue
    return st.median(vals) if vals else None


def state_factor(ref, base):
    """Machine state of a run relative to the baseline state: reference GB/s over the stored baseline GB/s."""
    return ref / base


def normalise(raw, factor):
    """Decode tok/s the run would have reached in the baseline machine state."""
    return raw / factor


def factor_valid(factor):
    return FACTOR_MIN <= factor <= FACTOR_MAX


def macmon(path):
    op = gzip.open if path.endswith(".gz") else open
    busy, cpu = [], []
    for line in op(path, "rt"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        mhz, util = d["gpu_usage"]
        if util >= 0.5:
            busy.append(mhz)
            cpu.append(d.get("cpu_usage_pct", 0.0))
    if not busy:
        return None, None
    return st.median(busy), st.median(cpu)


def definition_change(a, size, ours, admitted, best, ok, frac, flat_floor, floor, ndec=None):
    """Decode floor at one size under the pre-2026-10-21 definition (raw decode of the best intarsia arm against 85 % of the
    flat weights-only floor, no admission) and under the current one (normalised decode of the best valid arm against 85 %
    of the context-aware floor; STATE-INVALID when no arm is valid). Informational: gate.py's verdict does not read it."""
    raws = [(k, v["rows"][size]["decode_tok_s"]["median"], v) for k, v in ours.items() if size in v["rows"]]
    ok_old, old_metric, old_arm = None, None, None
    if raws:
        old_arm, old_metric, _ = max(raws, key=lambda x: x[1])
        ok_old = old_metric >= FLOOR_FRAC * flat_floor
    metrics = {"flat_floor_tok_s": flat_floor}
    if raws:
        k0, _, v0 = raws[0]
        metrics["context_floor_tok_s"] = floor(v0["rows"][size], v0["gen"])
    if best is not None:
        k, v = best
        metrics.update(raw_tok_s=v["rows"][size]["decode_tok_s"]["median"], normalised_tok_s=ndec(v, size), context_floor_tok_s=floor(v["rows"][size], v["gen"]))
        if v["factor"] is not None:
            metrics["state_factor"] = v["factor"]
        cur = dict(metric_name="normalised_decode_tok_s" if v["factor"] is not None else "raw_decode_tok_s", metric=ndec(v, size),
                   limit=FLOOR_FRAC * metrics["context_floor_tok_s"], verdict="PASS" if ok else "FAIL", arm=k,
                   definition=f"decode of the best valid arm (clock-normalised when a reference ran) >= {FLOOR_FRAC:.0%} of the context-aware byte floor; GPU/CPU admission or machine-state factor in [{FACTOR_MIN:.2f}, {FACTOR_MAX:.2f}]")
    else:
        cur = dict(metric_name="normalised_decode_tok_s", metric=None, limit=FLOOR_FRAC * metrics.get("context_floor_tok_s", flat_floor), verdict="STATE-INVALID", arm=None,
                   definition="no valid intarsia arm (machine state outside the admission window): the floor gate cannot judge")
    return {
        "gate": f"metal_decode_floor[{size}]",
        "changed_on": CHANGED_ON,
        "log_ref": LOG_REFS,
        "metrics": metrics,
        "old_def": dict(metric_name="raw_decode_tok_s", metric=old_metric, limit=FLOOR_FRAC * flat_floor, verdict=None if ok_old is None else ("PASS" if ok_old else "FAIL"), arm=old_arm,
                        definition=f"raw decode of the best arm >= {FLOOR_FRAC:.0%} of the flat weights-only floor ({flat_floor:.2f} tok/s), no admission"),
        "current_def": cur,
    }


def definition_line(c):
    f = lambda v: "-" if v is None else f"{v:.2f}"
    o, n = c["old_def"], c["current_def"]
    refs = "; ".join(r.replace("docs/decisions/LOG.md row ", "") for r in c["log_ref"])
    return (f"gate-definition: {c['gate']}: old def: {o['verdict']} {f(o['metric'])} tok/s (limit {f(o['limit'])}) | "
            f"current def: {n['verdict']} {f(n['metric'])} tok/s (limit {f(n['limit'])}) [changed {c['changed_on']}; LOG {refs}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--weights-gb", type=float, default=15.1368)
    ap.add_argument("--kv-bytes", type=float, default=147456)
    ap.add_argument("--bw-gbs", type=float, default=614.0)
    ap.add_argument("--ref-baseline-gbs", type=float, default=None, help="override ref-baseline.json")
    ap.add_argument("--verdict-json", default=None, help="where to write the old/current definition records (default <dir>/gate-verdict.json)")
    a = ap.parse_args()
    base = a.ref_baseline_gbs if a.ref_baseline_gbs else (json.load(open(BASELINE_FILE))["gbs"] if os.path.exists(BASELINE_FILE) else None)
    arms = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*-localeval.json"))):
        label = os.path.basename(f)[: -len("-localeval.json")]
        mm = [p for p in (os.path.join(a.dir, f"{label}-macmon.jsonl"), os.path.join(a.dir, f"{label}-macmon.jsonl.gz")) if os.path.exists(p)]
        mhz, cpu = macmon(mm[0]) if mm else (None, None)
        d = json.load(open(f))
        gen = d["speed"].get("gen_tokens", 256)
        rows = {r["target_prompt_tokens"]: r for r in d["speed"]["rows"]}
        ref = ref_gbs(os.path.join(a.dir, f"{label}-ref.jsonl"))
        fac = state_factor(ref, base) if ref and base else None
        arms[label] = dict(mhz=mhz, cpu=cpu, gen=gen, rows=rows, ref=ref, factor=fac)
    ours = {k: v for k, v in arms.items() if k.startswith("intarsia")}
    mlx = {k: v for k, v in arms.items() if k.startswith("mlx-lm")}
    lla = {k: v for k, v in arms.items() if k.startswith("llamacpp")}
    if not ours:
        print("no intarsia-* arm with a localeval json", file=sys.stderr)
        return 2
    sizes = sorted(next(iter(ours.values()))["rows"])
    floor = lambda r, gen: a.bw_gbs * 1e9 / (a.weights_gb * 1e9 + a.kv_bytes * (r["prompt_tokens"] + gen / 2))
    dec = lambda arm, s: arm["rows"][s]["decode_tok_s"]["median"] if s in arm["rows"] else None
    ndec = lambda arm, s: normalise(dec(arm, s), arm["factor"]) if arm["factor"] else dec(arm, s)  # clock-normalised when a reference ran
    best = lambda group, s: max((x for x in (dec(v, s) for v in group.values()) if x), default=None)

    print(f"Metal speed gate on {a.dir}: floor {FLOOR_FRAC:.0%} of context-aware bytes, decode clock-normalised by the refbw.py reference (baseline {base if base else '?'} GB/s, valid factor {FACTOR_MIN:.2f}-{FACTOR_MAX:.2f}); arms without a reference use the GPU >= {MIN_MHZ:.0f} MHz and CPU <= {MAX_CPU:.2f} admission; ratios >= {MLX_FRAC:.2f}x mlx-lm, >= {LLAMA_FRAC:.2f}x llama.cpp\n")
    print("| arm | GPU MHz | CPU | ref GB/s | state factor | valid | " + " | ".join(f"{s}: raw tok/s, % cf, norm tok/s, % cf" for s in sizes) + " |")
    print("|---|---:|---:|---:|---:|---|" + "---:|" * len(sizes))
    for k, v in arms.items():
        if v["factor"] is not None:
            adm = factor_valid(v["factor"])
        else:
            adm = v["mhz"] is not None and v["mhz"] >= MIN_MHZ and v["cpu"] <= MAX_CPU
        v["admitted"] = adm
        cells = []
        for s in sizes:
            r = v["rows"].get(s)
            if not r:
                cells.append("-")
                continue
            c = f"{dec(v, s):.1f}, {100 * dec(v, s) / floor(r, v['gen']):.1f} %"
            if v["factor"] is not None:
                c += f", {ndec(v, s):.1f}, {100 * ndec(v, s) / floor(r, v['gen']):.1f} %"
            cells.append(c)
        mhz = f"{v['mhz']:.0f}" if v["mhz"] is not None else "?"
        cpu = f"{v['cpu']:.2f}" if v["cpu"] is not None else "?"
        ref = f"{v['ref']:.1f}" if v["ref"] is not None else "-"
        fac = f"{v['factor']:.3f}" if v["factor"] is not None else "-"
        print(f"| {k} | {mhz} | {cpu} | {ref} | {fac} | {'yes' if adm else 'no'} | " + " | ".join(cells) + " |")

    ratio_ok, floor_ok, any_adm = True, True, False
    changes = []
    flat_floor = a.bw_gbs * 1e9 / (a.weights_gb * 1e9)  # the pre-2026-10-21 floor: weights only, no KV bytes
    print()
    for s in sizes:
        m, l = best(mlx, s), best(lla, s)
        for k, v in ours.items():
            x = dec(v, s)
            if x is None:
                continue
            if m:
                ok = x >= MLX_FRAC * m
                ratio_ok &= ok
                print(f"{s}: {k} / mlx-lm {x / m:.3f} {'ok' if ok else 'FAIL'}")
            if l:
                ok = x >= LLAMA_FRAC * l
                ratio_ok &= ok
                print(f"{s}: {k} / llama.cpp {x / l:.3f} {'ok' if ok else 'FAIL'}")
    for s in sizes:
        adm = [(k, v) for k, v in ours.items() if v["admitted"] and s in v["rows"]]
        any_adm |= bool(adm)
        if not adm:
            print(f"{s}: floor gate: no valid intarsia arm")
            changes.append(definition_change(a, s, ours, adm, None, None, None, flat_floor, floor))
            continue
        k, v = max(adm, key=lambda kv: ndec(kv[1], s) / floor(kv[1]["rows"][s], kv[1]["gen"]))
        fl = floor(v["rows"][s], v["gen"])
        fr = ndec(v, s) / fl
        ok = fr >= FLOOR_FRAC
        floor_ok &= ok
        how = (f"raw {dec(v, s):.2f} tok/s ({100 * dec(v, s) / fl:.1f} %), ref {v['ref']:.1f} GB/s / baseline {base:.1f} = factor {v['factor']:.3f}, "
               f"normalised {ndec(v, s):.2f} tok/s") if v["factor"] is not None else f"raw {dec(v, s):.2f} tok/s (not normalised: no reference)"
        print(f"{s}: floor gate: best valid {k}: {how}: {100 * fr:.1f} % of the context-aware floor ({fl:.2f} tok/s) {'ok' if ok else 'FAIL'}")
        changes.append(definition_change(a, s, ours, adm, (k, v), ok, fr, flat_floor, floor, ndec))
    print()
    for c in changes:
        print(definition_line(c))
    out = a.verdict_json or os.path.join(a.dir, "gate-verdict.json")
    if not ratio_ok or (any_adm and not floor_ok):
        final, rc = "FAIL", 1
    elif not all(any(v["admitted"] and s in v["rows"] for v in ours.values()) for s in sizes):
        final, rc = "STATE-INVALID", 3
    else:
        final, rc = "PASS", 0
    try:
        with open(out, "w") as fh:
            json.dump({"format": "intarsia-metal-speed-gate-v1", "verdict": final, "definition_changes": changes}, fh, indent=2)
            fh.write("\n")
    except OSError as e:
        print(f"note: could not write {out}: {e}", file=sys.stderr)
    if final == "FAIL":
        print("\nVERDICT: FAIL")
    elif final == "STATE-INVALID":
        print("\nVERDICT: STATE-INVALID (ratios pass; re-run the floor gate in a valid machine state)")
    else:
        print("\nVERDICT: PASS")
    return rc


if __name__ == "__main__":
    sys.exit(main())

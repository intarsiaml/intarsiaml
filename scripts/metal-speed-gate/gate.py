#!/usr/bin/env python3
"""Metal speed gate (restated by the user 2026-10-21 12:25 EDT; docs/benchmarking.md "Metal speed gate").

usage: gate.py <dir>
  <dir> holds, per arm, `<label>-localeval.json` (a localeval `speed` result.json) and `<label>-macmon.jsonl[.gz]`
  (`macmon pipe` samples taken while that arm ran). Arms are recognised by label prefix: `intarsia*`, `mlx-lm*`,
  `llamacpp*` (`sameslot.sh` writes exactly this layout).

Gates, per prompt size:
  1. decode >= 85 % of the context-aware byte floor: (weights + KV bytes at the mean context) / bandwidth, where the
     mean context is prompt + gen / 2. Only intarsia arms that are *admitted* count: over the arm, the median GPU clock
     of busy samples (GPU utilisation >= 50 %) is >= 1300 MHz and the median macmon CPU load is <= 0.30. An arm outside
     that window is STATE-INVALID (re-run it), not a failure. The gate passes when at least one admitted intarsia arm
     passes at every size; with no admitted arm the verdict is STATE-INVALID.
  2. same-slot ratios (clock-invariant, every intarsia arm against the same slot's other engines): decode >= 0.90x
     mlx-lm and >= 1.00x llama.cpp.
Exit 0 = PASS, 1 = FAIL, 3 = STATE-INVALID (ratios passed, no admitted arm), 2 = usage / missing data.

Model constants default to Qwen3-8B BF16 (15.1368 GB of weights streamed per token, 147,456 KV bytes per position,
614 GB/s on endor's M5 Max); override with --weights-gb, --kv-bytes, --bw-gbs.
"""
import argparse, glob, gzip, json, os, statistics as st, sys

MIN_MHZ, MAX_CPU, FLOOR_FRAC, MLX_FRAC, LLAMA_FRAC = 1300.0, 0.30, 0.85, 0.90, 1.00


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--weights-gb", type=float, default=15.1368)
    ap.add_argument("--kv-bytes", type=float, default=147456)
    ap.add_argument("--bw-gbs", type=float, default=614.0)
    a = ap.parse_args()
    arms = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*-localeval.json"))):
        label = os.path.basename(f)[: -len("-localeval.json")]
        mm = [p for p in (os.path.join(a.dir, f"{label}-macmon.jsonl"), os.path.join(a.dir, f"{label}-macmon.jsonl.gz")) if os.path.exists(p)]
        mhz, cpu = macmon(mm[0]) if mm else (None, None)
        d = json.load(open(f))
        gen = d["speed"].get("gen_tokens", 256)
        rows = {r["target_prompt_tokens"]: r for r in d["speed"]["rows"]}
        arms[label] = dict(mhz=mhz, cpu=cpu, gen=gen, rows=rows)
    ours = {k: v for k, v in arms.items() if k.startswith("intarsia")}
    mlx = {k: v for k, v in arms.items() if k.startswith("mlx-lm")}
    lla = {k: v for k, v in arms.items() if k.startswith("llamacpp")}
    if not ours:
        print("no intarsia-* arm with a localeval json", file=sys.stderr)
        return 2
    sizes = sorted(next(iter(ours.values()))["rows"])
    floor = lambda r, gen: a.bw_gbs * 1e9 / (a.weights_gb * 1e9 + a.kv_bytes * (r["prompt_tokens"] + gen / 2))
    dec = lambda arm, s: arm["rows"][s]["decode_tok_s"]["median"] if s in arm["rows"] else None
    best = lambda group, s: max((x for x in (dec(v, s) for v in group.values()) if x), default=None)

    print(f"Metal speed gate on {a.dir}: floor {FLOOR_FRAC:.0%} of context-aware bytes; admission GPU >= {MIN_MHZ:.0f} MHz and CPU <= {MAX_CPU:.2f}; ratios >= {MLX_FRAC:.2f}x mlx-lm, >= {LLAMA_FRAC:.2f}x llama.cpp\n")
    print("| arm | GPU MHz | CPU | admitted | " + " | ".join(f"{s}: tok/s, % cf" for s in sizes) + " |")
    print("|---|---:|---:|---|" + "---:|" * len(sizes))
    for k, v in arms.items():
        adm = v["mhz"] is not None and v["mhz"] >= MIN_MHZ and v["cpu"] <= MAX_CPU
        v["admitted"] = adm
        cells = []
        for s in sizes:
            r = v["rows"].get(s)
            cells.append(f"{dec(v, s):.1f}, {100 * dec(v, s) / floor(r, v['gen']):.1f} %" if r else "-")
        mhz = f"{v['mhz']:.0f}" if v["mhz"] is not None else "?"
        cpu = f"{v['cpu']:.2f}" if v["cpu"] is not None else "?"
        print(f"| {k} | {mhz} | {cpu} | {'yes' if adm else 'no'} | " + " | ".join(cells) + " |")

    ratio_ok, floor_ok, any_adm = True, True, False
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
            print(f"{s}: floor gate: no admitted intarsia arm")
            continue
        fr = max(dec(v, s) / floor(v["rows"][s], v["gen"]) for k, v in adm)
        ok = fr >= FLOOR_FRAC
        floor_ok &= ok
        print(f"{s}: floor gate: best admitted {100 * fr:.1f} % of the context-aware floor {'ok' if ok else 'FAIL'}")
    if not ratio_ok or (any_adm and not floor_ok):
        print("\nVERDICT: FAIL")
        return 1
    if not all(any(v["admitted"] and s in v["rows"] for v in ours.values()) for s in sizes):
        print("\nVERDICT: STATE-INVALID (ratios pass; re-run the floor gate in an admitted state)")
        return 3
    print("\nVERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

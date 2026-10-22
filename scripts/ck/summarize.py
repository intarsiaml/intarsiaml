#!/usr/bin/env python3
"""Lane ck: summarize one backend's chunk-job output (speed, plan scratch, memory gauges).

    summarize.py RAWDIR            # RAWDIR holds speed/<model>/{bench-c*.txt,plan-c*.json,mem-c*-*.txt}, possibly nested one level
Prints, per model: prefill tok/s (median of the 3 reps; all reps) per size and chunk, the change vs chunk 2048 (the mean of
the 2048 runs if the drift-control repeat exists), the plan's scratch_bytes and the load-time memory gauges.
"""
import glob, json, os, re, statistics, sys


def main():
    root = sys.argv[1]
    models = sorted({p.split("/speed/")[1].split("/")[0] for p in glob.glob(f"{root}/**/speed/*/bench-c*.txt", recursive=True)})
    for m in models:
        d = glob.glob(f"{root}/**/speed/{m}", recursive=True)[0]
        med, reps = {}, {}  # (size, chunk) -> [median per serve run], [all printed reps]
        for f in sorted(glob.glob(f"{d}/bench-c*.txt")):
            c = int(re.search(r"bench-c(\d+)", f).group(1))
            txt = open(f).read()
            for sz, rate in re.findall(r"^\s+(\d+)\s+\d+\s+[\d.]+\s+(\d+)\s+[\d.]+\s+\d+\s*$", txt, re.M):
                med.setdefault((int(sz), c), []).append(float(rate))
            for sz, rate in re.findall(r"size\s+(\d+) rep: prompt=\d+ gen=\d+ ttft=[\d.]+s prefill=(\d+) tok/s", txt):
                reps.setdefault((int(sz), c), []).append(float(rate))
        print(f"== {m}")
        chunks = sorted({c for _, c in med})
        sizes = sorted({s for s, _ in med})
        for s in sizes:
            base = med.get((s, 2048), [])
            b = statistics.mean(base) if base else float("nan")
            for c in chunks:
                v = med.get((s, c), [])
                print(f"  size {s:>6} chunk {c:>5}: {statistics.mean(v):8.1f} tok/s (median of 3 reps; runs {v})  vs 2048: {100 * (statistics.mean(v) / b - 1):+5.1f} %  reps {reps.get((s, c), [])}")
        for f in sorted(glob.glob(f"{d}/plan-c*.json")):
            p = json.load(open(f))
            print(f"  plan {os.path.basename(f)}: prefill_chunk {p['prefill_chunk']} scratch_bytes {p['scratch_bytes']} ({p['scratch_bytes'] / 2**20:.1f} MiB)")
        for f in sorted(glob.glob(f"{d}/mem-c*.txt")):
            print(f"  {os.path.basename(f)}: " + " | ".join(l.strip() for l in open(f)))


main()

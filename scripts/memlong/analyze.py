#!/usr/bin/env python3
"""memlong analysis: python3 analyze.py <raw-dir> <tag> [--oneline REASON]
Reads mem-<tag>.csv (memmix/sampler.sh, 60 s rows), marks-<tag>.tsv (base start settle load stop) and serve-<tag>.log (libmemstat.c malloc_info blocks, no trims).
Series: RssAnon (sampler), live heap (arena read-write part - free chunks + directly mmapped chunks), arena committed size (read-write part), free chunks in arenas, mmapped chunks.
Prints slope +- SE (MB/h, autocorrelation-corrected like memmix) from minute 30 and from hour 3 of the load, and a plateau test: slope over the last third vs the first third of
the window from minute 30. Plateau = the last-third slope is under 1/3 of the first-third one, or within 2 SE of zero. Stdlib only."""
import csv, math, pathlib, re, sys

MB = 1 << 20


def fit(xs, ys):
    n = len(xs)
    if n < 4:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    r1 = sum(res[i] * res[i - 1] for i in range(1, n)) / max(1e-9, sum(e * e for e in res))
    infl = math.sqrt(max(1.0, (1 + r1) / max(0.05, 1 - r1))) if r1 > 0 else 1.0
    return b, math.sqrt(sum(e * e for e in res) / (n - 2) / sxx) * infl


def slope(series, a, b):  # series: [(t_hours, value)]; window in hours
    pts = [(t, v) for t, v in series if a <= t <= b and not math.isnan(v)]
    return fit([p[0] for p in pts], [p[1] for p in pts]), len(pts)


def blocks(log):
    """[(epoch, arenas, rw_MB, free_MB, mmap_MB)] for the pid with the most heaps"""
    text = open(log, errors="replace").read()
    per = {}
    for part in re.split(r"(?m)^\[memstat\] ", text)[1:]:
        m = re.match(r"pid=(\d+) t=(\d+) ", part)
        if not m:
            continue
        heaps = re.findall(r"<heap nr=\"(\d+)\">(.*?)</heap>", part, re.S)
        if not heaps:
            continue
        rw = free = 0
        for _, body in heaps:
            for typ in ("fast", "rest"):
                mm = re.search(rf'<total type="{typ}" count="\d+" size="(\d+)"', body)
                free += int(mm.group(1)) if mm else 0
            mm = re.search(r'<aspace type="mprotect" size="(\d+)"', body)
            rw += int(mm.group(1)) if mm else 0
        tail = part[part.rfind("</heap>"):]
        mm = re.search(r'<total type="mmap" count="\d+" size="(\d+)"', tail)
        per.setdefault(int(m.group(1)), []).append((int(m.group(2)), len(heaps), rw / MB, free / MB, (int(mm.group(1)) if mm else 0) / MB))
    if not per:
        return []
    return per[max(per, key=lambda p: max(r[1] for r in per[p]))]


def load(raw, tag):
    rows = []
    for r in csv.DictReader(open(raw / f"mem-{tag}.csv")):
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = float("nan")
        rows.append(d)
    marks = {}
    mp = raw / f"marks-{tag}.tsv"
    if mp.exists():
        for line in mp.read_text().splitlines():
            if "\t" in line:
                t, name = line.split("\t")
                marks[name] = int(t)
    return rows, marks


def oneline(raw, tag, reason):
    rows, marks = load(raw, tag)
    live = [r for r in rows if r.get("st_RssAnon", float("nan")) == r.get("st_RssAnon") and r["st_RssAnon"] > 0 and r["epoch"] >= marks.get("load", 0)]
    if not live:
        return f"memlong {tag}: stop reason '{reason}'; no load-phase samples"
    t0 = marks.get("load", live[0]["epoch"]); last = live[-1]
    mina = min(r["mi_MemAvailable"] for r in live) / 1024
    reqs = max((r["requests_total"] for r in live if not math.isnan(r["requests_total"])), default=float("nan"))
    thr = [r["st_Threads"] for r in live if not math.isnan(r["st_Threads"])]
    fds = [r["fds"] for r in live if not math.isnan(r["fds"])]
    return (f"memlong {tag}: stop reason '{reason}'; load {(last['epoch'] - t0) / 3600:.2f} h, {len(live)} samples, requests {reqs:.0f}, RssAnon {live[0]['st_RssAnon']:.0f} -> {last['st_RssAnon']:.0f} MB, "
            f"min MemAvailable {mina:.1f} GB, threads {min(thr, default=0):.0f}-{max(thr, default=0):.0f}, fds {min(fds, default=0):.0f}-{max(fds, default=0):.0f}")


def main():
    raw = pathlib.Path(sys.argv[1]); tag = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "--oneline":
        print(oneline(raw, tag, sys.argv[4] if len(sys.argv) > 4 else "")); return
    rows, marks = load(raw, tag)
    t0 = marks.get("load", rows[0]["epoch"])
    hrs = lambda e: (e - t0) / 3600
    ser = {"RssAnon": [(hrs(r["epoch"]), r["st_RssAnon"]) for r in rows if r["epoch"] >= t0],
           "MemAvailable": [(hrs(r["epoch"]), r["mi_MemAvailable"]) for r in rows if r["epoch"] >= t0],
           "store host tier": [(hrs(r["epoch"]), r["prefix_store_tier_bytes_tier__host__"] / 1e6) for r in rows if r["epoch"] >= t0],
           "engine GPU": [(hrs(r["epoch"]), r["gpu_mb_intarsia"]) for r in rows if r["epoch"] >= t0]}
    bl = blocks(raw / f"serve-{tag}.log") if (raw / f"serve-{tag}.log").exists() else []
    bl = [b for b in bl if b[0] >= t0]
    ser["live heap"] = [(hrs(b[0]), b[2] - b[3] + b[4]) for b in bl]
    ser["arena committed"] = [(hrs(b[0]), b[2]) for b in bl]
    ser["arena free chunks"] = [(hrs(b[0]), b[3]) for b in bl]
    ser["mmapped chunks"] = [(hrs(b[0]), b[4]) for b in bl]
    # outside malloc (approx.): RssAnon at the sample nearest each block minus the arenas' committed size and the mmapped chunks; with the prefault default the store buffers are in here
    rs = ser["RssAnon"]
    ser["anon outside malloc"] = [(hrs(b[0]), min(rs, key=lambda p: abs(p[0] - hrs(b[0])))[1] - b[2] - b[4]) for b in bl] if rs else []
    end = max((p[0] for s in ser.values() for p in s), default=0)
    print(f"{tag}: {len(rows)} samples, {len(bl)} malloc_info blocks ({len(bl[0:1]) and bl[0][1]} arenas first block), load {end:.2f} h; marks " + ", ".join(f"{k}={(v - marks['start']) / 60:.1f}min" for k, v in sorted(marks.items(), key=lambda kv: kv[1]) if 'start' in marks))
    names = ["RssAnon", "live heap", "arena committed", "arena free chunks", "mmapped chunks", "anon outside malloc", "store host tier", "MemAvailable", "engine GPU"]
    print("\nvalues at minutes of load (nearest sample, MB; RssAnon from the 60 s sampler, malloc numbers from the malloc_info blocks)")
    print("min\t" + "\t".join(names))
    for mn in (5, 10, 20, 30, 60, 120, 180, 240, 360, 480, 600, 660):
        if mn / 60 > end + 0.02:
            break
        cells = []
        for n in names:
            pts = ser[n]
            p = min(pts, key=lambda q: abs(q[0] - mn / 60)) if pts else None
            cells.append(f"{p[1]:.0f}" if p and abs(p[0] - mn / 60) < 0.05 else "-")
        print(f"{mn}\t" + "\t".join(cells))
    print("\nmean per hour of load (MB)")
    print("hour\t" + "\t".join(names))
    for h in range(int(math.ceil(end))):
        cells = []
        for n in names:
            v = [y for t, y in ser[n] if h <= t < h + 1 and not math.isnan(y)]
            cells.append(f"{sum(v) / len(v):.0f}" if v else "-")
        print(f"{h}-{h + 1}\t" + "\t".join(cells))
    print("\nslopes MB/h +- SE (autocorrelation-corrected); window = minute 30 / hour 3 of the load to its end")
    print("series\tfrom 30 min\tn\tfrom 3 h\tn")
    for n in names[:7] + ["MemAvailable"]:
        (b1, s1), n1 = slope(ser[n], 0.5, 1e9); (b3, s3), n3 = slope(ser[n], 3.0, 1e9)
        print(f"{n}\t{b1:+.0f} +- {s1:.0f}\t{n1}\t" + (f"{b3:+.0f} +- {s3:.0f}\t{n3}" if n3 >= 4 else "-\t-"))
    print("\nplateau test: slope over the first third vs the last third of the window from minute 30 (MB/h +- SE)")
    print("series\tfirst third\tlast third\tverdict")
    span = end - 0.5
    for n in ("RssAnon", "live heap", "arena committed", "anon outside malloc"):
        if span < 0.3:
            print(f"{n}\t-\t-\twindow too short"); continue
        a, b, c = 0.5, 0.5 + span / 3, 0.5 + 2 * span / 3
        (f1, e1), _ = slope(ser[n], a, b); (f3, e3), _ = slope(ser[n], c, end)
        if math.isnan(f1) or math.isnan(f3):
            v = "too few points"
        elif abs(f3) <= 2 * e3 or (f1 > 0 and f3 < f1 / 3):
            v = "plateau (last third flat or under a third of the first)"
        else:
            v = "still growing" if f3 > 0 else "shrinking"
        print(f"{n}\t{f1:+.0f} +- {e1:.0f}\t{f3:+.0f} +- {e3:.0f}\t{v}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analysis half of attn-tfblock-bench.sh: calibration gate, A/B table, verdicts. Pure stdlib; runs on the Spark and on a laptop.

  tfblock-table.py calib  --ab LOG --rope LOG --expected expected-hashes.txt     exit 0 pass, 3 fail
  tfblock-table.py ab     --variant NAME=LOG ... [--blocks base=1,NAME=2] [--stop NAME=12] [--expected F]
  tfblock-table.py rope   --base LOG ... --variant NAME=LOG ... [--expected F]

Attention logs are what ab-run.sh prints ("## pair <i> <role> <time>", the probe line, one `attn <shape> hash ...` line per shape) with role
base or var; several --variant logs may be given, one per variant. Rope logs are rope_ip_bench output, one file per process (base) or
"## pair" separated (variant), see attn-tfblock-bench.sh. A hash mismatch is NEGATIVE, never tuning.
"""
import argparse, re, statistics, sys, collections

ATTN = re.compile(r"^attn (.+?)\s+hash (\w+) min ([\d.]+) med ([\d.]+) max ([\d.]+) ms")
ROPE = re.compile(r"^rope (.+?)\s+hash (\w+)(?:\s+min ([\d.]+) med ([\d.]+) ms|\s+([\d.]+) ms)")
PROBE_BLOCKS = re.compile(r"active blocks per SM (\d+)")
PROBE_REGS = re.compile(r"regs/thread (\d+)")
PROBE_LOCAL = re.compile(r"local (\d+) B")
PROBE_TOTAL = re.compile(r"total (\d+) B")
HDR = re.compile(r"^## pair (\d+) (\w+)")


def parse_ab(path):
    """-> {role: {shape: [(hash, min, med, max), ...]}}, {role: probe-info}"""
    d = {"base": collections.defaultdict(list), "var": collections.defaultdict(list)}
    probe = {}
    role = None
    for line in open(path):
        line = line.rstrip("\r\n")
        m = HDR.match(line)
        if m:
            role = m.group(2)
            continue
        if line.startswith("probe:") and role:
            probe[role] = {"blocks": int(PROBE_BLOCKS.search(line).group(1)), "regs": int(PROBE_REGS.search(line).group(1)),
                           "local": int(PROBE_LOCAL.search(line).group(1)) if PROBE_LOCAL.search(line) else None,
                           "total": int(PROBE_TOTAL.search(line).group(1))}
            continue
        m = ATTN.match(line)
        if m and role:
            d[role][m.group(1)].append((m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5))))
    return d, probe


def med(xs):
    return statistics.median(xs)


def load_expected(path):
    e = {}
    if path:
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#"):
                kind, rest = line.split(" ", 1)
                name, h = rest.rsplit(" ", 1)
                e[(kind, name)] = h
    return e


def parse_rope(paths):
    """-> {shape: [(hash, min, med), ...]} over all processes in the files (a "## pair" header only separates processes)"""
    d = collections.defaultdict(list)
    for p in paths:
        for line in open(p):
            m = ROPE.match(line.rstrip("\r\n"))
            if m:
                if m.group(3) is not None:
                    d[m.group(1)].append((m.group(2), float(m.group(3)), float(m.group(4))))
                else:
                    d[m.group(1)].append((m.group(2), float(m.group(5)), float(m.group(5))))
    return d


def cmd_calib(a):
    exp = load_expected(a.expected)
    ok = True
    d, probe = parse_ab(a.ab)
    base = d["base"]
    print("calibration (unmodified kernel; recorded: ab-run-1.log hashes, 1.30 ms at CSA T=512, 5.04 ms at CSA T=2048)")
    for shape, want_ms in (("CSA T512 Nc1024 Kw512", 1.30), ("CSA T2048 Nc1024 Kw512", 5.04)):
        runs = base.get(shape, [])
        if not runs:
            print(f"  FAIL {shape}: no result"); ok = False; continue
        got = med([r[2] for r in runs])
        good = abs(got - want_ms) / want_ms <= a.tol
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {shape}: median {got:.4f} ms vs recorded {want_ms:.2f} ms (tolerance {a.tol * 100:.0f}%)")
    for shape, runs in base.items():
        want = exp.get(("attn", shape))
        got = {r[0] for r in runs}
        good = want is not None and got == {want}
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} hash {shape}: {','.join(sorted(got))} vs recorded {want}")
    if a.rope:
        r = parse_rope([a.rope])
        for shape, runs in r.items():
            want = exp.get(("rope", shape))
            got = {x[0] for x in runs}
            good = want is not None and got == {want}
            ok &= good
            print(f"  {'ok  ' if good else 'FAIL'} hash rope {shape}: {','.join(sorted(got))} vs recorded {want}")
        if not r:
            print("  FAIL rope: no result"); ok = False
    if "base" in probe:
        print(f"  probe base: {probe['base']}")
        if probe["base"]["blocks"] != 1:
            print("  FAIL base should report 1 active block per SM"); ok = False
    print("CALIBRATION " + ("PASS" if ok else "FAIL: stop, do not read the variants"))
    return 0 if ok else 3


def cmd_ab(a):
    exp = load_expected(a.expected)
    blocks = dict(kv.split("=") for kv in a.blocks.split(",")) if a.blocks else {}
    stops = dict(kv.split("=") for kv in a.stop) if a.stop else {}
    summary = []
    meds = {}
    for spec in a.variant:
        name, path = spec.split("=", 1)
        d, probe = parse_ab(path)
        shapes = list(d["base"].keys())
        print(f"\n=== {name}  (log {path})")
        for role in ("base", "var"):
            p = probe.get(role)
            if p:
                print(f"  probe {role:4s}: {p['regs']} regs/thread, local {p['local']} B, {p['total']} B smem/block, active blocks per SM {p['blocks']}")
        print(f"  {'shape':26s} {'base ms':>8s} {'[range]':>17s} {'var ms':>8s} {'[range]':>17s} {'var/base':>8s} {'min/min':>8s} {'gain':>7s}  hashes")
        bad = []
        gain_t2048 = None
        for s in shapes:
            b, v = d["base"][s], d["var"].get(s, [])
            if not v:
                print(f"  {s:26s} no variant result"); bad.append(s); continue
            bm, vm = [x[2] for x in b], [x[2] for x in v]
            hs = {x[0] for x in b} | {x[0] for x in v}
            want = exp.get(("attn", s))
            same = len(hs) == 1 and (want is None or hs == {want})
            if not same:
                bad.append(s)
            ratio = med(vm) / med(bm)
            mm = min(x[1] for x in v) / min(x[1] for x in b)
            if s.startswith("CSA T2048"):
                gain_t2048 = 1 - ratio
                meds[name] = med(vm)
            print(f"  {s:26s} {med(bm):8.4f} [{min(bm):.4f},{max(bm):.4f}] {med(vm):8.4f} [{min(vm):.4f},{max(vm):.4f}] {ratio:8.3f} {mm:8.3f} {(1 - ratio) * 100:+6.1f}%  "
                  f"{'identical ' + next(iter(hs)) if same else 'MISMATCH ' + ','.join(sorted(hs))}  (n base {len(b)}, var {len(v)})")
        verdict = None
        want_blocks = blocks.get(name)
        got_blocks = probe.get("var", {}).get("blocks")
        if bad:
            verdict = f"NEGATIVE: hash mismatch or missing result at {', '.join(bad)}. Bit identity is the gate; this is not tuning."
        elif want_blocks is not None and got_blocks is not None and int(want_blocks) != got_blocks:
            verdict = f"INVALID: probe reports {got_blocks} active blocks per SM, expected {want_blocks}; the experiment did not run as designed"
        elif gain_t2048 is None:
            verdict = "INVALID: no CSA T=2048 result"
        elif name in stops:
            lim = float(stops[name]) / 100
            verdict = (f"POSITIVE: gain {gain_t2048 * 100:+.1f}% at CSA T=2048 >= {lim * 100:.0f}% and all hashes identical" if gain_t2048 >= lim else
                       f"NEGATIVE (stop rule): gain {gain_t2048 * 100:+.1f}% at CSA T=2048 < {lim * 100:.0f}%")
        else:
            verdict = f"INFORMATIONAL: gain {gain_t2048 * 100:+.1f}% at CSA T=2048, all hashes identical (no stop rule set for {name})"
        loc = probe.get("var", {}).get("local")
        if loc:
            verdict += f"  [note: variant has {loc} B local memory per thread (spills): its time is not representative of the design]"
        print("  VERDICT " + name + ": " + verdict)
        summary.append((name, verdict))
    if "e2" in meds and "e2e5" in meds:
        print(f"\nE5 on top of E2 at CSA T=2048 (median of the two variants' process medians): e2e5 / e2 = {meds['e2e5'] / meds['e2']:.3f}")
    print("\nSUMMARY")
    for n, v in summary:
        print(f"  {n}: {v}")
    return 0


def cmd_rope(a):
    exp = load_expected(a.expected)
    base = parse_rope(a.base)
    base_ms = {s: med([x[2] for x in r]) for s, r in base.items()}
    summary = []
    for spec in a.variant:
        name, path = spec.split("=", 1)
        v = parse_rope([path])
        print(f"\n=== {name}  (log {path})")
        print(f"  {'shape':28s} {'base ms':>8s} {'var ms':>8s} {'var/base':>8s}  hashes")
        bad = []
        for s, r in base.items():
            vr = v.get(s, [])
            if not vr:
                print(f"  {s:28s} no variant result"); bad.append(s); continue
            hs = {x[0] for x in r} | {x[0] for x in vr}
            want = exp.get(("rope", s))
            same = len(hs) == 1 and (want is None or hs == {want})
            if not same:
                bad.append(s)
            vm = med([x[2] for x in vr])
            print(f"  {s:28s} {base_ms[s]:8.4f} {vm:8.4f} {vm / base_ms[s]:8.3f}  {'identical ' + next(iter(hs)) if same else 'MISMATCH ' + ','.join(sorted(hs))}")
        q, o = "q T2048 H32 hd512", "o^-1 T2048 H32 hd512"
        if not bad and q in base_ms and o in base_ms:
            vq, vo = med([x[2] for x in v[q]]), med([x[2] for x in v[o]])
            save = 80 * ((base_ms[q] - vq) + (base_ms[o] - vo))
            verdict = (f"bit-identical on all {len(base)} rope shapes; at T=2048 the pair (q, o^-1) goes {base_ms[q] + base_ms[o]:.3f} -> {vq + vo:.3f} ms per layer-chunk; "
                       f"extrapolated saving {save:.0f} ms per 4K request (80 layer-chunks at chunk 2048, from the fusion/tfblock report; not measured in situ)")
        else:
            verdict = f"NEGATIVE: hash mismatch or missing result at {', '.join(bad)}. Bit identity is the gate; this is not tuning."
        print("  VERDICT " + name + ": " + verdict)
        summary.append((name, verdict))
    print("\nSUMMARY (rope)")
    for n, vd in summary:
        print(f"  {n}: {vd}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib"); c.add_argument("--ab", required=True); c.add_argument("--rope"); c.add_argument("--expected", required=True); c.add_argument("--tol", type=float, default=0.06)
    b = sub.add_parser("ab"); b.add_argument("--variant", action="append", required=True); b.add_argument("--blocks"); b.add_argument("--stop", action="append"); b.add_argument("--expected")
    r = sub.add_parser("rope"); r.add_argument("--base", action="append", required=True); r.add_argument("--variant", action="append", required=True); r.add_argument("--expected")
    a = ap.parse_args()
    sys.exit({"calib": cmd_calib, "ab": cmd_ab, "rope": cmd_rope}[a.cmd](a))


if __name__ == "__main__":
    main()

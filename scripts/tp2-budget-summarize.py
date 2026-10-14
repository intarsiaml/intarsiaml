import re, sys, glob, os
d = sys.argv[1]
for f in sorted(glob.glob(d + '/[BCV]-*.log')):
    n = os.path.basename(f)[:-4]
    t = open(f).read()
    st = re.findall(r'SELF-STATS mode=(\w+) positions=(\d+) mean_kl21=(\S+) max_kl21=(\S+) flips=(\d+) flip_pct=(\S+) tail_abs=(\S+) worst_prompt_kl21=(\S+)', t)
    v = re.findall(r'self-golden verdict.*: (\w+)', t)
    out = n.ljust(20)
    for m, pos, kl, mx, fl, fp, ta, wp in st:
        out += f' | {m[:3]} kl={float(kl):.2e} max={float(mx):.1e} fl={fl}/{pos} tail={float(ta):.1e} wp={float(wp):.1e}'
    out += ' | ' + (v[0] if v else '?')
    print(out)
print()
for f in sorted(glob.glob(d + '/[AF]-*.log')):
    n = os.path.basename(f)[:-4]
    t = open(f).read()
    ov = re.findall(r'^\s*OVERALL\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', t, re.M)
    g = re.findall(r'golden gate: (\w+)', t)
    print(n.ljust(20), ' | '.join(f'n={a} top1={b} KLfull={c} KL21={e}' for a, b, c, e, _ in ov), '|', g[-1] if g else '?')

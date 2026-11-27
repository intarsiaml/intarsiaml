"""Per-kernel comparison of two device assemblies (isa_check.sh): for each TU, every function body (its label to .Lfunc_end) in
main's and the branch's .s, instructions only (comments, labels' numbering and the metadata sections dropped)."""
import os, re, sys

out, tus = sys.argv[1], sys.argv[2:]

def funcs(path):
    fs, cur, body = {}, None, []
    for line in open(path, errors="replace"):
        m = re.match(r"^([A-Za-z_$][\w$.]*):\s*(;.*)?$", line)
        if m and not m.group(1).startswith(".L"):
            if cur:
                fs[cur] = body
            cur, body = m.group(1), []
            continue
        if cur is None:
            continue
        if line.startswith(".Lfunc_end"):
            fs[cur] = body
            cur, body = None, []
            continue
        s = line.split(";")[0].strip()
        if not s or s.startswith("."):
            continue
        s = re.sub(r"\.LBB\d+_\d+", ".LBB", s)
        body.append(s)
    return fs

tot_same = tot_diff = 0
for t in tus:
    b = t[:-4]
    pm, pb = os.path.join(out, f"main-{b}.s"), os.path.join(out, f"br-{b}.s")
    if not (os.path.exists(pm) and os.path.exists(pb)):
        print(f"{t}: missing assembly")
        continue
    fm, fb = funcs(pm), funcs(pb)
    same = [k for k in fm if k in fb and fm[k] == fb[k]]
    diff = [k for k in fm if k in fb and fm[k] != fb[k]]
    only_m = [k for k in fm if k not in fb]
    only_b = [k for k in fb if k not in fm]
    tot_same += len(same); tot_diff += len(diff)
    print(f"{t}: {len(same)} kernels identical, {len(diff)} differ, {len(only_m)} only in main, {len(only_b)} only in the branch")
    for k in diff:
        print(f"  DIFF {k} ({len(fm[k])} -> {len(fb[k])} instructions)")
    for k in only_m:
        print(f"  ONLY-MAIN {k}")
    for k in only_b:
        print(f"  ONLY-BRANCH {k}")
print(f"total: {tot_same} identical, {tot_diff} differ")

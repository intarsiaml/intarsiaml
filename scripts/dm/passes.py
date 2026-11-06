"""Lane dm: summarise an INTARSIA_SPEC_TRACE=1 generate log: passes, tokens a pass, mean draft / verify+rollback ms (passes after
the first 4), the DFlash2 draft's phases, and plain steps' share. Usage: passes.py <stderr log> [--json]"""
import re, sys, json

def summarise(path):
    emits, draft, ver, phases = [], [], [], []
    for line in open(path, errors="replace"):
        m = re.match(r"spec: token \d+ pos \d+ drafted (\d+) accepted (\d+) emits (\d+)", line)
        if m:
            emits.append(int(m.group(3)))
            continue
        m = re.match(r"spec: mtp draft ([\d.]+) ms, verify \+ rollback ([\d.]+) ms", line)
        if m:
            draft.append(float(m.group(1))); ver.append(float(m.group(2)))
            continue
        m = re.match(r"spec: dflash2 draft flush ([\d.]+) block ([\d.]+) head ([\d.]+) selproj ([\d.]+) dl ([\d.]+) select ([\d.]+) total ([\d.]+) ms", line)
        if m:
            phases.append([float(x) for x in m.groups()])
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    d, v, p = draft[4:], ver[4:], phases[4:]
    out = {"passes": len(emits), "tokens_per_pass": mean(emits), "draft_ms": mean(d), "verify_rollback_ms": mean(v),
           "pass_ms": mean(d) + mean(v)}
    if p:
        out["dflash2"] = dict(zip(["flush", "block", "head", "selproj", "dl", "select", "total"], [mean(c) for c in zip(*p)]))
    return out

if __name__ == "__main__":
    s = summarise(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(s)); sys.exit()
    if not s["passes"]:
        print("no spec passes"); sys.exit()
    txt = f"passes {s['passes']} tok/pass {s['tokens_per_pass']:.2f} draft {s['draft_ms']:.2f} verify+rb {s['verify_rollback_ms']:.2f} pass {s['pass_ms']:.2f} ms"
    if "dflash2" in s:
        txt += " [" + " ".join(f"{k} {v:.2f}" for k, v in s["dflash2"].items()) + "]"
    print(txt)

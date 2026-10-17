#!/usr/bin/env python3
"""Median-over-rounds tables from tools/m3v_project.py JSON outputs (proj-ctx<N>.json in the trace dirs)."""
import json, statistics, sys, os

def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None

def table(path, scen, keys):
    d = json.load(open(path))
    out = {}
    for sc in scen:
        rr = d.get(sc)
        if not rr:
            continue
        row = {"rounds": len(rr), "on_agg": med([r["agg"] for r in rr.values()]), "off_agg": med([r["measured_off_agg"] for r in rr.values()])}
        row["on_over_off"] = med([r["agg"] / r["measured_off_agg"] for r in rr.values() if r["measured_off_agg"]])
        for k in keys:
            if all(f"proj_agg_{k}" in r for r in rr.values()):
                row[k] = med([r[f"proj_agg_{k}"] for r in rr.values()])
                row[k + "_vs_on"] = med([r[f"proj_agg_{k}"] / r["agg"] for r in rr.values()])
                row[k + "_vs_off"] = med([r[f"proj_agg_{k}"] / r["measured_off_agg"] for r in rr.values() if r["measured_off_agg"]])
                row[k + "_vs_on_rounds"] = [round(r[f"proj_agg_{k}"] / r["agg"], 4) for r in rr.values()]
        row["today_model_s"] = med([r["today_ms"] / 1000 for r in rr.values()])
        row["wall_s"] = med([r["wall_s"] for r in rr.values()])
        row["comp_r1"] = list(rr.values())[0]["comp"]
        out[sc] = row
    return out

if __name__ == "__main__":
    R = sys.argv[1]
    res = {}
    for d in ("trace", "trace-gateoff"):
        for c in (512, 1024, 4096):
            p = f"{R}/{d}/proj-ctx{c}.json"
            res[f"{d}/ctx{c}"] = table(p, ["grounded-n2", "grounded-n3", "open-n2", "open-n3"], ["g0b", "meas", "none", "kcap", "hypo"])
    json.dump(res, open(f"{R}/summary.json", "w"), indent=1)
    for k, t in res.items():
        print("##", k)
        for sc, r in t.items():
            s = f"{sc}: on {r['on_agg']:.1f} off {r['off_agg']:.1f} on/off {r['on_over_off']:.3f}"
            for key in ("g0b", "meas", "none", "kcap", "hypo"):
                if key in r:
                    s += f" | {key} {r[key]:.1f} (vs on {r[key+'_vs_on']:.3f}, vs off {r[key+'_vs_off']:.3f})"
            print(s)

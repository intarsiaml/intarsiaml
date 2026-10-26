"""Aggregate the dflash_ref.py run JSONs into the small result files under raw/ref/.
  python make_results.py RUNS_DIR OUT_DIR
RUNS_DIR holds dflash_r15.json, dflash_r16s.json, dflash_zlab.json, dcoco_grid1.json, dcoco_combined.json,
first_diff_margin.json, transition_acc.log, plain192.json."""
import json, sys
from pathlib import Path

runs, out = Path(sys.argv[1]), Path(sys.argv[2])
L = lambda n: json.loads((runs / n).read_text())
slim = lambda r: {"variant": r["variant"], "layout": r["layout"], "aggregate": r["aggregate"], "prompts": [
    {k: q[k] for k in ("id", "kind", "prompt_len", "n_rounds", "tau", "tau_committed", "equals_plain_greedy",
                       "first_diff", "sec")} for q in r["prompts"]]}

d15, d16, dz = L("dflash_r15.json"), L("dflash_r16s.json"), L("dflash_zlab.json")
paper = {"dflash_tau": 3.913, "dcoco_tau": 4.985, "note": "ten-suite average, greedy, thinking off (arXiv 2610.12570)"}
dfl = {
    "what": "plain DFlash (argmax drafts) with Qwen3-8B target and the TD-Labs/Qwen3-8B-DCOCO draft (dcoco.* ignored), "
            "greedy, 16 prompts x up to 192 new tokens, tau = mean(accepted drafts per round) + 1",
    "paper_reference": paper,
    "layout_used": "r15",
    "layout_comparison_tau_pooled": {r["runs"][0]["layout"]: r["runs"][0]["aggregate"]["tau_pooled"] for r in (d15, d16, dz)},
    "layout_notes": {"r15": "block [anchor]+14 masks (15 rows), drafts = rows 0..14, row r predicts token anchor+r+1 (USED)",
                     "r16s": "block [anchor]+15 masks (16 rows), drafts = rows 0..14",
                     "zlab": "block [anchor]+15 masks (16 rows), drafts = rows 1..15 (z-lab convention): tau ~1, wrong for this checkpoint"},
    "meta": d15["meta"], "result": d15["runs"][0],
    "target_greedy_ids": {k: v["ids"] for k, v in d15["plain"].items()},
}
(out / "dflash.json").write_text(json.dumps(dfl))

g, c = L("dcoco_grid1.json"), L("dcoco_combined.json")
allruns = g["runs"] + c["runs"] + (L("dcoco_lmax.json")["runs"] if (runs / "dcoco_lmax.json").exists() else [])
rows = [{"variant": r["variant"], **{k: r["aggregate"][k] for k in ("tau_pooled", "tau_mean_of_prompts", "rounds",
         "equal_plain_greedy", "sec")}} for r in allruns]
base = d15["runs"][0]["aggregate"]["tau_pooled"]
best = max(rows, key=lambda x: x["tau_pooled"])
ties = [r["variant"] for r in rows if abs(r["tau_pooled"] - best["tau_pooled"]) < 1e-9]
second = max((r for r in rows if r["tau_pooled"] < best["tau_pooled"] - 1e-9), key=lambda x: x["tau_pooled"])
# are the tied variants identical round by round?
byname = {r["variant"]: r for r in allruns}
ident = {}
for n in ties:
    if n != "default":
        pairs = [(x, y) for a, b in zip(byname["default"]["prompts"], byname[n]["prompts"])
                 for x, y in zip(a["rounds"], b["rounds"])]
        ident[n] = {"rounds_compared": len(pairs), "rounds_with_different_drafts": sum(x["drafts"] != y["drafts"] for x, y in pairs),
                    "rounds_with_different_accepted": sum(x["accepted"] != y["accepted"] for x, y in pairs)}
res = {
    "what": "DCOCO corrector variants, in-loop speculative decoding, same 16 prompts / 192 tokens / layout r15 as dflash.json",
    "paper_reference": paper, "plain_dflash_tau_pooled": base,
    "summary": rows, "winner": "default (eps=1e-5, gelu=erf, order=A [s;za;zb;la;lb], hnorm=post, logit=raw)",
    "tied_at_top": ties, "tied_variants_vs_default_round_diff": ident,
    "winner_tau_pooled": best["tau_pooled"], "gain_over_plain_dflash": best["tau_pooled"] - base,
    "gain_over_plain_dflash_pct": 100 * (best["tau_pooled"] / base - 1),
    "margin_to_next_distinct_variant": {"variant": second["variant"], "delta_tau": best["tau_pooled"] - second["tau_pooled"]},
    "meta": g["meta"], "runs": [slim(r) for r in allruns],
}
(out / "dcoco_variants.json").write_text(json.dumps(res, indent=1))
(out / "first_diff_margin.json").write_text((runs / "first_diff_margin.json").read_text())
(out / "transition_acc.txt").write_text((runs / "transition_acc.log").read_text())
print(json.dumps({k: res[k] for k in ("summary", "tied_at_top", "tied_variants_vs_default_round_diff",
                                      "gain_over_plain_dflash_pct", "margin_to_next_distinct_variant")}, indent=1))
print(dfl["layout_comparison_tau_pooled"])

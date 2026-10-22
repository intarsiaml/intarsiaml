#!/usr/bin/env python3
"""fx6: aggregate the R9700 interleaved visits (raw/hip/<pass>/<arm>-<pass>.result.json, localeval `speed` results).

usage: aggregate-hip.py <raw/hip dir> [--out table.md]

Per arm and prompt size: each visit (one server start, 3 reps) contributes the median of its 3 reps; the cell is the median of the
visit medians, with the band min..max over visits (n = number of visits). Also prints the ratio of every arm to intarsia and whether
the band of the two arms overlap (a ratio whose bands overlap is called a tie). Prefill is localeval's per-request column; the
marginal column (delta tokens / delta TTFT) is reported when localeval marks it reliable.
"""
import glob, json, os, statistics as st, sys

ARM_ORDER = ["intarsia", "intarsia-old", "llama-vk", "llama-vk4", "llama-hip", "hipfire"]
LABEL = {
    "intarsia": "intarsia HIP (origin/main cd617de2)",
    "intarsia-old": "intarsia HIP (2026-10-21 standings binary, pre-J1)",
    "llama-vk": "llama.cpp v0.5.0 Vulkan MXFP4_MOE",
    "llama-vk4": "llama.cpp 0.4.0-dev Vulkan MXFP4_MOE (old comparator)",
    "llama-hip": "llama.cpp v0.5.0 HIP MXFP4_MOE",
    "hipfire": "hipfire ad10b3d MQ4R",
}
SIZES = [128, 1024, 4096, 16384]


def load(d):
    arms = {}
    for f in sorted(glob.glob(os.path.join(d, "*", "*.result.json"))):
        base = os.path.basename(f)[: -len(".result.json")]
        if "-mmlu-" in base:
            continue
        arm, _, ps = base.rpartition("-")
        j = json.load(open(f))
        rows = {r["target_prompt_tokens"]: r for r in j["speed"]["rows"]}
        arms.setdefault(arm, []).append((ps, rows))
    return arms


def cell(visits, size, key):
    vals = []
    for _, rows in visits:
        r = rows.get(size)
        if not r:
            continue
        v = r.get(key)
        if isinstance(v, dict):
            v = v.get("median")
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return st.median(vals), min(vals), max(vals), len(vals)


def fmt(c, nd=0):
    if c is None:
        return "-"
    m, lo, hi, n = c
    return f"{m:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f}, n={n})"


def main():
    d = sys.argv[1]
    arms = load(d)
    order = [a for a in ARM_ORDER if a in arms]
    out = []
    for title, key, sizes, nd in (("Decode tok/s, single stream", "decode_tok_s", SIZES, 1),
                                  ("Prefill tok/s (per request; 1k prompt has 990 tokens so TTFT overhead is included)", "prefill_tok_s", SIZES[1:], 0),
                                  ("Marginal prefill tok/s (delta tokens / delta TTFT)", "marginal_prefill_tok_s", SIZES[1:], 0),
                                  ("TTFT s", "ttft_s", SIZES[1:], 2)):
        out.append(f"\n**{title}**: median of visit medians (min-max over visits, n = visits of 3 reps each)\n")
        out.append("| engine | " + " | ".join(str(s) for s in sizes) + " |")
        out.append("|---|" + "---|" * len(sizes))
        for a in order:
            out.append(f"| {LABEL[a]} | " + " | ".join(fmt(cell(arms[a], s, key), nd) for s in sizes) + " |")
    out.append("\n**Ratio to intarsia (new binary), medians; 'tie' when the min-max bands of the two arms overlap**\n")
    out.append("| engine | metric | " + " | ".join(str(s) for s in SIZES[1:]) + " |")
    out.append("|---|---|" + "---|" * len(SIZES[1:]))
    for a in order:
        if a == "intarsia":
            continue
        for key, nm in (("decode_tok_s", "decode (ours / theirs)"), ("prefill_tok_s", "prefill (ours / theirs)")):
            cells = []
            for s in SIZES[1:]:
                o, t = cell(arms["intarsia"], s, key), cell(arms[a], s, key)
                if not o or not t:
                    cells.append("-")
                    continue
                tie = not (o[1] > t[2] or t[1] > o[2])
                cells.append(f"{o[0] / t[0]:.2f}x" + (" tie" if tie else ""))
            out.append(f"| {LABEL[a]} | {nm} | " + " | ".join(cells) + " |")
    txt = "\n".join(out)
    print(txt)
    if "--out" in sys.argv:
        open(sys.argv[sys.argv.index("--out") + 1], "w").write(txt + "\n")


if __name__ == "__main__":
    main()

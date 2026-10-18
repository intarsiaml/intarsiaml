import json, glob, sys, statistics as st
P = sys.argv[1] if len(sys.argv) > 1 else "main"  # "main" = first batch, "retry" = second batch
R = "/opt/code/intarsiaml/docs/results/raw/2026-10-18-metal-sameslot-main/localeval-results"
order = [f"{P}-1", f"{P}-mlx-lm", f"{P}-llamacpp", f"{P}-2"]
D = {}
for k in order:
    p = glob.glob(f"{R}/*adhoc-{k}/result.json")[0]
    d = json.load(open(p))
    D[k] = (d["run_id"], d["speed"]["rows"])


def fmt(x, d=1):
    return f"{x['median']:.{d}f} ({x['min']:.{d}f}-{x['max']:.{d}f})"


print("| run | run id | prompt tok (128/1k/4k) | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k | prefill tok/s 1k | prefill tok/s 4k |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k in order:
    rid, rows = D[k]
    print(f"| {k} | {rid} | {'/'.join(str(r['prompt_tokens']) for r in rows)} | "
          + " | ".join(fmt(r['decode_tok_s']) for r in rows)
          + f" | {rows[1]['marginal_prefill_tok_s']:.0f} | {rows[2]['marginal_prefill_tok_s']:.0f} | {fmt(rows[1]['prefill_tok_s'], 0)} | {fmt(rows[2]['prefill_tok_s'], 0)} |")


def agg(names, i, f):
    return st.mean(f(D[n][1][i]) for n in names)


M = [f"{P}-1", f"{P}-2"]
FLOOR = 40.6
print()
mlx = D[f"{P}-mlx-lm"][1]
lla = D[f"{P}-llamacpp"][1]
for nm, g in (("MAIN", M),):
    print(f"## {nm} (mean of two runs)")
    for i, lab in ((1, "1k"), (2, "4k")):
        m = agg(g, i, lambda r: r["marginal_prefill_tok_s"])
        p = agg(g, i, lambda r: r["prefill_tok_s"]["median"])
        print(f"prefill {lab}: marginal {m:.0f} = {100*m/mlx[i]['marginal_prefill_tok_s']:.1f}% mlx-lm ({mlx[i]['marginal_prefill_tok_s']:.0f}), "
              f"{100*m/lla[i]['marginal_prefill_tok_s']:.1f}% llama.cpp ({lla[i]['marginal_prefill_tok_s']:.0f}); "
              f"prefill tok/s {p:.0f} = {100*p/mlx[i]['prefill_tok_s']['median']:.1f}% mlx ({mlx[i]['prefill_tok_s']['median']:.0f}), "
              f"{100*p/lla[i]['prefill_tok_s']['median']:.1f}% llama ({lla[i]['prefill_tok_s']['median']:.0f})")
    for i, lab in ((0, "128"), (1, "1k"), (2, "4k")):
        d = agg(g, i, lambda r: r["decode_tok_s"]["median"])
        print(f"decode {lab}: {d:.2f} = {100*d/FLOOR:.1f}% floor, {100*d/mlx[i]['decode_tok_s']['median']:.1f}% mlx ({mlx[i]['decode_tok_s']['median']:.1f}), "
              f"{100*d/lla[i]['decode_tok_s']['median']:.1f}% llama ({lla[i]['decode_tok_s']['median']:.1f})")
    a, b = g
    for i, lab in ((0, "128"), (1, "1k"), (2, "4k")):
        d1, d2 = D[a][1][i]["decode_tok_s"]["median"], D[b][1][i]["decode_tok_s"]["median"]
        line = f"drift {lab}: decode {d1:.1f}->{d2:.1f} ({100*(d2-d1)/d1:+.1f}%)"
        if i > 0:
            m1, m2 = D[a][1][i]["marginal_prefill_tok_s"], D[b][1][i]["marginal_prefill_tok_s"]
            line += f"; marginal prefill {m1:.0f}->{m2:.0f} ({100*(m2-m1)/m1:+.1f}%)"
            p1, p2 = D[a][1][i]["prefill_tok_s"]["median"], D[b][1][i]["prefill_tok_s"]["median"]
            line += f"; prefill tok/s {p1:.0f}->{p2:.0f} ({100*(p2-p1)/p1:+.1f}%)"
        print(line)

print()
print("## Marginal prefill bounds (per-rep TTFT is not stored by localeval, only min/median/max, so the range is the extreme bound:")
print("## dtok / (ttft_max[i] - ttft_min[i-1]) .. dtok / (ttft_min[i] - ttft_max[i-1]); the median column is the localeval figure)")
print("| run | marginal 1k median (bound) | marginal 4k median (bound) |\n|---|---|---|")
for k in order:
    rows = D[k][1]
    cells = []
    for i in (1, 2):
        dt = rows[i]["prompt_tokens"] - rows[i-1]["prompt_tokens"]
        lo = dt / (rows[i]["ttft_s"]["max"] - rows[i-1]["ttft_s"]["min"])
        hi = dt / (rows[i]["ttft_s"]["min"] - rows[i-1]["ttft_s"]["max"])
        cells.append(f"{rows[i]['marginal_prefill_tok_s']:.0f} ({lo:.0f}-{hi:.0f})")
    print(f"| {k} | " + " | ".join(cells) + " |")

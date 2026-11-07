Coordinator go-ahead 2026-11-07 04:25 EST (user: 'keep building overnight'; radiance takedown approved 11-05, max 75 min per window): one traced serve window on tatooine to settle whether INTARSIA_SPEC_GRAPH_TP engaged in gt2's speed runs; no default flips; radiance back by 06:00.

# Traced serve window: did `INTARSIA_SPEC_GRAPH_TP=1` engage in the speed runs? (lane gt3, 2026-11-07)

Measurement only, no code change, no default flip. Question left by lane gt2 (`2026-11-07-spec-graph-tp2.md`): its localeval serve speed arms were untraced, so
"35B tp2 MTP k2 at 4k, 223.7 eager vs 219.1 with the knob (0.98x)" did not prove the captured step ran.

## Method

One fleetq job (#9024, both R9700s, `--takes-down radiance`), `scripts/gt3/window.sh` (copy of gt2's): a fresh `serve --max-batch 1` per arm, Qwen3.6-35B-A3B-NVFP4,
`--tp 2 --devices 0,1 --spec mtp --spec-k 2 --max-seq 20480 --mem-budget 29`, `localeval speed --gen 128 --reps 3`, both arms with `INTARSIA_SPEC_TRACE=1`
(the `spec: mtp graph step` line is printed per captured step only; the eager step prints its `spec: token .. drafted ..` line only). The knob arm adds
`INTARSIA_SPEC_GRAPH_TP=1`. Engagement is counted from the container log of each arm (the whole arm, warm-up request included). Build: tree `~/intarsiaml-gt3` at
main `0f0102a22` + the scripts (separate `--load cpu` job #9023, removed afterwards). Order: round 1 knob alone (1k + 4k), then 4k pairs BA, AB, BA.

## Engagement (knob arms)

| arm | verified steps | captured (`mtp graph step`) | capture failures | eager-fallback lines |
|---|---|---|---|---|
| r1 (1k + 4k, 6 requests) | 332 | 331 | 0 | 0 |
| r2 (4k, 3 requests) | 170 | 170 | 0 | 0 |
| r3 | 161 | 160 | 0 | 0 |
| r4 | 161 | 160 | 0 | 0 |

The captured step is engaged in the serve path at tp2: 1321 of 1324 steps captured (the one non-captured step per arm is the first, right after prefill), no capture
failure, no fallback. Captured step time about 10.7-10.9 ms. The eager arms print no graph-step line (0 of 328, 314, 328 steps). So the cause is not "not engaged":
serve and `generate` take the same `spec_step_ok` path (max-batch 1, `st.mtp` active, `verify_attn_per_row`).

## Speed, 4k decode tok/s (3 reps per arm; `eager` = knob off, `knob` = on)

| round | eager reps | eager mean | knob reps | knob mean |
|---|---|---|---|---|
| r1 | not run | | 208.3 200.8 224.7 | 211.3 |
| r2 (eager first) | 216.3 194.9 217.9 | 209.7 | 193.2 195.4 226.0 | 204.9 |
| r3 (knob first) | 202.7 226.6 215.2 | 214.8 | 184.9 185.4 203.2 | 191.2 |
| r4 (eager first) | 194.5 188.8 226.1 | 203.1 | 187.4 233.8 217.5 | 212.9 |

Paired rounds r2-r4: eager 209.2 mean, knob 203.0 mean, ratio 0.97x (per-round ratios 0.98, 0.89, 1.05). Rep-to-rep spread within one arm is up to 40 tok/s, larger
than the arm difference: with engagement confirmed, the captured step at tp2 is **not measurably faster** than the eager MTP step (and not measurably slower).
This confirms gt2's verdict (0.98x) with the engagement question closed. 1k, knob only (r1): 227.8 226.2 206.3. The gt2 raw 1k/16k eager numbers were not repeated.

Draft acceptance is the same in both: about 0.64 accepted per proposed token (knob 210-219 per 320-340 proposed; eager 215-223 per 314-327).

## Radiance

Stopped 04:27:09, answering 04:30:52 (guard.log: 20 s after start): 3 min 43 s downtime. Raw: `2026-11-07-spec-graph-tp2-traced/raw/w1/` (per-arm full container logs,
speed logs, spec-metrics, guard.log). Note: the first run of `window.sh` wrote into gt2's raw directory (path copied from gt2); the files were moved here and gt2's restored
from git, and the script path is fixed.

No default flip; the knob stays opt-in. The next lever for tp2 MTP speed is not the replay of the step but what it costs (the id all-gather and thread barrier per step, gt2).

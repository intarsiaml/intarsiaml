# Metal continuous batching: M3 stage 0, batched speculative verify go / no-go

Date 2026-10-17. Branch `metal-cb-m3-verify` (local, not pushed) off `spike/metal` d793300. Machine endor (Apple M5 Max), Qwen3-8B BF16. Measurement and projection only: nothing under `crates/*/src` changed. New: example `crates/intarsia-models/examples/m3_verify_cost.rs`, tools `tools/m3v_*.py` and `tools/m3v_cost_slot.sh`, raw data `docs/results/raw/2026-10-17-metal-cb-m3-verify-go/`. Binary for the traces: d793300, `cargo build --release -p intarsia-cli --features metal` (sha256 prefix `4b7c728de731fd2f`), Xcode toolchain.

## Verdict: NO-GO

Against the bar fixed before the run (GO needs grounded aggregate gain >= 10 % over d793300 spec-on at B=2 or B=3 **and** open-ended >= 0.98x of spec off):

| criterion | projected | bar | result |
|---|---|---|---|
| grounded B=2, batched verify (k = 4, R <= 6) against d793300 spec-on | **+1.3 %** (1.013; rounds 1.013 / 1.012 / 1.013) | >= +10 % | **fails** |
| grounded B=3 | **+1.2 %** (1.012; 1.013 / 1.012 / 1.012) | >= +10 % | **fails** |
| open-ended B=2, spec left on, against spec off | 0.988 | >= 0.98 | passes (narrowly; 0.984 with no attention correction) |
| open-ended B=3, spec left on, against spec off | **0.852** | >= 0.98 | **fails** |

The grounded gain is small because of how the scheduler schedules, not because the batched pass is slow. A batched pass of 6 rows is cheap (section 2: for one speculating sequence plus its neighbours it would save 41-47 % of that iteration). But in `grounded-n2` and `grounded-n3` the sequences almost never need the GPU in the same iteration: a sequence that verifies emits the rest of its 4-5 tokens over the next iterations from its pending queue with no row (`Seq::prepare`), the streams settle into the same period at different offsets, and only 6 of 124 iterations with GPU work (B=2) and 16 of 178 (B=3) hold two or more verify groups. Everything else is one solo verify pass alone in its iteration, which batched verify would run unchanged. Two 5-row groups (R = 10) cannot share a pass under `R_step_max` = 6 either. Where a group does meet a plain row (open-ended with speculation forced on, section 4), batched verify recovers most of a loss, but that is not a gain over spec off.

Nothing above was tuned after the data: the method (ctx 512 for the primary costs, the attention correction, the packing rule, the metric) is the commit before the data, `04f6629`.

## 1. What was measured

All GPU work in three exclusive fleetq slots on endor (`--exclusive --mem 24`): **#2260** (cost slot: `m3_verify_cost` at ctx 512 / 1024 / 4096, 10 rounds x 12 reps, kinds interleaved in rotated order; `g0b_bench attn` at the same contexts), **#2261** (d793300 iteration traces, default gate: arms off and on, 3 rounds, scenarios solo-grounded, solo-open, grounded-n2/n3, open-n2/n3, `INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1`, via `tools/cbm3_slot.sh`), **#2262** (the same with the gate disabled, `INTARSIA_SPEC_GATE=0,2,32,2,0,0`, so open-ended streams keep drafting; solo-open, open-n2/n3, grounded-n2/n3). All three finished exit 0 and left no runner or server (`no runner left`, `no intarsia serve left`). macmon at 100 ms in each.

**Clock.** Cost slot (#2260): GPU clock during the samples median 1475 / 1475 / 1489 MHz at ctx 512 / 1024 / 4096 (p10 1364 / 1331 / 1339, p90 1616 / 1600 / 1617). Samples whose macmon clock was below 1400 MHz were dropped as pre-registered (84 % / 75 % / 79 % kept); kept and unfiltered medians differ by at most 0.5 ms (0.9 ms for verify6 at 4k). Trace slots: busy-GPU median 1447 MHz (#2261, p10 1290, p90 1617) and 1544 MHz (#2262, p10 1308, p90 1617); per arm, round and scenario the median clock ranges 1248-1617 MHz (`trace/clock.txt`, `trace-gateoff/clock.txt`). The trace walls were not clock-filtered (rule 8 applied to the cost tables only). The round-to-round differences in the tables below (grounded-n2 on: 127.9 / 119.8 / 124.7 tok/s) follow that clock spread.

## 2. Component times

#### Component times (ms, median)

| pass | ctx 512 | ctx 1024 | ctx 4096 |
|---|---:|---:|---:|
| plain1 | 28.01 | 27.79 | 28.30 |
| plain2 | 28.63 | 28.22 | 29.94 |
| plain3 | 29.64 | 28.96 | 31.53 |
| plain4 | 30.31 | 30.23 | 33.29 |
| plain5 | 31.60 | 31.53 | 35.53 |
| plain6 | 33.22 | 33.55 | 38.30 |
| verify2 | 27.98 | 28.24 | 29.20 |
| verify3 | 28.25 | 28.72 | 29.85 |
| verify4 | 28.83 | 29.10 | 30.65 |
| verify5 | 29.52 | 29.62 | 31.54 |
| verify6 | 30.93 | 31.01 | 33.64 |

#### Today's iterations: measured composite against the sum of its parts (ms, ctx 512 / 1024 / 4096)

| iteration | measured | sum of parts | ratio |
|---|---|---|---|
| iter_b2_k2 | 56.19 / 56.37 / 58.08 | 56.26 / 56.50 / 58.16 | 0.999 / 0.998 / 0.999 |
| iter_b2_k3 | 56.76 / 56.52 / 58.71 | 56.83 / 56.89 / 58.95 | 0.999 / 0.993 / 0.996 |
| iter_b2_k4 | 57.64 / 57.02 / 59.47 | 57.53 / 57.41 / 59.84 | 1.002 / 0.993 / 0.994 |
| iter_b3_k2 | 56.99 / 56.91 / 59.32 | 56.88 / 56.94 / 59.79 | 1.002 / 1.000 / 0.992 |
| iter_b3_k3 | 57.53 / 57.39 / 60.45 | 57.46 / 57.32 / 60.58 | 1.001 / 1.001 / 0.998 |
| iter_b3_k4 | 58.08 / 58.13 / 61.80 | 58.15 / 57.84 / 61.47 | 0.999 / 1.005 / 1.005 |
| iter_vv2_k4 | 59.09 / 59.47 / 63.62 | 59.05 / 59.24 / 63.07 | 1.001 / 1.004 / 1.009 |
| iter_vv3_k4 | 88.90 / 89.41 / 95.72 | 88.57 / 88.86 / 94.61 | 1.004 / 1.006 / 1.012 |

#### Attention: what a q-row verify group saves against q single-row launches inside a plain step (ms per pass)

| q | g0b-derived ctx 512 | g0b-derived ctx 1024 | g0b-derived ctx 4096 | measured P(q)-V(q) ctx 512 | measured P(q)-V(q) ctx 1024 | measured P(q)-V(q) ctx 4096 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.37 | 0.14 | 0.76 | 0.65 | 0.00 | 0.74 |
| 3 | 0.61 | 0.36 | 1.61 | 1.38 | 0.25 | 1.68 |
| 4 | 0.86 | 0.79 | 2.34 | 1.48 | 1.13 | 2.65 |
| 5 | 1.10 | 1.18 | 3.12 | 2.08 | 1.91 | 4.00 |
| 6 | 1.40 | 1.58 | 3.82 | 2.30 | 2.53 | 4.66 |

#### g0b_bench attn, us per layer (this slot): T1 = serving 1-row; T(q) = rows kernel

| ctx | T1 | T(2) | T(3) | T(4) | T(5) | T(6) |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 11.3 | 12.2 | 16.8 | 21.3 | 25.8 | 28.8 |
| 1024 | 14.6 | 25.3 | 33.7 | 36.3 | 39.9 | 43.4 |
| 4096 | 36.6 | 52.2 | 65.2 | 81.5 | 96.5 | 113.5 |

Reading: the costs add (today's iteration = V(1+k) + P(B-1) within 1 %, lockstep verifies = N x V(5) within 1.2 %). A solo verify of q rows is cheaper than a plain step of q sequences because the q rows share one KV cache; the measured difference P(q) - V(q) is larger than the g0b-derived estimate, so the pre-registered g0b correction is the smaller of the two (the conservative one for a GO). Weight pass R = 6 is 33.2 ms against 28.0 ms for one row at ctx 512 (+19 %); R = 5 is +13 %.

### One speculating sequence among B (the brief's model), today vs one batched pass

ms, ctx 512 / 1024 / 4096:
| B | k | rows R=B+k | today V(1+k)+P(B-1) | batched pass P(R) - corr(g0b) | saving |
|---|---|---|---|---|---|
| 2 | 2 | 4 | 56.3 / 56.5 / 58.2 | 29.7 / 29.9 / 31.7 | 47% / 47% / 46% |
| 2 | 3 | 5 | 56.8 / 56.9 / 59.0 | 30.7 / 30.7 / 33.2 | 46% / 46% / 44% |
| 2 | 4 | 6 | 57.5 / 57.4 / 59.8 | 32.1 / 32.4 / 35.2 | 44% / 44% / 41% |
| 3 | 2 | 5 | 56.9 / 56.9 / 59.8 | 31.0 / 31.2 / 33.9 | 46% / 45% / 43% |
| 3 | 3 | 6 | 57.5 / 57.3 / 60.6 | 32.4 / 32.8 / 36.0 | 44% / 43% / 41% |
| 3 | 4 | 7 | does not fit one pass (R > 6): 2 passes | | |

## 3. Grounded projection (the bar)

Method: the DecodePhase / `spec:` / Decode records of each `on` run of #2261 give every scheduler iteration its verify groups (q = 1 + drafted) and plain rows; each iteration is costed as today (sum V(q) + P(plain)) and as batched verify (first-fit-decreasing packing into passes of <= 6 rows; a pass of one group alone costs V(q), a mixed pass P(R) minus the attention correction). Projected wall = measured wall minus (modelled today minus modelled batched). Median of 3 rounds, ctx 512 costs:

| | spec off | d793300 spec-on (measured) | on / off | projected batched verify (g0b correction) | gain over spec-on | gain, measured correction | gain, no correction | gain at ctx 1024 / 4096 |
|---|---|---|---|---|---|---|---|---|
| grounded B=2 | 67.5 | 124.7 | 1.847 | **126.3** | **+1.3 %** | +1.3 % | +1.2 % | +1.2 % / +1.2 % |
| grounded B=3 | 95.9 | 123.3 | 1.286 | **124.8** | **+1.2 %** | +1.3 % | +1.2 % | +1.2 % / +1.2 % |

(tok/s, aggregate tokens over wall, prefill included, as the baseline. The busy-probe doc's 129.3 / 128.8 tok/s and 1.891 / 1.326 are the same arm in another slot; these three rounds ran with the trace on and clocks of 1248-1617 MHz.)

Iteration compositions of round 1 (verify groups + plain rows : iterations): B=2 `[5]+0` 114, `[5,5]+0` 6, `[5]+1` 2, `[]+1` 2; B=3 `[5]+0` 157, `[5,5]+0` 14, `[5,5,5]+0` 1, `[5]+1` 2, `[5,5]+1` 1, `[]+1` 3 (a group is written by its row count q = 5 = 1 + 4 drafts). Passes today 132 / 198, passes batched 130 / 195 (rounds 2 and 3 identical, round 1 of B=3 199 / 196). The three rounds have the same shape (`trace/proj-ctx*.txt`, `.json`).

Model check: modelled today decode time 3.89 s (B=2) and 5.84 s (B=3) against measured walls 4.11 s and 6.23 s (medians of rounds); the remainder is prefill, host work and the trace's own cost, 5-6 %. On/off 1.847 / 1.286 here against 1.891 / 1.326 in the busy-probe doc.

### Labelled variants (not the k = 4 design, not the bar)

Both change the draft length or the schedule; neither is part of the verdict.

* **k shrunk only when an iteration does not fit** (largest group first until R <= 6; accepted = min(a, k'); a rate approximation): +4.1 % (B=2), +6.7 % (B=3).
* **Hypothetical aligned pass**: every speculating sequence verifies in the same pass with k capped so R <= 6 (k = 2 at B=2, k = 1 at B=3), which needs the scheduler to align the verifies (today they are staggered and it does not). A pass that accepted a drafts at k = 4 is assumed to become ceil((a+1)/(k'+1)) capped passes of the same tokens (a matched run continues across the cut; 2.32 and 1.63 tokens per capped pass per stream). Result: +10.1 % (B=2; 1.104 / 1.096 / 1.101 per round) and +15.2 % (B=3) at ctx 512, +7.8 % / +12.2 % at ctx 1024. It rests on the run-splitting assumption, on an alignment mechanism that does not exist, and on the same unbuilt mixed shape; it states what would have to be true and is not a projection of the code that was asked about.

## 4. Open-ended projection

d793300 already turns drafting off after the warm-up pass for open-ended streams (busy rule), so its open-ended aggregate is spec off to within noise (this slot: 1.021 at B=2 and 0.997 at B=3, medians of 3 rounds; the busy-probe doc 1.016 / 1.001). Batched verify cannot improve on that; the question is whether it lets drafting stay on without loss. #2262 disables the gate so the open-ended streams draft whenever the n-gram has a match (min suffix 2, k = 4), and the same projection is applied to those iterations against the spec-off arm of the same slot:

| | spec off | gate off, today (measured) | today on / off | projected batched verify, spec left on | / spec off (g0b) | measured corr. | no corr. | ctx 1024 / 4096 |
|---|---|---|---|---|---|---|---|---|
| open-ended B=2 | 70.3 | 61.9 | 0.885 | 69.1 | **0.988** | 0.992 | 0.984 | 0.987 / 0.985 |
| open-ended B=3 | 99.3 | 84.2 | 0.840 | 85.0 | **0.852** | 0.858 | 0.847 | 0.851 / 0.852 |

At B=2 a verify group meets the other stream's plain row in 38 of 255 iterations (`[5]+1`, R = 6, fits) and the loss shrinks from 11.5 % to 1.2 %; it does not go away, because a 6-row pass costs 16 % more than a 2-row pass (P(6) 33.2 against P(2) 28.6 ms) and open-ended text accepts about 0.06 tokens per draft (11 of 179 proposed at B=2). At B=3 `[5]+2` (R = 7) occurs in 46 iterations and does not fit, so it is two passes, and the loss stays at 15 %. The k-shrink variant (never more than 6 rows) would bring B=3 to 1.011 of spec off, but only by drafting less; it gains nothing over spec off either.

## 5. Where batched verify would pay (analytic, outside the bar)

The iteration the brief describes, one speculating sequence next to plain rows, is where the single pass wins (section 2 table: 41-47 % off that iteration). It does not occur in `grounded-nK`, where every stream speculates. It occurs when a grounded stream runs beside open-ended ones. An analytic cycle model (not traced, not measured end to end, not one of the pre-registered scenarios): the grounded stream verifies once per 3.95 tokens and drains the rest with no row, each open stream needs a row in every iteration; ctx 512: B=2 (1 grounded + 1 open) 56.4 -> 68.8 tok/s (**+22 %**); B=3 (1 grounded + 2 open) 83.1 -> 82.0 tok/s (-1 %: R = 7 splits into two passes at k = 4; k = 3 would fit and was not modelled because it changes the acceptance). If the decision rests on this mixed case rather than on `grounded-nK`, it needs its own traced run first.

## 6. Exactness (Q1): mixed shapes are not shown row-exact

Every projected mixed pass (B plain rows next to a 1+k verify group, up to 6 rows) is an **exactness risk until shown**. Established, individually:

* the R-row GEMVs, bitwise equal to the 1-row kernel per row for R <= 8 (G0, 16-byte aligned and 8-mod-16 shards);
* the row-parallel small ops (RmsNorm, RmsNormAdd, Add, SiluGlu, Embedding, Argmax, QkNormRopeKv including the KV write; `g0b_small`, k 2..8);
* the rows split-KV attention reproducing the serving 1-row kernel row by row on a single cache (7 `g0b_attn` shapes, k 2..8, host positions).

These are in use today in `forward_batch` and `forward_verify`, each in its own pass. **Not shown anywhere, and not built:**

* one plan that dispatches the per-row 1-row `AttnDecode` path for the plain rows and the `AttnPrefill` rows kernel (priority -1, `T` rows over one slot's cache) for the group in the same step, under one `exact_rows` scope (today `forward_batch` and `forward_verify` each set and clear the flag for their own pass: `model/batch.rs:147,153`, `model.rs:788,793`);
* the QkNormRopeKv / KV-write layout of a mixed step (a q-row group at `pos0..pos0+q` of one slot, 1-row positions for the others);
* the verify head (rows `[0,q)` argmax into `verify_tok`) combined with the batch head for the plain rows;
* the rows attention in device-position mode (g0b tested host position only).

The timings above do not touch any of this: they time the two existing paths separately and add them (and the composite check shows they add), so they cannot show it either. The bitwise gate for a future build is the existing one: mixed step vs each request's solo run, verify rows and plain rows, at every composition up to R = 6.

## 7. Assumptions and uncertainty

* **Proxy.** No mixed pass was run (the code does not exist). A mixed pass of R rows is costed as the measured plain R-sequence step minus the attention correction. The plain step is R single-row attention launches over R caches; the real pass has q rows over one cache for the group, so the proxy overstates attention. The g0b-derived correction (0.4-1.4 ms at ctx 512 for q 2..6) is the smaller of two estimates; the end-to-end one (P(q) - V(q), 0.65-2.3 ms) is larger. The verdicts do not move between "none", g0b and measured (grounded +1.2 / +1.3 / +1.3 %, open B=3 0.847 / 0.852 / 0.858). The g0b figures are chain-measured and cache-warm; they were re-measured in slot #2260 at the same contexts, not taken from the older doc.
* **Schedule held fixed.** The projection re-costs the iterations d793300 actually ran, with the same drafts and acceptance. Batched verify would not change the tokens produced, but it would change timing and therefore which streams meet in an iteration; that second-order effect is not modelled (the hypothetical aligned variant in section 3 is the only look at it).
* **Contexts.** Primary costs at ctx 512 because the prompts are 385 tokens plus up to 256 generated (grounded) and about 30 plus 256 (open). Sensitivity at 1024 and 4096 moves the grounded gain by < 0.1 point and open B=2 by 0.3 point.
* **Clock and noise.** Cost medians are from kept samples above 1400 MHz (clock median 1475-1489 MHz); per-pass p10-p90 spread is 3-6 %. The trace runs sit at 1248-1617 MHz per window, so measured aggregates move 5-8 % between rounds. The projected grounded gains are model deltas applied to the same run's wall and are stable across rounds (1.012-1.013). The open-ended ratios compare an on-arm run with an off-arm run of the same slot and carry the +-3-5 % per-pair scatter of the earlier docs (per-round open B=2 g0b ratios to off: 0.988 median, rounds 1.004 / 0.987 / 0.983).
* **Trace cost.** `INTARSIA_CB_TRACE=1` prints one or two lines per iteration; on/off here (1.847 / 1.286) is 2-3 % under the busy-probe doc's untraced run, within the clock spread. The iteration compositions are the same in all three rounds.
* **Single model.** Qwen3-8B BF16 only; the two m3 prompts with a nonce line per stream; 256 tokens.
* Not done: no mixed grounded + open run (section 5 is analytic), no direct measurement of a mixed pass (nothing to run), no repeat of the cost slot.

## fleetq jobs and commits

Jobs (all mine, `--host endor --exclusive --mem 24`): #2260 cost slot, #2261 default-gate traces, #2262 gate-off traces. Commits on `metal-cb-m3-verify`: `04f6629` pre-registered method, cost example and slot script; the result commit follows it (`git log`).

## Reproduce

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
CARGO_TARGET_DIR=$S/target cargo build --release -p intarsia-cli --features metal
CARGO_TARGET_DIR=$S/target cargo build --release -p intarsia-models --example m3_verify_cost -p intarsia-backend-metal --example g0b_bench
EX=$S/target/release/examples/m3_verify_cost G0B=$S/target/release/examples/g0b_bench fleetq run --host endor --exclusive --mem 24 --label "..." -- tools/m3v_cost_slot.sh OUT/cost
BIN=$S/target/release/intarsia ARMS="off on" SCEN=solo-grounded,solo-open,grounded-n2,grounded-n3,open-n2,open-n3 COOLDOWN=20 MAXMIN=150 SERVER_ENV="INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1" fleetq run ... -- tools/cbm3_slot.sh client OUT/trace 3
(same with SCEN=solo-open,open-n2,open-n3,grounded-n2,grounded-n3 and SERVER_ENV="... INTARSIA_SPEC_GATE=0,2,32,2,0,0" for OUT/trace-gateoff)
python3 tools/m3v_cost.py OUT/cost --json OUT/cost/cost.json --md OUT/cost/cost.md
python3 tools/m3v_project.py trace --dir OUT/trace --cost OUT/cost/cost.json --ctx 512 --out OUT/trace/proj-ctx512.json   # also trace-gateoff, ctx 1024 / 4096
python3 tools/m3v_summary.py OUT ; python3 tools/m3v_tables.py OUT/cost/cost.json ; python3 tools/m3v_clock.py OUT/trace
```

## Method as committed before the data

(`04f6629`, unchanged. Rule 8, the 1400 MHz filter, was applied to the cost tables and not to the trace walls; the traces report the clock per window instead.)

Go bar (given by the coordinator, fixed): **GO** only if the projected grounded aggregate gain is >= 10 % over d793300 spec-on at B=2 or B=3 AND the projected open-ended aggregate is >= 0.98x of spec off; anything else is **NO-GO**, numbers reported, the user decides.

Projection rules fixed here, so that no assumption is chosen after seeing the result:

1. **Iteration compositions come from d793300 itself**, not from a model of it: `INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1`, scenarios grounded-n2/n3 and open-n2/n3 through `tools/cbm3_slot.sh`. Per scheduler iteration the trace gives the verifying sequences (their drafted count d, so q = 1 + d rows each) and the plain rows. A sequence that verified, or that drains pending accepted tokens, has no row in the batched step of that iteration (`Seq::prepare`, `SeqAdapter::prepare`).
2. **Today's cost of an iteration** = sum of solo verify passes V(q) + one batched step of the plain rows P(n) (P(1) = the solo step). Checked against the measured composite iterations and against the measured wall of the same run.
3. **Batched-verify cost** of the same iteration (same drafts, same acceptance, k = 4 as shipped): rows R = sum q + plain rows. R <= 6: one pass costing P(R) minus the attention correction below. R > 6: groups are packed first-fit-decreasing into passes of <= 6 rows (plain rows fill the gaps), each pass costed the same way; a pass that holds one group and nothing else costs V(q), a pass of plain rows only costs P(n). `R_step_max` is 6 (Q14).
4. **Attention correction** (explicit assumption): a plain R-row step runs R single-row split-KV launches over R separate caches; a verify group of q rows runs the rows kernel over one cache, which g0b measured cheaper than q single-row launches. The correction per group is 36 x (q x T1 - T(q)) with T1 and T(q) from `g0b_bench attn` re-run at the same context in the same slot. A second, measured correction V(q) - P(q) (end to end) is reported next to it as a cross-check, and "no correction" as a pessimistic bound.
5. **Context for the primary numbers**: the grounded prompt is 371 tokens plus up to 256 generated, the open prompt about 30 plus 256, so the primary costs are those at ctx 512. ctx 1024 and 4096 are reported as sensitivity only.
6. **Metric**: the baseline 129.3 / 128.8 tok/s are aggregate tokens over wall (prefill and TTFT included). Projected wall = measured wall of the same run - (modelled today decode time - modelled batched decode time). A decode-only speedup is never divided into the aggregate.
7. **Open-ended**: d793300 turns drafting off after the warm-up pass, so its traces hold no draft statistics. Drafting behaviour with the gate disabled (`INTARSIA_SPEC_GATE=0,2,32,2,0,0`) is traced separately and used to simulate batched verify with speculation left on (rule 3 for packing and cost), against spec off.
8. Clock: samples taken below 1400 MHz GPU (macmon, 100 ms) are dropped from the cost tables; the clock of every run used is reported.
9. Exactness (Q1): every mixed shape is listed with what is and is not shown row-exact; none is claimed covered by this stage.

# Metal continuous batching: M3 stage 0, batched speculative verify go / no-go

Date 2026-10-17. Branch `metal-cb-m3-verify` off `spike/metal` d793300. Measurement and projection only: no file under `crates/*/src` changed.

## Pre-registered method (committed before any data was taken)

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

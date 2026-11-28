# Inkling-2-Mini on Metal: DFlash2 candidates by the device top-k (lane ink-metal-dtopk, opt-in, 2026-11-28)

Status: **all gates PASS (slot D, endor exclusive #12927); opt-in.** Candidates bit-identical on every row (1429 checked passes), ids
= goldens, serve unchanged. Decode: the pre-registered rule says GO at 1.025 (bar 1.01), but two speed steps fell on default -> top-k
transitions; runs side by side in the same power state give **+0.5-1.0 %**, which matches the host time saved (D0: 0.09 ms a pass).
Default-on is the user's call, and the gain is small.

## Why

Each speculative pass on Metal takes the DFlash2 drafter's 16 candidates per block row on the host. With the Metal default since #557
(k7 + cut 0.3), the host fetches one row's f32 logits at a time (200058 x 4 bytes, a download and a synchronisation each) until
the cut stops the walk, and takes each row's top 16 on the CPU. HIP has taken them on the device since lane dh item 2
(`INTARSIA_DFLASH_DEVICE_TOPK`, on unless `0`), downloading only 16 (id, value) pairs a row. Metal has had a working
`Backend::topk_rows` since lane ink-mtp (`crates/intarsia-backend-metal/src/topk.rs`: the sampler's order, value descending, ties
by ascending index, NaN rows flagged), but `supports_topk()` stays false there, because device sampling (`batch.rs`) and the DCoCo
corrector (`dcoco.rs`) key on it. The DFlash2 draft therefore never used it on Metal.

The candidates do not decide the output (greedy verification keeps the target's tokens), only which drafts are proposed, so ids
cannot change. The candidates must still be bit-identical to the host's, or the acceptance rate (and so the speed) would move.

## The change (branch `ink/metal-dtopk`, on #557's head `9bafcdf02`)

`crates/intarsia-core/src/model/dflash2.rs` `device_topk_resolve`: a backend with `supports_topk()` (HIP) keeps "on unless `0`";
Metal takes the device top-k only with `INTARSIA_DFLASH_DEVICE_TOPK=1` or `=check` (opt-in; the trait is unchanged, so device
sampling and DCoCo are untouched); every other backend stays on the host. A Metal load prints
"dflash: candidates by the Metal device top-k (...)". Unit test: `dflash2.rs` `cut_drafts_are_a_prefix_of_the_picks` (beside the cut resolve asserts).

## Gates (pre-registered before slot D)

One binary, one endor `--exclusive` fleetq slot, `raw/slotD.sh`. Arms: `dflt` = the Metal default (no flags, no env); `dtopk` =
`INTARSIA_DFLASH_DEVICE_TOPK=1`; `chk` = `=check`; `off` = `--spec off`.

| # | Check | Rule |
|---|---|---|
| D0 | Draft phases (`INTARSIA_SPEC_TRACE=1`, speed.json) | reported, no bar: median `dl + select` a pass for `dflt` (today's baseline, not measured before: the cut lane's traces ran with the conf trace on) and `dtopk` |
| D1 | Candidates bit for bit | `chk` over g3.json (256 tokens, `--ignore-eos`) and speed.json: every `dflash2 topk check` line reports 0 rows differing |
| D2 | Greedy ids | g3.json md5 `off` = `dflt` = `dtopk` = the goldens (`c69d1438...`, `9e85ba11...`, `89994b87...`) |
| D3 | Decode | ABBA x 4 over speed.json (`raw/rule.py dtopk dflt`, `RULE_BAR=1.01`): ids identical in every process. Opt-in PR: no cell with a CI upper bound < 0.99. Default-on recommendation (the user's call): also mean of the six cells >= 1.01 |
| D4 | Serve | `serve --max-batch 1`, `dtopk` vs `--spec off`: 12 / 12 turns' ids and `cached_tokens` equal |

Sampled requests do not draft on Metal by default, so the change cannot reach them (no sampled gate). 45 s GPU idle before every
timed run, macmon logged.

**Decision rule:** D1, D2, D4 pass and D3 shows no slow cell: opt-in PR, merged on green CI. Any candidate difference (D1) or ids
mismatch: no merge, report. D3 mean >= 1.01: ask the user in the morning whether to make it the Metal default.

## Results (slot D, endor exclusive fleetq #12927, 00:00:00-00:21:01 EST 2026-11-28)

Binary `intarsia-dt` md5 `978535cb0e6db1bcae90b7d00114e921` (`648dc9722`). Log `raw/slotD-12927.log`, per-run files `raw/D/`,
build `raw/build-12926.log` (core dflash2 3/3, models `inkling2_dflash2` 14/14).

| # | Result | Evidence |
|---|---|---|
| D0 | reported | median a pass (1231 passes each): `dflt` dl+select **0.421 ms** (the cut's one-row-at-a-time fetch: download + host top-k inside `dl`), `dtopk` **0.328 ms** (dl 0.306 + select 0.021). Draft total 5.86 vs 6.51 ms: single runs, first and second in the slot, so block / head differ by power state, not by the change |
| D1 | **PASS** | `chk`: g3.json 198 checked passes, speed.json 1231, **0 rows differing**; acceptance identical to `dflt` (74.5 % / 3.93 a pass on g3, 71.4 % / 3.54 on speed.json) |
| D2 | **PASS** | code `c69d1438...`, len4k `9e85ba11...`, len16k `89994b87...` in `off`, `dflt` and `dtopk`; peak 20.29 / 20.32 / 20.32 GB |
| D3 | **PASS** (opt-in bar: no slow cell) | rule: mean **1.025** [0.988-1.063], every cell faster by the point estimate, none with CI upper < 0.99, ids identical in all 8 processes; GO at `RULE_BAR=1.01` |
| D4 | **PASS** | 12 / 12 turns' ids and `cached_tokens` equal to `--spec off` |

**D3 per run, with endor's power state** (`raw/power.py`, macmon 250 ms; `raw/D/power.md`):

| run | tok/s | GPU MHz (median, busy) | GPU W |
|---|---:|---:|---:|
| dflt@r1 | 110.40 | 1476 | 39.5 |
| dtopk@r1 | 116.53 | 1617 | 48.3 |
| dtopk@r2 | 116.31 | 1617 | 48.0 |
| dflt@r2 | 115.73 | 1617 | 48.7 |
| dflt@r3 | 117.46 | 1617 | 48.1 |
| dtopk@r3 | 124.43 | 1617 | 48.6 |
| dtopk@r4 | 123.97 | 1617 | 48.0 |
| dflt@r4 | 122.99 | 1617 | 49.5 |

- The speed rose in two steps, at dflt@r1 -> dtopk@r1 (the GPU clock went 1476 -> 1617 MHz) and at dflt@r3 -> dtopk@r3 (same GPU
  clock and power, so another state: not identified). Both are default -> top-k transitions, which ABBA cannot cancel: reps 1 and 3
  read 1.034 / 1.054, reps 2 and 4 (top-k first) 1.004 / 1.008.
- Within one state: dtopk@r1-r2 vs dflt@r2 **1.006**, dtopk@r4 vs dflt@r4 **1.008**. With D0's 0.09 ms a pass (0.3 % of a ~32 ms
  pass) plus one fewer synchronisation a pass, **+0.5-1.0 % is the estimate**; 1.025 is not.

## Slot P: where a k7 + cut pass's GPU time goes (endor exclusive #12928, diagnostic)

`raw/slotP.sh`: `INTARSIA_METAL_PROFILE=cb` with `INTARSIA_METAL_COMMIT_EVERY=1` (one command buffer per dispatch, the backend's
labels), `INTARSIA_SPEC_TRACE=1`, the Metal default, `raw/prof4.json` (warm, 1k, prose 300, 4k ids), 128 tokens; 166 drafting passes,
857 dispatches and **31.4 ms of GPU work a pass** (the span, 49 ms, is inflated by a command buffer per dispatch). Table `raw/P/prof-dflt.md`
(`raw/prof.py`); the 27 MB trace is not committed (`.scratch/prof-dflt.jsonl.gz` in the lane worktree, 3.8 MB).

| Part | ms a pass | share | ops |
|---|---:|---:|---|
| Target MoE experts | 10.3 | 33 % | `moe_experts_situ_bf16_rows` (3 kernels) + `moe_experts_mxfp8_bf16_rows` (2), rows 2-7 |
| Draft (5 layers + head, block 8) | ~5.9 | 19 % | `matmul_bf16_gemv_rows` at t 8 (MLP, attention projections) 3.7, `logits_bf16_gemv_rows 200058x2304` 1.71 (539 GB/s: at bandwidth), window attention 0.18, `dyn_conv` 0.08 |
| Target projections | ~5.8 | 18 % | `ink_in_proj_*` 4.7, `matmul_p12_gemv_rows` 0.86, NAX GEMMs 0.7 |
| Target attention + conv mix | ~5.2 | 17 % | `conv_mix_bf16` (4 kernels) 2.7, `rel_sink_attn_bf16` 1.52, `ink_kv_prologue_bf16` 0.54, `ink_attn_decode_bf16` 0.64 |
| Target head | 1.21 | 4 % | `matmul_p12_logits_gemv_rows` |

**Next lever found:** the draft's `matmul_bf16_gemv_rows 6912x2304` at t 8 (the MLP gate and up, 10 dispatches a pass) takes
**2.42 ms a pass, 0.242 ms each for 31.9 MB = 132 GB/s**, while its down projection `2304x6912` (same bytes, t 8) takes 0.069 ms
(460 GB/s) and `2304x4096` 430 GB/s. At the down projection's rate the pair would cost ~0.7 ms: **~1.7 ms a pass, ~5 %**. The draft
does not decide the output (greedy verification), so a faster kernel there cannot change ids, only acceptance (which D1-style checks
measure). Lane ink-metal-dgemv.

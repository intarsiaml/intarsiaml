# Lane WMR2: HIP WMMA rows family re-measured on main 5f7ec5996 (2026-10-30)

**Docs only. No code, no default changed.** This updates the evidence for the pending decision on PR #272 (lane WMR, "DO NOT MERGE", `docs/results/2026-10-30-wmr-hip-wmma-rows-evidence.md` on branch `lane-wmr/evidence`).

Question: WMR measured `INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all` ("the pair") at +34.6 % (TP=1) / +30.0 % (TP=2) at 16 streams against main d2fe64a8a. Since then HGV (#271: small bf16 GEMVs over the grid, shared expert 4 tokens/block, AR LL 128 KiB) and HFP8 (#274: fp8 one-pass rows kernel for M >= 3) merged, and both speed up ops the pair replaces. What does the pair still add on current main, and what does it cost single-stream?

## 0. Answer

**On current main (5f7ec5996) the WMMA pair adds +9.5 % at 16 streams on TP=1 (464.3 -> 508.3 tok/s) and +6.9 % on TP=2 (652.3 -> 697.2 tok/s), and costs 2.3 to 3.7 % single-stream decode on TP=1 (1k/4k/16k: -3.4 / -2.3 / -3.7 %) and 2.0 to 2.6 % on TP=2 (-2.6 / -2.1 / -2.0 %).** The gain at 16 streams fell from +34.6 / +30.0 % to +9.5 / +6.9 % because the scalar kernels it competed with got much faster (off arm 16 streams: TP=1 370.9 -> 464.3, TP=2 512.7 -> 652.3). The single-stream cost did not shrink (WMR: -4.2 / -3.6 / -3.0 % TP=1, -2.8 / -2.7 / -1.6 % TP=2). At 4 streams the pair is neutral (0.0 / -0.1 %), at 8 streams +3.2 % on both TP, at 1 stream through the batched server -3.0 % (TP=1) / -1.2 % (TP=2). Only the fp8 half carries the remaining gain (TP=1, 16 streams: `fp8a8=all` alone +8.7 %, `bf16=all` alone +0.4 %, which is noise). Quality was not re-run (see section 5).

| result | now (main 5f7ec5996) | WMR (main d2fe64a8a) |
|---|---|---|
| 16 streams, TP=1 | **+9.5 %** [+8.6..+10.4] (464.3 -> 508.3) | +34.6 % (370.9 -> 499.1) |
| 16 streams, TP=2 | **+6.9 %** [+6.5..+7.2] (652.3 -> 697.2) | +30.0 % (512.7 -> 666.5) |
| 8 streams, TP=1 / TP=2 | +3.2 % / +3.2 % | +14.7 % / +10.5 % |
| 4 streams, TP=1 / TP=2 | 0.0 % / -0.1 % | +2.0 % / +2.4 % |
| 1 stream on `--max-batch 16` server, TP=1 / TP=2 | -3.0 % / -1.2 % | -4.6 % / -2.2 % |
| single-stream decode `--max-batch 1`, TP=1 (1k / 4k / 16k) | **-3.4 / -2.3 / -3.7 %** | -4.2 / -3.6 / -3.0 % |
| single-stream decode `--max-batch 1`, TP=2 (1k / 4k / 16k) | **-2.6 / -2.1 / -2.0 %** | -2.8 / -2.7 / -1.6 % |
| TP=1 16 streams, only `fp8a8=all` / only `bf16=all` | **+8.7 % / +0.4 %** | not measured (WMR: ROWS=1 alone, W8A16 table, -0.7 %) |
| cross-stream greedy identity, TP=1, flag on | 16/16 at 4, 8, 16 streams and solo repeat | same |

Prefill and TTFT are noise for the pair (both TP, 1k/4k/16k, within the round-to-round spread; tables in `raw/tables-win*.md`).

## 1. Setup

- Tree: the existing `~/intarsiaml-tp2-hfp8new2` on tatooine (HFP8 branch on 05451fe66). Its `crates/` and `kernels/` are byte-identical to `git archive 5f7ec5996 crates kernels` (525 files, per-file sha256 compared, no diff), so its `tp2-intarsia` binary is main 5f7ec5996 code and **no new build was done** (no new target volume, no new tree on tatooine).
- Hardware/model as WMR: 2 x R9700 (gfx1201) on tatooine, Qwen3.6-35B-A3B-NVFP4, spec off, `serve --backend hip --spec off --profile-every 0 --mem-budget 29 --max-seq 18432 --no-supervisor`, docker image `intarsia-hip-dev:rocm7.2.4-rust1.90.0`. TP=1 is card 0, TP=2 is `--tp 2 --devices 0,1`.
- Method: `scripts/wmr/window.sh` from lane WMR (copied from `origin/lane-wmr/evidence`, plus two small additions in this PR: arms `fp8` = `ROWS=1 TABLE=fp8a8=all` and `bf16` = `ROWS=1 TABLE=bf16=all`, and a `PLAN=1` step that runs `plan show`). One fresh server per (phase, round, arm), `localeval speed` with 1k prompt, gen 512, streams 1/4/8/16 (`--max-batch 16`) or 1k/4k/16k prompts at `--max-batch 1`; 3 rounds; arm order alternates by round so the two arms of a pair are adjacent. Arms: `off` = main default, `on` = the pair (`INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all`).
- Windows (each one fleetq job holding both R9700 cards, `--takes-down radiance --quiet gpu,cpu --expect 20`, guard on tatooine restores radiance on exit; every window <= 11 min):

| window | fleetq job | content | radiance down -> answers |
|---|---|---|---|
| A | #7382 | TP=1 off/on: g512 x3 rounds, b1 x3 rounds, identity (round 1) | 04:10:01 -> 04:20:43 (after 15 s) |
| B | #7383 | TP=2 off/on: g512 x3, b1 x3 | 04:21:43 -> 04:29:00 (after 15 s) |
| C | #7384 | TP=1 off/fp8/bf16: g512 x3, `plan show` x4 | 04:29:43 -> 04:39:13 (after 15 s) |

  Raw: `2026-10-30-wmr2-wmma-rows-on-new-main/raw/win{A,B,C}/` (localeval logs, `steps.log`, `downtime.log`, ident json/logs, plan-show output), tables from `scripts/wmr/summarize.py` in `raw/tables-win{A,B,C}.md`. Radiance is up after every window and after the last (`:8080/v1/models` answers; `fleetq status` shows `radiance-rank0` and `radiance-rank1` up).
- Window A and B `plan show` calls failed (exit 2: `plan show` takes no `--max-batch`; same fallback WMR's `gates.sh` had); the empty `plan-*.txt` and the `.err` there are kept as the record. Window C ran the fixed call.
- Drift control: the `off` arm at 16 streams TP=1 is 464.3 (window A) and 464.8 (window C), 465 in the HFP8 doc; TP=2 652.3 vs 652 there. Window to window drift is < 0.2 %.

## 2. Served A/B, `--max-batch 16`, aggregate decode tok/s (1k prompt, gen 512, 3 rounds, interleaved)

| config | off (rounds; mean) | pair (rounds; mean) | pair vs off |
|---|---|---|---|
| TP=1, 1 stream | 155.5 / 157.5 / 157.6; **156.9** | 152.4 / 152.3 / 151.9; **152.2** | -3.0 % [-3.6..-2.0] |
| TP=1, 4 streams | 302.8 / 298.7 / 301.1; **300.9** | 300.3 / 302.2 / 299.7; **300.7** | -0.0 % [-0.8..+1.2] |
| TP=1, 8 streams | 406.6 / 396.5 / 397.8; **400.3** | 412.5 / 414.7 / 412.3; **413.2** | +3.2 % [+1.5..+4.6] |
| TP=1, 16 streams | 465.8 / 461.5 / 465.5; **464.3** | 509.5 / 509.6 / 505.7; **508.3** | **+9.5 %** [+8.6..+10.4] |
| TP=2, 1 stream | 180.8 / 185.8 / 185.9; **184.2** | 182.2 / 181.7 / 181.9; **181.9** | -1.2 % [-2.2..+0.8] |
| TP=2, 4 streams | 370.2 / 368.2 / 370.4; **369.6** | 368.9 / 369.1 / 369.3; **369.1** | -0.1 % [-0.4..+0.2] |
| TP=2, 8 streams | 528.2 / 527.6 / 528.5; **528.1** | 548.0 / 543.0 / 543.9; **545.0** | +3.2 % [+2.9..+3.7] |
| TP=2, 16 streams | 654.9 / 655.5 / 646.6; **652.3** | 700.4 / 698.0 / 693.2; **697.2** | **+6.9 %** [+6.5..+7.2] |

(`[min..max]` is the per-round paired delta.) The pair's gain is a high-occupancy effect: it appears from 8 streams and grows with M, as in WMR, but its size is a quarter of what WMR measured.

## 3. Single-stream decode, `--max-batch 1` (3 rounds, interleaved), decode tok/s

| config | off (rounds; mean) | pair (rounds; mean) | pair vs off |
|---|---|---|---|
| TP=1, 1k prompt | 162.4 / 164.6 / 164.2; **163.7** | 156.4 / 158.4 / 159.6; **158.1** | **-3.4 %** [-3.8..-2.8] |
| TP=1, 4k prompt | 157.2 / 160.5 / 160.4; **159.4** | 154.9 / 155.0 / 157.2; **155.7** | **-2.3 %** [-3.4..-1.5] |
| TP=1, 16k prompt | 145.8 / 146.8 / 145.8; **146.1** | 140.9 / 140.3 / 141.0; **140.7** | **-3.7 %** [-4.4..-3.3] |
| TP=2, 1k prompt | 193.8 / 192.5 / 192.2; **192.8** | 187.8 / 188.0 / 187.7; **187.8** | **-2.6 %** [-3.1..-2.3] |
| TP=2, 4k prompt | 190.4 / 190.7 / 190.4; **190.5** | 186.7 / 186.4 / 186.2; **186.4** | **-2.1 %** [-2.3..-1.9] |
| TP=2, 16k prompt | 180.7 / 181.3 / 181.2; **181.1** | 177.7 / 177.5 / 177.3; **177.5** | **-2.0 %** [-2.2..-1.7] |

Prefill tok/s (TP=1 1k/4k/16k: +2.4 / -3.7 / +0.4 %; TP=2: +4.6 / -0.3 / 0.0 %) and TTFT are within the round-to-round spread (the 1k/4k prefill rounds have single outliers: TP=1 off round 1 6042 vs 7325, pair round 3 6351 and 9204); nothing to read into them (`raw/tables-winA.md`, `raw/tables-winB.md`).

## 4. `plan show` with the pair set, and what the pair replaces

`plan show --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq 8192` (card 0), four arms (`raw/winC/plan-{off,on,fp8,bf16}.txt`). `plan show` plans the one-row (M = 1) decode step, so it lists the ops whose M = 1 kernel changes; the M = 2..16 batched selection comes from the registry `when` clauses (read in `crates/intarsia-backend-hip/src/quantmoe.rs:427-473` and `kernels.rs:225-259`), see below.

Diff of `plan-off.txt` against `plan-on.txt` (calls/step in the 40-layer 35B plan):

| op | off | pair | calls/step |
|---|---|---|---|
| matmul_fp8 | `matmul_fp8_gemv_v2` | `matmul_fp8_wmma` | 10 |
| matmul_fp8_pair | `matmul_fp8_pair_gemv` | `matmul_fp8_pair_wmma` | 30 |
| gated_rmsnorm_matmul_fp8 | `..._gemv` | `..._wmma` | 30 |
| rmsnorm_add_matmul_fp8_pair | `..._gemv` | `..._wmma` | 10 |
| sigmoid_mul_matmul_fp8 | `..._gemv` | `..._wmma` | 10 |
| matmul_pair (bf16) | `matmul_pair_bf16_gemv` | `matmul_pair_bf16_wmma` | 1 |
| rmsnorm_add_matmul_pair (bf16) | `..._bf16_gemv` | `..._bf16_wmma` | 69 |
| matmul (bf16) | `matmul_bf16_gemv` | `matmul_bf16_wmma` | 0 (registered, not in the decode step) |

Nothing else changes: `moe_block_bf16`, `matmul_nvfp4_gemv_v2` (shared expert), `gdn_conv_delta`, `attn_decode_bf16_splitkv` and the AR path are identical in all four arms. `plan-fp8.txt` differs from `plan-off.txt` in the five fp8 rows only (`matmul_fp8`, `matmul_fp8_pair` and the three fp8 folds). `plan-bf16.txt` differs in the two bf16 pair rows, `matmul_bf16` and also the same three fp8 folds (the fold kernels switch under either table entry, as in WMR; `matmul_fp8` and `matmul_fp8_pair` switch only under `fp8a8`).

Does the pair override HFP8's one-pass kernel and HGV's grid GEMV? **Yes, in the batched range it replaces both**, by selection rather than by being slower:
- The HFP8 one-pass kernel (`fp8_gemv_rows_kernel`, `kernels/hip/fp8.hip`, `INTARSIA_HIP_FP8_ONEPASS_MIN`) lives under `matmul_fp8_gemv_rows`. Its registry clause is `!wmma_takes(fp8_kind(s), s.m) && (1..=ROWS_MAX).contains(&s.m) ...` (`quantmoe.rs:445`), so with `fp8a8=all` it is excluded for every M in 1..=16 and `matmul_fp8_wmma_rows` takes the call. The pair therefore replaces the new one-pass kernel for all batched fp8 dense projections.
- HGV's small bf16 GEMV over the grid is in the bf16 GEMV family (`matmul_bf16_gemv`, `matmul_bf16_gemv_rows`, `matmul_pair_bf16_gemv`, `rmsnorm_add_matmul_pair_bf16_gemv`); every one of these has `!wmma_takes(WmmaKind::Bf16, s.m)` in its clause (`kernels.rs:225,242`; `quantmoe.rs:737,832`), so `bf16=all` removes them from M 1..=16 and the `_wmma` twins (`kernels.rs:252,259`; `quantmoe.rs:839,846`) take over. This was only inferred from the registry clauses and the plan diff, not from per-kernel timing.
- Not replaced by the pair, so still contributing in the `on` arm: HGV's shared expert 4 tokens/block (nvfp4 / MoE path), HGV's AR LL default, the MoE block, GDN and attention.

## 5. Which half carries the gain (TP=1, `--max-batch 16`, window C, 3 rounds interleaved)

| config | off | only `bf16=all` | only `fp8a8=all` | bf16 vs off | fp8 vs off |
|---|---|---|---|---|---|
| 1 stream | 156.3 / 157.6 / 157.8; **157.2** | 152.9 / 153.7 / 154.5; **153.7** | 152.4 / 152.8 / 150.8; **152.0** | -2.2 % [-2.5..-2.1] | -3.3 % [-4.4..-2.5] |
| 4 streams | 296.5 / 298.4 / 301.6; **298.8** | 301.1 / 299.3 / 303.7; **301.4** | 295.4 / 295.8 / 293.9; **295.0** | +0.8 % [+0.3..+1.6] | -1.3 % [-2.6..-0.4] |
| 8 streams | 397.1 / 402.3 / 397.4; **398.9** | 397.9 / 400.8 / 402.3; **400.3** | 406.4 / 409.8 / 410.3; **408.8** | +0.4 % [-0.4..+1.2] | +2.5 % [+1.9..+3.2] |
| 16 streams | 465.1 / 464.0 / 465.2; **464.8** | 464.3 / 469.1 / 466.3; **466.6** | 504.8 / 507.6 / 503.3; **505.2** | +0.4 % [-0.2..+1.1] | **+8.7 %** [+8.2..+9.4] |

Read against window A (pair at 16 streams: 508.3, +9.5 % over its own off 464.3): the fp8 half alone gives 505.2 and so about 92 % of the pair's gain; the bf16 half alone is +0.4 % (inside the spread) at 16 streams. The bf16 half is not worth taking on current main: HGV's grid GEMV already serves those router matrices well. The 1-stream-through-batched-server loss shows in both halves (-2.2 % bf16, -3.3 % fp8); single-stream `--max-batch 1` was not run per half. Not run for TP=2 (the task made this arm optional; one TP was enough to see which half carries the gain).

## 6. Cross-stream greedy identity (TP=1, flag on)

`scripts/wmr/ident.py`, `--max-batch 16`, temperature 0, 160 tokens, 16 distinct prompts of different lengths: solo (sequential), solo repeat, then 4, 8 and 16 concurrent. Pair on: **16/16 identical to solo at solo repeat, 4, 8 and 16 streams** (`raw/winA/ident-g512-tp1-on-r1.{log,json}`); main default arm: also 16/16 at all four (`ident-g512-tp1-off-r1.*`). Nothing regressed relative to WMR (which found 16/16 as well).

## 7. Not done / limits

- No KL gate and no in-process bitwise tests re-run (the task skipped them: WMR ran them; HGV and HFP8 were bitwise against their own main, and the WMMA family's numerics do not depend on either). The quality numbers in the WMR doc (KL21 2.70e-3 short / 3.89e-3 long vs vLLM noise, TP=2 similar) stand as measured on d2fe64a8a; the pair's kernels are unchanged since.
- TP=2 identity not re-run (task: TP=1 only). TP=2 half-table arms not run. Single-stream per half not run.
- 3 rounds per arm; window to window drift is small (section 1) but the 1k/4k prefill numbers have occasional single-round outliers.
- `plan show` shows the M = 1 plan only; the M = 2..16 coverage is read from the registry clauses, not from a per-step kernel trace.
- Failed/empty: `winA/plan-*.txt` and `winB/plan-*.txt` (exit 2, wrong flag, see section 1).

## Files

- `scripts/wmr/` (copied from `origin/lane-wmr/evidence`; `window.sh` extended with the `fp8` / `bf16` arms and `PLAN=1`).
- `docs/results/2026-10-30-wmr2-wmma-rows-on-new-main/raw/` (all windows, raw total under 1 MB so not gzipped).

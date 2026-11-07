# 27B DFlash2 k7 vs MTP k2 at tp=1 on one R9700, rerun on main `ea10f27c7` (lane dt1, 2026-11-06)

Rerun of `docs/results/2026-10-26-dfh-hip-dflash.md` section 10, window 5 (lane dfh, 10-26) on current main. Results and scripts only: no engine change, no default change.

* Binary: `origin/main` `ea10f27c7` (code tree `5515fd9d20f7`), `serve --backend hip`, built on tatooine as `dt1-intarsia` (tree `~/intarsiaml-dt1`, volume `intarsia-cargo-dt1`).
  Since 10-26 main gained: device top-k (#323), fp8 `lm_head` (11-06), the HIP DFlash pass-cost work (lane dh, #323), the split-KV draft window attention (lane dw, #325), pack defaults.
  `INTARSIA_DFLASH_TP` acts only at tp>1. `INTARSIA_DFLASH_WIN_SPLIT` is read without a tp condition (`model/dflash2.rs:394`, default on) so it is on at tp=1, but it only changes the
  draft's attention once the context passes the draft's 2048-key window: none of the category files (short prompts, 512-token answers) get there, the 4k ids prompts do.
* Setup as window 5: 27B (`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`), tp=1 on card 0 only, a fresh `serve --max-batch 1 --mem-budget 29 --max-seq 20480` per arm and visit, `localeval quality --judge none`,
  512 forced tokens (`ignore_eos`, `min_tokens`), the 7 category files of `scripts/dfh/cats/` (36 rows). Arms: `--spec off`, `--spec dflash --spec-k 7` (DFlash2 draft), `--spec mtp --spec-k 2`.
* Two rounds in one radiance window: round 1 order off, mtp2, dfl7; round 2 order dfl7, mtp2 (off once: its two-round spread was not needed; the 10-26 off arm was one round too). 35 localeval runs, 0 problems (every row 512 tokens, same rows in every arm).
  Table = decode tok/s pooled per category (tokens / sum of decode times, `tools/dfh/cat_table.py raw/w1`), mean of the two rounds for the spec arms; tokens per pass (t/p) from the server's counters, round 1.
* Tooling: `scripts/dt1/` (copies of lane dw's `scripts/dw/`, repointed to `~/intarsiaml-dt1`, volume `intarsia-cargo-dt1`, container `dt1-serve`; `stages-ids.sh` is `scripts/dfh/stages-ids.sh` sourcing the dt1 `lib.sh`). Raw: `2026-11-06-27b-dflash-hip-tp1/raw/w1/`.
* fleetq (owner `claude-e7360a-dt1`): build #8873 (cpu, 19:46-19:48), ids stage dry run #8874, window #8875 (`--takes-down radiance`, 19:48:53 to 20:22:43 = 33 min 50 s down, radiance answering `:8080/v1/models` 20 s after the guard started it and confirmed again at 20:23).

## Per-category speed (decode tok/s; x vs off of the same window; t/p = tokens per verify pass)

| category | off 10-26 | off now | DFlash2 k7 10-26 | DFlash2 k7 now | MTP k2 10-26 | MTP k2 now | DFlash2 / MTP now |
|---|---:|---:|---|---|---|---|---:|
| ho-chat | 33.0 | 35.9 | 37.9, 1.15x, 3.07 t/p | **61.0, 1.70x**, 3.08 t/p | 51.5, 1.56x, 2.30 t/p | **63.9, 1.78x**, 2.25 t/p | 0.95 |
| ho-code | 33.0 | 35.9 | 68.7, 2.08x, 5.58 t/p | **95.4, 2.66x**, 4.82 t/p | 62.8, 1.90x, 2.80 t/p | 80.2, 2.23x, 2.79 t/p | **1.19** |
| ho-prose | 33.0 | 35.9 | 30.2, 0.92x, 2.47 t/p | **49.4, 1.38x**, 2.51 t/p | 46.4, 1.41x, 2.08 t/p | **59.0, 1.64x**, 2.08 t/p | 0.84 |
| ho-recall | 33.0 | 35.8 | 43.0, 1.30x, 3.51 t/p | 77.3, 2.16x, 3.93 t/p | 52.1, 1.58x, 2.34 t/p | 69.7, 1.94x, 2.44 t/p | **1.11** |
| rcp-code | 33.0 | 35.8 | 62.7, 1.90x, 5.11 t/p | **101.9, 2.84x**, 5.15 t/p | 61.0, 1.85x, 2.72 t/p | 78.0, 2.18x, 2.73 t/p | **1.31** |
| rcp-prose | 33.0 | 35.9 | 30.9, 0.94x, 2.53 t/p | **50.7, 1.41x**, 2.58 t/p | 46.4, 1.41x, 2.09 t/p | **59.9, 1.67x**, 2.11 t/p | 0.85 |
| rcp-recall | 33.0 | 35.9 | 48.5, 1.47x, 3.94 t/p | 80.6, 2.25x, 4.10 t/p | 54.1, 1.64x, 2.43 t/p | 70.1, 1.95x, 2.46 t/p | **1.15** |

Round to round the spec arms differ by 0.0-0.5 tok/s (e.g. rcp-code DFlash2 102.1 / 101.7, MTP 78.0 / 77.9), so no ratio in the last column is inside the noise.
Tokens per pass moved by -14 percent (ho-code 5.58 to 4.82) to +12 percent (ho-recall 3.51 to 3.93) since 10-26 (the other categories within 6 percent; the fp8 `lm_head` default changes the target's greedy text, so acceptance shifts per category), so the gain is mostly pass cost, with acceptance shifting a little either way.

## Pass cost

`tests/dflash2_real.rs` on card 0, 3000-token context (the same probe as 10-26, `raw/w1/mtest-probe27.log`), and the in-engine trace of the ids runs (`generate`, `INTARSIA_SPEC_TRACE=1`, mean of the passes after the first 4, one run each):

| ms | 10-26 probe | now probe | now trace short-0 (41 tok prompt) | now trace 4k-1 (3135 tok prompt) |
|---|---:|---:|---:|---:|
| plain decode step | 30.7 | 28.39 | 1000 / 36.03 = 27.8 | 1000 / 35.37 = 28.3 |
| draft (7 drafts) | 23.5 | **11.10** | 11.82 | 12.41 |
| verify8 (row-exact, with rollback in the trace) | 65.1 | **43.35** | 37.90 | 41.73 |
| pass (draft + verify) | 88.5 | **54.45** | 49.72 | 54.15 |
| break-even tokens per pass (pass / plain) | 2.9 | **1.92** | 1.79 | 1.91 |

Draft ops now (probe): `matmul` 56 calls 6.66 ms (10-26: 10.4), the draft head `matmul_fp8` 1 call 3.22 ms (10-26: 7.8 ms bf16 `[8, V]` head), `attn_decode` 40 calls 1.67 ms (4.4), `dyn_conv` 0.41, rope 0.25, rmsnorm 0.25, qk_norm 0.21, add 0.20, kv_append 0.20, silu_glu 0.10, argmax 0.06 ms; 13.27 ms of ops in a 14.18 ms profiled wall.
The draft more than halved (23.5 to 11.1 ms), verify8 fell by a third (65.1 to 43.4 ms), and the plain step by 8 percent. In the served runs a DFlash2 pass costs 2.51 / 49.4 = 50.8 ms (prose) to 4.82 / 95.4 = 50.5 ms (code); an MTP k2 pass costs about 35 ms (2.25 / 63.9).
The 4k-1 draft is only 0.6 ms above short-0's (draft block with the split-KV window attention, see above; the whole-draft 4k penalty without it was 3.2 ms at tp=2 in `2026-11-06-dflash-win-split.md`). Not isolated here with an A/B.

## Greedy ids

`generate --chat --no-think --ignore-eos --show-ids --max-tokens 256`, card 0 (short-0) and card 1 (4k-1, same tp=1 code path), arms compared with `--spec off` of the same prompt:

| prompt | `--spec dflash --spec-k 7` | `--spec mtp --spec-k 2` |
|---|---|---|
| short-0 (41-token prompt) | PASS, 256 ids identical (3.83 t/p) | PASS, identical (2.49 t/p) |
| 4k-1 (3135-token prompt) | PASS, identical (3.70 t/p) | PASS, identical (2.48 t/p) |

4 of 4 (`raw/w1/ids/`). Only two prompts, as the brief asked; the full 18-prompt gate of 10-26 section 12 was not rerun.

## Reading

Against `--spec off`, DFlash2 k7 now wins in every category at tp=1 (1.38x on prose to 2.84x on code, against 0.92x-2.08x on 10-26): its pass fell from 88.5 to about 50 ms, so the break-even dropped from 2.9 to 1.9 tokens per pass and prose (2.5 t/p) clears it.
MTP k2 also sped up (1.64x-2.23x against 1.41x-1.90x) and its pass (about 35 ms for 2.1-2.8 t/p) is still much cheaper, so DFlash2 beats MTP k2 only where it accepts many drafts: code (1.19x and 1.31x of MTP's speed), recall (1.11x and 1.15x), and not on prose (0.84x, 0.85x) or chat (0.95x; 3.08 t/p against about 3.2 needed to tie).
DFlash2 therefore still does not win everywhere against MTP k2, so nothing changes: it stays opt-in and no default is touched. The remaining gap on prose and chat is the pass cost (50 against 35 ms), mostly the 43 ms row-exact verify8; a per-prompt choice between the two (acceptance-driven) would collect the code and recall wins, but that is a feature, not part of this rerun.

Limits: one window, two rounds (off once), tp=1 card 0 only, 36 rows over 7 categories, ids on two prompts. The off arm ran at 35.9 tok/s here against 33.0 on 10-26, so ratios to off use each window's own off.

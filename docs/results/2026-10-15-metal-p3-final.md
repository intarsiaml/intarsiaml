# metal-p3-final: P3 (chunk 2048) merge bar on spike/metal 98f506e + golden-4k + p3-chunk

Branch `metal-p3-final` (worktree `/opt/code/intarsiaml-p3-final`) = `spike/metal` 98f506e (tail-fix merged) + `metal-golden-4k`
(baseline-gated `acc-4k`, b71ee56 and the docs after it) + `metal-p3-chunk` (177da74, `[tuning.metal] prefill_chunk = 2048`), then
the re-recorded Metal baselines (da8f69a). Qwen3-8B BF16, endor (Apple M5 Max). **Not merged into spike/metal, nothing pushed.**
Raw data: `docs/results/raw/2026-10-15-metal-p3-final/` (`run.sh` is the script; `attempt1/` is a first run stopped at its section 7
for two script bugs, kept for the record: an old envelope file blocked `--write-noise-envelope`, and a zsh `$G4:chunk` expansion broke
the 4k recording; sections 1-3 of that run are the ones reported below).

## Verdict

**All six items of the merge bar pass at chunk 2048.** Fresh `qualify --backend metal`: **PASS** (11 PASS, `ab`/`tp` SKIP).
Two things to know: the long golden (longest prompt 700 tokens) crosses no 2048 boundary, so only the 4k golden and the id checks
exercise P3's chunking; and the 4k golden has no absolute top-1 floor by decision (it is baseline-gated; 99.28% prefill top-1 is
bf16 noise at fp32 near-ties, see `2026-10-15-metal-golden-4k.md`).

| # | check | result |
|---|---|---|
| 1 | `cargo test -p intarsia-cli -p intarsia-models` | PASS: 0 failed in every suite (intarsia-cli 75 + 1; `tests/cargo-test.txt`) |
| 2 | `cargo test -p intarsia-backend-metal --release -- --test-threads=1` | PASS: 0 failed, 14 suites (`tests/cargo-test-metal.txt`) |
| 3 | `intarsia kernels test --backend metal -v` | PASS: 23 kernels, 291 cases, all ADMITTED, no FAIL/REJECT (`kernels/kernels-test.txt`) |
| 4 | accuracy at chunk 2048, below | PASS, long top-1 99.72% / KL 3.70e-4 |
| 5 | greedy ids spike/metal (chunk 256) vs final (chunk 2048) | identical in all 6 comparisons, below |
| 6 | fresh full `qualify --backend metal --bandwidth-gbs 614`, goldens short, long, 4k | **PASS** |

Items 1-3 ran on the tip before the baseline commit (head 9eeb96c); the later commits change `golden/baselines/*.json` and docs only
(no `crates/`, `recipes/`, `kernels/` difference: `git diff 9eeb96c HEAD -- crates kernels recipes tools scripts` is empty).

## 4. Accuracy at chunk 2048 (default, no `--prefill-chunk`)

| golden | mode | top-1 | KL | gate |
|---|---|---|---|---|
| short (8 prompts, 434 positions, full-vocab fp32) | prefill | 100.00% | 2.832e-4 | PASS (floor + baseline/envelope) |
| short | decode | 100.00% | 3.313e-4 | PASS |
| **long** (10 prompts, 1434 positions) | prefill | **99.72%** | **3.699e-4** | PASS; merge bar (99.6%, 5e-4) met |
| **long** | decode (64) | **99.80%** | **3.538e-4** | PASS; merge bar met |
| 4k (3 prompts, 12685 positions, top-20, 21-way KL) | prefill | 99.28% | 2.274e-4 | PASS (no top-1 floor, baseline PASS, tail ok) |
| 4k | decode (64) | 98.96% | 2.548e-4 | PASS |
| 4k | prefix reuse (chat_reuse_4k, 2048) | | | PASS (`prefix-reuse verdict: PASS`) |

The short decode KL 3.313e-4 and long decode 3.538e-4 match the tail-fix numbers the coordinator gave (3.313e-4 short; long 3.371e-4 was
the decode figure of another window). "Prompts spanning more than one chunk" at 2048: short 0/8, long 0/10, **4k 3/3**.
Control for the baseline gate (`accuracy/4k-control`): a copy of the 4k baseline made better than this build by 1.5 top-1 points and half the
KL makes `baseline verdict` **FAIL** (exit 1), so the gate is not vacuous. The recorded-baseline gates themselves pass.

## 5. Greedy ids, `generate --temperature 0 --ignore-eos --show-ids --max-seq 8192`, spike/metal 98f506e (chunk 256) vs final (2048)

| prompt | 48 tokens | 256 tokens |
|---|---|---|
| short (`prompt-short.txt`) | identical | identical |
| 4k (`prompt-4k.txt`, 4062 tokens) | identical | identical |
| w4107 (`identity-logits/w4107.txt`, 4107 tokens, a 2048+2048+11 chunk split) | identical | identical |

(`identity/summary.txt`, `first_divergence.py`; the logs show `prefill chunk 256 (default)` for spike/metal and `2048 (recipe qwen3, metal)` for final.)

## 6. Qualify (fresh report `qualify/full-report.md`, binary head=da8f69a code-tree=5096c3c96c53 dirty=no)

doctor, recipe, plan, kernels (23 / 291), gate, acc-short (100.00% / 2.83e-4; 100.00% / 3.31e-4), bench (36.4 tok/s at 128, 90% of the byte
floor), acc-long (`--mode both`: 99.72% / 3.70e-4; 99.80% / 3.54e-4), **acc-4k** (99.28% / 2.44e-4; 98.96% / 2.57e-4; gated against the
recorded baseline and envelope of commit 9eeb96c, reuse gate PASS), envelope, bench-long (36.4 tok/s at 1k): PASS. ab, tp: SKIP.
These speeds are qualify output, not benchmark data.

## Baselines re-recorded at chunk 2048 on this tip (`golden/baselines/`, recorded by the stamped binary head=9eeb96c code-tree=1e0dfabdf84a)

| golden | perturbation | self-check |
|---|---|---|
| `qwen3-8b-fp32` (short), `--mode both --decode-positions 16 --max-seq 4096` | chunk 2048 -> 64 | floor PASS, baseline PASS |
| `qwen3-8b-long-fp32`, `--decode-positions 64 --max-seq 8192` | chunk 2048 -> 256 | floor PASS, baseline PASS |
| `qwen3-8b-4k-fp32-top20` (replaces the 256 pair of `metal-golden-4k`) | chunk 2048 -> 256 | floor top-1 FAIL (99.5 class floor, expected), baseline PASS |

The three pre-tail-fix files from `metal-p3-chunk` (short/long baseline + envelope, e8be186 stamps) are overwritten.

## fleetq and golden

fleetq jobs, all released: #1986 (acc-4k baseline check on `metal-golden-4k`), #1987 (first run, stopped on the script bugs, released by hand),
#1988 (this run). Golden copied (file only) to `/opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors`, sha256
`7371a77f417c86b9ca50518fe9872e05dc5935a744254e6040a1307cb1a394f8`.

## Remaining before P3 is merge-ready

Nothing from the accuracy/identity/qualify bar. P3's speed and same-slot criteria are in `2026-10-14-metal-prefill-p3-final.md` and
`2026-10-15-metal-final-sameslot.md` (not re-measured here). Merge order: `metal-golden-4k` is contained in this branch; merging
`metal-p3-final` into spike/metal brings the wiring, the golden-4k docs, the re-recorded baselines and P3 together.

# Metal: 8B fp32 4k golden, `acc-4k` stage, qualify at prefill chunk 2048

Qwen3-8B BF16 on endor (Apple M5 Max, macOS 27.0). Branch `metal-golden-4k` = `metal-tail-fix` (b1c38d9, `NAX_MIN_M` 16 -> 2)
+ the qualify wiring below (f67c379, cd4c7d3). The P3 build is a scratch merge of that branch with `metal-p3-chunk`
(`metal-golden-4k-p3-scratch`, 03ee339, not kept). Raw data: `docs/results/raw/2026-10-15-metal-golden-4k/`. Nothing merged or pushed.

## Outcome

* **The 4k golden exists** and `qualify` uses it: with it, `qualify --backend metal` **PASSES at chunk 2048** (P3 build, 11 stages
  PASS, `ab`/`tp` SKIP) and at chunk 256 (tail-fix base). Without it, the same P3 build FAILs `acc-long` on the prefix-reuse gate
  exactly as before (`round2/p3-no4k.log`: "prefix reuse differs from a cold run", the SKIPPED-counts-as-fail rule).
* **The prefix-reuse gate PASSES at chunk 2048** on the 4k golden, in all three cases (continue, restore, aligned) of `chat_reuse_4k`.
* **Accuracy is identical for the two builds, and for chunk 256 and 2048 on either build**: the same 12685 prefill and 192 decode
  positions give bit-for-bit the same top-1 and KL in all four runs (SAFE at 256 and 2048, P3 at 2048 and 256). The noise
  envelope of the chunk perturbation (256 <-> 2048, every 4k prompt crosses at least one boundary) is zero, delta ~1e-20.
  P3 changes no 4k number.
* **The 4k golden does not meet the class floor top-1 >= 99.5% with any build** (see "The top-1 floor"), including the unchanged
  tail-fix base at chunk 256 and the CPU engine. `acc-4k` therefore gates top-1 at 98.0%, KL/tail/non-finite at the class values.
  **This is a threshold decision for the user** (below); it is not caused by P3.

## 1. The golden

| | |
|---|---|
| file | `/opt/code/intarsiaml-golden-4k/golden/qwen3-8b-4k-fp32-top20.safetensors` (git-ignored; 2,086,924 bytes) |
| sha256 | `7371a77f417c86b9ca50518fe9872e05dc5935a744254e6040a1307cb1a394f8` |
| format | `intarsia-golden-topk-v1`, top-20, 3 prompts, 12685 positions (`long_docs_4k` 4000+64, `long_code_4k` 4000+64, `chat_reuse_4k` 4493+64, `reuse_split` 4447) |
| reference | HF transformers 5.19.0, torch 2.14.1, **fp32 weights and activations, CPU**, model `/Users/josef/Models/Qwen3-8B`, ChatML, 64 greedy continuation tokens per prompt |
| command | `fleetq run --host endor --mem 50 -- uv run python -u golden.py --model /Users/josef/Models/Qwen3-8B --device cpu --dtype float32 --topk 20 --prompts prompts/golden_4k.jsonl --out ../golden/qwen3-8b-4k-fp32-top20.safetensors` (fleetq #1977, in `tools/`) |
| runtime / memory | generation 303 s (99 / 104 / 100 s per prompt); the whole job 4220 s of wall clock, almost all of it queued behind #1975; max RSS 41.5 GB (budget 50 GB) |

Top-20 follows `docs/accuracy.md` ("HF (8B), top-20 only"). Consequence: the gated KL on this golden is the 21-way KL (a lower bound
of the full-vocabulary KL, doc "How a top-k golden is compared"), with the tail-mass check; it is not the exact full-vocabulary KL
that the guardrails on the 8 short / 10 long full-vocabulary goldens use. The file is outside git like every other golden. It lives in this worktree's
ignored `golden/`; `qualify` finds it by name only from there, from `--golden-dir`, or from the main checkout's `golden/` (which
I was told not to touch). Copy it to `/opt/code/intarsiaml/golden/` to make it the default; removing the worktree deletes it.
Also reproducible: the command above is deterministic CPU fp32 (not re-run to confirm bit-identity).

## 2. Qualify wiring (files)

* `crates/intarsia-cli/src/cmd/qualify/facts.rs`: `find_golden_4k()` (a `<model>-4k...` file, fp32 first); `find_golden` shares one
  `find_scored` with it, scoring and first-directory-wins ties unchanged. Unit test extended.
* `crates/intarsia-cli/src/cmd/qualify.rs`: `--golden-4k FILE` (else found by name in the golden dirs, never made: the HF run is
  ~5 min of CPU at 41 GB); new stage `acc-4k` (`accuracy --mode all`, `--decode-positions 64 --max-seq 8192`, top-1 floor
  `ACC_4K_MIN_TOP1 = 98.0`); `acc-long` runs `--mode both` instead of `all` **only when** a 4k golden is found and differs from
  the long golden (the reuse gate then lives in `acc-4k`).
* `crates/intarsia-cli/src/cmd/qualify/report.rs`: `acc-4k` listed with `acc-short` / `acc-long`.
* `docs/adding-a-model.md`, `docs/accuracy.md`: stage table, golden discovery, where the golden lives.

**Backward compatibility (CUDA / HIP / cpu / other models):** `intarsia-core` and the engine are untouched; only `qualify` changed.
With no `<model>-4k...` golden in the golden dirs (true for every other model and box today; the planned spark1 8B top-20 was never
built) `acc-long` is unchanged (`--mode all`, same golden) and `acc-4k` is SKIP, which does not change the overall verdict. Verified:
`round2/safe-no4k` (acc stages only, no 4k golden) PASS with `acc-4k` SKIP, `acc-long` numbers equal to the 4k-golden run. The one
visible difference on those boxes is an extra SKIP row in the report. Someone copying a 4k golden to a CUDA box changes `acc-long`
there to `--mode both`, with the reuse gate moving to `acc-4k` (where it also runs at chunk 256).

## 3. Accuracy, `intarsia accuracy --mode all --decode-positions 64 --max-seq 8192` on the 4k golden

| build, chunk | prefill top-1 | prefill KL(21) | decode top-1 | decode KL(21) | reuse gate | class floor (99.5 / 1e-3) |
|---|---|---|---|---|---|---|
| SAFE (tail-fix + wiring), 256 (default) | 99.28% | 2.274e-4 | 98.96% | 2.548e-4 | PASS (9 cases: all three prompts at 256; at 2048 only `chat_reuse_4k` runs, the raw prompts print "skipped", split 2000 < 2048) | top-1 FAIL, KL ok |
| SAFE, `--prefill-chunk 2048` | 99.28% | 2.274e-4 | 98.96% | 2.548e-4 | PASS | same |
| P3, 2048 (default) | 99.28% | 2.274e-4 | 98.96% | 2.548e-4 | PASS | same |
| P3, `--prefill-chunk 256` | 99.28% | 2.274e-4 | 98.96% | 2.548e-4 | PASS | same |

Per prompt (prefill top-1 / KL21): `long_docs_4k` 99.34% / 1.83e-4, `long_code_4k` 99.53% / 1.83e-4, `chat_reuse_4k` 99.01% / 3.07e-4.
The one reuse row beyond the rounding tolerance, `chat_reuse_4k` continue, is accepted by the existing logit criterion (warm and cold
diverge at token 24, rounding-scale KL; identical in all four runs). The existing goldens (qualify stages `acc-short` / `acc-long`),
both builds: short prefill 100.00% / 2.83e-4, decode 100.00% / 3.31e-4; long prefill 99.72% / 3.70e-4, decode 99.80% / 3.54e-4,
all above the merge bar (KL 5e-4, top-1 99.6%). **The merge bar (top-1 99.6%) cannot be applied to the 4k golden**: no build reaches it.

### The top-1 floor

* Positions are near-ties for the fp32 reference. Of the 91 prefill misses, the reference's top-1 minus top-2 gap is under 0.1
  nats at 82 (90%) and under 0.02 at 43 (47%); median 0.022 nats, max 0.45. Such gaps hold 4.2% of all 12685 positions
  (`posdump.tsv`, 91 = 12685 x (1 - 0.9928)). A bf16 logit noise of a few 0.01 nats flips them.
* The intarsia **cpu** engine (bf16 weights, independent kernels, `--backend cpu --mode prefill`, fleetq #1981): top-1 99.45%,
  KL21 2.05e-4, also below 99.5. Metal is 0.17 points lower (21 more of 12685 flips) at 1.1x the CPU KL.
* The decode window is 192 positions: one flip is 0.52 points, so a floor of 99.5 allows 0 flips; the measured two flips is
  98.96%.

So the class floor top-1 is not informative on 4k prompts, while KL (2.3e-4, 4x below the 1e-3 floor and below the long golden's
3.7e-4), the tail mass (excess -1e-4) and the zero non-finite rows are. `acc-4k` keeps those and sets top-1 at 98.0%: 1.4 points
under the CPU engine's prefill figure, three decode flips allowed (98.44%). The regression guard for this golden is the baseline
and envelope below, not the floor. **Decision for the user:** if 98.0 is not acceptable, the alternatives are a baseline-gated
`acc-4k` (needs the recorded files wired into `qualify`) or an HF bf16 CPU 4k reference to calibrate against.

## 4. Baselines and envelopes (Metal, on tail-fix)

Recorded like the P3 baselines (`--mode both --decode-positions 64 --max-seq 8192`, perturbation = the other chunk, `--allow-same-code`,
self-check with `--pre-change <head>`; `run.sh`):

| file | build / chunk | contents |
|---|---|---|
| `golden/baselines/qwen3-8b-4k-fp32-top20.metal.baseline.json` | SAFE, 256 (tracked, head=f67c379 code-tree=b7b7b80a8f67) | prefill top-1 99.283%, KL21 2.2737e-4; decode 98.958%, 2.5475e-4 |
| `golden/baselines/qwen3-8b-4k-fp32-top20.metal.envelope.json` | perturbation chunk 256 -> 2048 | nothing moved: every per-prompt delta is 0 to 1e-19, and the tool warns that the perturbation did not touch the prompts, so the fixed margins (KL x1.75 +1e-5, top-1 -1 pt) apply |
| `docs/results/raw/2026-10-15-metal-golden-4k/p3-baselines/*.json` | P3, 2048 (head=0af824d), perturbation 2048 -> 256 | same numbers; for P3 to adopt on its branch (a second file with the same name would conflict) |

Self-checks: `baseline verdict ... PASS` for both, while the floor verdict is FAIL (top-1 only), as in section 3. The baselines
were recorded before the `ACC_4K_MIN_TOP1` commit; that commit touches only `qualify`, not the engine or `accuracy`. **Not redone by me:**
the P3 branch's own `golden/baselines/qwen3-8b-{,long-}fp32.metal.*.json` (short, long) were recorded on plain spike/metal at chunk 2048,
before the tail-fix; the coordinator's note says decode KL moved a little (short 3.496 -> 3.313e-4, long 3.517 -> 3.371e-4), so P3
should re-record them on tail-fix at merge time. Nothing was recorded on plain spike/metal in this work (the first tail-fix base was
in place before any build).

## 5. Qualify verdicts (fresh, `--backend metal --bandwidth-gbs 614`, goldens: short, long, 4k)

| build | chunk | verdict | stages |
|---|---|---|---|
| SAFE (tail-fix + wiring) | 256 | **PASS** | doctor, recipe, plan, kernels (23 / 291), gate, acc-short, bench, acc-long (`--mode both`), **acc-4k**, envelope, bench-long PASS; ab, tp SKIP |
| P3 (SAFE + `metal-p3-chunk`) | 2048 | **PASS** | same; `acc-4k` prefix-reuse PASS at 2048 |
| SAFE, no 4k golden | 256 | PASS | `acc-long` `--mode all` PASS, `acc-4k` SKIP (acc stages only) |
| P3, no 4k golden | 2048 | **FAIL** | `acc-long`: prefix reuse (SKIPPED counts as fail), unchanged from before |

Round 1 (`safe/`, `p3/`, before the 98.0 floor): both builds FAIL only on `acc-4k` top-1 (99.28% / 98.96% < 99.5), which found the
issue in section 3. Speeds in the qualify reports (decode ~36.5 tok/s, 88-91% of the byte floor) are not benchmark data (qualify runs
once; clock state varies); see the P3 and final same-slot documents for speed.

## 6. What blocks merging P3

1. The 4k golden must be copied to the main checkout's `golden/` (or `--golden-dir`/`--golden-4k` used) so `qualify` finds it; without it
   P3's qualify FAILs as before.
2. `metal-golden-4k` (the wiring) must merge together with or before P3, and the `acc-4k` top-1 floor (98.0) needs the user's OK.
3. P3's committed short/long Metal baselines predate the tail-fix and should be re-recorded on it; the 4k baseline/envelope pair at 2048
   from `p3-baselines/` should be added to `golden/baselines/` on the P3 side.
4. Not a blocker found: accuracy at 2048 vs 256 is bit-identical on all three 4k prompts, the reuse gate passes, KL is well inside the
   floor. The remaining P3 criteria (speed, the four-engine same-slot) are in P3's own documents and were not touched here.

## 7. Method and ids

* fleetq jobs (all released): #1977 golden generation (CPU, 50 GB); #1979 accuracy + baselines + round-1 qualify (exclusive); #1981 CPU
  engine on the 4k golden (30 GB); #1982 Metal position dump (12 GB); #1984 round-2 qualify (exclusive). (#1976 was my first launch
  with full-vocabulary output, cancelled before it started: the doc says top-20.)
* Builds, stamped (`scripts/tree-stamp.sh`, `dirty=no`): SAFE round 1 head=f67c379 code-tree=b7b7b80a8f67; P3 round 1 head=0af824d
  code-tree=fdde68198b2c; round 2 SAFE head=cd4c7d3 code-tree=42a65ddc0934, P3 head=03ee339 code-tree=336fc7542763 (all
  `cargo build --release -p intarsia-cli --features metal`).
* `cargo test -p intarsia-cli`: 74 + 1 passed, 0 failed (includes `stage_table_is_consistent`, the golden discovery test, report tests).
  `cargo fmt --check` shows pre-existing differences in `accuracy/baseline.rs` only; none in the files I changed.

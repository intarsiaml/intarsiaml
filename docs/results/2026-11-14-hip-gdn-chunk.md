# HIP: chunked WMMA GatedDeltaNet prefill scan (lane gdn-chunk)

**Update 2026-11-15: default-on by user decision (docs/results/2026-11-15-hip-gdn-default.md); `INTARSIA_GDN_CHUNK=0` disables. The `scripts/gdn-chunk` runs with no knob (`E=""`) now run the chunked kernel.**

Status: built and measured 2026-11-14 (branch `gdn-chunk`, not pushed). Opt-in only: the change-against-base accuracy gate fails
(section 4), so the default is NOT flipped. The lane brief is kept below; results follow it.

## Lane brief (kept here so it survives context compaction)

Goal: port radiance's chunked WMMA Gated DeltaNet prefill scan (v2.0.0, ddcb9c6, Apache-2.0;
`libr4d/r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip` + `r4d_gdn_kkt_solve_k128_c64_bf16.hip` + `r4d_gdn_wmma.h`) to the
HIP backend, so multi-token prefill on the R9700s (gfx1201) stops running the delta rule token by token. Baseline:
`docs/results/2026-10-17-hip-prefill.md` puts the delta rule at ~30 % of 35B HIP prefill (~100 ms for 30 layers x 4408
tokens). Expected (an estimate, to be measured): 15-25 % faster 4k prefill on Qwen3.6-35B-A3B NVFP4; also applies to the 27B.

Constraints:

- Work only in the worktree `/opt/code/intarsiaml/.worktrees/gdn-chunk` (branch `gdn-chunk`, from origin/main 0f42fbfad);
  the radiance clone `.worktrees/gdn-chunk-radiance` is read-only. No other local work folders. Do not push, open a PR,
  merge, or delete the worktree / clone. Small logical commits.
- Opt-in: `INTARSIA_GDN_CHUNK=1` selects the chunked path; the sequential kernels stay the default. Decode (T = 1) and
  small T stay on the existing kernels; the min-T threshold comes from measurement.
- Must handle: initial state in, final f32 state out, chunk tails (T not a multiple of the chunk), the per-row snapshot
  variant (keep the sequential kernel there if chunking is awkward, and say so), packed multi-prompt prefill.
- Gates before "done": (1) `intarsia kernels test` (HIP) cases vs the CPU reference at T in {1..few, 63, 64, 65, 200,
  4096}, with and without initial state, outputs and final state, the repo's bf16-out / f32-state tolerances;
  (2) 35B HIP accuracy / qualify gates knob on vs off: KL and greedy-id divergence point (bit-identical ids not
  expected), flag any failure; (3) speed: 35B NVFP4 tp1 prefill tok/s at 1k / 4k / 16k, knob on vs off, same build,
  alternating, >= 3 runs each, plus the GDN kernel time from a profile if cheap, plus a 4k tp2 number if tp2 uses the
  same op; (4) `cargo test` for the touched crates and `cargo build --release -p intarsia-cli --features hip` in the
  tatooine image.
- Fleet: build / test on tatooine in `deploy/tatooine/dev.sh`, tree rsynced to `tatooine:~/intarsiaml-gdn-chunk/`
  (NOT `~/intarsiaml/`; `INTARSIA_REPO` points dev.sh at it). Every tatooine command through fleetq (`fleetq status`
  first). Builds: `--load cpu --mem 0`. GPU work: `fleetq acquire --host tatooine --takes-down radiance --mem 30
  --expect <MIN> --label "intarsiaml hip: gdn-chunk <what>" --wait`, `ssh tatooine systemctl --user stop radiance`,
  commands prefixed `FLEETQ_JOB=<id>`, then ALWAYS `ssh tatooine systemctl --user start radiance`, check
  `curl -s http://tatooine:8080/v1/models`, `fleetq release <id>`. GPU windows under 45 minutes, batched in scripts.
  Never leave a slot held or radiance down; never `--force` / `--no-queue`; honest `--expect`.
- No single tool call or wait over 4 minutes: long runs go under nohup on tatooine with a log, polled.
- Remove scratch on tatooine except `~/intarsiaml-gdn-chunk/` (say if left).
- Final report under 600 words: files, selection, each gate with numbers, what is not done / not verified, and a
  recommendation on flipping the default.

## 1. What was built

- `kernels/hip/gdn_chunk.hip`: `ik_gdn_delta_rule_bf16_chunk`, `Op::GdnDeltaRule` on 64-token chunks with gfx12 WMMA. Four
  dispatches per call: gates (within-chunk running sum G of the log decay, beta), q / k norm factors, the per-chunk inverse
  `(I + strict_lower(diag(beta) K K^T e^{G_i - G_j}))^-1` (radiance's kkt solve), and the scan (radiance's chunk scan: the f32
  state in WMMA accumulators across the chunk loop, the `<= 1` decay-factor guard kept). dk = dv = 128, Hv / Hk in
  {1, 2, 3, 4, 6, 8}, any T >= 1 (short last chunk), state read and written in place (f32). Compiled `-mcumode` (build.rs).
- Differences from radiance: one sequence per call, no slot-indexed / fp16 state cache; q, k, v are read straight from the conv
  output rows, and the l2 normalisation is folded into the per-token f32 scales (log scales `lq`, `lk` in scratch), so the WMMAs
  see the exact bf16 q / k the conv stored (the first version rounded normalised copies to bf16: kernel error 4.7e-3 -> 3.8e-3,
  scratch 51 MB -> 17.8 MB at the 35B's 4096-token chunk).
- Registration `gdn_delta_rule_bf16_chunk` (`crates/intarsia-backend-hip/src/kernels.rs`), priority 12 (above v3), scratch
  `gdn_chunk_scratch`; ffi + dispatch; `ik_hip_abi.h`; NOTICE-radiance; knobs in `docs/tuning.md` and `ENV_KNOBS`.
- Selection: `INTARSIA_GDN_CHUNK=1` (exactly `1`; default off) and T >= `INTARSIA_GDN_CHUNK_MIN` (default 64). Decode, verify
  rows and the fused conv + delta (T <= 32, including its snapshot variant) are untouched: the snapshot variant only exists
  for the fused op, which never reaches this kernel, so it stays sequential. Packed multi-prompt prefill calls the op once per
  sequence on its own state (`gdn_block_segs`), so it takes the chunked kernel per sequence with no extra work. tp2 shards call
  the same op with (Hk 8, Hv 16) and take it too.
- `deploy/tatooine/dev.sh` forwards only `INTARSIA_HIP_*` / `INTARSIA_TP_*`: the knob needs `dev.sh env INTARSIA_GDN_CHUNK=1 ...`
  (docker runs: `-e`). The brief named the knob; the HIP convention would be `INTARSIA_HIP_GDN_CHUNK`.
- Tests: `intarsia-core` admission sweep gets `gdn_chunk_cases` (own RNG stream, existing cases keep their data);
  `tests/quick_coverage_gdn_chunk.rs`; `kernels::tests::gdn_chunk_knobs`; `tests/gdn_timing.rs::gdn_chunk_vs_sequential_timing`.
- Scripts: `scripts/gdn-chunk/` (window wrapper that always restores radiance, kernel / timing / accuracy / speed windows).

## 2. Gate 1: kernel tests (HIP vs the cpu reference), raw/v2

`kernels test --backend hip --op gdn_delta_rule`, tolerance `BF16_TOL` (max_abs 1.5e-2, rel_l2 6e-3) for the bf16 output and
the f32 state (the GDN registrations set no separate f32 budget).

| run | cases | fail | max_abs | rel_l2 |
|---|---:|---:|---:|---:|
| chunk, `INTARSIA_GDN_CHUNK_MIN=1`: T in {1, 2, 3, 63, 64, 65, 200, 4096} x {S0 zero, S0 random} at (Hk 1, Hv 2); 35B shape T 1 / 7 / 64 / 65 / 130 / 200; tp2 shard, ratio 1 and 3 at T 130 | 30 | 0 | 5.25e-3 | 3.80e-3 |
| chunk, default min T 64 | 20 | 0 | 4.95e-3 | 3.69e-3 |
| sequential, knob off: gdn_delta_rule_bf16 / _v2 / _v3 on the extended sweep | 46 / 34 / 22 | 0 | 1.22e-4 | 5.5e-5 |

The chunked kernel passes with a thin margin on rel_l2 (63 % of the budget) and is ~70x less exact than the sequential
kernels; the error is flat in T (T = 1: 2.8e-3, T = 4096: 3.7e-3), i.e. bf16 operand rounding (state as read by the chunk's
matmuls, W, A, V', decayed k), not drift. The state master copy stays f32.

## 3. Min-T threshold and kernel time, raw/v2/timing.out, trace-*.err

Per launch, tight loop, zero data (us; sequential = v2 below 64 tokens, v3 from 64):

| T | 35B shape seq | chunk | ratio | tp2 shard seq | chunk | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 33 | 23.5 | 27.3 | 0.86 | 21.5 | 25.5 | 0.84 |
| 48 | 33.9 | 27.7 | 1.23 | 29.3 | 26.1 | 1.12 |
| 64 | 42.8 | 29.3 | 1.46 | 36.5 | 26.7 | 1.37 |
| 128 | 79.9 | 46.9 | 1.70 | 67.6 | 32.6 | 2.08 |
| 512 | 302.0 | 84.6 | 3.57 | 254.0 | 72.3 | 3.51 |
| 1024 | 605.4 | 155.4 | 3.89 | 501.6 | 124.8 | 4.02 |
| 2048 | 1196.3 | 338.8 | 3.53 | 1006.3 | 234.9 | 4.28 |
| 4096 | 2379.1 | 911.5 | 2.61 | 1997.1 | 464.6 | 4.30 |

Crossover between 33 and 48 tokens; `GDN_CHUNK_MIN_TOKENS = 64` (one full chunk, v3's own bound). Not understood: at the full
35B shape the per-chunk cost more than doubles past T = 1024 (the tp2 shard stays linear). Not a queue stall (trace gaps are
3.8 us). Untested ideas: scan pieces (`ik_split_pieces`, the launcher has the cut points; its 14 us / chunk estimate never
triggers a split at 4096), a 2048-token prefill chunk.

`INTARSIA_HIP_TRACE=1`, 4420-token prompt, 35B tp1: delta rule 88.1 ms (v3, 60 launches, 20.7 % of kernel time) -> 33.1 ms
(chunk, 8.9 %), 2.66x; prefill 443 -> 391 ms in the traced runs.

## 4. Gate 2: 35B NVFP4 accuracy, knob on vs off (one card), raw/v2

| check | knob off | knob on | verdict |
|---|---|---|---|
| short golden (vLLM), prefill: top-1 / KL21 scored | 98.61 % / 5.98e-3 | 98.47 % / 4.70e-3 | golden gate PASS both (floor 3e-2) |
| short golden, decode | 98.75 % / 1.21e-2 | 98.75 % / 1.18e-2 | PASS both |
| long golden (to 16k tokens), prefill | 94.32 % / 1.420e-2 | 94.27 % / 1.430e-2 | PASS both |
| long golden, decode | 93.75 % / 1.28e-2 | 92.50 % / 1.22e-2 | PASS both |
| **change vs base, short** (`--self-golden --self-vs-ref-noise`), prefill mean KL21 | 2.6e-12 (control) | **3.455e-3, limit 3.161e-3** | **FAIL** (1.09x); 10 flips of 1441 (0.69 %), max position KL 0.44 |
| change vs base, short, decode | - | 1.01e-3 (limit 3.16e-3) | mean ok, one prompt (`code`) over 3x its own noise |
| **change vs base, long**, prefill mean KL21 | - | **2.791e-2, limit 2.218e-2** | **FAIL** (1.26x); 1633 flips of 36887 (4.43 %), max position KL 24.1 |
| change vs base, long, decode | - | 7.31e-3 (limit 2.218e-2) | ok |
| tight self-golden budgets (mean KL <= 1e-4) | PASS | FAIL | expected for any non-bitwise kernel |
| `qualify --quick-only` | PASS (all stages; acc-short 98.61 % / 7.63e-3) | **FAIL**: acc-short (recorded baseline gate), acc-oracle (logits moved vs the recorded HIP dump: mean KL 9.64e-3, limit 1e-4); doctor, recipe, plan, kernels (2202 cases), gate, bench PASS | baselines would have to be re-recorded |
| greedy ids, 256 tokens, short prompt | - | identical | |
| greedy ids, 256 tokens, 4k prompt | - | first difference at token 132 | |

Reading: against the vLLM reference the knob changes nothing measurable (KL within +-1e-3 either way), but the logits move
against the default build by about as much as vLLM's own run-to-run noise, and the repo's A/B rule ("a change must be no
larger than the reference's run-to-run KL") fails on both goldens. The first version (normalised q / k rounded to bf16,
min T 128) measured 3.109e-3 (short, 0.98x, pass) and 2.868e-2 (long, fail): raw/v1. Removing the q / k rounding did not move
the model-level KL visibly (the short-golden pair is confounded: the min T went from 128 to 64 at the same time, so more calls
are chunked; the long golden is dominated by full chunks either way). The remaining sources (state / W / A / V' in bf16) are
inherent to bf16 WMMA operands. Whether a much smaller perturbation would land well under the noise limit was not measured.

## 5. Gate 3: prefill speed, raw/speed (localeval, same binary, rounds interleaved, 3 each)

35B NVFP4, `serve`, prefill tok/s = prompt tokens / TTFT:

| arm | prompt | off (3 runs) | on (3 runs) | median off -> on |
|---|---:|---|---|---|
| tp1 | 1k (~991) | 7248 / 7357 / 7068 | 7929 / 8331 / 7949 | 7248 -> 7949, **+9.7 %** |
| tp1 | 4k (~4089) | 10507 / 10414 / 10277 | 10629 / 11840 / 11952 | 10414 -> 11840, **+13.7 %** |
| tp1 | 16k (~16670) | 9232 / 9455 / 9463 | 10689 / 10652 / 10682 | 9455 -> 10682, **+13.0 %** |
| tp2 | 4k (~4089) | 12534 / 12594 / 12651 | 14780 / 14785 / 14610 | 12594 -> 14780, **+17.4 %** |

(1k is under 1024 uncached tokens, where TTFT includes the request overhead.) Below the brief's 15-25 % estimate at tp1,
inside it at tp2. Decode is unchanged (165 tok/s in both qualify runs).

## 6. Gate 4: builds and tests

- Mac: `cargo test -p intarsia-backend-hip` (stub build) and `cargo test -p intarsia-core -p intarsia-backend-cpu`: pass.
- tatooine image: `cargo build --release -p intarsia-cli --features hip`: ok (raw/build.log is the first build);
  `cargo test --release -p intarsia-backend-hip --test gdn_timing`: ok.

## 7. Not done / not verified

- The 27B (Qwen3.8-27B MXFP4) was not run; its GDN shape is expected to be accepted (dk = dv = 128) but is unmeasured.
- No tp2 accuracy run (tp2 speed only); no `serve --max-batch` / packed-prefill accuracy run (the op is per sequence there).
- The superlinear launch time at the 35B shape past T = 1024 (section 3) is not explained.
- CUDA / Metal: untouched (the new sweep cases run on every backend's `GdnDeltaRule` kernels; cpu passes locally).
- Radiance downtime: five fleetq windows, about 30 minutes in total (the first kernel-test window roughly 2 min, not logged to
  the second; then 1.4 / 9.9 / 13.6 / 3.4 min from the window logs), each restored and checked before the release.

## 8. Recommendation

Keep it opt-in. Do not flip the HIP default: the change-vs-base gate fails on both goldens and `qualify --quick-only` fails
on its recorded baselines. If the +13 % (tp1) / +17 % (tp2) prefill is wanted as default, it is a decision to accept a
reference-noise-sized logit change and re-record the HIP baselines, not a kernel fix; revert is unsetting the knob.

# Metal 35B: MTP (k=2) on by default

Lane `lane-fm/metal35-mtp-default`, off main `602a1fcfe`. Qwen3.6-35B-A3B NVFP4 on Metal (endor, M5 Max), single sequence, `--spec mtp --spec-k 2`.

## Approval

User, 2026-11-05 17:06 EST: "flip 1" (relayed by the coordinator): make MTP speculation (k=2) the default for the Qwen3.6-35B-A3B NVFP4 MoE on Metal. The gate set below is the coordinator's brief; a greedy-id difference against `--spec off` stops the merge and goes back to the user.

## Change

* `recipes/qwen3_5_moe.toml` `[tuning.metal] mtp_default_k = 2` (comment cites this doc). No Rust change: `Recipe::resolve_mtp_default(backend)` and `intarsia_models::load::default_k` are backend-generic and carry every CUDA exclusion already: `--spec off`, `INTARSIA_SPEC_DEFAULT=0`, `--max-batch` above 1, a configured prefix store, a checkpoint without the MTP head (all covered by `single_rank_mtp_default_needs_no_spec_one_rank_the_backend_one_sequence_and_no_opt_out`).
* `crates/intarsia-models/tests/tp_mtp.rs` `only_the_measured_recipe_backend_pairs_default_to_mtp` lists `qwen3_5_moe metal tp1 2` (the G0 pair test: every other recipe and backend still runs without `--spec`).
* Docs: `docs/tuning.md` (MTP row), `docs/recipes.md` (`mtp_default_k` row), `docs/decisions/LOG.md` (default row and a dated decision row quoting the approval).
* No `mtp_draft_vocab` key: not asked for, and the 35B on CUDA has none either (the full head). The pipelined step `INTARSIA_METAL_MTP_GPU_ACCEPT` (docs/results/2026-10-29-mmtp-metal-gpu-accept.md) is on by default and is part of what is measured.

## Method

Binary `.scratch/intarsia.branch` (head `133bbf3f4`, release, `--features metal`), `intarsia serve --backend metal --model Qwen3.6-35B-A3B-NVFP4 --max-seq 20000 --profile-every 0 --no-supervisor --no-ui`, no `--max-batch` (single sequence). Arm **mtp** = no `--spec` flag at all (the recipe default must engage: server stderr carries `--spec mtp --spec-k 2: the default of recipe qwen3_5_moe on metal at tp=1`, present in both mtp arms' server logs, absent in both off arms); arm **off** = `--spec off`. Same binary, one fleetq slot (`--exclusive`, job #8542), one server per arm, ABAB: r1 mtp, r2 off, r3 mtp, r4 off.

Per arm: (1) `localeval speed --sizes 1024,4096,16384 --gen 256 --reps 3 --thinking off` (filler text, nonce first; TTFT, prefill, decode as localeval reports them); (2) the **natural-v1** prompts (suite.v1.json, sha256-checked: 1k technical prose, 4k Python module with tests and a bug report, 16k mixed packet with code) sent through localeval's own `Client` (`raw/scripts/nat.py`, 1 warmup + 3 reps per size, nonce first, thinking off, 256 forced tokens). `localeval suite speed-spec` cannot do this: it runs only on a spec lane and aborts the `--spec off` arm, and `localeval speed` sends filler only, so the natural text needed a thin driver on the same client. Peak memory: `/usr/bin/time -l` (peak memory footprint) and `footprint <pid>` sampled every 3 s.

Coverage gap, stated: natural-v1 has one category per size (1k prose, 4k code, 16k mixed), so there is no code prompt at 1k and no chat prompt at 4k. The filler rows are the same at every size but are repetitive text.

## Results

localeval run ids (unanimis): r1 mtp `20261105-173429-adhoc-flipmtp-r1-mtp` (aa8baf95), r2 off `...-173610-...-r2-off` (e86f2b60), r3 mtp `...-173808-...-r3-mtp` (05ae276f), r4 off `...-173937-...-r4-off` (eac9206d). fleetq #8542. Raw: `raw/ab/`, `raw/localeval/`.

**Decode tok/s** (median of 3 reps per arm run; the two runs per arm in order):

| text | context | MTP default | `--spec off` | ratio (mean of 2) |
|---|---|---|---|---|
| natural-v1 prose | 1k | 149.7 / 163.4 | 112.4 / 112.9 | **1.39** |
| natural-v1 code | 4k | 149.6 / 150.4 | 113.8 / 113.8 | **1.32** |
| natural-v1 mixed (code, prose, data) | 16k | 122.9 / 127.5 | 106.6 / 107.3 | **1.17** |
| filler | 1k | 140.4 / 148.9 | 121.9 / 120.6 | 1.19 |
| filler | 4k | 136.0 / 140.5 | 118.9 / 114.2 | 1.19 |
| filler | 16k | 109.9 / 119.6 | 102.7 / 102.8 | 1.12 |

Every MTP run beats every off run at every size and text. Acceptance in the generate runs of the ids gate is 57 to 100 % (2.1 to 3.0 tokens per pass); the served runs print no spec stats line.

**Prefill and TTFT** (first window; the 4k filler row is the concern, see "Prefill window"):

| text | context | prompt tokens | TTFT s, MTP | TTFT s, off | prefill tok/s, MTP | prefill tok/s, off | ratio |
|---|---|---|---|---|---|---|---|
| natural-v1 | 1k | 1027 | 0.27 / 0.23 | 0.24 / 0.24 | 3861 / 4466 | 4216 / 4243 | 0.98 |
| natural-v1 | 4k | 4074 | 0.94 / 0.93 | 0.93 / 1.07 | 4342 / 4371 | 4370 / 3812 | 1.07 |
| natural-v1 | 16k | 16401 | 4.61 / 4.83 | 4.76 / 4.95 | 3561 / 3392 | 3445 / 3316 | 1.03 |
| filler | 1k | 990 | 0.27 / 0.26 | 0.22 / 0.22 | 3708 / 3862 | 4564 / 4580 | 0.83 |
| filler | 4k | 4089 | 0.97 / 0.93 | 0.70 / 0.76 | 4224 / 4409 | 5831 / 5362 | 0.77 |
| filler | 16k | 16654 | 4.81 / 4.59 | 4.88 / 5.06 | 3461 / 3626 | 3412 / 3291 | 1.06 |

**Peak memory** (process, endor, max-seq 20000): peak memory footprint 22.81 / 22.42 GB with MTP against 20.59 / 20.58 GB off (+2.0 GB with the MTP head loaded; the split between head weights, KV and workspaces was not traced); maximum resident set 41.8 GB against 38.6 GB. Sampled `footprint` maxima 21 GB against 19 GB.

**Served greedy ids** (`/v1/completions`, raw natural-v1 text, 256 tokens, `ignore_eos`): r1 (mtp) against r2 (off) at 987, 4035 and 16357 prompt tokens: **identical** in all three.

## Greedy ids, generate (raw/ids-summary.txt, fleetq #8542)

`intarsia generate --backend metal`, 256 greedy ids with `--ignore-eos --show-ids`, three arms per prompt: `--spec off`, the recipe default (no flag; the stderr line and a `spec:` stats line are required), explicit `--spec mtp --spec-k 2`. 24 prompts: the 8 `tools/prompts/golden.jsonl` prompts (4 raw, 4 chat reduced to the last user turn), `golden_4k` and `golden_12k` corpora (character slices of `corpus_docs.txt` / `corpus_code.txt`, prompt token counts in the logs), a 700-token prose slice, the 16384- and 32764-token long-context prompts, and `scripts/cmtp/ids-prompts` short-0..2, 4k-0..2, 16k-0..2 (chat, no think).

**48 of 48 comparisons are identical** (24 default against off, 24 explicit k2 against off); no divergence anywhere, including 12k, 16k and 32k contexts (acceptance 62 to 100 %).

## Prefill window (fleetq #8613): the filler-4k cost was noise

The first window showed filler prefill at 4k 0.77x of off (TTFT 0.93 to 0.97 s against 0.70 to 0.76 s) while natural text showed 1.07x. A second ABAB (r1 mtp, r2 off, r3 mtp, r4 off; 4k and 16k, 5 reps per size, filler and natural-v1, `GEN=16`; raw `raw/ab-prefill/`, localeval runs `20261105-194717-adhoc-flipmtp-r1-mtp` (cf12224c), `...-194840-...-r2-off` (dcd8c3d2), `...-195022-...-r3-mtp` (4112b513), `...-195146-...-r4-off` (cd73d996)):

| text | context | TTFT s, MTP | TTFT s, off | prefill tok/s, MTP | prefill tok/s, off | ratio |
|---|---|---|---|---|---|---|
| filler | 4k | 0.788 / 0.827 | 0.832 / 0.829 | 5188 / 4946 | 4916 / 4936 | 1.03 |
| filler | 16k | 3.84 / 4.00 | 4.38 / 3.95 | 4337 / 4168 | 3800 / 4219 | 1.06 |
| natural-v1 | 4k | 0.708 / 0.708 | 0.767 / 0.704 | 5757 / 5761 | 5315 / 5786 | 1.04 |
| natural-v1 | 16k | 4.01 / 3.86 | 4.18 / 3.75 | 4093 / 4248 | 3921 / 4373 | 1.01 |

**Verdict: no measurable prefill cost.** With 5 reps per cell the arms are within 6 % of each other in both directions (MTP ahead in 3 of 4 cells on both runs), and the whole window ran about 25 % faster than the first one (natural 4k: 5760 against 4350 tok/s with MTP), so the first window's filler 4k off arm was a fast outlier (0.70 s, against 0.83 s here), not an MTP cost. Decode figures of this window (`GEN=16`) are not used. The first window's other prefill cells (natural 1k/4k/16k, filler 16k) agreed with this one.

## Gates

| gate | result |
|---|---|
| `cargo test --workspace --no-fail-fast` on the merged head | **PASS**: 1353 passed, 0 failed, 32 ignored (debug, fleetq #8725, `raw/cargo-test-workspace.txt`; on the pre-merge head 1322 / 0 / 29, #8562) |
| greedy ids, 24 prompts x (default, explicit k2) against `--spec off`, pre-merge binary | **48 / 48 identical** (#8542, `raw/ids-summary.txt`) |
| same, merged head `1cb31e6bb` + recipe-comment edit | **48 / 48 identical** (#8726, `raw/ids-merged-summary.txt`) |
| served greedy ids, natural-v1 1k / 4k / 16k, mtp against off | identical (#8542) |
| `plan show` (text and JSON), main against branch: 8B, Llama-3.1-8B, Qwen2.5-7B, 27B | **byte-identical** (#8613, `raw/gates/plan/`) |
| `plan show` 35B | differs by **8 added lines and 0 removed**: the MTP head's ops (`add_bf16`, `matmul_bf16_gemv`, `matmul_bf16_nax`, `attn_prefill_bf16_splitkv_rows`, `matmul_fp8_gemv_rows`) now appear because the head loads by default; expected, and the stderr shows `--spec mtp --spec-k 2: the default of recipe qwen3_5_moe on metal` |
| `kernels test --backend metal`, main and branch | both 49 kernels, 980 cases, 0 FAIL |
| greedy ids 256, 8B and 27B, short and ~3.5k prompt, main against branch | **identical** (4 of 4), plan json and txt byte-identical (`raw/gates/noop/`) |
| CUDA `plan show` on spark1 | **not run**: the footprint check lists only `crates/intarsia-models/tests/tp_mtp.rs` (a test) and `recipes/` (a `[tuning.metal]` key, embedded recipe), spark1 was held exclusively by another job and has no git checkout to build the two CUDA binaries from |

## Merge with main 1a2a246f6 (#312 batched MTP N=1, #313 sampled speculation)

`git merge origin/main` was automatic, no conflicts; main added `batched_mtp = 1` to `[tuning.cuda]` and `[tuning.metal]` of the same recipe beside the keys here. Two comment lines were stale after it and are rewritten: main's "Metal has no `--spec mtp` default here" and this branch's "`--max-batch` above 1 keeps plain decode" (a `--max-batch` server now takes the default through `batched_mtp`; main's test `batched_mtp_default_only_where_measured_and_env_wins` covers it and passes). The combination, run with no flags on the merged binary (`raw/comb/`, `raw/comb.log`, fleetq #8726):

* `serve`: startup lines `--spec mtp --spec-k 2: the default of recipe qwen3_5_moe on metal at tp=1`, `speculative decoding: Mtp, up to 2 drafts per verify pass` and `sampled requests (temperature > 0): speculation Match (default; single sequence, one rank)`; a T=0 and a T=0.7 chat request both complete.
* `generate` with no flags at T=0.7 (seed 3): `spec: verify passes=41 proposed=82 accepted=54 (acceptance 65.9 %)`, so a sampled request speculates.
* `serve --max-batch 4` with no `--spec`: the same default line and `batched MTP N=1 (recipe qwen3_5_moe, [tuning.metal] batched_mtp)`.
* Greedy ids against `--spec off`: 48 / 48 identical, as before the merge.

Not run on the merged head: the served ABAB speed (the speed table is from the pre-merge binary; the merge adds no kernel change to the `--max-batch 1` path) and the CUDA plan-show.

## Recommendation

Ready to merge. Decode x1.17 to x1.39 on natural text (x1.12 to x1.19 on filler), prefill unchanged within noise, +2.0 GB peak footprint, greedy ids identical on 48 of 48 prompts including 12k, 16k and 32k contexts. Opt-outs: `--spec off`, `INTARSIA_SPEC_DEFAULT=0`.

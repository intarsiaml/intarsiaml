# NMD: the new-model drill on Qwen3.5-4B, Metal, endor (2026-10-27 morning)

Lane NMD, branch `lane-nmd/new-model-drill`, build origin/main `1f8764f92` plus the one engine fix below. All times are the box clock
(EDT), `date` stamps in `raw/2026-10-27-nmd/stopwatch.log`. fleetq jobs on endor: #6099, #6100, #6102, #6107, #6125 to #6128, #6135, #6136,
#6142 (next to `endor-kev` and `endor-tensorfold`, which stayed up). Requirement #1 of the user, as quoted in
`2026-10-27-g4d-gemma4-drill.md`: a basic test within 1 to 2 hours of the weights, in-depth within 2 to 4, quick numbers publishable while
the in-depth tier continues.

**Model.** `Qwen/Qwen3.5-4B` (bf16, 8.7 GB, `Qwen3_5ForConditionalGeneration`: 24 GatedDeltaNet + 8 gated-attention layers, tied embeddings,
MTP head present but unused), from the HF cache on endor. Why this one: the engine had never run it (only mlx-lm had, in old bench
files), `intarsia recipe new` says `covered: builtin:qwen3_5_vl already handles this model (45/45 tensor templates resolve, quant: bf16)`,
so it is a new size of a supported family; it is a recent Qwen the user cares about; it fits the 60 GB free next to the standing services.
Nothing was downloaded. The other candidates on disk were either already onboarded (Qwen3-8B, Qwen2.5-7B, Llama-3.1-8B, Qwen3.8-27B,
Qwen3.6-35B), needed a new block (Nemotron-3.5-Lightning is `nemotron_h`, LFM2.5 is `lfm2_moe`) or are 18 GB+ bases (Qwen3.5-9B-Base).

## Verdict

**The time targets are met, but only after a fix and with a FAIL that is reference noise.** Stopwatch T0 = 04:53:34 (weights on disk).
Quick numbers published at +48 min, in-depth report at +51 min, the explanation of the one FAIL at +57 min; Metal speed from localeval at +58 min.
Two things made it cost 45 minutes before the first token, neither of them an onboarding-pipeline fault: a 34-minute CPU golden for the
4k set (a hybrid model gets no golden from `qualify`) and one engine gap (f32 `A_log` and gated-norm weights, fixed here, 5 minutes once found).
With the golden running beside the first load, first token would have been at about +15 min (section "What I would do differently").

| Stage | Start | End | Elapsed | Note |
|---|---|---|---|---|
| T0, build `cargo build --release` (no `--features metal`) | 04:53:34 | 04:53:59 | 25 s | wrong build: friction 4 |
| fleetq wait for onboard run 1 (`--quiet gpu,cpu`) | 04:54:24 | 04:58:53 | 4 min 29 s | behind another lane's `cargo test` |
| onboard run 1: `recipe` PASS, `smoke` ERROR "built without the Metal backend" | 04:58:53 | 04:58:53 | 0 s | friction 4 |
| rebuild with `--features metal` | 04:59:04 | 04:59:18 | 14 s | |
| HF fp32 golden, short + 700-token set (`tools/golden.py`, CPU) | 04:58:53 | 04:59:58 | 1 min 5 s | by hand: friction 3 |
| HF fp32 golden, 4k set `--topk 20` (3 prompts, CPU) | 04:59:58 | 05:33:53 | **33 min 55 s** | 548 s, 753 s, 729 s per prompt; holds the CPU |
| onboard run 2: `smoke` ERROR "no kernel for gdn_delta_rule ... w=F32" | 05:33:58 | 05:33:58 | 0 s | friction 1 |
| read code, fix `A_log`, rebuild, run 3: `smoke` ERROR "metal w: expected bf16, got f32" | 05:34 | 05:37:31 | 3 min 30 s | friction 1 |
| fix gated-norm weight, rebuild, resubmit | 05:37:31 | 05:39:07 | 1 min 36 s | |
| onboard run 4 (same `--out`, resumed): `smoke` PASS, **first token** | 05:39:07 | 05:39:10 | 1 s (load 0.3 s, prefill 5 tokens 204 ms) | **+45 min 36 s** |
| `quick` tier (qualify `--quick-only`) | 05:39:10 | 05:41:43 | 2 min 34 s | **quick.md published, +48 min 9 s**; verdict FAIL (acc-short) |
| `in-depth` tier | 05:41:43 | 05:45:02 | 3 min 19 s | **in-depth.md published, +51 min 28 s**; verdict FAIL (acc-short, acc-long) |
| HF bf16 golden (noise check, by hand) + comparison | 05:48:57 | 05:50:37 | 1 min 40 s | explains the FAIL: +57 min |
| `intarsia serve` load + localeval speed sweep (128, 1k, 4k, 16k) | 05:51:08 | 05:52:23 | 1 min 15 s | unanimis 22be5b7f; +58 min 49 s |
| localeval MMLU `--limit 10` (n = 570, 5-shot, thinking off) | 05:57:04 | 06:31:43 | 34 min 39 s | unanimis b84bcc40; +1 h 38 min |

Targets: basic test within 1 to 2 h: **48 min (quick.md) / 57 min (with the FAIL explained)**. In-depth within 2 to 4 h: **51 min** for what
onboard's in-depth tier contains (it is 3 minutes of machine time here; see friction 6).

## Numbers

| Result | Value | Where |
|---|---|---|
| `onboard` verdict | FAIL (exit 1): 15 stages, 11 pass or skip, `acc-short` FAIL, `acc-long` FAIL, all on the absolute top-1 floor | `raw/2026-10-27-nmd/quick.md`, `in-depth.md` |
| Accuracy vs HF fp32 golden, short set (10 prompts) | prefill top-1 99.72 % (1441 rows), KL21 1.43e-4; decode top-1 **98.75 %** (2 flips in 160 rows, floor 99.5), KL21 1.70e-4 | `acc-short.out` |
| Accuracy vs HF fp32 golden, 4k set (3 prompts, top-20) | prefill top-1 **99.08 %** (12759 rows), KL21 1.20e-4 (floor 1e-3, PASS); decode top-1 **99.48 %** (192 rows), KL21 1.46e-4 | `acc-long.out` |
| **Noise check**: HF bf16 against HF fp32 on the same 1441 short-set prefill rows | top-1 **98.96 %**, full-vocabulary KL 3.2e-4 | this run, section "Reading the FAIL" |
| Chat-form perplexity gate | 2.02, 5/5 prompts, no non-finite rows | `gate` |
| Decode (localeval, forced 128 tokens, 3 reps) | 61.5 tok/s at 114 prompt tokens, 61.5 at 992, 61.0 at 4089, 57.6 at 16.7k | unanimis 22be5b7f |
| Prefill | 2258 (114 tok, overhead-bound), 4835 (992), 4623 (4089; marginal 4554), 2767 (16.7k; marginal 2448) tok/s; TTFT 6.0 s at 16.7k | same |
| Decode vs the byte floor (8.41 GB per token, 614 GB/s) | 84 % at 1k and 4k (onboard's `intarsia bench`: 54.1 / 60.5 / 60.7 tok/s at 128 / 1k / 4k) | `quick.md` |
| MMLU `--limit 10` (n = 570) | **0.751** (SE about 0.018), 0 truncated, mean reply 153 tokens although thinking is off (the model explains its answer), 2051 s | unanimis b84bcc40 |

Every speed cell is one run (3 reps). The decode figure is eager (`decode ran eagerly` in the bench warnings, no graph replay on Metal), at 84 %
of the byte floor. Not compared with mlx-lm on the same weights (no time).

## Reading the FAIL (not an engine fault, a gate that no bf16 run can meet)

`acc-short` and `acc-long` fail only the **absolute top-1 floor of 99.5 %** that applies to a bf16 model with no committed baseline; the KL
gates pass by a factor of 7 or more (KL21 1.2e-4 to 1.7e-4 against 1e-3). The golden is HF **fp32**, as the docs ask for hybrid models.
HF's own bf16 run agrees with its fp32 run on only 98.96 % of the 1441 short-set prefill rows (KL 3.2e-4), so the engine (99.72 % on the same
rows, KL 1.43e-4) is closer to the fp32 reference than a bf16 HF run is. On the 160 decode rows one flip is 0.625 %, so a single near-tie
flip already fails the floor. The same step is how `qualify` handles KL (`ref_noise`), but top-1 has no such record.

Caveat I did not close: the engine rounds `A_log` and the gated-norm weight to bf16 at load (the fix below). The conversions from f32
that the MLX and ModelOpt checkpoints went through do the same, and the KL numbers are at the class level of the other Qwen3.5 runs, but I did not
run the cpu backend (f32 `A_log`) on the same goldens to isolate the rounding.

## Friction, ranked, each with the fix

1. **Engine gap, fixed here: the original HF Qwen3.5 checkpoints store `A_log` and the GatedDeltaNet norm weight as f32.** The 27B and 35B
   MLX/ModelOpt checkpoints store bf16. The Metal (and CUDA) GatedDeltaNet kernels are registered with `weights(bf16)`, so the load stopped
   with `no kernel for gdn_delta_rule ... (m=1 n=128 k=128 act=Bf16 w=F32)`, which does not name the tensor; after the first fix the plan
   admitted, but the first generate failed with `invalid op call: metal w: expected bf16, got f32` (the gated norm), which names no op. Any
   unquantised HF Qwen3.5 checkpoint hits it, on every GPU backend. Fix in this branch: `gpu_f32_to_bf16` in `crates/intarsia-core/src/model.rs`
   rounds both tensors to bf16 once at load on non-cpu backends (the cpu reference keeps f32). It only changes a case that used to fail to load;
   bf16 checkpoints go through unchanged. Larger, not built: the admission at load time should check the weight dtype of every op (the
   gated-norm mismatch passed `plan` and failed at run time), and the error should name the tensor.
2. **A correct bf16 model FAILs the absolute top-1 floor against an fp32 golden (above).** The verdict of `onboard` for a healthy model is
   FAIL (exit 1), and a reader must run the noise check by hand to know. Proposed fix: have `qualify` take the top-1 floor from the golden's
   own reference spread (extend `ref_noise` with a top-1 field, `intarsia ref-noise` already compares two goldens) and make the HF bf16 golden for
   a dense bf16 model as part of the same step (94 s here); also report the number of rows behind the floor (one flip in 160 is 0.6 %).
3. **`qualify` makes no golden for a hybrid model, so the user makes it by hand, and the 4k set costs 34 minutes of CPU on a 4B model.**
   The doc said so (`wf.hybrid` is excluded in `qualify.rs`, "no MoE / hybrid / quantisation"), but gave no command, file name or
   timing; I read `qualify.rs` and `facts.rs` to learn the name rule (`qwen3.5-4b-fp32`, found by the directory name or the golden's `model`
   metadata) and `--golden-dir` after `--`. Fixed in the doc (new bullet with the commands). Larger, not built: (a) let `qualify` auto-make
   the short golden for a dense bf16 hybrid model (1 minute here; the HF path works for it), (b) the 4k golden runs HF's PyTorch fallback
   for the GatedDeltaNet recurrence (`flash-linear-attention` and `causal_conv1d` are not installed), 548 to 753 s per prompt; try
   `--device mps` or fewer 4k prompts for hybrids. While the CPU golden runs, a `--quiet gpu,cpu` timing job cannot start.
4. **The doc never says the binary needs `--features metal` (or `cuda`, `hip`).** A plain `cargo build` ends `smoke` with a clear message,
   but only after fleetq queueing (4.5 min). Fixed in the doc (a Gotchas bullet); `scripts/deploy.sh` already does it.
5. **A 4k golden given alone is taken as the long golden and `acc-4k` reads "covered by acc-long".** Correct, but it reads like a skipped
   stage in the report; `acc-long` also ran the prefix-reuse check on it (bit-identical, PASS). Noted in the doc bullet of item 3.
6. **The in-depth tier is 3 minutes here.** With no `--reference` server, no CPU-oracle golden and no `--tp`, `ab`, `acc-oracle`, `acc-format` and
   `tp` are SKIPs. The requirement's "2 to 4 hours of in-depth" is therefore the localeval quality set (MMLU, GSM8K) that the user runs by hand
   after onboard; onboard could end by printing the exact `localeval` commands (and the server command) for a model it just qualified.
   MMLU `--limit 25` is 1425 questions at 3.6 s each (mean reply 153 tokens at 61 tok/s), about 85 min on this 4B; I started it, stopped it at 5 subjects and ran `--limit 10` (n = 570, 34 min) instead.
7. **Reading code or guessing, complete list:** `qualify.rs` golden rules (item 3); `kernels.rs`, `gdn.rs`, `model.rs` for the no-kernel error
   (item 1); the checkpoint's tensor dtypes (a Python script over the safetensors headers); the `intarsia serve` flags and `localeval --url` for a
   local server (no doc for "serve the model I just qualified and benchmark it"; `docs/benchmarking.md` was not needed).

## What I would do differently

Run `smoke` first and the golden beside it: the load error would have appeared at 04:59 instead of 05:34, and with the 5-minute fix the
first token would have been at about +15 min, quick numbers at about +20 min. The 34-minute 4k golden is then the only thing between
there and the in-depth report; it can run while the quick numbers are published, which is what the requirement asks for.

## Not done

* `cpu` backend run on the same goldens to separate the bf16 rounding of `A_log`/norm from the Metal numerics (above).
* GSM8K, mlx-lm / llama.cpp same-weights comparison, `--spec mtp` (the checkpoint carries an MTP head; the recipe has `[mtp]` for the 27B).
* The goldens are not committed (the short fp32 golden is 716 MB); the commands to remake them are in `docs/adding-a-model.md`.
* No CUDA/HIP no-op ids for the loader change, on this evidence: the cast fires only for f32 `A_log` / gated-norm tensors on non-cpu backends
  (the cpu backend reports its name as `cpu`, `intarsia-backend-cpu/src/lib.rs:109`). The `A_log` and `linear_attn.norm.weight` headers of
  every Qwen3.5-family checkpoint on endor's disk that the fleet runs (`Qwen3.6-35B-A3B-NVFP4`, `Qwen3.8-27B-Quark-AWQ-MXFP4`, `qwen38-27b`)
  are all BF16, so for them the helper returns the tensor untouched; the CUDA and HIP `gdn_delta_rule` entries are registered `weights(bf)` like
  Metal's, so f32 `A_log` could not load there. Not checked: whether a CUDA or HIP gated-rmsnorm kernel accepts an f32 norm weight (a checkpoint
  with bf16 `A_log` and f32 norm would then change numerics there); none is known. `cargo test --release --locked -p intarsia-core` (fleetq #6143): 133 + 2 + 34 passed, 0 failed.

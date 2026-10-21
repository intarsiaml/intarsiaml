# M3: accuracy debt — the 35b-long prefill-floor FAIL and HIP prefix reuse at chunk 1024 (2026-10-21)

Lane M3 of `docs/plans/2026-10-21-quota-burn-plan.md`. Base `origin/main` 4be7b7d4, the same engine code as a7febf48
(`git diff --stat a7febf48 4be7b7d4 -- crates kernels` is empty). Model `nvidia/Qwen3.6-35B-A3B-NVFP4`. Branch
`accuracy/m3-debt`: docs, analysis scripts (`scripts/m3-accuracy/`) and raw logs only. **No engine code changed, and no
gate changed.** Raw material is in `docs/results/2026-10-21-m3-accuracy-debt/raw/`.

## 1. Headline

- **The 35b-long prefill-floor FAIL is not an engine defect. It comes from the system and user-turn rows of one prompt.**
  - `chat_reuse_4k` is 12.6 % of the rows and carries 71-73 % of the KL21 sum.
  - Inside that prompt, the KL sits in the 4,491 rows of the system and user turns. Serving never asks the model to
    predict those tokens. The assistant turns of the same prompt score 5e-3 to 2.7e-2.
  - Without those rows, HIP main scores **1.420e-2** against the 3.0e-2 floor (tool-exact per-row dump; 4.356e-2 with
    them).
  - The plain-text prompts do not grow with position. `long_docs_12k` stays at 1.6-3.5e-2 in every 2k bin up to 12k.
    That rules out the long-position candidates: rope/yarn, accumulation drift and GDN chunk-boundary carry.
- **The f64 CPU oracle sides with our GPU backends, not with the golden.**
  - On `chat_reuse_4k` the CPU reference (f32 accumulation; NVFP4, MoE and GDN in f64; bf16 KV) scores **2.628e-1**
    against the golden. HIP main scores 2.479e-1.
  - Offline, CPU sits 0.14-0.15 from HIP and from CUDA, and 0.20-0.21 from every vLLM sample.
  - So the remaining candidates, the GPU accumulation precision and the NVFP4 activation path, are not the cause either.
    There is no direction for a per-layer bisect: the error does not grow with position, and the most exact
    implementation is not closer to the golden. Section 2.7 says why the bisect was not run.
- **Mechanism.** In a long user turn the model keeps weighing "end the user message now".
  - All implementations put mean P(`<|im_end|>`) of 0.26-0.29 on user-turn rows that are not the end of the turn:
    vLLM 0.289, CUDA 0.269, HIP 0.270, CPU 0.256.
  - On the short chat prompts, `<|im_end|>` is vLLM's top-1 on 42-51 % of the rows. On raw-text prompts it is 0 %.
  - Single rows flip between P ≈ 1 and P ≈ 0.002 under last-bit differences. The top 5 % of rows carry about 80 % of the
    KL, and most of them have `<|im_end|>` mass.
- **The golden is one noisy sample on these rows.** I re-queried the same vLLM server and version with the stored ids.
  - vLLM against its own stored golden on `chat_reuse_4k`: **0.10-0.12 KL21**. Two fresh samples differ by 0.09-0.10.
  - Whole 35b-long set, offline KL21 against the stored golden: fresh vLLM samples 2.0-2.7e-2, HIP main 3.22e-2. Without
    system and user-turn rows: vLLM 8.3-8.7e-3, HIP 1.36e-2.
- **The verdict does not depend on which vLLM sample is the golden.** HIP main against the stored golden and two fresh
  samples scores 4.356e-2, 4.545e-2 and 4.557e-2. All three FAIL, so regenerating the golden does not fix it.
- **HIP prefix reuse at `--prefill-chunk 1024` is not a chunk-1024 bug, and the default chunk never tests it.**
  - The failing case is `long_docs_4k continue` (split 2000). At the default chunk of 4096 the tool **skips** this case
    ("split 2000 is below one prefill chunk"). "Default passes" therefore only means "default does not run it".
  - In the scripted replay (`--reuse-diag`), warm with the reply prefilled equals cold **bit for bit**, 32/32 rows.
    Cold is bit-identical across chunks 512, 1023, 1024 and 2048. The whole replayed warm-vs-cold difference is that
    the 15 reply tokens went through the decode kernels: KL mean 1.26e-2, max 8.45e-2, max |dlogit| 2.28.
  - The cold run sits on a 0.0037-nat near-tie at token 8. The judge's tolerance (0.25 nats / KL 1e-2) was calibrated on
    bf16 Qwen3-8B, so the 35B's decode-vs-prefill difference counts as "beyond rounding".
  - **Open caveat:** the engine's own `generate` rows differ from its scripted replay by KL about 8e-3, on both the warm
    and the cold run, with and without graphs. The bit-exactness above is proven for the replay, not for `generate`
    (section 2.8).
- **Proposals (no gate changed):** gate the 35b-long floor on predicted rows only, and change the prefix-reuse judge.
  Both are in section 7.

## 2. Numbers

### 2.1 Per prompt (HIP main, tool, `--prefill-chunk 4096`; `raw/tatooine/long-c4096.log`, `long-c4096.pos.tsv.gz`)

| prompt | rows | KL21 | share of the KL sum |
|---|---|---|---|
| long_docs_4k | 4064 | 1.674e-2 | 4 % |
| long_code_4k | 4064 | 7.775e-3 | 2 % |
| **chat_reuse_4k** | 4631 | **2.479e-1** | **71 %** |
| long_docs_12k | 12064 | 2.071e-2 | 16 % |
| long_code_12k | 12064 | 9.045e-3 | 7 % |
| OVERALL | 36887 | **4.356e-2** (floor 3.0e-2: FAIL) | |

The same run aggregated from its per-row dump (`posmask.py`; `raw/tatooine/posmask-main.txt`):
- without the system and user-turn rows of chat prompts: **1.420e-2** (32,349 rows);
- without `chat_reuse_4k` entirely: 1.422e-2.

The 2026-10-14 HIP build gave 1.393e-2 on the same measure (`raw/vllm/posmask-hip-1014.txt`).

### 2.2 No growth with position (HIP main, tool-exact rows, mean KL21 per 2048-row bin; `raw/tatooine/bins-main.txt`)

| prompt | bins in order |
|---|---|
| long_docs_12k | 1.6e-2, 1.7e-2, 2.5e-2, 3.5e-2, 1.6e-2, 1.6e-2 |
| long_code_12k | 7.4e-3, 9.6e-3, 6.4e-3, 9.8e-3, 1.1e-2, 1.0e-2 |
| chat_reuse_4k | 1.1e-1, 3.4e-1, 4.3e-1 |

The plain-text prompts score the same 12,000 positions in as at position 0. A precision loss in long-context
attention, in the GDN state carry across chunks, or in rope/yarn at long positions would grow along the prompt, and
none does. `chat_reuse_4k` is already high in its first bin, and its system turn (rows 0-22) scores 2.4-6.0e-2.

### 2.3 Inside chat_reuse_4k: the turns (offline KL21 against the stored golden; `split.py`, `raw/vllm/split-cuda-w3.txt`, `raw/cpu/split-cpu.txt`)

| rows | n | KL21 CUDA / CPU | P(`<|im_end|>`) on non-end rows: vLLM / CUDA / CPU | top-1 agreement CUDA / CPU |
|---|---|---|---|---|
| system turn [0,23) | 23 | 5.4e-2 / 2.4e-2 | 0.469 / 0.421 / 0.482 | 91 / 100 % |
| **user turn with the document [25,4516)** | 4491 | **1.91e-1 / 2.13e-1** | 0.289 / 0.269 / 0.256 | 88 / 87 % |
| assistant "Understood…" [4518,4536) | 18 | 2.7e-2 / 2.1e-2 | 0.505 / 0.542 / 0.521 | 89 / 89 % |
| user question [4538,4562) | 24 | 9.0e-4 | 0.932 / 0.928 | 100 % |
| assistant `<think>` + continuation [4564,4631) | 66 | 4.9e-3 | 0.003 / 0.007 | 98.5 % |

How often `<|im_end|>` is vLLM's top-1 on a row whose real next token is something else (`imend.py`;
`raw/vllm/imend.txt`):
- `chat_reuse_4k`: 31.7 %.
- `chat_short`, `chat_system`, `chat_multiturn` and `chat_reasoning`: 47 %, 42 %, 51 % and 42 %.
- Every raw-text prompt: 0 %.

All implementations reproduce this. CPU on the short set scores 5.8e-3 to 2.5e-2 on those chat prompts, which would be
in the nats if it did not. It is how the model behaves in user turns, which SFT does not train it to predict.

The same rows contain confident misses (`confwrong.py`):
- On **4.23 %** of `chat_reuse_4k` rows, vLLM puts more than 0.9 on one token while the real next token is outside its
  top-20. On the raw-text prompts over the same corpus the rate is 0.03-0.25 %.
- One example: after `--sizes 128,102`, vLLM puts 0.77 on `scribe` and CUDA puts 0.999 on the real `4`.
- CUDA has its own such rows, for example P(`<|im_end|>`) = 1.0 where vLLM has 0.002.

Both predict the user text equally badly. Top-1 equals the real token on 35.0 % of rows for vLLM, 34.8 % for CUDA and
35.7 % for CPU.

### 2.4 Is the golden reproducible? (spark1 vLLM `guidance`, left up; `vllm_requery*.sh`)

I re-sent the stored ids (not retokenised text) to the same server, vLLM 0.26.1rc1.dev602, the same version the golden
records. Every response reported `cached_tokens: 0`, salted or not. So `prompt_logprobs` requests do not read the prefix
cache, which resolves the golden metadata's "unverified" note.

Offline KL21 on `chat_reuse_4k` (`pairs.py`; vLLM samples are top-20, our dumps top-64; `raw/vllm/pairs-chat.txt`):

| pair | all rows | user turn | other rows |
|---|---|---|---|
| stored golden vs fresh sample (same salt twice / two salts / unsalted) | 0.103 / 0.123 / 0.114 / 0.109 | 0.106-0.127 | 4.7e-3 to 1.4e-2 |
| two fresh samples | 0.089-0.104 | 0.092-0.105 | 4e-3 to 8e-3 |
| stored golden vs CUDA 208ed00 / HIP main | 0.186 / 0.188 | 0.19 | 1.5-1.6e-2 |

On `long_docs_4k`, the stored golden vs a fresh sample is 1.20e-2, and vs CUDA 1.68e-2.

Whole 35b-long set, offline KL21 against the stored golden, every candidate truncated to top-20 (`setkl.py`;
`raw/vllm/setkl.txt`):

| candidate | all rows | without system/user-turn rows |
|---|---|---|
| vLLM sample A | 2.37e-2 | 8.6e-3 |
| vLLM sample B | 2.66e-2 | 8.7e-3 |
| vLLM sample A as reference, B as candidate | 2.00e-2 | 8.3e-3 |
| HIP main (c4096 dump) | 3.22e-2 | 1.36e-2 |
| CUDA 208ed00 (W3 dump) | 3.22e-2 | 1.37e-2 |

The same check on 35b-short reads 2.5e-3 and 2.9e-3: the short set is reproducible, and only one prompt of the long set
is not.

### 2.5 Tool-exact: HIP main against three vLLM samples (`raw/tatooine/acc-long-*.log`)

| golden | OVERALL KL21 | verdict |
|---|---|---|
| stored (sha256 `7614f2a3…`) | 4.356e-2 | FAIL |
| fresh sample A | 4.545e-2 | FAIL |
| fresh sample B | 4.557e-2 | FAIL |

### 2.6 CPU reference on chat_reuse_4k (endor, fleetq #2902, 25 min; `raw/cpu/cpu-chat4k.log`)

`--backend cpu --mode prefill --prefill-chunk 256`, top-64 dump. The tool reads **2.628e-1** against the golden (top-1
87.52 %); HIP main reads 2.479e-1. Offline KL21 among the implementations (`raw/cpu/pairs-cpu.txt`):

| pair | all rows | user turn | other rows |
|---|---|---|---|
| vLLM vs vLLM (stored, A, B) | 0.102-0.129 | 0.105-0.133 | 4e-3 to 1.2e-2 |
| HIP main vs CUDA 208ed00 | 0.120 | 0.123 | 9.5e-3 |
| CPU vs HIP main / CPU vs CUDA | 0.139 / 0.146 | 0.142 / 0.150 | 1.5e-2 / 1.8e-2 |
| vLLM (stored, A, B) vs CPU | 0.199-0.207 | 0.204-0.213 | 1.6e-2 to 2.3e-2 |
| vLLM stored vs HIP main / CUDA | 0.188 / 0.186 | 0.19 | 1.5-1.6e-2 |

On this prompt's user turn, every pair of correct implementations differs by 0.10-0.21, even two runs of one server.
The CPU oracle groups with our GPU backends, and all three sit about 0.2 from vLLM. Measured this way the golden cannot
rank implementations: it is measuring a chaotic quantity.

There is also a systematic offset between vLLM and the rest, about 0.2 against 0.10-0.15 within each group. It shows on
the other rows too (1.36e-2 against 8.6e-3). Section 7.4 covers it.

### 2.7 Why no per-layer bisect

The task asked for per-layer hidden-state dumps at long positions, bisected against the golden's source. That source is
the vLLM server, whose API exposes logits only, so a hidden-state bisect could only run against the CPU oracle. Two
measurements make it pointless here:
- the error does not grow with position (2.2);
- the CPU oracle is not closer to the golden than the GPU backends; it is slightly further (2.6).

A per-layer bisect toward the CPU would therefore chase the 0.14 chaos between two correct implementations, not a defect.
The CPU result also covers the NVFP4 activation path and GPU accumulation precision, because the CPU does those in
f32/f64.

### 2.8 Item 2: HIP prefix reuse at `--prefill-chunk 1024` (`--mode reuse --reuse-diag`, `long_docs_4k`-only golden)

`raw/tatooine/reuse-docs4k-c1024.log` (graphs) and `reuse-docs4k-c1024-nographs.log` (`--no-graphs`) carry identical
digits. Results:

| comparison, 32 rows, teacher-forced on the cold stream | bitwise | KL mean / max | max \|dlogit\| |
|---|---|---|---|
| warm vs warm again; cold vs cold again | 32 / 32 | 0 | 0 |
| cold, chunk 1024 vs 512 / 1023 / 2048 (replay) | 32 / 32 each | 0 | 0 |
| **cold vs warm with the reply prefilled, not decoded (replay)** | **32 / 32** | 0 | 0 |
| warm vs cold, same context (replay) | 0 | 1.26e-2 / 8.45e-2 | 2.277 |
| warm vs warm with the reply prefilled, decode-KV only (replay) | 0 | 1.26e-2 / 8.45e-2 | 2.277 |
| engine `generate` warm vs its scripted replay | 0 | 8.77e-3 / 4.47e-2 | 2.347 |
| engine `generate` cold vs its scripted replay | 0 | 8.41e-3 / 4.98e-2 | 3.740 |

The first divergence is at token 8. Warm takes 29832 by +0.183 nats, and cold takes 283 by **+0.0037 nats**. The gate
note (round 9 w6, same build) gives "shared rows differ by 1.238 (before) / 0.771 (here) nats … rounding scale 0.250 …
(re-chunked cold run: 0.000)".

What this shows:
1. **Restore is bit-exact in the replay.** The bookkeeping, the recurrent-state snapshot and the KV restore are exact:
   with the reply prefilled, warm equals cold bit for bit.
   - The replayed warm-vs-cold difference equals, to every digit, the cost of writing the 15 reply tokens' KV and GDN
     state with the decode kernels (T=1) instead of the prefill kernels.
   - On the 35B that cost reaches KL 8.5e-2 and 1.2 nats in the top-5 log-probs. The dense bf16 8B the tolerance was
     fitted on moved 0.09-0.14 nats (`2026-10-13-prefix-reuse.md`).
   - Plausible cause, not proven: the decode and prefill paths use different MoE and GEMM kernels, and a last-bit
     difference can flip an expert choice.
2. **The judge has no rounding scale on HIP.** Its rounding control is "twice what re-chunking the cold run moves the
   rows". HIP prefill re-chunks bit-exactly, so the control is 0 and the scale falls to the 0.25-nat / 1e-2 floor. A
   0.0037-nat tie then counts as "beyond rounding".
3. **The default skips this case.** At chunk 4096 the tool reports "split 2000 is below one prefill chunk (4096):
   nothing to reuse across chunk boundaries" (`raw/tatooine/reuse-docs4k-c4096.log`; round 9 w6 likewise). The
   `continue` cases of the 4k prompts only run at chunks below 2000.
4. **Open caveat: `generate` does not match its own scripted replay on the 35B.**
   - The gap is KL about 8e-3, the same on warm and cold, with graphs on and off.
   - On the 8B (2026-10-13), `generate` matched its replay bit for bit, and that was the anchor of the "not an engine
     bug" argument there.
   - Here it is a common-mode difference between `generate` and `forward_script`, of the same size on both runs. It is
     not a warm/cold asymmetry, so it does not explain the reuse divergence, but its cause is unverified. Candidates: a
     different last-chunk or lm_head row path in `generate`, or a different chunk split from the cached offset. It is
     worth a follow-up (section 7.3).

`--reuse-diag` on a `restore` case also aborts, with "cannot roll recurrent state back to 1024 tokens (no snapshot
there; length 2043)". The replay truncates where `forward_script` took no recurrent snapshot. This is a diagnostic-only
bug in the shared `crates/intarsia-cli/src/cmd/accuracy.rs`, and the gate itself had passed that case 32/32.

**CUDA** was not rerun for item 2. spark1 had 25 GB free (vLLM at 81 GB), and the 45 GB slot (#2906) would have blocked
G1 and G1b in the queue, so I cancelled it. CUDA has failed the same case before (`long_docs_4k continue`, ADR-0007,
phase 1b 7.3).

## 3. Notes on method

1. **KL21** is the tool's definition (`accuracy.rs` around 549-640): the reference's top-k plus a rest bucket on both
   sides.
   - Offline, a candidate's mass on a reference id outside its own top-k is approximated by its k-th probability, as
     `scripts/acc35b/xcompare.py` does. That underestimates the KL.
   - HIP main reads 3.22e-2 offline against 4.356e-2 in the tool. Compare offline numbers only with each other.
2. **Rows.** Row t is the distribution over token t+1. A "system/user-turn row" is any row from a non-assistant
   `<|im_start|>` up to its `<|im_end|>`.
3. **Builds.**
   - HIP main: this branch's tree on tatooine, built from the r9 target volume, unstamped.
   - CUDA rows: the 2026-10-14 W3 dump (`208ed00`, top-64), the only CUDA long dump available. Its tool KL was 4.273e-2;
     main reads 4.576e-2.
   - CPU: this tree on endor.

## 4. Gates

No engine code and no gate changed, so the GPU gates were run as measurements only:
- KL21 35b-long on HIP main: 4.356e-2, the same digits as round 9;
- KL21 against two fresh goldens;
- prefix reuse on HIP at 1024 and 4096.

The branch touches `docs/` and `scripts/` only. The footprint check against `origin/main` is empty, and
`cargo test` / `cargo check` are unaffected.

## 5. Files

- `docs/results/2026-10-21-m3-accuracy-debt.md` and `raw/`:
  - `raw/tatooine/`: HIP logs and the per-row dump `long-c4096.pos.tsv.gz`;
  - `raw/cpu/`: the CPU log and comparisons;
  - `raw/vllm/`: the offline analysis outputs;
  - `raw/downtime.log`.
- `scripts/m3-accuracy/`:
  - offline analysis: `split.py`, `pairs.py`, `setkl.py`, `posmask.py`, `bins.py`, `imend.py`, `imend_ab.py`,
    `confwrong.py`, `nll.py`, `head.py`;
  - vLLM re-query: `vllm_requery.py` and `vllm_requery*.sh`;
  - fleetq window wrappers: `cpu_chat4k*.sh`, `cuda_w*.sh` (not run), `hip_*.sh`.

The scripts carry this session's scratch paths (`D=`), so edit them before rerunning. The re-queried goldens and the
top-64 dumps (2-19 MB each) are not committed.

## 6. Fleet and downtime (`raw/downtime.log`)

- **radiance:** two windows, **7 min 21 s** of the lane's 20-minute share.
  - #2910, 09:29:53 → 09:35:18;
  - #2924, 09:57:01 → 09:58:57.
- **spark1 vLLM:** never taken down. Non-exclusive slots:
  - #2903: build;
  - #2904 and #2905: about 30 `prompt_logprobs` requests, each under 6 s;
  - #2906: cancelled while queued.
- **endor:** #2902, the CPU reference, 25 min, 40 GB. #2914 and #2921 (the fp8-KV CPU run) were cancelled while queued.

Every process I started has exited, and no slot is held.

## 7. Decisions for the user (proposals; nothing was changed)

1. **35b-long floor: gate only the rows the model is asked to predict.** There are three options.
   - **(a)** Exclude system and user-turn rows of chat prompts from the floor aggregate. The chat structure is in the
     golden ids. HIP main would read 1.42e-2, about 2x headroom.
   - **(b)** Drop `chat_reuse_4k` from the floor aggregate and keep it for prefix reuse, which needs its
     `reuse_split`. Also 1.42e-2.
   - **(c)** Keep the rows, with a noise-aware floor such as max(3e-2, k × the vLLM resampling spread on the same set).
     This keeps a metric that a handful of `<|im_end|>` flips dominates.

   **I recommend (a).** It also cleans the short set's chat prompts. The 3.0e-2 class default was never calibrated on
   this set: the pre-change build already failed it at phase 1b (4.49e-2).
2. **Golden metadata.** This is optional, and the verdict does not change. Record that `prompt_logprobs` requests bypass
   the prefix cache on vLLM 0.26.1rc1 (`cached_tokens: 0`), and that vLLM's own resampling KL21 on `chat_reuse_4k` is
   about 0.1.
3. **Prefix-reuse judge on hybrid MoE models.**
   - **Gate the bookkeeping exactly.** The diag line "cold vs warm with the reply prefilled" is a bitwise test of
     restore. It is 32/32 here, and a stale row or a wrong KV position would break it. Make it the reuse gate's hard
     check.
   - **Do not derive a tolerance from the decode-vs-prefill line per run.** That line grows with whatever the decode path
     does, so a decode-path bug would widen its own allowance. Decode-vs-prefill numerics are already gated by the
     golden gate's decode mode and the kernel tests.
   - If a nats tolerance stays for the stream comparison, measure it once on a known-good build and keep it per model in
     a baseline.
   - Run the `continue` case at the default chunk too, so that "default passes" is not vacuous.
   - Follow up the `generate`-vs-replay difference from 2.8 point 4, and fix the `--reuse-diag` restore abort.
   - All of this touches the shared `accuracy.rs` or the engine, so it is left to the coordinator.
4. **The systematic offset to vLLM is untested.** Our three implementations sit about 1.6x further from vLLM than vLLM
   sits from itself, on every prompt: 1.36e-2 against 8.6e-3 outside the user turns.
   - The obvious candidate is vLLM's fp8 KV cache: the golden's server runs `cache_dtype fp8`, and ours is bf16.
   - It was not tested. Our `--kv-dtype fp8_e4m3` path ignores the checkpoint's FP8 KV scales ("scales not used" in
     `cpu-chat4k.log`), and vLLM probably applies them, so a null result would not have ruled it out.
   - A clean test needs a bf16-KV vLLM instance, which means a spark1 vLLM window agreed with agent-harness.

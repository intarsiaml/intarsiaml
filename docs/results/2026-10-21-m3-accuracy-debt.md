# M3: accuracy debt — the 35b-long prefill-floor FAIL and HIP prefix reuse at chunk 1024 (2026-10-21)

Lane M3 of `docs/plans/2026-10-21-quota-burn-plan.md`. Base `origin/main` 4be7b7d4, the same code as a7febf48
(`git diff --stat a7febf48 4be7b7d4 -- crates kernels` is empty). Model `nvidia/Qwen3.6-35B-A3B-NVFP4`. Branch
`accuracy/m3-debt`: docs, analysis scripts (`scripts/m3-accuracy/`) and raw logs only. **No engine code changed, no gate
changed.** Raw logs are in `docs/results/2026-10-21-m3-accuracy-debt/raw/`.

## 1. Headline

- **The 35b-long prefill-floor FAIL is not an engine defect. It is one prompt's system and user turns.**
  - `chat_reuse_4k` is 12.6 % of the rows and carries 73 % of the KL21 sum.
  - Inside it, the KL sits in the 4,491 rows of the system and user turns, where the model is never asked to predict
    anything in serving. The assistant turns are at 5e-3 to 2.7e-2.
  - Without those rows, HIP scores **1.39e-2** against the 3.0e-2 floor (tool-exact per-row dump). The other four
    prompts never grow with position: `long_docs_12k` stays at 1.5e-2 to 3.3e-2 in every 2k bin up to 12k.
  - That rules out the long-position candidates: rope/yarn, accumulation drift, and GDN chunk-boundary carry.
- **Mechanism: in a long user turn the model hovers on "end the user message now".** Every implementation, vLLM included,
  puts mean P(`<|im_end|>`) ≈ 0.27-0.29 on rows that are not the end of the turn. On the short chat prompts `<|im_end|>`
  is vLLM's top-1 on 42-51 % of the rows. Single rows flip between P ≈ 1 and P ≈ 0.002 under last-bit differences.
  Rows with P(`<|im_end|>`) > 0.05 carry 54-73 % of the KL, and the top 5 % of rows carry about 80 %.
- **The golden is one noisy sample on exactly these rows.** I re-queried the same server and version (vLLM
  0.26.1rc1.dev602) with the stored ids:
  - vLLM against its own stored golden: **0.10-0.12 KL21 on `chat_reuse_4k`**. Two fresh vLLM samples differ by 0.09.
  - On the whole 35b-long set, vLLM against its own golden reads 2.4-2.7e-2 offline. In tool units (×1.35, see 3.3)
    that is about 3.2-3.6e-2, so the reference would fail its own floor.
  - Excluding system and user-turn rows, vLLM against itself is 8.6e-3 and HIP is 1.36e-2.
- **The floor verdict does not depend on the vLLM sample**, so this is not "regenerate the golden". HIP main against
  the stored golden and two fresh samples: 4.356e-2, 4.545e-2 and 4.557e-2, all FAIL.
- **Proposal (no gate changed):** gate the 35b-long prefill floor on rows the model is asked to predict. That means
  excluding system and user-turn rows of chat prompts, or dropping `chat_reuse_4k` from the floor aggregate while keeping
  it for prefix reuse. Details are in section 6.
- **HIP prefix reuse at `--prefill-chunk 1024` is not an engine bug, and the default chunk does not test it.**
  - The failing case is `long_docs_4k continue` (split 2000). At the default chunk of 4096 this case is **skipped**
    ("split 2000 is below one prefill chunk"), so "default passes" means "default never runs it".
  - Measured with `--reuse-diag`: the prefix restore is **bit-exact**. Warm with the reply prefilled equals cold, 32/32
    rows bitwise, and cold is chunk-invariant across 512, 1023, 1024 and 2048.
  - The whole warm-vs-cold difference comes from the 15 reply tokens that went through the decode kernels instead of the
    prefill kernels: KL mean 1.26e-2, max 8.45e-2, max |dlogit| 2.28.
  - That is the known decode-vs-prefill spread of this MoE model. It exceeds the tolerance calibrated on bf16 Qwen3-8B
    (0.25 nats, KL 1e-2), and the cold run sits on a 0.0037-nat near-tie at token 8, so the tie flips.

## 2. Numbers: where the 35b-long KL is

### 2.1 Per prompt (HIP main, tool, `--prefill-chunk 4096`, `raw/tatooine/long-c4096.log`)

| prompt | rows | KL21 | share of the KL sum |
|---|---|---|---|
| long_docs_4k | 4064 | 1.674e-2 | 4 % |
| long_code_4k | 4064 | 7.775e-3 | 2 % |
| **chat_reuse_4k** | 4631 | **2.479e-1** | **71-73 %** |
| long_docs_12k | 12064 | 2.071e-2 | 15 % |
| long_code_12k | 12064 | 9.045e-3 | 6 % |
| OVERALL | 36887 | **4.356e-2** (floor 3.0e-2: FAIL) | |

Tool-exact per-row KL21 (HIP, 2026-10-14 `hip-longpre.pos.tsv`, overall 4.583e-2; `scripts/m3-accuracy/posmask.py`):
- all rows: 4.58e-2;
- without the system and user-turn rows of chat prompts: **1.393e-2** (32,349 rows);
- without `chat_reuse_4k` entirely: 1.394e-2.

### 2.2 No growth with position

KL21 by 2048-row bin (HIP, tool-exact rows):
- `long_docs_12k`: 1.7e-2, 1.7e-2, 2.3e-2, 3.3e-2, 1.6e-2, 1.5e-2.
- `long_code_12k`: 7.5e-3, 6.9e-3, 6.7e-3, 1.0e-2, 1.1e-2, 9.0e-3.

Twelve thousand positions in, the prompts that are plain text sit where they sit at position 0. A precision loss in
long-context attention, the GDN state carry across chunks, or rope/yarn at long positions would grow along the prompt.
None does. `chat_reuse_4k`'s user turn is high from the very first rows (system-prompt rows 0-23: 5.4e-2), not late.

### 2.3 Inside chat_reuse_4k: the turns

CUDA (W3 dump, top-64) against the stored vLLM golden, offline KL21 (`scripts/m3-accuracy/split.py`):

| rows | n | KL21 | P(`<|im_end|>`) vLLM / CUDA on non-end rows | top-1 agreement |
|---|---|---|---|---|
| system turn [0,23) | 23 | 5.4e-2 | 0.469 / 0.421 | 91 % |
| **user turn with the document [25,4516)** | 4491 | **1.91e-1** | 0.289 / 0.269 | 88 % |
| assistant "Understood" [4518,4536) | 18 | 2.7e-2 | 0.505 / 0.542 | 89 % |
| user question [4538,4562) | 24 | 9.0e-4 | 0.932 / 0.928 | 100 % |
| assistant `<think>` + continuation [4564,4631) | 66 | 4.9e-3 | 0.003 / 0.007 | 98.5 % |

`<|im_end|>` as vLLM's top-1 on rows whose real next token is not `<|im_end|>` (`imend.py` on both goldens):
- `chat_reuse_4k`: 31.7 %.
- `chat_short`, `chat_system`, `chat_multiturn`, `chat_reasoning`: 47 %, 42 %, 51 % and 42 %.
- Every raw-text prompt: 0 %.

Our engines reproduce this: CPU on 35b-short scores 5.8e-3 to 2.5e-2 on those chat prompts, which would be nats if it
did not. This is how the model behaves in user turns, which SFT does not train it to predict. It is not a vLLM artifact.

Two other things on the same rows:
- vLLM's top-1 has a probability above 0.9 while the real next token is outside its top-20 on **4.23 %** of
  `chat_reuse_4k` rows. The rate is 0.03-0.25 % on the raw prompts over the same corpus.
- Example: after `--sizes 128,102` vLLM puts 0.77 on `scribe`, while CUDA puts 0.999 on the real `4`. CUDA has rows
  like this too, with P(`<|im_end|>`) = 1.0 where vLLM has 0.002.
- Both implementations predict the user text equally badly (top-1 equals the real token on 35.0 % and 34.8 % of rows).

### 2.4 Is the golden reproducible? (spark1 vLLM, guidance, stays up; `scripts/m3-accuracy/vllm_requery*.sh`)

The re-query sends the stored ids, not retokenised text. Every response reports `cached_tokens: 0`, salted or not. So
`prompt_logprobs` requests do not read the prefix cache, which resolves the golden metadata's "unverified" note.

| pair (offline KL21, `pairs.py`: vLLM samples have top-20, our dumps top-64) | chat_reuse_4k | its user turn | other rows |
|---|---|---|---|
| stored golden vs fresh sample, same salt twice / two salts / unsalted | 0.103 / 0.123 / 0.114 / 0.109 | 0.106-0.127 | 4.7e-3 to 1.4e-2 |
| fresh sample A vs fresh sample B | 0.089-0.104 | 0.092 | 7.8e-3 |
| stored golden vs CUDA (W3) | 0.186 | 0.191 | 1.5e-2 |
| long_docs_4k: stored golden vs fresh sample / vs CUDA | 1.20e-2 / 1.68e-2 | | |

Whole 35b-long set, offline KL21 against the stored golden (`setkl.py`; `raw/vllm/setkl.txt`):

| candidate | all rows | without system/user-turn rows |
|---|---|---|
| vLLM sample A | 2.37e-2 | 8.6e-3 |
| vLLM sample B | 2.66e-2 | 8.7e-3 |
| vLLM A vs B | 2.00e-2 | 8.3e-3 |
| HIP main (c4096 dump) | 3.22e-2 | 1.36e-2 |
| CUDA 208ed00 (W3 dump) | 3.22e-2 | 1.37e-2 |

35b-short, vLLM against its stored golden: 2.5e-3 and 2.9e-3. The short set is reproducible, and the long set is not,
because of one prompt.

### 2.5 Tool-exact: HIP main against three vLLM samples of the reference (`raw/tatooine/acc-long-*.log`)

| golden | OVERALL KL21 | chat_reuse_4k | verdict |
|---|---|---|---|
| stored (`7614f2a3`) | 4.356e-2 | 2.479e-1 | FAIL |
| fresh sample A | 4.545e-2 | | FAIL |
| fresh sample B | 4.557e-2 | | FAIL |

### 2.6 CPU reference on chat_reuse_4k

Setup: the `cpu` backend (f32 accumulation; NVFP4, MoE and GDN in f64; bf16 KV) on endor, on a `chat_reuse_4k`-only golden.
`--mode prefill`, chunk 256, top-64 dump; 25 min under fleetq #2902 (`raw/cpu/cpu-chat4k.log`).

- **Tool:** KL21 against the stored vLLM golden is **2.628e-1** (top-1 87.52 %). HIP main reads 2.479e-1. The most exact
  implementation we have is *further* from the golden on this prompt than the GPU backends.

Offline KL21 among all the implementations on `chat_reuse_4k` (`raw/cpu/pairs-cpu.txt`):

| pair | all rows | user turn | other rows |
|---|---|---|---|
| vLLM vs vLLM (stored / A / B) | 0.102-0.129 | 0.105-0.133 | 4e-3 to 1.2e-2 |
| HIP main vs CUDA 208ed00 | 0.120 | 0.123 | 9.5e-3 |
| CPU vs HIP main / CPU vs CUDA | 0.139 / 0.146 | 0.142 / 0.150 | 1.5e-2 / 1.8e-2 |
| vLLM (stored / A / B) vs CPU | 0.207 / 0.199 / 0.199 | 0.204-0.213 | 1.6e-2 to 2.3e-2 |
| vLLM stored vs HIP main / CUDA | 0.188 / 0.186 | 0.19 | 1.5-1.6e-2 |

**Reading.** On this prompt's user turn, every pair of correct implementations differs by 0.10-0.21. That holds even
for two runs of one server. The CPU oracle sits with our GPU backends (0.14-0.15 from them), and all three sit about
0.19-0.21 from vLLM. Nothing here points at a GPU-kernel defect.
- The golden cannot rank implementations on these rows. The quantity it measures is chaotic: a few `<|im_end|>`-or-not
  rows flipping between near-certainties.
- The remaining offset between the vLLM cluster and ours (0.2 against 0.1-0.15 within each) is systematic. Section 6.4
  covers it.

## 3. Notes on method

1. **KL21** follows the tool (`accuracy.rs` 549-640): the reference's top-20 plus a rest bucket on both sides.
   - Offline, a candidate's mass on a reference id outside its own top-k is approximated by its k-th probability (as
     `scripts/acc35b/xcompare.py` does).
   - Candidates are truncated to top-20, so that vLLM samples (top-20) and our dumps (top-64) are treated alike.
2. **Rows.** Row t is the distribution over token t+1. A "system/user-turn row" is any row from a non-assistant
   `<|im_start|>` up to its `<|im_end|>`.
3. **Offline vs tool.** The HIP main dump reads 3.22e-2 offline and 4.356e-2 in the tool, a factor of 1.35. The offline
   numbers compare candidates with each other, and the ×1.35 conversion is an estimate.

## 4. Item 2: HIP prefix reuse at `--prefill-chunk 1024`

Measured with `--mode reuse --reuse-diag` on a `long_docs_4k`-only golden. The window ran HIP main
(`raw/tatooine/reuse-docs4k-c1024.log`).

| comparison, 32 rows, teacher-forced on the cold stream | bitwise | KL mean / max | max \|dlogit\| |
|---|---|---|---|
| warm vs warm again; cold vs cold again | 32 / 32 | 0 | 0 |
| cold chunk 1024 vs 512 / 1023 / 2048 | 32 / 32 each | 0 | 0 |
| **cold vs warm with the reply prefilled, not decoded** | **32 / 32** | 0 | 0 |
| warm vs cold, same context | 0 | 1.26e-2 / 8.45e-2 | 2.277 |
| warm vs warm with the reply prefilled (decode-KV only) | 0 | 1.26e-2 / 8.45e-2 | 2.277 |

The first divergence is at token 8. The warm run takes 29832 by +0.183 nats, and the cold run takes 283 by **+0.0037
nats**.

**Reading.** The reuse bookkeeping, the recurrent-state snapshot and the KV restore are bit-exact on this case. The only
numerical difference between warm and cold is that the 15 reply tokens' KV rows and GDN state were written by the decode
kernels (T=1) instead of the prefill kernels. The two lines are identical to every digit. On this MoE hybrid that
difference reaches KL 8.5e-2 and 1.2 nats in the top-5 log-probs. For scale, the dense bf16 8B the tolerance was fitted on
moved 0.09-0.14 nats (`2026-10-13-prefix-reuse.md`). On the 35B, the decode and prefill paths use different MoE and GEMM
kernels, and a last-bit difference can flip an expert choice. That is consistent with this size, though not proven to be
the cause.

The judge's rounding scale is "twice what re-chunking the cold run moves the rows". On HIP, re-chunking moves nothing
(bitwise), so the scale falls to the 0.25-nat / 1e-2 floor calibrated on bf16 Qwen3-8B. The 0.0037-nat tie then counts as
"beyond rounding".

**Why the default passes.** At chunk 4096 the tool skips the case: "split 2000 is below one prefill chunk (4096): nothing
to reuse across chunk boundaries" (`raw/tatooine/reuse-docs4k-c4096.log`, and round 9's w6 log). The `continue` case of
`long_docs_4k` and `long_code_4k` only runs at chunks below 2000.

**Side finding (diagnostic tool only).** `--reuse-diag` on a `restore` case aborts with "cannot roll recurrent state back
to 1024 tokens (no snapshot there; length 2043)". The scripted replay (`script_rows`) truncates where `forward_script`
took no recurrent snapshot. The gate itself had already passed that case, 32/32. This needs a fix in the shared
`crates/intarsia-cli/src/cmd/accuracy.rs`, so it is not in this branch.

**CUDA.** Not rerun this round. spark1 had only 25 GB free with vLLM at 81 GB, and my 45 GB slot (#2906) would have
blocked G1 and G1b, so I cancelled it. CUDA has failed the same case before: `long_docs_4k continue`, ADR-0007 and
phase 1b 7.3. The 2026-10-13 prefix-reuse doc put it down to decode-vs-prefill rounding.

## 5. Gates

Nothing in the engine or the gates changed, so the GPU gates were run only as measurements:
- the KL21 above;
- prefix reuse at 1024 and 4096 on HIP.

The branch adds only `docs/` and `scripts/m3-accuracy/`, so `cargo test` / `cargo check` are unaffected. They were not
rerun.

## 6. Decisions for the user (proposals; no gate was changed)

1. **35b-long floor: gate the rows the model is asked to predict.** Pick one of three:
   - **(a)** Exclude system and user-turn rows of chat prompts from the floor aggregate. The tool knows the chat
     structure from the golden ids. HIP would read about 1.4e-2, about 2x headroom.
   - **(b)** Drop `chat_reuse_4k` from the floor aggregate, but keep it for the prefix-reuse gate, which needs its
     `reuse_split`. The result is the same 1.39e-2.
   - **(c)** Keep the rows and set a noise-aware floor: max(3e-2, k × the vLLM-vs-vLLM spread measured on the same
     set). This keeps a gate that a 1-row flip can move by 1e-3.

   I recommend (a): it also cleans the short set's chat prompts, where `<|im_end|>` rows are 42-51 %. Whichever is
   chosen, the 3.0e-2 class default was never calibrated on this set. The pre-change build at phase 1b already failed
   it (4.49e-2).
2. **Golden hygiene.** It is optional and does not change the verdict.
   - Record in the golden metadata that `prompt_logprobs` requests bypass the prefix cache on 0.26.1rc1 (`cached_tokens:
     0`).
   - Note that vLLM's own run-to-run KL on `chat_reuse_4k` is about 0.1.
3. **Prefix-reuse judge on hybrid MoE models.** For the `continue` case, the right rounding control is the diag line
   "warm vs warm with prefilled reply (decode-KV only)". It is the exact numerical cost of decode-written reply rows.
   Re-chunking the cold run is bitwise on HIP and gives 0.
   - Proposal: `judge_streams` takes max(rechunk control, decode-vs-prefill control) as the scale for `continue`, or the
     35B gets measured tolerances.
   - Separately: should the `continue` case run at the default chunk at all? A split below the chunk could still
     exercise decode-written reply rows.
   - This is a change in a shared file (`accuracy.rs`), so it is left to the coordinator or user.
4. **Remaining systematic gap.** Our engines sit about 1.6x further from vLLM than vLLM sits from itself, on every
   prompt (rows outside the user turn: 1.36e-2 against 8.6e-3).
   - Candidates are vLLM's fp8 KV cache (the golden's server runs `cache_dtype fp8`; ours is bf16) and kernel
     differences.
   - FP8_RESULT

## 7. Files

- `docs/results/2026-10-21-m3-accuracy-debt.md` (this doc) and `raw/` (tatooine logs, CPU log, analysis outputs).
- `scripts/m3-accuracy/`:
  - offline analysis: `split.py`, `pairs.py`, `setkl.py`, `posmask.py`, `imend_ab.py`, `confwrong.py`, `nll.py`,
    `head.py`;
  - vLLM re-query: `vllm_requery.py` and `*.sh`;
  - the fleetq window wrappers: `cpu_chat4k*.sh`, `cuda_w*.sh`, `hip_*.sh`.

  The scripts carry this session's scratch paths. Edit `D=` before rerunning. The re-queried goldens and the top-64
  dumps (6-19 MB each) were not committed.

## 8. Fleet and downtime

`raw/downtime.log`.

- **radiance:** one window, fleetq #2910, 09:29:53 → 09:35:18. That is **5 min 25 s** of the lane's 20-minute share.
- **spark1 vLLM:** never taken down. It was used non-exclusively:
  - #2903: build;
  - #2904 and #2905: about 30 `prompt_logprobs` requests, each under 6 s;
  - #2906: cancelled while queued, for lack of memory.
- **endor CPU:** #2902, 25 min, 40 GB, no role taken down. The fp8-KV CPU run (#2921) was queued; see 6.4.

Every process I started has exited, and no slot is held.

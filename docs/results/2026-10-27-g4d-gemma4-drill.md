# G4D: the onboarding drill on Gemma-4-26B-A4B-NVFP4, CUDA, spark2 (2026-10-26 night)

Lane G4D, branch `lane-g4d/gemma4-drill`, fleetq jobs #5853 (50 GB) and #5964 (68 GB) on spark2, both `--quiet gpu,cpu`, next to the
running `spark2-vllm` (retro, 38 GB). Build: origin/main `e0a04220` (the code under test) and `6887ec7cc` (the one fix below).
Times are EDT. Requirement #1 of the user, quoted: "A new model comes out, we should be able to complete a basic test it within a
reasonable time from the weights finish downloading. Lets say 1-2 hours for a basic test and 2-4 for more in-depth. If these test can be
done in one pass back-to-back thats better. Quick test finishes and numbers can be published while the in-depth tests keeps going."

## Verdict

**The drill passes the time targets: quick numbers 30 min after T0 (accuracy gate included), the quick quality number (MMLU) 62 min
after, the whole in-depth set 1 h 48 min after. Nothing in the recipe or the engine blocked the pass.** The drill found one real fault
(the decode byte floor of every Gemma-4 report was wrong, fixed here), four pipeline frictions that cost 35 minutes of my clock, and
two engine gaps that are the real finding about this model: **Gemma-4 on CUDA decodes at 34.5 tok/s (about half of the byte floor),
prefills at 369 tok/s at 4k and 211 tok/s at 16k, and cannot batch.** One mistake of mine took spark2 down at 01:25 (section 6).

| Result | Value | Where |
|---|---|---|
| `onboard` verdict (second run, golden found) | PASS, exit 0, 15 min 44 s, 15 stages: 9 pass, 0 fail, 6 skip | `quick.md`, `in-depth.md` |
| Accuracy, HF f32 golden (10 prompts, 1468 prefill positions, 160 decode rows) | prefill top-1 86.65 %, KL21 2.12e-1; decode top-1 89.38 %, KL21 1.57e-1; PASS (the 4bit gate class with the reference-noise factor of the golden, `2026-10-22-g4g-gemma4-gates.md`) | `acc-short` |
| Chat-form perplexity (recipe gate) | 35.97, HF bf16 35.14; 5/5 sanity prompts | `gate` |
| Decode (localeval, forced 128 tokens, 3 reps) | 34.5 tok/s at 136 prompt tokens, 31.4 at 1k, 25.0 at 4k, 13.6 at 16.7k | unanimis 5964e6de, 17a336cf |
| Prefill | 576 (136 tok, overhead-bound), 522 (1k), 369 (4k, marginal 338), 211 (16.7k, marginal 204) tok/s; TTFT 79 s at 16.7k | same |
| Multi-stream decode, 4 and 8 streams | aggregate 21.6 tok/s at both: the requests are served one after the other (recipe `max_batch = 1`), so this is no scaling figure | 17a336cf |
| MMLU `--limit 25` (n = 1425, 5-shot, thinking off) | **0.811**, 0 truncated, 1194 s | unanimis 84d78b12 |
| GSM8K, localeval default prompt, n = 400 | 0.470 `gsm8k_flexible`; mean reply 9 tokens: the model answers with a bare number, so this measures no reasoning | unanimis 9fae475e |
| GSM8K, `--pin-gsm8k` (few-shot chain of thought), n = 400 | **0.973** `gsm8k_flexible`, mean reply 162 tokens, 0 truncated, 34 min | unanimis e6056f82 |
| Spec decode | none for this model: the recipe has no drafter, `--spec mtp` needs `mtp.*` tensors (this checkpoint has none) | not run |
| vLLM on the same box | not run (retro must stay up). Reference: the tatooine record (R9700, Radiance vLLM, MXFP4 checkpoint, MTP draft: 77.6 decode, 8238 prefill) is a different GPU, checkpoint and drafter, so it is not a comparison | |

GSM8K n = 400 is the first 400 questions, not the full 1319 (the pinned run alone took 34 min). MMLU and GSM8K have no same-box reference
number; they are the first quality numbers of the model on this engine. Every cell is one run (3 reps for the speed cells, 2 for 16k).

## Timeline (T0 = 23:22:52, the first action on the box; the weights had been on spark2 since 2026-10-22)

| Time | Elapsed | Event |
|---|---|---|
| 23:17 | | session start; fleetq status, docs, found the weights (`~/models/Gemma-4-26B-A4B-NVFP4`, 18.8 GB, on spark2 only; 5 min, before T0) |
| 23:22:52 | 0 | `scripts/deploy.sh spark2 --backend cuda` (lease #5852, `--load cpu`): rsync, cargo 47 s, doctor, kernels-test; done 23:24 (**build 2 min**) |
| 23:24:55 | 2 min | lease #5853 granted without waiting; `intarsia onboard` run 1: recipe PASS 0 s, smoke PASS 61 s (loads the model), quick PASS 6 min 07 s |
| 23:32:04 | 9 min | **first `quick.md` published**: speed 618/523/367 prefill, 34.7/31.4/24.8 decode, gate PASS. All three accuracy stages SKIP (friction 1) |
| 23:38:13 | 15 min | run 1 ends: in-depth 6 min 09 s, **verdict PASS with zero accuracy stages run** |
| 23:41 | | run 2: ERROR after 29 s (friction 2) |
| 23:45:18 | 22 min | run 3, `--model-name gemma4-26b-a4b-nvfp4` (the committed goldens are found) |
| **23:52:28** | **30 min** | **`quick.md` with the accuracy gate: acc-short PASS** |
| 00:01:03 | 38 min | `in-depth.md`, onboard PASS, 15 min 44 s. acc-long, acc-4k, acc-oracle, ab, tp SKIP (no golden / no reference / no `--tp`) |
| 00:01:38 | | `intarsia serve` up (55 s load); localeval speed 128/1k/4k 00:02:50-00:04:33 |
| 00:04:33 to 00:25 | 62 min | **MMLU `--limit 25` done** (20 min; calls the end of the quick tier of the requirement) |
| 00:26 to 00:31 | | 16k prefill and 4/8-stream decode |
| 00:31 to 00:35 | | GSM8K, default prompt (3 min) |
| 00:36 to 01:10 | 1 h 47 min | GSM8K pinned chain of thought (34 min): **the in-depth set is complete** |
| 01:12 | | fix deployed (lease #5964), HF f32 4k golden made (192 s) |
| ~01:25 | | 12k golden attempt; **spark2 stops answering** (section 6) |

## Friction, each with the fix it needs

1. **The golden is looked up by the directory name, and a miss is a SKIP, not a failure.** `~/models/Gemma-4-26B-A4B-NVFP4` does not
   match the committed `golden/gemma4-26b-a4b-nvfp4-fp32.safetensors`, so run 1 skipped `acc-short`, `acc-long`, `acc-4k`, the envelope and
   `acc-oracle`, and still printed `ONBOARD PASS`. Cost: 13 min for a green that checked no accuracy. `docs/adding-a-model.md` already warns
   about it, but the recipe knows the model and the committed golden names; a recipe key for the golden stem, or onboard printing
   `PASS (accuracy not checked)` when `acc-short` is SKIP, would make the first run right. Workaround: `--model-name gemma4-26b-a4b-nvfp4`.
2. **`--model-name` after `--` makes qualify exit 2** ("cannot be used multiple times"), because onboard always forwards its own. It is
   an onboard flag. 4 min. Fix: say so in the doc (done below), or have onboard drop a second `--model-name`.
3. **The decode byte floor was wrong for Gemma-4 (fixed, `6887ec7cc`).** `weight_facts` reads the routing width from
   `num_experts_per_tok`; Gemma-4 calls it `top_k_experts`, so no expert was scaled: 17.64 GB per token, a floor of 12.5 tok/s, and every
   report said "277 % of the byte floor" and, for 4k, "above 100 %: the estimate is too high". The fix reads either key (unit test
   `moe_top_k_reads_both_config_spellings`; `cargo test -p intarsia-cli`: 181 passed). Not verified on the box: see section 6. The
   engine and the plan read `top_k_experts` correctly (`recipe.rs:912`), and `quant/moe.rs:46` reads `num_experts_per_tok` for the Qwen
   path only, so no other reader has the gap. No threshold depends on the floor (the report only adds a note above 100 %), so the fix
   changes text and a percentage, not a verdict. It is a CLI reporting change: no engine math, so no CUDA no-op ids.
4. **The report prints the scaffold TODO list of `recipe new` even when a recipe covers the model.** `quick.md`, section "Plan and
   recipe", lists "sliding-window attention is not implemented", "activation other than silu", "NVFP4 W4A4 unsupported" and 1542 unclaimed
   tensors beside `recipe: PASS (builtin:gemma4 covers the model)`. All of it is the analysis of a model the recipe already handles; a
   reader sees a wall of TODOs on a PASS. Fix: print the analysis only when no recipe covers the model.
5. **`scripts/deploy.sh` rsyncs with `--delete`, so a deploy into a tree wipes every output kept in it.** My second deploy (to get the fix
   onto the box) deleted `~/intarsiaml-g4d/runs/` with the raw logs and `state.json` of both onboard runs. The Markdown reports had been
   copied out; the per-stage logs and bench JSON were not. Keep run output outside the deployed tree, or exclude it in the rsync.
6. **The HF golden tool needs Python headers that the HF venv does not have.** `tools/gemma4_nvfp4_golden.py` on a prompt of 4k tokens
   reaches torch 2.14's `torch._native` Triton op `bmm_outer_product`, which compiles a C helper and fails with `Python.h: No such file`.
   Workaround: `C_INCLUDE_PATH` at the uv-managed Python's `include/python3.12`. The short prompts of the original golden did not reach it.
7. **The memory a lease declares is not what the 12k golden needs** (section 6).

## Engine findings (named, not chased)

* **Decode runs eagerly, no CUDA graph** (`bench: warning: size N: 0 replays measured reps (decode ran eagerly)` at every size). 34.5 tok/s
  is 29 ms per token. The floor with the fix is a few GB per token at 221 GB/s, so the engine is at roughly half of it; eager launches
  of a 30-layer model with a 128-expert router are a likely part. Not profiled.
* **Long-context cost.** Decode drops from 34.5 to 13.6 tok/s between 136 and 16.7k tokens and prefill from 576 to 211 tok/s. The window
  of the 25 sliding layers is applied in `kernels/cuda/attention_ext.cu`, so the cost sits in the 5 global layers (head_dim 512) or in the
  kernel itself; not profiled. For scale, the same engine prefills Qwen3-8B at 2.4k tok/s at 4k with its default chunk (`2026-10-22-ck-cuda8b-chunk.md`).
* **No batching and no speculative decoding** for Gemma-4: `max_batch = 1` in `recipes/gemma4.toml`, and the model has no drafter in this
  checkpoint. A served multi-user load is therefore one request at a time (21.6 tok/s aggregate at 4 and 8 streams including the prefill of
  each).
* **The long-context accuracy of the sliding window is not gated.** The committed golden has prompts up to 700 tokens, below the 1024
  window, so no accuracy stage exercises the window cutting keys. The 4k golden (`acc-4k`) is what would; it was made (section 6) but
  could not be used.

## 6. What went wrong, and what is not done

* **spark2 stopped answering at about 01:25** (ping 100 % loss, ssh timeout, `spark2-vllm` retro DOWN) while the 12k-token HF f32 golden
  (`tools/prompts/golden_12k.jsonl`, `--dtype float32 --device cuda`) was running under lease #5964 (68 GB). The 4k golden before it took
  192 s and 68 GB of the 121 GB (unified memory, retro holds 38). The 12k prompt needs more attention memory than the lease declares, and
  on a unified-memory box the shortfall is the whole machine. My mistake: I should have run the 12k golden alone with a measured peak, or
  not at all. I did not touch the box after that; it needs a power cycle or its own recovery. Whoever restarts it must bring
  `spark2-vllm` (retro) back.
* **Not done** because of that: the re-run of `onboard` with the fixed binary (the floor figure of `6887ec7cc` is unit-tested only), `acc-4k`
  and `acc-long` (the 4k golden `golden-new/gemma4-26b-a4b-nvfp4-4k-fp32.safetensors` is on spark2's disk, 2.1 MB), the cleanup of
  `~/intarsiaml-g4d` on spark2, and a Gemma-4 vLLM comparison. There is no `acc-oracle` golden for this model either (an f64 CPU run of
  the 26B, `2026-10-22-g4g-gemma4-gates.md` section C).
* Raw reports: `quick.md` and `in-depth.md` of both runs are summarised above; the raw directories were lost to friction 5.

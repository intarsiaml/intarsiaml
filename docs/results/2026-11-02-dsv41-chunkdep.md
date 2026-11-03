# DeepSeek-V4.1 prefill: how much does the chunk size move the numerics (job B of the prefill lane)

Branch `dsv41/chunkdep` (from origin/main c33851575, after PR #282). Follows [2026-11-02-dsv41-prefill.md](2026-11-02-dsv41-prefill.md), which left "the cause of the chunk-size dependence on long prompts" unchecked and kept `INTARSIA_DSV41_CHUNK=2048` opt-in because 32K and 65K prompts moved the logprobs (0.19 nats at 32K).

**Question.** Is the difference between chunk 512 and chunk 2048 a bug (a state, boundary or indexer problem that makes one of them wrong) or rounding noise that the model amplifies? **Short answer: rounding noise, with the same size as an equivalent op-fusion toggle. No chunk-boundary effect was found. At the last prompt position (the only row a request samples from) the chunk change moves the next-token distribution by 0.013 to 0.029 nats (KL) at 8K (position 7999) and 0.016 to 0.018 at 32K (position 32767), one prompt each; the top-1 token at that position did not change in those 5 comparisons, and the 16 greedy tokens are identical to chunk 512 in 4 of 5 runs (chunk 1024 at 8K diverges at token 9). The rounding-order toggle `HC_FUSED=0` is not smaller: KL 0.023 at 7999, top-1 flipped, and its 16 greedy tokens differ from the first. (The other chunk-end rows inside the prompt, KL 0.0000 to 0.03, are not sampled by anyone.) The plan's indexer-Jaccard gate (>= 0.98 at 32K) is not met (0.76 at position 32767). This does not show that 2048 is as accurate as 512; there is no reference run (see Not checked).**

All numbers are for the 8000-token prompt `prompts/doc8k.txt` (and the 32K prompt `prompts/long32k.txt`, below), rank 0, greedy, one binary: `bin/intarsia.dump` (branch head 3c9ff4ed9 plus the row-dump hook in `runtime.rs`; the hook only copies rows out after a step). Raw output is under `2026-11-02-dsv41-chunkdep/raw/`.

## Method

`INTARSIA_DSV41_DUMP=<dir> INTARSIA_DSV41_DUMP_POS=<absolute positions> INTARSIA_DSV41_DUMP_WHAT=<step name substrings>` writes, after each graph step, only the rows at those absolute token positions (so runs with different chunk sizes can be compared file by file): the embedding, `hc_attn` / `hc_ffn` (normalised sublayer input, mixes), indexer output (selected sets), `sparse_attn`, router gate (expert ids), head (logits), at positions 3, 100, 255, 256, 511, 512, 513, 1023, 1024, 2047, 2048, 2049, 4095, 4096, 7999. `tools/dsv41/chunkstudy.py` compares runs: relative rms of the last hidden state and logits, of the normalised FFN input per layer, Jaccard of the indexer selected sets and of the routed expert ids, and a bit-equality count for the parts with no incoming noise.

## What the numbers are measured on

- **Relative rms** = `||a - b||_2 / ||b||_2` over one row (a vector of 5120 for hidden states, 64640 for the logits row of rank 0's half of the vocabulary), `chunkstudy.py rel()`. "Logits @7999" is the logits row at the last prompt position (the next-token distribution); "@4095" the row at position 4095. "n(ffn) at layer L" is the normalised input of layer L's FFN sublayer (`layers.L.hc_ffn` output 0), a 5120 row, averaged over the 15 dumped positions. The logits exist only at the last row of each chunk, so the five positions every chunk size has are 511, 1023, 2047, 4095, 7999.
- **In nats and top-1 changes at the final logits** (`tools/dsv41/logitdist.py`, both ranks' vocabulary halves joined into the full 129280-entry row; reference = chunk 512; `raw/logitdist-8k.out`):

| run vs chunk 512 | positions compared | relative rms of the logits row (mean) | KL(512 \|\| run) in nats, mean (range) | argmax (top-1) changed |
|---|---|---|---|---|
| chunk 1024 | 4 (1023, 2047, 4095, 7999) | 0.129 | 0.012 (0.0006 to 0.029) | 0 of 4 |
| chunk 2048 | 3 (2047, 4095, 7999) | 0.088 | 0.0045 (0.0002 to 0.013) | 0 of 3 |
| chunk 4096 | 2 (4095, 7999) | 0.119 | 0.013 (0.0007 to 0.026) | 0 of 2 |
| `HC_FUSED=0` (chunk 512, rounding-order change only) | 5 | 0.098 | 0.0085 (0.0000 to 0.023) | **1 of 5** (position 7999) |

  So the "0.09 to 0.15" of the table below is a relative rms of the logits row; as a distribution change it is 0.0002 to 0.03 nats of KL over these chunk-end positions. **Only position 7999 is a distribution a request samples from**: there KL is 0.029 (chunk 1024), 0.013 (2048), 0.026 (4096) and 0.023 for the fusion toggle, top-1 unchanged for the three chunk sizes and flipped for the toggle. The top-1 token is unchanged in all nine chunk-size comparisons over the chunk-end rows, but those are 3 positions that matter little; the greedy continuation is the better test (next section). These are 9 comparisons at one prompt: a size indication, not a rate. They compare chunk sizes to chunk 512, not to the truth (see Not checked). Only the 7999 row (the last prompt position) is a distribution a request would sample from; the others are logits at the end of a chunk inside the prompt.

## Results, 8K prompt (windows: B1 job 8039 and B2 job 8072, same binary)

| pair | logits @7999 | last hidden @7999 | logits @4095 | indexer Jaccard (mean, min) | router-id Jaccard |
|---|---|---|---|---|---|
| c512 vs c512 again (job 8072 `r512`) | 0 | 0 | 0 | 1.000 | 1.000 |
| c512 vs c512 with `INTARSIA_DSV41_F16W_ROWS=1` | 0 | 0 | 0 | 1.000 | 1.000 |
| c512 vs c512 with `INTARSIA_DSV41_EXL3_SPLITK=0` | 0 | 0 | 0 | 1.000 | 1.000 |
| c512 vs c512 with `INTARSIA_DSV41_CAP=32768` | 0 | 0 | 0 | 1.000 | 1.000 |
| **c512 vs c512 with `INTARSIA_DSV41_HC_FUSED=0`** (separate HcMixes + HcPre + RmsNorm, rounding only) | 1.30e-01 | 8.90e-02 | 3.56e-02 | 0.921, min 0.603 | 0.902 |
| c512 vs c1024 | 1.23e-01 | 8.91e-02 | 1.01e-01 | 0.919, min 0.595 | 0.897 |
| c512 vs c2048 | 8.79e-02 | 7.47e-02 | 4.06e-02 | 0.923, min 0.625 | 0.908 |
| c512 vs c4096 | 1.39e-01 | 1.01e-01 | 8.23e-02 | 0.920, min 0.600 | 0.897 |
| c1024 vs c2048 | 1.26e-01 | 8.97e-02 | 7.98e-02 | 0.922, min 0.636 | 0.917 |
| c1024 vs c4096 | 9.20e-02 | 9.41e-02 | 4.22e-02 | 0.918, min 0.603 | 0.917 |
| c2048 vs c4096 | 1.54e-01 | 1.19e-01 | 5.97e-02 | 0.920, min 0.646 | 0.911 |

What this says:

1. **The harness is exact.** A second run at chunk 512 is bit-identical to the first at every dumped row, and three settings that must not change numerics (`F16W_ROWS`, `EXL3_SPLITK`, `CAP`) are bit-identical too. So every difference below is a real numeric difference between kernel/order choices, not run-to-run noise.
2. **Changing the chunk size moves the output by the same amount as switching off the HC fusion**, which is a pure change of rounding order (1.3e-1 for `HC_FUSED=0` against 0.9e-1 to 1.5e-1 for the chunk pairs; router-id Jaccard 0.90 against 0.90 to 0.92). Nobody calls `HC_FUSED=0` a bug. Each of the six chunk pairs is about equally far apart, with no ordering by chunk size: 2048 is not an outlier (against the mean of the others, c2048 is 8.5e-2 and the `HC_FUSED=0` run is 1.3e-1).
3. **The difference starts at layer 0 as rounding and grows with depth.** Embedding, Engram lookups and layer 0's `hc_attn` normalised input are bit-equal between chunk sizes. Layer 0 `hc_ffn` input is bit-equal for 190 of 225 position/pair cases, the layer 0 router weights for 190 of 315: the first deviation is the GEMM tile shape (rows per tile) changing the sum order. The relative rms of the FFN input then grows 2e-5 (L0), 3.2e-3 (L1), 1.3e-2 (L2), 3.2e-2 (L5), 6e-2 (L8), 1.1e-1 (L12), 1.5e-1 (L16), 1.8e-1 (L30). Expert ids differ for about 10 % of picks, indexer selected sets for about 8 % (min 0.60): a near-tie flips to the neighbour, and the flipped expert or block changes the next layer's input by more than rounding does.
4. **No chunk-boundary effect.** The normalised FFN input at the first row of a chunk (p512, p1024, p2048, p4096) is not further from the other run than the row before it (c512 vs c2048: 1.09e-01 at p511, 1.22e-01 at p512, 9.9e-02 at p513; 9.4e-02 at p1023, 9.1e-02 at p1024). A stale compressor / sliding-window state at a boundary would show as a step there. The indexer Jaccard falls with position (1.00 at 511, 0.92 at 1023, 0.83 at 4095) because there are more candidate blocks to choose from, not because of chunk edges (it is flat across a boundary).

## Greedy text (what a user sees)

Same 8000-token prompt, 16 greedy tokens (`raw/ours-doc8k-ab*.json`, job 8039): chunk 512, 2048 and 4096 give the same 16 tokens as each other (`"... was started twice in one window, A"`), chunk 1024 differs from token 9 (`"... ran twice in each window, A ="`). The logprob distance of the top-5 (lpdist.py) to chunk 512 is 0.19 nats at 1024, 0.06 at 2048, 0.11 at 4096. For comparison, `HC_FUSED=0` (job 8072) gives a different first token than chunk 512 (0 of 16 tokens agree), so the rounding-order toggle disturbs this prompt's greedy text more than chunk 2048 or 4096 do; with one prompt this says the sizes are comparable, not that a chunk change is the safer one. TensorFold's text for the same prompt shares the first 9 tokens with all four and then differs (`"... was started twice in a window, A = fused"`), so TF and our engine at any chunk are not token-identical beyond about 9 tokens either.

## TensorFold

- TensorFold's chunk is a **compile-time constant**: `pub const chunk_rows = 2048;` in `src/tensorfold/zig/src/families/dsv41/model.zig`, startup line "prompts in 2048-row chunks". `TF_DS_PREFILL_CHUNK` appears in comments and in `tools/dsv41-zig/build_inputs.sh`, but the binary does not read it. The attempt to run TF at 512 (`TF_CHUNK=512` in `tf/bf-node-prof.sh`, job 8087) started the same 2048 build; its startup log says 2048. That run is therefore a repeatability check: 7 of 7 prompts that completed in both starts gave identical 64-token texts (`raw/tf-text/`).
- TensorFold's server refuses `logprobs` ("logprobs are not supported by this model or backend"), so the planned TF logprob comparison (job 8064) returned only errors; the retry (job 8087) compares text only.
- So TF's own chunk sensitivity was not measured: that needs a TF rebuild with another `chunk_rows`.

## 32K prompt

One 32768-token prompt (`prompts/long32k.txt`; job 8115, the `CAP=33000` rerun), chunk 512 / 1024 / 2048, same binary, dumps at 15 positions up to 32767 (`raw/study-32k-chunks.out`, `raw/logitdist-32k.out`, `raw/indexer-jaccard-by-position-32k.out`).

| run vs chunk 512 | relative rms of the logits row, position 32767 | KL(512 \|\| run) nats at 32767 | KL mean over the 5-6 chunk-end positions | top-1 changed | 16 greedy tokens | top-5 mean abs dlogprob of the 16 tokens (lpdist) |
|---|---|---|---|---|---|---|
| chunk 1024 | 0.063 | 0.0163 | 0.0087 (6 positions, max 0.028 at 16383) | 0 of 6 | identical | 0.126 |
| chunk 2048 | 0.068 | 0.0176 | 0.0110 (5 positions, max 0.029 at 16383) | 0 of 5 | identical | 0.079 |

- The final-position distance at 32K (rel rms 0.06 to 0.07, 0.016 to 0.018 nats) is **no larger than at 8K** (0.09 to 0.14, 0.01 to 0.03 nats): in this one prompt the shift does not grow with length. The earlier doc's 0.19 nats at 32K is a different statistic (logprob differences of generated tokens, taken from that doc, not re-derived here).
- Normalised FFN input per layer grows with depth as at 8K (3e-3 at L1, 4-5e-2 at L5, 1.5e-1 at L20 and L30); layer 0's attention and `hc_attn` input are bit-equal across chunk sizes (45/45), layer 0 `hc_ffn` input and router weights in 33 of 45 cases. No step at chunk boundaries (c512 vs c2048: p511 7.4e-2, p512 6.1e-2; p1023 1.47e-1, p1024 1.75e-1; p1023 / p1024 straddle a boundary of c512 and c1024 but not of c2048; indexer Jaccard 0.919 and 0.902).
- **Indexer selected-set Jaccard (the plan's gate, >= 0.98 at 32K): not met.** Chunk 2048 vs chunk 512, mean over the 8 indexer layers: 1.00 up to position 511, 0.92 at 1023, 0.87 at 2047, 0.84 at 4095, 0.79 at 8191, 0.77 at 16383, **0.76 at 32767** (min over layers 0.60); chunk 1024 vs 512 is the same (0.874 mean against 0.875 over all positions). The sets are the top 512 of the candidate blocks, so near-tied scores swap members. The `HC_FUSED=0` toggle at 8K gives the same kind of drop (0.77 at position 7999 against chunk 512), so the Jaccard falls with rounding-level noise too and does not separate a bug from noise; but the gate as written fails, which is one more reason chunk 2048 stays opt-in.
- Memory: the 32K runs at chunk 2048 peaked with a 512 MiB scratch and did not trip memfine's 5 GiB line (`raw/spark1-memfine-214643.log`, 0 kills).

## Why there is no f32 reference (job 8039, ab4 `f32c512`)

The engine exits during warm-up with `deepseek_v41 step embed (embedding_bf16): backend error: ik_embedding_bf16 returned IK_ERR_DTYPE (2); inputs [I32, Bf16] shapes [[1], [129280, 5120]]` (`.scratch/winB1/ab4/serve-195737.log`, copy in `raw/winB1-ab4-f32-steps.log`). `INTARSIA_DSV41_ACT=f32` (`dsv41_engine.rs:64`) makes the graph's activations F32, but the CUDA backend names its generic ops by dtype and has them for bf16 only: `kernels/cuda/elementwise.cu:348` `ik_embedding_bf16` checks `IK_DTYPE_BF16` for the output, and `crates/intarsia-backend-cuda/src/device.rs` dispatches 21 `*_bf16` kernels against 2 `*_f32` (`argmax_f32`, `add_f32`). The embedding is only the first failing op: the rest of the generic elementwise ops, the EXL3 GEMM activation input and the MoE kernels would fail next, and `ACT=f32` exists for CPU parity runs (`crates/intarsia-models/tests/dsv41_runtime.rs`). Making a CUDA f32 graph is a kernel port, not a small fix, so I stopped without a fix attempt.

**What is missing for a reference, and substitute checks (none run):**

1. A CUDA f32-activation path for the generic ops (embedding, add, mul, rmsnorm, silu, cast), the EXL3 dense / MoE GEMMs with f32 inputs, and a check of each DeepSeek kernel for f32 (`dsv41_hc_sub.cu` templates over the float type; I did not check the attention, indexer, MoE and EXL3 kernels). Large.
2. **A long-prompt quality run at 32K/65K** with localeval on the engine at chunk 512 and 2048 (a needle-in-a-haystack or long-document QA set, 50 or more items each), then compare accuracy: a task-level answer to "is 2048 worse", at the lengths where the logprob shift was largest. MMLU is short-context and says little about it.
3. **Text comparison against TensorFold on 32K/65K prompts** (TF refuses logprobs, so text only): greedy 128 tokens from TF and from chunk 512 and 2048, the length of the common prefix. The chunk size whose text is closer to TF's is a weak signal that it is closer to a second implementation, not to the truth.
4. **A CPU f32 run** of a short prompt (a few thousand tokens) with `ACT=f32` through the test path in `dsv41_runtime.rs`, if a machine can hold the 99 GB of weights; unverified here whether any can, and its chunk sizes would have to be above the prompt length to differ.
5. **Chunk 2048 against 512 at 32K and 65K on the same prompt with the greedy continuations** (the prefill doc has the 32K logprob shift of 0.19 nats on one prompt) repeated over 20 or more distinct long prompts, to get a rate of changed greedy tokens.

## What did not work

- The whole-prompt-as-one-chunk control (chunk 8192 on the 8K prompt, job 8072) was **killed by memfine**: the workspace of an 8192-row chunk is 4.41 GiB, MemAvailable after warm-up was 5.15 GiB, and the 512 MiB kernel scratch that the first prefill pass allocates took it to 4.05 GiB, under the 5 GiB kill line (`spark1-memfine-201620.log` and `spark2-memfine-201620.log`: `KILLED under 5 GiB`; the client saw an empty reply after 1 s). Not a bug of the engine; not fixed, since chunk 4096 already bounds the study. (My first draft of this document said memfine did not fire; that was wrong, the log says it did.)
- The first 32K attempt (job 8072, `CAP=32768`) was rejected by the engine ("the prompt has 32768 tokens but the context window is 32768 tokens"); rerun with `CAP=33000` (job 8115).
- The TensorFold logprob run (job 8064) and the 512-chunk TF run (job 8087) did not measure what they were meant to (above).

## Not checked

- **Which chunk is closer to the truth.** All runs are bf16 activations. The `INTARSIA_DSV41_ACT=f32` reference runs cannot be made on the CUDA backend as it is (next section), so no run here is closer to the truth than the others. The relative-rms and KL figures say how far chunk sizes are from each other, and that the distance is no larger than an equivalent fusion toggle; they do not say 2048 is as accurate as 512. MMLU 0.879 at chunk 2048 (570 items, job 7948) is inside the gate, which is an accuracy statement at the task level only.
- Sampled (non-greedy) output, prompts of 65K, and rank 1 (only rank 0 dumps were read).
- Why the 0.19-nat 32K shift of the earlier doc is larger than the 8K 0.06: one prompt each; not enough to say it grows with length.

## Files kept out of the commit

`scripts/dsv41-tp2/run-full.sh` needs a three-line `LONG_PROMPTS_AB<i>` option (a different `longp` prompt list per `AB` config) for the 32K runs of `winB3.sh`. That edit is **not committed**; the copy is `.scratch/run-full.with-longprompts-ab.sh` in the work folder (`/opt/code/intarsiaml/.worktrees/dsv41-prefill/.scratch/`; the same change as a patch against main's `run-full.sh` is `.scratch/run-full-longprompts-ab.patch`), and the three `winB*.sh` copies in `raw/` need it.

## Reproduce

`.scratch/winB1.sh`, `winB2.sh`, `winB3.sh` (copies in `raw/`) are `AB=` lists for `scripts/dsv41-tp2/run-full.sh`, each a `fleetq run --host spark1,spark2 --takes-down spark1-vllm,spark2-vllm --quiet gpu,cpu` job. Compare: `uv run --no-project --with numpy python tools/dsv41/chunkstudy.py c512=<dir> c2048=<dir> ...` after copying `~/intarsia-dsv41-prefill/dump/<tag>` from spark1.

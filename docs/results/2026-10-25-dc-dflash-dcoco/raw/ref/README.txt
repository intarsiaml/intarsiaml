DFlash / DCOCO numerical reference (lane dc), 2026-10-25, host endor (Apple M5 Max, 128 GB)

Software: mlx 0.32.3, mlx-lm 0.31.3 (/opt/code/TensorFold/.venv/bin/python), Python 3.13.15, numpy 2.5.3.
Target: Qwen/Qwen3-8B HF bf16 (/Users/josef/models/Qwen3-8B). Draft: TD-Labs/Qwen3-8B-DCOCO rev 4080a604 (sha256 of
model.safetensors verified = df240fec...cb12c); dcoco.*, embed_tokens, lm_head are not loaded into the draft
(embed/lm_head are bound from the target; they are byte-identical, verified earlier by the lane).
Code: tools/dc/ref/{dflash_ref.py, dcoco_ref.py, zlab_dflash_mlx.py (MIT excerpt, notice kept), test_dcoco_ref.py,
transition_acc.py, first_diff_margin.py, make_results.py}; prompts: tools/dc/prompts_dc.jsonl (16: 6 math, 5 code, 5 chat;
Qwen3 chat template, enable_thinking=False; greedy, 192 new tokens max, stop at <|im_end|>/<|endoftext|>).
Every GPU run went through fleetq (`fleetq run --host endor --expect N --mem 28 --quiet gpu,cpu ...`).

COMMANDS (W=/opt/code/intarsiaml/.worktrees/dc-ref, R=.../dc/tools/dc/ref, C="--ckpt ckpt --target /Users/josef/models/Qwen3-8B
--prompts .../tools/dc/prompts_dc.jsonl --max-new 192 --plain-cache runs/plain192.json", cwd=W):
  python $R/dflash_ref.py $C --mode dflash --layout r15  --out runs/dflash_r15.json      # also r16s, zlab (layout comparison)
  python $R/dflash_ref.py $C --mode dcoco --variant grid1 --out runs/dcoco_grid1.json    # default, eps=1e-6, gelu=tanh, order=B, order=C, hnorm=pre, logit=logsoftmax
  python $R/dflash_ref.py $C --mode dcoco --variant "eps=1e-6,gelu=tanh" --out runs/dcoco_combined.json
  python $R/dflash_ref.py $C --mode dcoco --variant logit=lmax --out runs/dcoco_lmax.json
  python $R/dflash_ref.py $C --mode dcoco --variant default --skip-spec --fixture-dir out --fixture-ids math2,code1,chat2 --layout r15
  python $R/transition_acc.py --ckpt ckpt --target ... --prompts ... --plain runs/plain192.json --step 3
  python $R/first_diff_margin.py --target ... --prompts ... --run runs/dflash_r15.json --out runs/first_diff_margin.json
  python $R/make_results.py runs raw/ref
Wall times: plain greedy target on 16 prompts 96 s; each speculative pass over the 16 prompts 70-77 s (dflash and dcoco
alike); full grid1 job 509 s. Model load < 2 s (page cache warm).

FINDINGS THAT MATTER FOR THE RUST IMPLEMENTATION
1. BLOCK LAYOUT of this checkpoint differs from z-lab's MLX loop and from the lane note section 1. Decisive: the drafts
   are rows 0..14 INCLUDING the anchor row, and row r predicts the token at anchor_pos + r + 1 (shifted next-token
   convention; probe over 28 anchors showed row 0 = next token at 82%, row r = token +r+1). The z-lab convention (16 rows,
   drafts = rows 1..15, "zlab") is wrong here: pooled tau 1.041. Context = target features of positions 0..anchor_pos-1,
   anchor token not yet fed. Taps = OUTPUT of target layer i (0-based), i in 0,8,16,24,32: correct at the first try.
   UNRESOLVED: whether the query block has 15 rows ([anchor]+14 masks, config block_size 15; pooled tau 3.298) or 16 rows
   ([anchor]+15 masks, row 15 unused; pooled tau 3.261). In-loop gap 0.037 is inside the noise, and the paired
   teacher-forced comparison on 1373 anchors (layout_tf.txt) gives mean accepted 4.326 vs 4.323, paired diff
   -0.003 +- 0.016, argmax agreement 0.964, so the extra mask row changes little. r15 is the primary fixture set
   (out/fixture_*.npz); r16s is the hedge (out/r16s/). Verify is [anchor, d1..d15] = 16 rows in both.
2. DCOCO structure confirmed: teacher-forced conditional top-1 accuracy of slot p+1 given the true slot p (7379 transitions,
   transition_acc.txt): argmax 0.587 -> default corrector 0.748; order B 0.538, order C 0.565, pre-norm h 0.577,
   [g;b] fuse order 0.538, log-softmax logits 0.724, l-minus-top1 0.725. eps/GELU variants: 0.7477 / 0.7479.
   NOTE: this metric is conditioned on the TRUE token at slot p, not on a correctly drafted prefix, so it is a variant
   discriminator (relative comparison), not a predictor of the tau gain.
3. In-loop tau gain of the corrector on these prompts is small: DFlash 3.298 -> DCOCO 3.360 pooled (+1.9%), versus the
   paper's +27% (3.913 -> 4.985, ten suites, not these prompts). Mean tau per prompt (not pooled): 4.64 -> 4.78. The sample
   is small (904-920 rounds) and trajectories differ per variant, so differences below ~0.05 tau are noise. Chat prompts
   (tau 1.4-2.6) get nothing; math/code prompts (tau 4-8.5) get most of it.
4. Winner = default (eps 1e-5, erf GELU, order A = [s;za;zb;la;lb], h post-final-norm, raw logits). eps 1e-6 and tanh GELU
   are statistically indistinguishable (identical accepted counts in all 904 rounds; drafts differ in 18 / 3 rounds).
   Margin to the next DISTINCT variant (logit=logsoftmax) is only +0.007 tau; vs the clearly wrong ones +0.1 .. +0.33.
   Treat eps and GELU as unresolved by tau; resolve them (if needed) by a logit-level comparison against a real DCOCO run.

CAVEATS
- Spec output == plain greedy target in 9/16 prompts for every variant (even zlab, tau 1.04), 7 differ at the first
  difference position listed in first_diff_margin.json: every one is a bf16 logit tie or near-tie (gap 0 to 0.5 in the
  single-step target logits at logits of 24-37, i.e. 0 to 2 bf16 ulps; 4 of the 7 are exact ties); batched 16-row verify
  vs 1-row decode picks a different side. The first-difference index of each of the 7 prompts is IDENTICAL in all 13 runs
  (dflash r15/r16s/zlab and all dcoco variants), although drafts and round boundaries differ between them: this is
  multi-row-vs-single-row kernel numerics of the target on this GPU, not a draft/verify logic error; after the first difference the text
  diverges (the plain ids are NOT re-synced), later rounds are still valid speculation on the diverged text.
- MLX f32 matmul on the Apple GPU (M5 Max) is not full f32 (~1e-3 relative error vs f64). The corrector therefore runs
  on the CPU stream (Accelerate sgemm, true f32, validated against a float64 loop reference in test_dcoco_ref.py).
  Target and draft transformer math is bf16 MLX/GPU: use tolerance when comparing another implementation.
- Top-16 candidates are ordered by (logit desc, token id asc) from bf16 logits (many exact bf16 ties); plain-DFlash
  draft = candidate 0. With the corrector's MLP output zeroed the path is all zeros = plain argmax (unit test).
- tau definitions: tau_pooled = 1 + mean(accepted) over all rounds of all prompts (primary); tau_mean_of_prompts =
  mean of per-prompt tau; tau_committed = committed tokens per round (end of generation truncates the last round).
  Full 15-draft blocks are always run, even in the last round (commit is truncated to max_new).
- Fixtures (/opt/code/intarsiaml/.worktrees/dc-ref/out/fixture_{math2,code1,chat2}.npz + .json, P = prompt_len + 24, i.e.
  93 / 66 / 64; layout r15 so block hidden states are 15 x 4096, not 16 x 4096; a 16-row r16s set is in out/r16s/) are
  built from ONE target prefill of tokens[:P] (not from an in-loop trajectory) and a fresh draft cache.
- The lane note's description of the DCOCO query block (16 rows, row 0 dropped) is what the vLLM README suggests; this
  checkpoint does not behave that way (tau 1.04). Not checked: whether vLLM's own DFlash path implements it differently.

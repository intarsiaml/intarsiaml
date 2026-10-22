# ck: can the default prefill chunk go from 2048 to 4096 or 8192? (2026-10-22)

Lane ck, measurement only: no default, flag or kernel changed. Base `origin/main` a6217969 on all three backends (binaries built
from that tree plus this lane's `scripts/ck/`; `intarsia 0.0.1`, unstamped). Models: `nvidia/Qwen3.6-35B-A3B-NVFP4` and Qwen3-8B BF16.
Every number below comes from an EXCLUSIVE fleetq slot unless it says otherwise. Raw logs: `2026-10-22-ck-prefill-chunk/raw/`.

## 0. Answer

| Backend | Model | Bitwise vs 2048 (prefill logits, decode-after-prefill, greedy ids) | Memory (4096 / 8192 vs 2048) | Prefill tok/s at 4k (4096 / 8192) | at 16k (4096 / 8192) | Recommendation |
|---|---|---|---|---|---|---|
| CUDA spark1 (GB10) | 35B | **NO.** Every prompt longer than 2048 tokens differs from row 4 on; 28744 of 30200 rows, max abs d log-prob 13.5 nats, max row KL 10.2, row-mean KL 2.3e-2, 1359 argmax flips (4.5 %); greedy ids diverge at token 5 of 64 | scratch 80 -> 160 -> 320 MiB (+80 / +240) | +7.6 % / +5.9 % | +6.2 % / +9.6 % | **Keep 2048.** 4096 stays an opt-in flag (user decision 2026-10-19 21:53 stands) |
| CUDA spark1 (GB10) | 8B | yes (28685 + 434 rows, 0 differ) | scratch constant 0.7 MiB; free-memory gauge unusable on unified memory | -3.9 % / -5.8 % (inside the 2048 serves' own 3.8 % spread) | +0.4 % / +4.4 % (2048 serves differ by 3.6 %) | **Keep.** Bitwise, but no gain beyond noise. Its default today is 256 (no `[tuning.cuda]` in `recipes/qwen3.toml`), see section 5 |
| Metal endor (M5 Max) | 35B | yes: prefill (30200 rows), decode, 64 greedy ids on 5.8k / 16k-token prompts | free after load -350 MB / -1049 MB; scratch +80 / +240 MiB | +1.1 % / +1.1 % | +4.3 % / +8.3 % | 4096 is bitwise-neutral and the better trade (+4 % at 16k, +0.35 GB); 8192 buys +8 % at 16k for +1 GB. **But the Metal 35B default today is 256, not 2048** (section 5): the real question there is the missing `[tuning.metal]` line |
| Metal endor (M5 Max) | 8B | yes | -264 MB / -793 MB | +1.5 % / +0.3 % (noise) | **-10.5 % / -25.5 %** | **Keep 2048.** Bigger chunks are slower at 16k (matches lane MP's "bitwise but not faster") |
| HIP tatooine (R9700) | 35B | yes | free after load -348 MB / -1049 MB; scratch +80 / +240 MiB | +9.7 % / +8.9 % (second run +9.4 % / +9.3 %) | +7.9 % / +7.1 % (second run +7.5 % / +6.9 %) | **Keep 4096** (already the recipe value). 8192 gains nothing more and costs another +0.7 GB |
| HIP tatooine (R9700) | 8B | yes | -264 MB / -793 MB | +2.3 % / +2.3 % | +7.8 % / +5.3 % | 4096 is bitwise-neutral, +7.8 % at 16k for +0.26 GB: **optional change 2048 -> 4096**, the user's call (the recipe comment measured +-1.2 % at 1k / 4k only) |

Bitwise means exactly 0: every top-64 id and every log-prob bit equal, on every row. Only CUDA 35B fails. The other five combinations are
exactly 0 for all 13-14 prompts, all rows, in prefill and in decode mode.

## 1. Where the old "chunk 4096 changes numerics" note comes from, and why sp2 saw nothing

* **Source of the note:** `docs/results/2026-10-19-cuda-round6.md` (sections on "Opt-in, not bitwise: `--prefill-chunk 4096`" and decision 2) and
  LOG row 2026-10-19 21:53 ("CUDA `--prefill-chunk 4096` stays a flag ... opt-in, not bitwise"). Measured 2026-10-19 on spark1, 35B CUDA,
  by `scripts/cuda-round6/chunk-check.sh` (fleetq #2796, #2806): 128 greedy ids and the top-20 logits for corpus prompts of 5822 / 11692 /
  16014 tokens at chunk 2048 / 4096 / 8192: ids and top-20 DIFFER on 5 of 6 pairs (`raw/slots/2796-chunk-check.log`; only 8192 vs 2048 on
  the 5822-token prompt came out identical, which this lane reproduces); KL21 on the long golden 4.576e-2 -> 4.678e-2 (+2.2 %); greedy ids diverge at token 41 of 64 at 16k.
  The kill switches `INTARSIA_CUDA_FP8_GEMM_REG=0`, `MOE_W4A16=0`, `GDN_SPLIT=0` and `INTARSIA_MOE_BM=64` did not restore invariance
  (`2806-chunk-bisect.log`).
* **Why sp2's "byte-identical ids" did not contradict it:** `scripts/sp2/chunk-job.sh` compared a 1485-token prompt (256 ids; shorter than every
  chunk, so one chunk at 2048 / 4096 / 8192: nothing to compare) and two longer prompts with `--max-tokens 1` (a single greedy token; 4189 and
  16355 tokens). One greedy token is insensitive: here (section 2) the 64-token greedy ids of the 5822-token prompt are identical at 8192 vs 2048
  and still differ at 4096 vs 2048, and the first-token top-20 logits (printed to 4 decimals) can match while 4000+ rows of the dump differ in
  the last bits.
* **fx5b's "cold prefill bit-identical across chunk 512 / 1023 / 1024 / 2048"** is also right, and is the same boundary: everything up to
  2048 is one kernel regime. This lane's CUDA 35B data show the break lies between 2048 and 4096 (section 2).

## 2. Numerics (all rows, exclusive slots)

Method. `intarsia accuracy --mode prefill|decode --prefill-chunk C --write-self-golden` (top-64 log-probs per row) at C = 2048 (base), 4096,
8192 on:

* 35B: the short golden (10 prompts, 18-700 tokens) + the 4k golden (`long_docs_4k` 4064, `long_code_4k` 4064, `chat_reuse_4k` 4631 tokens)
  + one synthetic 16000-token prompt (`scripts/ck/make_synth.py`: the three 4k prompts' ids tiled; its vLLM top-k tensors are placeholders and only the
  engine's own dump is used). 14 prompts, 30200 prefill rows. The 35B 4k golden is `onboard-runs/35b-cuda/golden-fresh` from spark1.
* 8B: the short golden (8 prompts, 434 rows) and the 4k golden (4064 / 4064 / 4557 tokens) + the same synthetic 16000-token prompt.

Chunk coverage: at 4096 the 4k prompts are one chunk (the base has two); at 8192 the 16000-token prompt is [8192, 7808] (the base has
eight chunks). `scripts/ck/chunkcmp.py` compares each dump with the 2048 dump row by row (id sets and log-prob bits); "max abs d logprob" is over
ids both rows share (a log-prob is logit minus logsumexp, so this is the logit change when the logsumexp is unchanged; the dump keeps top-k
log-probs, not raw logits). KL is KL(2048 || C) over the shared top-k, renormalised. Full per-prompt tables: `raw/<backend>/compare/`.

CUDA 35B, prefill (the only failing combination; `raw/cuda/compare/prefill-35b-all.txt`):

| Prompt | tokens | chunk | rows differing | max abs d logprob | max row KL | mean row KL | argmax flips |
|---|---|---|---|---|---|---|---|
| all 7 prompts of 18-700 tokens | 434 | 4096 / 8192 | 0 | 0 | 0 | 0 | 0 |
| long_docs_4k | 4064 | 4096 / 8192 | 4060 | 7.73 | 1.04 | 1.18e-2 | 209 |
| long_code_4k | 4064 | 4096 / 8192 | 4061 | 8.90 | 0.96 | 5.77e-3 | 84 |
| chat_reuse_4k | 4631 | 4096 | 4627 | 12.7 | 7.80 | 6.29e-2 | 372 |
| chat_reuse_4k | 4631 | 8192 | 4627 | 12.7 | 7.80 | 6.38e-2 | 377 |
| synth_16k | 16000 | 4096 | 15996 | 13.5 | 10.2 | 2.04e-2 | 694 |
| synth_16k | 16000 | 8192 | 15996 | 10.6 | 8.90 | 2.22e-2 | 709 |
| all prompts | 30200 | 4096 / 8192 | 28744 | 13.5 / 12.7 | 10.2 / 8.90 | 2.28e-2 / 2.39e-2 | 1359 / 1379 |

(`prefill-35b-all.txt` has the 14 rows per chunk.) The first differing row is row 4 of every prompt that is longer than 2048 tokens, and
2044 of the first 2048 rows (the rows that sit in chunk 1 under both schedules) differ: the result of a row depends on how many rows the chunk
has, so it is a T-dependent kernel choice or schedule on CUDA and not a chunk-boundary (state carry, KV append) effect. 4096 and 8192 agree
with each other on the 4k prompts (both single-chunk). The break is above 2048: fx5b found 512 / 1023 / 1024 / 2048 identical. The cause was
not hunted here (candidates: the fp8 GEMM schedule or MoE tile choice by token count; round 6's four kill switches do not fix it).
The size is that of a different kernel, not of rounding noise on single rows: mean row KL 1e-2 to 6e-2 is the order of the engine-vs-vLLM
KL21 (4.7e-2 on the 4-bit floor of 3e-2).

CUDA 35B, greedy (`generate --show-ids --show-logits 20`, 64 tokens, ignore-eos, corpus prompts of 5822 and 16014 tokens, `raw/cuda/gen/`):

| Prompt | 4096 vs 2048: ids / first-token top-20 | 8192 vs 2048: ids / first-token top-20 |
|---|---|---|
| 5822 tokens | DIFFER (first at token 5, 57 of 64 differ) / DIFFER | identical / identical |
| 16014 tokens | DIFFER (first at token 5, 58 of 64 differ) / DIFFER | DIFFER (first at token 5, 59 of 64 differ) / DIFFER |

The 5822-token result at 8192 (a single chunk against three) matches round 6's one "identical" pair, so greedy ids on one prompt can agree
while the rows do not. The dump comparison above is the decisive one.

Everything else, all exactly 0 (`raw/<backend>/compare/`): CUDA 8B prefill (short 434 rows, 4k + synth 28685 rows) and its 64 greedy ids on both
prompts; Metal 35B and 8B prefill and decode mode (16 teacher-forced steps after the chunked prefill of each prompt, 16000-token one included)
and the 35B / 8B greedy ids; HIP 35B and 8B prefill, decode mode and greedy ids. CUDA decode mode: __CUDA_DEC__

## 3. Memory

`intarsia plan show` is the same document as `GET /v1/intarsia/plan` of the running server (saved per chunk in `raw/<backend>/speed/<model>/plan-c*.json`).
Its `scratch_bytes` is the largest kernel scratch of the plan; the activation workspace is sized by the chunk too and shows in the server's
`intarsia_device_memory_free_bytes` gauge right after load (`mem-c*-load.txt`; `--max-seq 17408`, so the KV size does not move with the chunk).

| Backend, model | scratch_bytes 2048 -> 4096 -> 8192 | free after load, change vs 2048 (4096 / 8192) | fits |
|---|---|---|---|
| CUDA spark1, 35B | 80.1 -> 160.1 -> 320.3 MiB | gauge unusable (unified memory with vLLM and other jobs: it moves the wrong way, 4.1 / 5.1 / 5.3 GB free) | ran at all three chunks under `--mem-budget 40`, vLLM (64 GB) up, 36 GB MemAvailable at job start |
| CUDA spark1, 8B | 0.7 MiB at every chunk | gauge unusable | ran under `--mem-budget 26` |
| Metal endor, 35B | 80.1 -> 160.1 -> 320.3 MiB | -350 MB / -1049 MB (73.11 -> 72.76 -> 72.06 GB free of 115.4) | yes, endor services (kev, tensorfold) up |
| Metal endor, 8B | 1.0 MiB at every chunk | -264 MB / -793 MB (96.22 -> 95.95 -> 95.43 GB free) | yes |
| HIP tatooine, 35B | 80.1 -> 160.1 -> 320.3 MiB | -348 MB / -1049 MB (12.19 -> 11.85 -> 11.14 GB free of 34.2) | yes under the 29 GB budget, max-seq 17408; radiance down |
| HIP tatooine, 8B | 0.7 MiB at every chunk | -264 MB / -793 MB (14.71 -> 14.45 -> 13.92 GB free) | yes; radiance down |

The workspace growth is the same engine formula on every backend: 8B +264 MB (4096) / +793 MB (8192), 35B +350 MB / +1049 MB, about 0.13 MB per
chunk row for the 8B and 0.17 MB for the 35B, against the +84 / +252 MB the plan's scratch shows. On tatooine radiance holds both cards
(31 of 32 GB each) while it is up, so "fits with the services up" does not apply there: HIP runs only inside a radiance-down window, and the
recipe's 29 GB budget leaves 11 GB free at 8192. The 35B plan on CUDA: 160.1 MiB at 4096 matches round 6's 167.9 MB (`plan35-chunk-memory.log`).

## 4. Speed (exclusive slots; fresh `serve` per chunk, `bench speed --at-most`, 3 reps, 1 warm-up, prompts <= 4096 / 16384 tokens)

Prefill tok/s (median of 3 reps; the 2048 value is the mean of two serves on CUDA and Metal, whose order was 2048, 4096, 8192, 2048;
HIP has no repeat, but the HIP 35B ran twice in two windows). The change is against that 2048 value.

| Backend, model | size | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| CUDA 35B | 4k | 5881 (runs 5924, 5838) | 6328 (+7.6 %) | 6229 (+5.9 %) |
| CUDA 35B | 16k | 5587 (5574, 5600) | 5932 (+6.2 %) | 6126 (+9.6 %) |
| CUDA 8B | 4k | 4444 (runs 4527, 4360) | 4269 (-3.9 %) | 4187 (-5.8 %) |
| CUDA 8B | 16k | 3602 (3666, 3538) | 3616 (+0.4 %) | 3759 (+4.4 %) |
| Metal 35B | 4k | 1794 (1795, 1792) | 1814 (+1.1 %) | 1813 (+1.1 %) |
| Metal 35B | 16k | 1539 (1538, 1539) | 1604 (+4.3 %) | 1666 (+8.3 %) |
| Metal 8B | 4k | 3868 (3867, 3868) | 3924 (+1.5 %) | 3879 (+0.3 %) |
| Metal 8B | 16k | 2680 (2716, 2644) | 2398 (-10.5 %) | 1996 (-25.5 %) |
| HIP 35B, window 1 | 4k | 9123 | 9980 (+9.4 %) | 9973 (+9.3 %) |
| HIP 35B, window 1 | 16k | 8310 | 8934 (+7.5 %) | 8880 (+6.9 %) |
| HIP 35B, window 2 | 4k | 9143 | 10030 (+9.7 %) | 9956 (+8.9 %) |
| HIP 35B, window 2 | 16k | 8309 | 8965 (+7.9 %) | 8902 (+7.1 %) |
| HIP 8B | 4k | 4528 | 4634 (+2.3 %) | 4634 (+2.3 %) |
| HIP 8B | 16k | 3348 | 3608 (+7.8 %) | 3524 (+5.3 %) |

Noise. The two 2048 serves agree within 0.2-1.5 % everywhere except Metal 8B at 16k (2716 vs 2644, 2.7 %) and CUDA 8B (4k 4527 vs 4360, 3.8 %; 16k 3666 vs 3538, 3.6 %, so its +-5 % changes are noise), and the HIP 35B windows agree
within 0.5 %. Metal 8B at 16k is well outside that: the 4096 reps are 2679, 2398, 2397 and the 8192 reps 2118, 1996, 1869 tok/s, falling rep by
rep, and the repeat of 2048 at the end is back at 2644-2716, so it is the chunk and not drift. Per-rep numbers are in the summary of each
`raw/<backend>/speed/*/bench-c*.txt` (`scripts/ck/summarize.py`). Decode tok/s is unchanged by the chunk (35B CUDA 72 tok/s, HIP 144, Metal
108-109 at 16k).

Gains are not monotonic in the chunk: CUDA 35B at 4k gains most at 4096 (+7.6 %) but at 16k most at 8192 (+9.6 %); HIP gets everything at 4096.

## 5. Findings outside the question

* **Metal 35B has no `[tuning.metal]`.** `recipes/qwen3_5_moe.toml` lists only `[tuning.cuda]` (2048) and `[tuning.hip]` (4096), so a Metal
  35B run without `--prefill-chunk` uses the built-in default 256 (LOG row "Prefill chunk": "unlisted backends: 256"). The same holds for the CUDA 8B:
  `recipes/qwen3.toml` has `[tuning.metal]` and `[tuning.hip]` (2048 each) but no `[tuning.cuda]`. This lane compared 2048 / 4096 / 8192
  only; it did not measure 256 against 2048 for those two (the Metal 35B 256 slot was queued behind 25 minutes of other work and cancelled
  before the 11:30 stop). Earlier m35 numbers put 2048 well ahead of 256 on Metal 35B prefill.
* **The accuracy tool cannot see this on the short golden.** Every prompt of the shipped short goldens (<= 700 tokens) is one chunk at every
  size, so `qualify` / `accuracy` on them are insensitive to the chunk; the tool warns about it ("prompts spanning more than one chunk: 0/13").
  Only the 4k goldens and the synthetic 16000-token prompt reach the CUDA 35B difference.
* The first HIP 35B speed window ran Qwen3-8B by mistake (a stage-name bug in `hip-stage.sh`, fixed; labelled `ck-hip-35-*`); its data was
  discarded and the stage rerun. Raw logs of the correct run are in `raw/hip/speed35` and `speed35-run1`.

## 6. Not done / limits

* CUDA decode-mode dumps were run last and __CUDA_DEC_STATE__
* Nothing here says why CUDA 35B above 2048 differs; it was not root-caused (no kernel change allowed in this lane).
* The 16000-token prompt is synthetic (tiled 4k prompts), so its vLLM-side KL columns are meaningless; only chunk-vs-chunk numbers are used.
* HIP and Metal 8B 2048 speed has one or two serves, 3 reps each (CUDA and Metal have the drift-control repeat of 2048, HIP does not).
* Dumps (15 MB each) are not committed; their sha256 and the per-prompt comparisons are in `raw/<backend>/`.
* Metal 35B and 8B memory is the free-memory gauge of a shared 128 GB Mac with other services; the deltas repeat to the byte on HIP
  (same formula), which is why they are trusted.

## 7. Fleet use and downtime

All GPU numbers exclusive. Jobs (owner `claude-f31ccf91-ck`): builds #4130 (HIP, CPU), #4131 (CUDA, CPU), #4132 (Metal, CPU); CUDA #4171
(exclusive, vLLM up; #4142 cancelled while queued to lower the memory claim; #4140 ran the script on the Mac by mistake with no binary and did nothing);
Metal #4135 (acc + speed), #4190 (greedy ids + decode mode); HIP (radiance down, exclusive): #4136 acc, #4137 acc with the 16k prompt, #4139
(the 8B-by-mistake run, discarded), #4155 35B speed, #4164 8B speed, #4191 greedy ids + decode mode, #4213 35B speed repeat with memory gauges.
Radiance downtime, from the `og-burst` "window total": 166 + 323 + 265 + 86 + 262 + 129 + 81 = 1312 s = 21.9 min (the `downtime.log` "wall"
column also counts the up-to-5-minute wait the burst script inserts before it stops radiance, so it overstates). Radiance answered within
15-25 s of every restart.

Reproduce: `scripts/ck/cuda-build-spark1.sh`, `hip-build-tatooine.sh`, `cargo build --release -p intarsia-cli --features metal`; then
`scripts/ck/cuda-run.sh`, `metal-run.sh`, `hip-window.sh acc|dec|speed35|speed8 <cap>` (each calls `chunk-job.sh`); `chunkcmp.py`, `compare-all.sh`,
`summarize.py` read the output.

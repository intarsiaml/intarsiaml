> Our arms (B, D) used build 22303d704 (#493), before T1 (#497) and the GPU sampler, so our columns understate main.

# mlx-serve v26.11.4-pre-release.1 vs intarsia Metal engine, Inkling-2-Mini, same box (endor, M5 Max 128 GB)

2026-11-22, one pass A B C D (the coordinator cancelled the planned D C B A reverse pass). localeval 0.4.0 `suite --tier adhoc
--cells speed-plain` and `--cells conc-plain-1k`, filler-v1 prompts, greedy, 256 forced output tokens, 3 reps. All timing ran in
one fleetq quiet slot (#11677, gpu+cpu quiet, 17:20 to 19:00). Every localeval result is in /opt/code/deepeval/results/<run id>.

## Setup

| | mlx-serve | intarsia |
|---|---|---|
| commit | 2fe9be834a601c70ef1a597563026340f0e722fe (tag v26.11.4-pre-release.1), `mlx-serve 26.11.4 (fast)`, MLX 0.32.5, zig 0.17.0, `zig build -Doptimize=ReleaseFast` server-only (NAX symbols asserted by build-mlx.sh) | 22303d704fb2c7cf68a1ffa5f8268cce26f4aafd (origin/main), `cargo build --release --bin intarsia --features metal` |
| checkpoint | release snapshot b85c3cfa615fc413cc9af3379f491b963f3b586a (NVFP4+MXFP8, `mtp.safetensors` beside it) | same |
| draft head | `mtp.safetensors` auto-loaded (DSpark, block 8, native stages) | `--spec dflash --spec-draft <snapshot> --spec-k 3` |

Exact server command lines (the ones recorded in the run dirs; `CK` = the snapshot path above):

- A: `/opt/code/intarsiaml/.worktrees/mlx-serve-src/zig-out/bin/mlx-serve --model $CK --serve --host 127.0.0.1 --port 18101 --ctx-size 20480`
  (draft head on = default; PLD is also on by default, its gate usually switches it off on these prompts)
- C: same binary, `--model /opt/code/intarsiaml/.worktrees/mlx-serve-bench/model-nodraft --serve --host 127.0.0.1 --port 18101 --ctx-size 20480 --no-pld`.
  `model-nodraft` is a directory of symlinks to the snapshot minus `mtp.safetensors`: the plugin loads the draft head only when that file exists
  (`src/plugins/inkling2/config.zig`). `--no-drafter` and `--no-mtp` do not govern the Inkling head. `--no-pld` removes the n-gram fallback.
  C's log says `spec=disabled (inkling2_loop serves serial-only)`, no `[spec-stats]` lines and `pld=off`.
- B (speed): `.../intarsia serve --model $CK --backend metal --max-batch 1 --max-seq 20480 --port 18102 --served-model-name inkling-2-mini --spec dflash --spec-draft $CK --spec-k 3`
- D (speed): same with `--spec off`.
- D (conc-plain-1k): `--max-batch 16 --max-seq 8192 --spec off` (restarted for the cell).
- B (conc-plain-1k): the batching engine refuses it (`unsupported: --spec dflash with the batching engine (use --max-batch 1)`), so the cell ran on the
  `--max-batch 1` server and the streams were serialised. mlx-serve (A, C) batches the same server with drafting on: its default `--max-concurrent 1` still
  batch-decoded 16 streams (help text: "Concurrent requests batch-decode either way"); localeval was told `--max-num-seqs 1`.

localeval flags: `--thinking default` (nothing sent) for all arms, `--tokenizer-repo thinkingmachines/Inkling-2-Mini --tokenizer-revision b85c3cfa...`,
`--max-context 20480` (conc D: 8192), `--lane plain` for A, C, D and `--lane spec:dspark` for B. Both servers answer with thinking on in that mode
(reasoning_content present). Their chat renderers differ by 2 prompt tokens (mlx-serve 32, intarsia 34 for the same short prompt: mlx-serve renders the
effort as the word `high`, intarsia as 0.9), so greedy chat text starts differently; speed prompts are built per engine from the same pinned tokenizer.
A is declared `plain` on purpose: mlx-serve's `/metrics` is vLLM-named, so localeval has no draft counters for it (a declared spec lane with no counters can
abort on the probe ratio). mlx-serve was started without `--metrics`, because its `vllm:num_requests_running` gauge reads 1 even when idle and localeval
refuses a "busy" server (suite has no `--ignore-busy`). Without it the idle check is `none`.

## Smoke test (arms A B C D, 40 raw greedy tokens, `/v1/completions`, 29-token prompt; logs/smoke3)

- A and C produced the identical 40 tokens; B and D the identical 40 tokens (draft on/off does not change greedy text in either engine).
- Across engines the text is near-identical but not byte-identical: mlx-serve `:\n\n"I am a lighthouse. I am a beacon of hope for those who are lost at sea. I am a symbol of safety and guidance. I am a reminder that even`,
  intarsia `:\n\n“I am a lighthouse. ... I am a symbol of strength and resilience. I am a reminder that even`. Same words for 24 tokens apart from a straight vs curly opening quote
  (a near-tie), then one different phrase. Both are coherent, so both run the model correctly; an exact match across two kernels stacks is not expected.
- Drafting confirmed in A: `spec=dspark (inkling2_loop native draft stages, block=8)` on every request and `[inkling2] draft verify cut 0.54 (a verified row 0.16 of a plain step)`.

## Box check (drift)

`localeval speed --sizes 128 --gen 128 --reps 2` on intarsia, spec off, decode tok/s (not the arm D row; engine-side normal is ~76):

| when | decode tok/s |
|---|---:|
| 17:08 (first, aborted attempt) | 74.5 |
| 17:21 (before arm A) | 76.4 |
| 18:58 (after arm D) | 76.3 |

No drift over the run. (Runs 20261122-172052 and 20261122-185815 in /opt/code/deepeval/results; the earlier 170819 is the aborted attempt.)

## Speed sweep (speed-plain), per arm

Median of 3 reps; decode tok/s (range). "marginal prefill" is localeval's delta-based figure (tok/s). TTFT in seconds.

### A: mlx-serve, draft head on. Run 20261122-172125-adhoc-mlxbench-p1-A-speed (no cell rejected)

| prompt tok | TTFT s | marginal prefill | decode tok/s |
|---:|---:|---:|---:|
| 144 | 0.27 | - | 70.7 (69.1-72.6) |
| 1021 | 0.53 | 3412 | 69.1 (68.3-69.9) |
| 4113 | 1.62 | 2821 | 67.7 (65.4-69.8) |
| 16454 | 7.12 | 2247 | 62.3 (62.2-63.6) |

### B: intarsia, spec on (dflash, k=3). Run 20261122-174201-adhoc-mlxbench-p1-B-speed (1024 row rejected by the prefill spread gate, shown)

| prompt tok | TTFT s | marginal prefill | decode tok/s |
|---:|---:|---:|---:|
| 144 | 0.15 | - | 100.7 (86.7-102.0) |
| 1022 | 0.51 | 2406 | 83.7 (76.2-87.4) |
| 4116 | 1.70 | 2618 | 91.4 (79.6-102.0) |
| 16455 | 6.67 | 2480 | 81.5 (81.3-87.7) |

### C: mlx-serve, draft off. Run 20261122-181432-adhoc-mlxbench-p1-C-speed (16k row rejected: prefill and decode spread, shown)

| prompt tok | TTFT s | marginal prefill | decode tok/s |
|---:|---:|---:|---:|
| 143 | 0.34 | - | 68.3 (67.8-71.5) |
| 1019 | 0.80 | 1913 | 68.6 (66.9-70.3) |
| 4110 | 2.10 | 2374 | 68.4 (68.0-69.7) |
| 16457 | 8.74 | 1860 | 64.1 (59.9-64.7) |

### D: intarsia, spec off. Run 20261122-183714-adhoc-mlxbench-p1-D-speed (1024 and 4096 rows rejected by the prefill spread gate, shown)

| prompt tok | TTFT s | marginal prefill | decode tok/s |
|---:|---:|---:|---:|
| 143 | 0.15 | - | 72.5 (71.1-73.2) |
| 1021 | 0.56 | 2147 | 70.7 (69.5-71.7) |
| 4109 | 1.76 | 2586 | 69.6 (69.0-70.7) |
| 16457 | 7.01 | 2352 | 65.7 (64.1-66.1) |

Prefill spread rejections are the known mid-length noise (up to +-39 %); the marginal prefill figures of both engines are therefore within noise of each
other at 1k-16k except mlx-serve's higher 1k-4k values in A. Over 4k to 16k: A 2247 / C 1860 vs B 2480 / D 2352.

## conc-plain-1k (aggregate decode tok/s at 1 / 4 / 16 streams)

| arm | run id | 1 | 4 | 16 | TTFT s at 16 | note |
|---|---|---:|---:|---:|---:|---|
| A mlx-serve, draft on | 20261122-172348-adhoc-mlxbench-p1-A-conc | 59.3 | 101.6 | 138.6 | 1.33 | batches with drafting on |
| B intarsia, spec on | 20261122-174410-adhoc-mlxbench-p1-B-conc | 63.5 | 68.2 | 65.6 | 58.4 | cannot batch with drafting: serialised on `--max-batch 1` |
| C mlx-serve, draft off | 20261122-181808-adhoc-mlxbench-p1-C-conc | 55.0 | 88.9 | 128.4 | 2.08 | batches |
| D intarsia, spec off | 20261122-183950-adhoc-mlxbench-p1-D-conc | 55.0 | 93.1 | 126.4 | 1.55 | `--max-batch 16 --max-seq 8192` |

## Draft acceptance

- B (localeval counters, speed cell, 20261122-174201): 3931 draft tokens proposed, 3036 accepted (77.2 %), 1959 verify passes (about 2.55 tokens per pass).
  B conc cell: 73265 proposed, 52344 accepted (71.4 %).
- A (mlx-serve log `[spec-stats] mode=dspark`, whole arm, logs/run/mlxlog_p1_A.log last section): 135 requests with a stats line, 2082 rounds, 1781 accepted drafts =
  0.86 accepted drafts per round (1.86 tokens per round). Those rounds cover only about 3.9k tokens, far fewer than the arm generated, so most decode steps were plain steps (the
  engine's own verify cut and gate, `draft verify cut 0.30 to 0.63`); treat the figure as indicative. A's decode is 0 to 3 % above C's
  (70.7/69.1/67.7/62.3 vs 68.3/68.6/68.4/64.1), i.e. no real draft speed-up on these prompts. localeval's own docs say filler-v1 understates draft acceptance, and
  this is the case for mlx-serve here; B on the same prompts shows 1.2 to 1.4x over D. The mlx-serve release chart numbers (200 tok/s on M5 Ultra) were not reproducible
  with filler prompts; natural-text prompts (`speed-spec`, natural-v1) were not run (outside the requested cells).

## Peak server memory (sampled every 30 s from the server's process tree)

| arm | max RSS (MB) | max `footprint` (MB) |
|---|---:|---:|
| A mlx-serve, draft on | 18102 | 31744 |
| C mlx-serve, draft off | 17486 | 29696 |
| B intarsia, spec on | 27060 | 16387 |
| D intarsia, spec off (speed server) | 26177 | 15363 |
| D intarsia, `--max-batch 16 --max-seq 8192` | 24699 | 19459 |

The two metrics disagree by engine: intarsia maps the checkpoint (RSS counts the mapped file, `footprint` does not), mlx-serve loads weights into
GPU buffers (small RSS, large `footprint` that includes the MLX buffer pool and KV). Do not compare the columns across engines; neither is a
clean peak GPU number. The D speed-server sampler caught only 5 samples (a short run).

## Not run / caveats

- One pass only (A B C D), per the coordinator; no reverse pass, so order effects are not separated (the box check before and after is flat at 76.4 / 76.3).
- B's conc-plain-1k cannot batch with drafting; the figure is the serialised `--max-batch 1` server.
- A's draft acceptance comes from mlx-serve's own log only; localeval recorded no spec stats for A and lane `plain`.
- Several mid-length prefill rows were rejected by the spread gate (listed above) and are shown, not re-run.
- `speed-spec` (natural-v1 prompts), which would show mlx-serve's drafted speed on natural text, was not run.
- The first quiet attempt (#11667) was cancelled: mlx-serve's `--metrics` made every localeval cell refuse a "busy" server, and the coordinator asked for one pass. Its logs
  are in logs/run_attempt1. No results from it are used.
- Nothing was pushed or committed to intarsiaml; the intarsia worktree `/opt/code/intarsiaml/.worktrees/mlx-serve-bench/intarsia` (git worktree of origin/main 22303d704) and the
  mlx-serve source tree are left in place; scripts: arms.sh, run.sh, smoke.sh, memsample.sh, build_*.sh; logs under logs/.

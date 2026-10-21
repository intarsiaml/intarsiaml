# Standings 2026-10-21: intarsiaml against the other engines, per platform

Measurement only. No engine code changed. Requirement 4 is "aim for the top 3 on the charts for each supported platform".
Lane: standings (quota-burn plan, `docs/plans/2026-10-21-quota-burn-plan.md`). Base: `origin/main` 789511c2 for tatooine,
f78f60be for endor, spark numbers from rounds 6 and 7. Raw data is in `docs/results/2026-10-21-standings/raw/`.

## 1. Headline

| platform | model | engines measured live | intarsia rank | state |
|---|---|---|---|---|
| spark (GB10, CUDA) | Qwen3.6-35B-A3B NVFP4 | intarsia, vLLM | **#2 of 2** on prefill (-8.3 / -9.0 / -15.3 % at 1k / 4k / 16k, same evening); **#1 of 2** on decode at 1k and 4k (+5.7 % / +2.1 %), #2 at 16k (-4.2 %) | no other 35B-A3B engine is installed on the Sparks |
| R9700 (HIP, tatooine, one card) | Qwen3.6-35B-A3B | intarsia, llama.cpp Vulkan | **#1 of 2** live: prefill 2.3x / 3.2x / 2.8x, decode 1.41x / 1.42x / 1.35x of llama.cpp; **#2 of 4** on decode if the second-hand hipfire figure counts | first same-model measurement on this card (round 3 said none existed) |
| endor (Metal, M5 Max) | Qwen3-8B BF16 | intarsia, mlx-lm, llama.cpp | **#1 of 3** on prefill at 4k and 16k and on decode at every length (tie with llama.cpp at 16k decode in one arm) | quality tier failed, see 4 |

"Top 3" is met numerically on every platform, but only because the field is small: two engines on two platforms. Where
the field is real (vLLM on the Spark, hipfire on the R9700), intarsia is second.

**Quality is the weakest column.** The quick tier (MMLU plus GSM8K, n=25) produced usable numbers only on tatooine (a
partial MMLU, n=225) and for vLLM. On endor both attempts failed and on the Spark intarsia was not served (section 4).

## 2. Numbers

Common method: `localeval` (`/opt/code/deepeval`, localeval 0.2.1, deepeval 4.2.6), `--sizes 128,1024,4096,16384 --gen 128
--reps 3 --thinking off`, median of 3 reps after one warmup, nonce-first prompts so no prefix-cache hit. Prefill is
localeval's "prefill tok/s" column (uncached prompt tokens / TTFT). **The 1k prompt has 990 tokens, under localeval's 1024
threshold, so its prefill column is TTFT-overhead-influenced** (marked `*`). Decode is tok/s. Single stream. Each table is
one platform and one model. Cells in italics are second-hand and were not measured today.

### 2.1 Spark (GB10, CUDA): Qwen3.6-35B-A3B

vLLM serves NVFP4 weights with fp8 attention and fp8 KV on `spark-guidance` (:8000). intarsia is the CUDA build of round 6
(default) and the opt-in kernel of round 7. Pairs are only valid within one day: round 6's own text notes vLLM's 1k
figure moved about 11 % between days.

| engine | build / date | quant | prefill 1k* | prefill 4k | prefill 16k | decode 1k / 4k / 16k | TTFT 1k / 4k / 16k (s) | quality |
|---|---|---|---|---|---|---|---|---|
| vLLM 0.26.1rc1 | today 10:24 (`raw/spark/vllm-summary.md`) | NVFP4 + fp8, fp8 KV | 5190 | 6361 | 6284 | 78.0 / 76.8 / 73.9 | 0.19 / 0.64 / 2.65 | MMLU 84.4 % (n=1425), GSM8K 52 % (n=25) |
| vLLM | 10-19 evening (round 6 pair) | same | 4699 | 6245 | 6308 | 78.7 / 79.1 / 76.3 | | |
| intarsia **default** | round 6, 10-19 (`raw/spark/r6-localeval-dev-vs-vllm-2026-10-19.log`) | NVFP4 + fp8 | 4310 | 5681 | 5344 | 83.2 / 80.8 / 73.1 | 0.23 / 0.72 / 3.12 | KL21 vs vLLM top-20 golden: short 5.43e-3 (top-1 97.92 %), long 4.576e-2 (top-1 94.17 %; fails the 3.0e-2 gate on base too) |
| vLLM | 10-21 10:56 (round 7 pair) | same | 5171 | 6353 | 6275 | 77.9 / 76.6 / 73.6 | | |
| intarsia opt-in `INTARSIA_CUDA_FP8_GEMM_LT=1` | round 7, 10-21 | same | 4689 | 6378 | 5964 | 82.4 / 79.9 / 72.3 | ~0.21 / ~0.64 / ~2.79 (derived) | KL21 long 1.355e-2 vs default 1.420e-2 on the M3 floor metric; `chat_reuse_4k` continue 3/32 (fails) |
| intarsia opt-in lt + `--prefill-chunk 4096` | round 7, 10-21 | same | 4671 | 6845 | 6354 | not reported per size (unchanged) | | not gated |

Deltas of intarsia against vLLM of the same day (positive = ahead):

| intarsia build | prefill 1k* / 4k / 16k | decode 1k / 4k / 16k |
|---|---|---|
| default (r6, vs 10-19 vLLM) | -8.3 % / -9.0 % / **-15.3 %** | +5.7 % / +2.1 % / -4.2 % |
| opt-in lt (r7, vs 10-21 vLLM) | -9.3 % / +0.4 % / -5.0 % | +5.8 % / +4.3 % / -1.8 % |
| opt-in lt + chunk 4096 | -9.7 % / +7.7 % / **+1.3 %** | unchanged |

Staleness: the CUDA default was last changed by round 6 (PR #31). Round 7 (PR #39, 11:13 today) is opt-in and leaves the
default bitwise, so the round-6 default row is still current. The round-7 rows are the only same-day pairing; no default arm
ran against vLLM today. Today's vLLM run agrees with 10-19 within 2 % at 4k and 16k (+10 % at 1k, the overhead-dominated
column), so the 10-19 pair is not distorted.

Other engines on the Sparks: `sglang` (`qwen38-27b`, `dev-nemotron3-5-lightning`), `vllm-openai:v0.28.0`, TensorFold and
EXL3 images are installed on spark1, but none has a Qwen3.6-35B-A3B NVFP4 setup, and a second vLLM does not fit next to
the serving one. Not measured.

### 2.2 R9700 (HIP, tatooine, one card): Qwen3.6-35B-A3B

One R9700 (PCI 03:00.0, gfx1201, HIP device 0), radiance down for one window (section 5). intarsia is origin/main
789511c2 (HIP defaults, rounds 9 and 10: the round-10 GEMM split is off), model `Qwen3.6-35B-A3B-NVFP4`, `--max-seq 17408`,
serve recipe chunk. llama.cpp is `~/.local/opt/llama.cpp` 0.4.0-dev build 10864 (commit 5d806aa25), Vulkan RADV, `-ngl 999
-fa on -c 20480`, GGUF `MXFP4_MOE` (the closest published quant class to NVFP4).

| engine | quant | prefill 1k* | prefill 4k | prefill 16k | decode 128 / 1k / 4k / 16k | TTFT 1k / 4k / 16k (s) | quality (MMLU, first 9 subjects, n=225) |
|---|---|---|---|---|---|---|---|
| **intarsia HIP** | NVFP4 + fp8 | 5726 | 9082 | 7626 | 159.9 / 159.7 / 155.8 / 142.1 | 0.17 / 0.45 / 2.18 | 87.6 % |
| llama.cpp Vulkan | MXFP4_MOE | 2537 | 2859 | 2681 | 113.1 / 112.5 / 110.2 / 105.0 | 0.39 / 1.43 / 6.21 | 87.6 % (26 subjects: 83.4 %, n=650) |
| llama.cpp Vulkan | UD-Q4_K_XL | not run, window cap | | | *tg128 117.8, pp512 2769 (2026-09-26)* | | |
| llama.cpp HIP (v0.5.0 build) | | not run, libraries unresolved in the dev image | | | *decode 82-85, prefill about 3350-3580 at pp512 (2026-09-26)* | | |
| hipfire MQ4R (`hipfire:ad10b3d`) | MQ4R | not run, no serve recipe found | | | *prefill 3198 / 2143 / 1235 at 512 / 6144 / 18293, decode 204 / 198 / 173 at 830 / 6144 / 18293 (2026-09-26)* | | *20/20 on a saturated 20-prompt check* |
| vLLM-radiance | Qwen3.5-35B-A3B MXFP4, **different model** | *prefill 14.9k / 12.6k / 10.6k at 4k / 16k / 34k* | | | *141 (single stream)* | | |

The MMLU subjects are the first nine that both runs completed; intarsia's run was cut by the 600 s cap in subject 10 and
GSM8K did not run, so this is a smoke check, not a ranking (SE about 2.2 points at n=225). The two engines tie.
The standard error of every quality figure here at n=25 per subject is about +-10 points per subject.

Rank: **#1 of 2** measured live. Counting the 2026-09-26 figures, which came from another session, another harness and a
different build, intarsia is first on prefill and second on decode behind hipfire MQ4R (204 against 160), and the
vLLM-radiance row is on a different model so it is context only. vLLM-ROCm was retired on 2026-10-11 and there is no image on the box.
radiance (native, serving only Qwen3.8-27B) cannot load the 35B, so it is not in this table; its head-to-head is lane M1.

### 2.3 Metal (endor, M5 Max): Qwen3-8B BF16

origin/main f78f60be (Metal code unchanged since 5eabb1a7 apart from tests; PR #12's earlier measurement stands), fleetq
#2960, 10:42-11:24 EDT, one exclusive slot, order intarsia, mlx-lm, llama.cpp, intarsia. mlx-lm 0.31.3 (mlx 0.32.1),
llama.cpp 0.6.0 build 11429 (BF16 GGUF, `-ngl 999 -fa on -c 20480`), intarsia `--max-seq 20480`, binary md5
8d0955e9fe5f3d1426dc833c4f2885a2. Marginal prefill (delta tokens / delta TTFT, localeval column) is in brackets.

| engine | arm | prefill 1k* (marginal) | prefill 4k (marginal) | prefill 16k (marginal) | decode 128 / 1k / 4k / 16k | TTFT 1k / 4k / 16k (s) | quality |
|---|---|---|---|---|---|---|---|
| **intarsia** | 1 | 3989 (4453) | 3917 (3893) | 2543 (2283) | 35.5 / 35.6 / 34.2 / 29.3 | 0.25 / 1.04 / 6.55 | invalid |
| **intarsia** | 2 | 3994 (4474) | 3919 (3898) | 2737 (2493) | 35.0 / 35.9 / 35.3 / 31.4 | 0.25 / 1.04 / 6.09 | not run |
| mlx-lm | | 2233 (4098) | 3079 (3499) | 2339 (2170) | 33.5 / 32.6 / 31.8 / 27.9 | 0.44 / 1.33 / 7.12 | not run |
| llama.cpp | | 2526 (2917) | 1821 (1673) | 1389 (1290) | 34.2 / 30.6 / 30.9 / 29.2 | 0.39 / 2.24 / 12.00 | not run |

intarsia against mlx-lm: prefill 4k +27 % (marginal +11 %), 16k +9 % to +17 % (marginal +5 % to +15 %), decode +6 % / +9 % /
+8 % / +5 % (arm 1) at 128 / 1k / 4k / 16k. Against llama.cpp: prefill 4k 2.2x, 16k 1.8x to 2.0x, decode +4 % / +16 % / +11 % /
0 % (arm 1; arm 2 +2 % / +17 % / +14 % / +8 %). At 16k decode arm 1 ties llama.cpp (29.3 against 29.2). The 1k marginal
figure (4453) is 9 % above mlx-lm (4098) and 53 % above llama.cpp (2917), which matches 2026-10-18.

Noise: GPU idle before the arms was 9, 1, 5 and 5 % (the desktop and a running Zoom contend for the GPU, load average
18-28). Run-to-run, intarsia's 16k prefill moved 8 % between its two arms and llama.cpp's decode spread up to 17 % at 1k.
Read single cells with +-10 %; the 16k deltas against mlx-lm are inside that band for arm 1. No 35B row exists for Metal:
the 35B checkpoint does not load on the Metal backend (no GatedDeltaNet, fp8, nvfp4 or MoE kernels, see
`2026-10-19-p2prime-moeblock.md`).

## 3. Rank summary by metric

| metric | spark (vs vLLM) | R9700 (vs llama.cpp live) | Metal (vs mlx-lm, llama.cpp) |
|---|---|---|---|
| prefill 1k (marginal on Metal) | 2 of 2 | 1 of 2 | 1 of 3 |
| prefill 4k | 2 of 2 (lt: 1 of 2, +0.4 %) | 1 of 2 | 1 of 3 |
| prefill 16k | 2 of 2 (lt+4096: 1 of 2, +1.3 %) | 1 of 2 | 1 of 3 (arm 1 within noise of mlx-lm) |
| decode, short | 1 of 2 | 1 of 2 | 1 of 3 |
| decode 16k | 2 of 2 | 1 of 2 | 1 of 3 (tie in arm 1) |
| quality | not comparable | tie (smoke) | not measured |

## 4. Gaps in this report (not measured, and why)

- **Spark intarsia was not served today.** spark1 reported 23 GB free of 116 GB (29 GB untracked by fleetq) while the 35B
  needs about 36 GB, and the G1 lane held spark1 exclusively with vLLM down from 10:00. Standings use round 6 and round 7
  numbers. The 8B on CUDA has no vLLM 8B to compare to (only the 35B is served) and starting one needs downtime.
- **Spark quality**: intarsia was not served, so no localeval MMLU / GSM8K for it. KL21 against vLLM's golden is the proxy and is
  not an accuracy score. vLLM's own KL against its stored golden is 8.42e-3 on long and 3.16e-3 on short (round 7 section 1),
  so the proxy cannot rank the two engines.
- **Metal quality**: attempt 1 (every MMLU task, `--limit 25`) would have taken about 2 hours and was aborted. Attempt 2
  (two subjects, 20 per subject, GSM8K 20) returned 0.0 for intarsia at about 30 s per call with a median of 470 completion
  tokens, some at the 1024 cap. The same server scored 0.56-0.96 per subject in attempt 1 at about 8 s per call. The
  cause is unexplained; the likely culprit is a mismatch between `--thinking off`, `--tasks` and the answer parsing, not the
  model. The run is kept as `raw/metal/st-intarsia-1-quality-attempt2-INVALID.txt` and is not used. The mlx-lm and
  llama.cpp quality runs were killed (llama.cpp's first subject looked sane at 0.7, n=20).
- **R9700**: llama.cpp UD-Q4_K_XL (window cap: arm 2 consumed the time), llama.cpp HIP (the v0.5.0 HIP build has `libhipblas.so.3`,
  `librocblas.so.5` and `libamdhip64.so.7` unresolved in the intarsia-hip-dev image), hipfire (no known serve recipe; `hipfire serve`
  needs its config dir and model registry set up), vLLM-ROCm (retired 2026-10-11, image gone), radiance on 35B (serves
  Qwen3.8-27B only). The 2026-09-26 figures are from another harness and build, and are shown only as context.
- **R9700 8B**: no GGUF or other-engine model for Qwen3-8B exists on tatooine; not measured.
- **R9700 quality**: MMLU is the first 9 subjects only (600 s cap hit in subject 10); GSM8K did not run on any arm.
- **Single pass, not ABBA**, on tatooine (one arm each). Metal had an ABBA for intarsia only. Sparks used the pairs of the
  source rounds.

## 5. What would move us up (the 3 largest gaps)

1. **CUDA prefill against vLLM at 16k: -15.3 % by default (-8.3 % at 1k, -9.0 % at 4k).** About 59 ms per 2048-token
   chunk, and round 6 attributes about 83 % of it to the fp8 projection GEMMs (bf16 MMA over e4m3-valued bf16 at 36-68 TFLOPS,
   against vLLM's e4m3 CUTLASS at 83-135). Round 7's opt-in e4m3 cuBLASLt path already closes it to -5.0 % at 16k and +0.4 %
   at 4k, and to +1.3 % at 16k with `--prefill-chunk 4096`; it is not the default because of the user decision on numerics
   and a failing `chat_reuse_4k` gate (3/32). Decode at 16k is also behind by 4.2 % (73.1 against 76.3). The user's call
   on round 7 is the largest single lever.
2. **R9700 decode against hipfire MQ4R: 160 against 204 tok/s, a 22 % deficit** (hipfire is +28 % ahead). The hipfire figure
   is second-hand (2026-09-26, no speculation, a different 4-bit format) so it has to be re-measured live before it is
   believed. What the profile says we can do: a decode step is about 8.1 ms with 1.83 ms (22 %) of launch gaps from 578
   launches, and the GEMVs already run at 85-92 % of the 627 GB/s roofline, so the only lever is fewer launches (fusion or a
   graph-level change), not faster kernels. Decode is also 11 % lower at 16k than at 128 (142.1 against 159.9).
3. **Long-context prefill on the R9700 and Metal.** HIP prefill falls from 9082 tok/s at 4k to 7626 at 16k (-16 %); on
   the 2026-09-26 vLLM-radiance row (Qwen3.5-35B, second-hand and a different model) prefill held 12.6k at 16k, which would be
   a 40 % deficit, to be confirmed on the same model. Metal falls from 3917 at 4k to 2543-2737 at 16k (-30 to -35 %), which
   leaves it only 5-15 % ahead of mlx-lm in marginal terms at 16k. For HIP the identified cause is the GDN delta rule (31 % of
   prefill) which needs a non-bitwise chunked formulation; for Metal the 16k path has not been profiled.

Measurement gaps to close first, because they decide the ranking: quality for intarsia on the 35B on Spark; the Metal
quality harness fault; hipfire and llama.cpp UD-Q4_K_XL on the R9700; the 8B on CUDA and HIP.

## 6. Files

- `docs/results/2026-10-21-standings.md` (this file).
- `docs/results/2026-10-21-standings/raw/hip/`: `hip-window.sh` (the single `fleetq run` window script, with preflight and a
  restore trap), `downtime.log`, `preflight.out`, `versions-tatooine.txt`, per-arm localeval text, `/metrics` snapshots.
- `docs/results/2026-10-21-standings/raw/spark/`: today's vLLM localeval `vllm-summary.md` and `vllm-result.json`, and the
  round-6 localeval log with the 10-19 intarsia numbers.
- `docs/results/2026-10-21-standings/raw/metal/`: `standings.sh`, `versions.txt`, `run.log`, per-arm localeval text and
  GPU-idle files, and the invalid quality run.
- Footprint: docs only. localeval result dirs are in `/opt/code/deepeval/results/` (`20261021-102415-spark-guidance-...`,
  `20261021-104317-adhoc-st-intarsia-1`, `20261021-110928-adhoc-st-mlx-lm`, `20261021-112006-adhoc-st-llamacpp`,
  `20261021-112255-adhoc-st-intarsia-2`). localeval's own unanimis capture failed with an ssh mux error on some runs.

## 7. Downtime and fleet

- **tatooine radiance**: one window, fleetq #2948, 10:21:55 to 10:40:42 EDT, 18.8 min including a 15 s restart; radiance
  answered `/v1/models` after 15 s. The window started about 5 min after the previous radiance window ended. Cap for this
  lane was 25 min. The trap restored radiance; no other process of mine was left on the box (the `intarsia-serve` container
  and the llama.cpp process were removed by pid / name). CPU-only jobs: #2938, #2940 (build, 36 s), #2943, #2944, #2946.
- **Sparks**: no downtime taken by this lane. The vLLM localeval (#2937) ran against the live server at 10:24 after
  waiting behind G1 and M2.
- **endor**: #2939 (an aborted first attempt, released) and #2960 (10:42-11:24, exclusive). endor-kev and endor-tensorfold
  were not touched. On the coordinator's order no further endor GPU or CPU work was queued after #2960 was released at 11:24.

## 8. Decisions

None taken. For the user: (a) whether to make round 7's e4m3 path (with chunk 4096) the CUDA default, which is the largest
lever above; (b) whether to approve a live hipfire and llama.cpp Q4_K_XL measurement in a second radiance window (about 20
min), since hipfire is the only engine ahead of intarsia on the R9700; (c) a Metal quality re-run once the endor hold is
lifted. Resume command for (c), after starting the intarsia server inside the same slot: `fleetq run --host endor --exclusive
--mem 40 --expect 15 --label "standings quality diag" -- zsh -c 'cd /opt/code/deepeval && uv run localeval quality --url
http://127.0.0.1:8093/v1 --model intarsia-metal --bench gsm8k --limit 10 --thinking off --label diag'`.

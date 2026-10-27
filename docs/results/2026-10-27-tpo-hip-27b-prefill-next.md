# Lane TPO: overlapping the tp2 all-reduce with the next rows' GEMM, Qwen3.8-27B prefill on the R9700s (HIP) (2026-10-27)

Branch `lane-tpo/hip-27b-prefill-next`, built on origin/main `563d7fa27` (PR #198, bf16 all-reduce partials as the recipe default). Host tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`, W4A8 (`mxfp4_act = fp8`), MTP k2 default in the server arms, the Qwen3.8-27B Quark MXFP4 checkpoint. Every GPU number comes from a fleetq slot that takes radiance down (`--takes-down radiance --quiet gpu,cpu`); radiance was restored and `/v1/models` answered before each release. 6 windows used (w1 profile, w2 microbench, w3 / w4 / w5 first overlap build, w6 gate run), none above 30 minutes. Raw logs: `docs/results/2026-10-27-tpo-hip-27b-prefill-next/raw/<window>/`; driver scripts `scripts/tpo/`.

## Headline

| | result |
|---|---|
| Lever chosen | (a) overlap the TP all-reduce with the next rows' compute (row pieces, all-reduce on a second stream). Not (b), the a8 GEMM rewrite. |
| Result | **draft PR, gate not met.** Bitwise identical to main, but prefill is **+4.5 %** at 4k in the 6 interleaved generate rounds (bar +5 %). The localeval server reading is +6.5 % at 4k, only 4 runs per arm, +3.7 % at 16k, and -0.7 % at 1k. |
| Behind | `INTARSIA_TP_OVERLAP=2` plus `GPU_MAX_HW_QUEUES=4` plus `--prefill-chunk 4096`. Default off, so main's behaviour is unchanged with no knob set. |
| Gap to radiance at 4k (6133 tok/s) | 1.74x before (3527), 1.63x with the overlap (3755). |

## 1. Where a 4k prefill goes on main (step 1)

One 4096-token-class prompt (3858 tokens, `scripts/cmtp/ids-prompts/4k-0.txt`), `generate --max-tokens 1 --tp 2`, rocprofv3 kernel trace (`raw/w1/p.csv.gz`, table by `scripts/h27p/trace_tp2.py`), rank 0 of 2 (rank 1 within 1 %). The profiled run is 1076 ms (3587 tok/s); the plain run is 1057 ms (3650 tok/s).

| class | ms | % of span | kernels |
|---|---|---|---|
| GEMM a8 (MXFP4 weights x fp8 activations) | 553.7 | 51.5 | 2304 |
| all-reduce (`ar2_kernel`, bf16 partials) | 231.9 | 21.6 | 1409 |
| GDN (delta rule, conv, gates) | 112.9 | 10.5 | 480 |
| activation quant (`mx_a8_quant`) | 39.7 | 3.7 | 2304 |
| norm | 32.1 | 3.0 | 579 |
| attention (prefill flash) | 30.6 | 2.8 | 64 |
| elementwise | 20.4 | 1.9 | 244 |
| runtime / other GEMM | 6.1 | 0.6 | |
| idle / gaps | 47.8 | 4.4 | |

Does the all-reduce overlap compute on main? No, it serialises. The comm class is 22 % of the span, and in the trace each `ar2_kernel` sits alone on the stream between the GEMM that produced the partial and the norm that consumes it (one stream, one hardware queue: `GPU_MAX_HW_QUEUES=1` is the launcher default). A run with the collective skipped (`INTARSIA_TP_NO_COMM=1`, wrong output, timing only) takes 903 ms against 1057 ms (rec) and 1258 ms (f32 partials): **the exposed all-reduce costs 154 ms, 14.6 % of the pass; a perfect overlap is worth at most +17 % prefill.** The all-reduces are the attention / GDN output projection and the FFN down projection of each of the 64 layers (1409 launches).

## 2. Lever choice (step 2)

(a) overlap, because its ceiling is measured and large: +17 % at perfect overlap, against (b) the a8 GEMM rewrite, whose best measured variant (lane TPB, prefetch distance 2 on the 27B shapes) was -5.7 % kernel time, about +2.4 % prefill, and whose next named lever (128 B line loads) has no measurement yet. The GEMM is already at 51 % of the fp8 peak on 51.5 % of the span; even a 10 % faster GEMM is +5 % prefill, with a new kernel (admission, mutants) to land. (a) needs no new GPU kernel. Only one lever was done.

## 3. What was built

- Backend trait: `Backend::stream_overlap()`, `stream_record(stream) -> token`, `stream_wait_for(waiter, token)`, `stream_wait(waiter, signal)` (defaults: off / unsupported). HIP: 64 pooled `hipEvent`s, `ik_rt_stream_wait_event` in `kernels/hip/ik_runtime.hip`.
- `Model` (`crates/intarsia-core/src/model.rs`): with `INTARSIA_TP_OVERLAP=K` a single-sequence prefill pass of at least `INTARSIA_TP_OVERLAP_MIN` rows (default 3072) inside `run_layers` runs the row-parallel projections in K row pieces (multiples of 256 rows; a last piece under 256 rows joins the previous one so the small-M kernels never run). After each piece's GEMM the piece's all-reduce is queued on stream 1 behind an event; stream 0 goes on with the next piece. The FFN is pipelined per piece (residual add + norm, gate, up, silu-glu, down, all-reduce) so a piece's norm waits only for its own rows' attention / GDN all-reduce. Every op that `exec_call_one` launches first joins whatever is still in flight, so nothing else can read an unreduced row.
- Eager, unprofiled, unrecorded, tp world 2, stream 0, not under the decode graph, not in the batch / MTP / DFlash paths (`in_layers` guard). `INTARSIA_TP_OVERLAP_FFN=0` keeps the FFN whole.
- Row-wise kernels and a per-element all-reduce: the result is bitwise the whole-pass one (reduction order unchanged), which the gates below confirm.
- Test hook `INTARSIA_TP_OVERLAP_MUTANT=skip_last|scale|offset` (planted bugs, tuning.md rows added) and `INTARSIA_CONTAINER_NAME` (dev.sh container name so a timed-out run can be killed).

## 4. Measurements

Base = `tpo-base`, main `563d7fa27` (default chunk). Branch = `tpo-dev4` with `INTARSIA_TP_OVERLAP=2 INTARSIA_TP_OVERLAP_MIN=3072 GPU_MAX_HW_QUEUES=4 --prefill-chunk 4096` ("overlap").

**Why chunk 4096 and 4 queues.** The overlap only pays when stream 1 can run beside stream 0 (hardware queues) and when the pieces are at least 2048 rows (a smaller piece runs the GEMM on a smaller, slower tile). At the default chunk (2048) a pass is split into pieces of 1024 and loses. Sweep, 4k prompt, 3 reps, `generate` (`raw/w6/sweep.log`, earlier variants in `raw/w3`, `raw/w5`):

| arm | prefill ms | tok/s | 48 greedy tokens md5 |
|---|---|---|---|
| main, chunk 2048 (default) | 1054 to 1059 | 3650 | d150c3504662 |
| main, chunk 4096, 4 queues (`ref4096q4`) | 1098 to 1101 | 3510 | same |
| overlap K=2, FFN whole (`v1c4096`) | 1026 to 1028 | 3760 | same |
| **overlap K=2, FFN pipelined (`v2c4096`)** | **1018 to 1021** | **3785** | same |
| overlap K=3 (pieces of 1536 rows) | 1043 to 1047 | 3690 | same |

Chunk 4096 without the overlap is slower than main (-3.8 %); the overlap on top of it is +7.8 %. Earlier windows (w3, dev1, attention / GDN outputs only): queues 1 gave 1092 ms (slower than base 1057: with one hardware queue the all-reduce waits in line behind the GEMM), queues 2 1100 ms, queues 4 1072 ms; K=4 pieces (2 queues) 1132 ms.

**Interleaved rounds, 4k prompt, `generate --max-tokens 1`** (`raw/w6/w6-r`, 6 rounds, order reversed every second round):

| | prefill ms per round | median tok/s |
|---|---|---|
| main | 1080, 1074, 1075, 1074, 1092, 1074 | 3591 |
| overlap | 1024, 1025, 1027, 1032, 1030, 1035 | 3751 |
| ratio | | **x1.0445** (bar x1.05) |

**localeval speed against the server** (`localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3`, MTP on, max-batch 1, 8 runs ABBA x2 = 4 per arm, `raw/w6d/speedtab.txt`; each server start confirmed `GPU_MAX_HW_QUEUES=4` for the overlap arm via `docker exec printenv`):

| prompt | prefill main | prefill overlap | ratio | decode main | decode overlap | ratio |
|---|---|---|---|---|---|---|
| 1k | 3280 | 3256 | x0.993 | 92.2 | 90.1 | x0.977 |
| 4k | 3527 | 3755 | **x1.065** | 83.5 | 85.5 | x1.024 |
| 16k | 3418 | 3545 | x1.037 | 74.8 | 75.4 | x1.008 |

Radiance reference: 6133 tok/s at 4k, so the gap goes from **1.74x to 1.63x** (localeval) or 1.71x to 1.63x (generate rounds). The two 4k readings differ in prompt (4084 vs 3858 tokens, so pieces of 2048 + 2036 rather than 2048 + 1810) and in path (server vs `generate`); neither alone is the 6-round interleaved localeval median the gate asks for, and the stricter reading misses the bar.

Decode: the overlap never engages for decode (single-token passes, graph replay), the decode kernels are unchanged, but `GPU_MAX_HW_QUEUES=4` is a process-wide setting. 1k decode reads x0.977 (MTP on, run spreads 8 %), 4k x1.024, 16k x1.008; with that spread I read it as noise, but it is unproven and the 1k number is below 1.0. The launchers pin `GPU_MAX_HW_QUEUES=1` as a known-bad-default fix (docs/ARCHITECTURE.md:182, h1 note on comm-test deadlocks with 1 queue); this lane did not look into why 4 queues were ruled out, and that has to be settled before the overlap is recommended.

Why the overlap recovers only a quarter of the 154 ms: in the rocprof trace of an overlapped run (`raw/w4/ov2q4.csv.gz`, `scripts/tpo/ovl.py`) 139 ms of the 260 ms of all-reduce time runs under other kernels, so about half is hidden, but the net gain is far below the hidden time (the likely cause, not measured: the all-reduce kernels take CUs and their P2P stores contend with the GEMM), and the first piece's GEMM and the last piece's all-reduce cannot overlap with anything (2 pieces: at best 1/2 of the comm hides). More pieces lose more to the smaller tile than they hide. A stream-overlap microbenchmark with spin kernels (`raw/w3/overlap.txt`) shows the hardware does not run a 16-block kernel beside a full-width GEMM freely: B on a second stream was slower than B in line there at every queue count, which is why the engine-level result was not predictable from it.

## 5. Gates

| gate | result |
|---|---|
| prefill >= +5 % at 4k, median of >= 6 interleaved rounds | **not met**: x1.0445 (6 generate rounds); localeval x1.065 over 4 runs per arm (fewer than 6) |
| decode not slower | decode kernels unchanged; localeval x0.977 / x1.024 / x1.008 (1k / 4k / 16k, noisy); not proven for the queue setting |
| greedy ids identical to main (bitwise) | met: 48-token md5 equal in every arm (`raw/w6/sweep.log`); 18/18 gate prompts x 256 forced tokens equal, main vs overlap (`raw/w6d/texts-9`, `texts-10`) |
| kernel admission, >= 3 planted mutants | no new GPU kernel (a runtime shim `ik_rt_stream_wait_event` only), so admission is N/A. Overlap mutants, 48 tokens md5 vs the candidate d150c3504662: `skip_last` 2ba540f3009f, `scale` a2b8dd259525, `offset` 5ac06c5f10f6, all three caught |
| batched == solo, MTP spec-on == spec-off | met: texts 18/18 equal for overlap spec-on and spec-off (`texts-9`, `texts-11`); `mixids` (a stream decoding while a second prompt prefills, max-batch 16 vs 1, spec off): A and B texts equal for the three B prompts (`steps.log` items 12, 13) |
| `plan show` byte-identical, 35B and 8B on HIP | met (1346 B and 3184 B, `raw/w6/w6-g`) |
| Mac `cargo test --workspace` | met: 195 test-binary results ok, 0 failed (final commit) |
| CI green | pending (draft PR) |
| behind an `INTARSIA_*` knob | met: `INTARSIA_TP_OVERLAP` (default off) |

No MMLU / KL runs: the change is scheduling only and bitwise, so the accuracy gate of a numerics change does not apply.

## 6. Outcome and what is left

Draft PR, not merged. The prefill gain is real (+4.5 % to +6.5 % at 4k, +3.7 % at 16k) and bitwise, but under the +5 % bar on the strict reading, and it depends on a process-wide `GPU_MAX_HW_QUEUES=4` whose launcher default of 1 is not understood here.

Left:
- Settle why the launchers pin `GPU_MAX_HW_QUEUES=1`, and measure decode / the 35B / radiance-adjacent servers with 4. If it is only a historic comm-test deadlock (h1 note), the knob can ship with the queue count set by the recipe.
- A 3-stream variant (compute, all-reduce, the next layer's non-dependent GDN work) or comm kernels on fewer CUs (a CU-masked stream: `hipExtStreamCreateWithCUMask`, microbench `raw/w3/overlap.txt` shows no gain on spin kernels) could hide the other half of the comm time.
- The overlap needs a pass of at least 3072 rows (pieces of at least 2048): at 1k nothing overlaps, so 1k gets nothing.
- Lever (b), the a8 GEMM line-load rewrite, is untouched.
- No window left: 6 of 6 used.

Commits on the branch carry `Co-Authored-By: Claude Opus 5.5` (first five) and `Claude Sonnet 5.5` (later ones): the attribution line given to the session changed mid-lane.

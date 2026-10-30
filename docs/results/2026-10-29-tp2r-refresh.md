# tp=1 vs tp=2 refresh on main 694b519 (lane TP2R, 2026-10-29)

Qwen3.6-35B-A3B NVFP4, CUDA, tp=2 across the two Sparks (rank 0 on spark1 with `--tp-remote`, rank 1 on spark2 with
`intarsia tp-worker`, NCCL 2.30.7 over ConnectX-7) against tp=1 on spark1. Re-measures
[2026-10-21-tp2-cuda-graphs.md](2026-10-21-tp2-cuda-graphs.md) on current origin/main `694b519cf`, and this time
reports prefill next to decode, with row-local fusion (`INTARSIA_TP_FUSION=1`) set explicitly. Branch
`lane-tp2r/refresh`; raw logs `2026-10-29-tp2r-refresh/raw/`; scripts `scripts/tp2r/`. No engine code changed.

## Headline

tp=2 still beats tp=1 on prefill (+10 to +12 %) and on MTP-off decode (+8 to +18 %), but both margins are smaller
than on 10-21 because tp=1 got faster. **tp=1 with its default MTP beats tp=2 (which cannot run MTP) on decode at 1k and
4k, and ties it at 16k.**

Build `head=694b519 code-tree=abec75fa23b5 dirty=no` on both Sparks. Interleaved A/B ×3 (order reversed in round 2),
`intarsia bench speed --at-most`, gen 128, 2 reps + 1 warmup per size, sizes 1024 / 4096 / 16384. Cells are the median
of the three round medians, in tok/s. All five arms are `--spec off` except tp1m.

| arm | prefill 1k\* | 4k | 16k | decode 1k | 4k | 16k |
|---|---|---|---|---|---|---|
| tp1g: tp=1, graphs, `--spec off` | 5476 | 7142 | 6857 | 87.1 | 84.6 | 77.2 |
| **tp2g: tp=2, graphs, `INTARSIA_TP_FUSION=0`** | **6039** | **7880** | **7574** | **94.0** | **92.3** | **87.0** |
| **tp2gf: tp=2, graphs, `INTARSIA_TP_FUSION=1`** | **6047** | **7941** | **7653** | **98.0** | **96.5** | **90.9** |
| tp1m: tp=1, graphs, default spec (MTP k2 on); rounds 1 and 3 only | 5452 | 7037 | 6823 | 103.7 | 109.5 | 92.4 |
| tp2g vs tp1g | +10.3 % | +10.3 % | +10.5 % | +7.9 % | +9.2 % | +12.7 % |
| tp2gf vs tp1g | +10.4 % | +11.2 % | +11.6 % | +12.6 % | +14.1 % | +17.7 % |
| tp2gf vs tp2g | +0.1 % | +0.8 % | +1.0 % | +4.3 % | +4.5 % | +4.4 % |
| tp2gf vs tp1m (tp=2 MTP-off vs tp=1 MTP-on) | +10.9 % | +12.8 % | +12.2 % | -5.4 % | -11.8 % | -1.6 % |
| tp1m vs tp1g | -0.4 % | -1.5 % | -0.5 % | +19.1 % | +29.4 % | +19.7 % |

\* 1k prefill includes the first-token overhead (TTFT-based), as in 10-21; the bench also notes that each size includes
2 to 4 profiled eager steps (per-op profiling synchronises around every op), the same in every arm.

- **Prefill.** tp=2 is +10 to +12 % over tp=1 at every size. Fusion adds nothing measurable to prefill (+0.1 to +1.0 %,
  inside the spread).
- **Decode.** Fusion adds +4.3 to +4.5 % over unfused tp=2, as on 10-21 (+3.8 to +5.3 %). tp=2 + fusion is +12.6 / +14.1 /
  +17.7 % over tp=1 MTP-off.
- **MTP changes the decode ranking.** The shipped tp=1 default on this model is now `--spec mtp --spec-k 2`
  (`[tuning.cuda] mtp_default_k`), which tp=2 cannot use (below). Against it, tp=2 + fusion decodes 5 % (1k), 12 %
  (4k) slower and 1.6 % slower (16k, inside the MTP spread). The tp1m cells are two rounds, and the bench prompts are
  synthetic filler, which inflates MTP acceptance (the 10-28 scoreboard notes the same); do not compare them one to one
  with localeval's 114.0 at 4k.
- **tp1g is `--spec off` on purpose.** On 10-21 the tp=1 default had no MTP, so "tp1g (shipped)" meant MTP off. Today
  the shipped default is MTP on, so the like-for-like arm needs `--spec off` explicitly (rank 0 of every tp=2 arm also
  gets `--spec off`); the shipped default is the separate tp1m row.

### Against the 10-21 window (same method, `871f912` then, `694b519` now)

| arm | prefill 1k | 4k | 16k | decode 1k | 4k | 16k |
|---|---|---|---|---|---|---|
| tp1g, 10-21 | 4510 | 5858 | 5505 | 81.7 | 79.3 | 72.9 |
| tp1g, now | 5476 | 7142 | 6857 | 87.1 | 84.6 | 77.2 |
| change | +21.4 % | +21.9 % | +24.6 % | +6.6 % | +6.7 % | +5.9 % |
| tp2g, 10-21 | 5272 | 7132 | 6857 | 91.6 | 90.0 | 85.0 |
| tp2g, now | 6039 | 7880 | 7574 | 94.0 | 92.3 | 87.0 |
| change | +14.5 % | +10.5 % | +10.5 % | +2.6 % | +2.6 % | +2.4 % |
| tp2gf, 10-21 | 5415 | 7130 | 6907 | 96.5 | 94.5 | 88.2 |
| tp2gf, now | 6047 | 7941 | 7653 | 98.0 | 96.5 | 90.9 |
| change | +11.7 % | +11.4 % | +10.8 % | +1.6 % | +2.1 % | +3.1 % |

tp=1 gained more since 10-21 (prefill +21 to +25 %, decode +6 to +7 %) than tp=2 (prefill +10 to +15 %, decode +2 to
+3 %), so the tp=2 lead over tp=1 shrank: prefill +17/+22/+25 % to +10/+10/+11 % (tp2g), decode (tp2gf) +18/+19/+21 %
to +13/+14/+18 %. This lane did not bisect which PRs since 10-21 moved the numbers.

### Run-to-run spread (max over min of the three round medians)

| arm | prefill 1k | 4k | 16k | decode 1k | 4k | 16k |
|---|---|---|---|---|---|---|
| tp1g | 0.6 % | 1.5 % | 0.2 % | 0.2 % | 0.5 % | 0.8 % |
| tp2g | 1.1 % | 0.5 % | 1.2 % | 6.0 % | 5.5 % | 3.6 % |
| tp2gf | 5.9 % | 1.9 % | 0.5 % | 1.0 % | 2.5 % | 1.4 % |
| tp1m (2 rounds) | 0.3 % | 0.0 % | 1.0 % | 1.2 % | 3.7 % | 1.8 % |

- Per round, tp2g decode 1k / 4k / 16k: 92.0, 94.0, 97.5 / 90.4, 92.3, 95.3 / 85.5, 87.0, 88.6 (rounds 1, 2, 3). It rises
  each round; tp2gf decode is flat (98.6, 98.0, 97.7 at 1k). Round 3 of tp2g (97.5 at 1k) overlaps tp2gf, so the
  tp2gf-over-tp2g gain is +4.3 % on medians but not cleanly separated at every size. 10-21 saw 2 to 3 % cross-host
  run-to-run on tp=2.
- Per round tp2gf prefill at 1k: 5770, 6047, 6112 (round 1 is the low outlier).
- tp=1 is tight (under 1.5 % on everything except tp1m decode 4k, 3.7 %).
- Every tp=2 graphs arm passed `--require-graphs` and the rank-1 replay check (rank 1 replays 788 to 1042 per
  worker log, six `REPLAY-CHECK ok` lines in `steps.log`).

## Streams (one round)

`--max-batch 16 --spec off`, `--max-seq 8192 --mem-budget 40`, `bench speed --sizes 1024 --concurrency 1,4,16
--conc-size 1024`, gen 128 (`raw/tp2r-w1/tp1s-r1/`).

| arm | streams | per-stream tok/s | aggregate tok/s | TTFT s |
|---|---|---|---|---|
| tp1s: tp=1 | 1 | 87.4 | 87.4 | 0.18 |
| tp1s: tp=1 | 4 | 47.5 | 169.0 | 0.50 |
| tp1s: tp=1 | 16 | 21.5 | 278.0 | 1.53 |
| tp2s: tp=2 + `--max-batch 16` | | refused at load | | |

**tp=2 cannot batch across the two Sparks.** `serve --tp 2 --tp-remote ... --max-batch 16` stops at model load, so
there are no tp=2 stream rows:

```
Error: loading <35B snapshot> with tp=2 (rank 1 at 10.100.152.3:29555)
Caused by:
    unsupported: batched decode needs a single-rank model or rank 0 of an in-process tensor-parallel group (no remote rank)
```

(`raw/tp2r-w1/tp2s-r1/serve.log`; the guard is `Model::enable_batch`, `crates/intarsia-core/src/model/batch.rs`.
Separately, that function allows batching under `--tp` on the hip backend only, so CUDA tp=2 would be refused even
in-process.) The single-stream row at 1k (87.4) matches tp1g's 87.1 from the A/B.

## MTP under tp=2

Refused, as the code says, but with a different message than the two the lane brief quoted. Rank 0 with `--spec mtp`
on `--tp 2 --tp-remote` (workers also started with `--spec mtp`) fails at load:

```
Error: loading <35B snapshot> with tp=2 (rank 1 at 10.100.152.3:29555)
Caused by:
    unsupported: --spec mtp with a tensor-parallel rank in another process: the MTP drafter runs on in-process ranks only (--tp N on one host)
```

(`raw/tp2r-w1/tp2mtp-r1/serve.log`.) Not the `mtp.rs` "MoE head not sharded" line or the `model.rs` "speculative verify"
line: the cross-process check fires first. Nothing in the 35B tp=2 path can use MTP across hosts, so the best tp=2
decode on the Sparks is the MTP-off row above, and the tp=1 MTP row (tp1m) is the comparison that matters for decode.

## Method and provenance

- fleetq jobs: build **#7255** (spark1, `scripts/tp2r/build.sh base`, exit 0, 3 min 31 s of compile, base =
  `694b519cf` via `git archive`); speed window **#7260** (both Sparks, exclusive, `MODE=window OUT=tp2r-w1`, exit 0).
  #7256 was the first window attempt: its pre-flight (binary stamp check, before any takedown) aborted because
  `--version` does not print the stamp; the check now reads the stamp from the binary and #7260 was the re-queue. No
  Spark was touched by #7256.
- The window ran 23:04:48 to 23:16:00; the 55 min work budget was not needed (11 min).
- **vLLM restore:** both containers (`missioncntrl-guidance` on spark1 and spark2) were stopped at 23:04:48 and restarted
  by the script's exit trap; spark1 answered `/v1/models` at 23:19:13 (down 865 s) and spark2 at 23:19:43 (down 895 s).
  `raw/downtime.log`.
- Arms (per round, rank-0 env / worker env identical): tp1g `serve --spec off`; tp2g `INTARSIA_TP_FUSION=0` on both
  ranks `--tp 2 --tp-remote --spec off`; tp2gf the same with `INTARSIA_TP_FUSION=1` (explicit, though it is the CUDA
  default); tp1m `serve` with no `--spec` (log: `--spec mtp --spec-k 2: the default of recipe qwen3_5_moe on cuda at tp=1`).
  Order: round 1 tp1g, tp2g, tp2gf, tp1m; round 2 reversed (tp2gf, tp2g, tp1g); round 3 as round 1.
- `--max-seq 20480 --mem-budget 45` on all A/B arms, prefill chunk 2048 (recipe default), fresh `serve` per arm.
- Not covered: localeval (the A/B uses `intarsia bench speed` only), accuracy (no code changed), vLLM on the same day.
- Commit trailers in this lane say Sonnet 5.5, matching the session's model, not the Opus 5.5 named in the lane brief.

## Raw data

- `2026-10-29-tp2r-refresh/raw/tp2r-w1/steps.log` (run log), `ab-table.txt` (`scripts/tp2r/ab_table.py` output).
- `raw/tp2r-w1/<arm>-r<round>/` : `tp2r-<arm>-r<round>.json` (bench result), `bench.log`, `serve.log`; rank 1 logs in
  `raw/tp2r-w1/spark2/worker-<arm>-r<round>.log`.
- `raw/build.log`, `raw/downtime.log`, `raw/window.stdout`.
- Scripts: `scripts/tp2r/xg.sh` (window), `build.sh`, `ab_table.py`. Derived from `scripts/tp2-graphs/`.

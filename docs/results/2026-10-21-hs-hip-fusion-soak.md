# HIP tp=2 row-local fusion: soak, then default on (lane HS, 2026-10-21)

Qwen3.6-35B-A3B NVFP4, `intarsia serve --backend hip --tp 2 --devices 0,1` on tatooine's two R9700s (one process, in-process
ranks, HIP P2P all-reduce, decode graphs). Fusion is `INTARSIA_TP_FUSION=1`, added to HIP in PR #59 (H1,
[2026-10-21-h1-tp2-hip.md](2026-10-21-h1-tp2-hip.md), +14.7 / +17.6 / +22.2 % decode over tp=1). The user decided: soak it first,
and make it the HIP tp=2 default if the soak passes. **It passed both windows, so this branch makes it the default.**

Branch `lane-hs/hip-fusion-default`, based on `lane-h1/tp2-hip-graphs` (PR #59, still open at `5c0b9f57` when this branch was
cut, so the PR's base branch is `lane-h1/tp2-hip-graphs`). Raw logs: `2026-10-21-hs-hip-fusion-soak/raw/{soak,flip}/`.
Scripts: `scripts/hs-hip-fusion/` (copies of H1's `lib.sh`, `window.sh`, `build.sh`, with this lane's paths) and `tools/hs_soak.py`.

## 1. Headline

| criterion (user's) | window 1: PR #59 head, `INTARSIA_TP_FUSION=1` | window 2: flip branch, default path (no env var) |
|---|---|---|
| no crash, hang or NaN | **PASS**: 0 errors, 0 hang candidates, 0 poll violations, 0 suspect serve-log lines, no amdgpu kernel lines, container never restarted | **PASS**: same |
| greedy ids at start and end identical to H1's fusion-on reference | **PASS**: CLI generate before and after the soak (short, 4k) identical to H1's `gen-head-35b-tp2gf-*.ids`; in-server start and end ids identical to each other and to the reference | **PASS**: in-server start and end ids identical to each other and to the reference |
| decode tok/s drift start to end <= 5 % | **PASS**: short 185.6 to 185.5 (-0.02 %), 4k 180.2 to 179.6 (-0.34 %) | **PASS**: short 186.3 to 186.3 (-0.03 %), 4k 179.2 to 178.8 (-0.22 %) |
| memory stable | **PASS**: VRAM spread 32 MiB per GPU after warmup, RSS 532 to 548 MiB (+3 %) | **PASS**: VRAM spread 32 MiB, RSS 536 to 555 MiB (+3.6 %) |

Load, mixed, at concurrency 1, 4 and 8, short and long prompts, streamed and not, with client cancels:

| | window 1 | window 2 |
|---|---|---|
| lane wall clock (sentinels, reference, load, drain, reference, sentinels) | 805 s (13.4 min) | 901 s (15.0 min) |
| load phase | 747 s | 843 s |
| requests served (server counter) | 728 (555 length, 135 stop, 38 cancelled) | 843 (632 length, 162 stop, 49 cancelled) |
| closed-loop replays (`cbidentity`, 28-case sample each) | 14: B=1 x4, B=4 x5, B=8 x5; stream on / off / mixed; 392 cases | 17: B=1 x5, B=4 x6, B=8 x6; 476 cases |
| open-loop (`cbsoak`, 0.3 req/s, 128 to 4k-token prompts, 30 % cancels planned) | 251 sent, 232 ok, 19 cancelled, 0 errors | 279 sent, 253 ok, 26 cancelled, 0 errors |
| replays identical to the solo reference (token ids) | yes (rc 0) | yes (rc 0) |
| second solo reference identical to the first | yes (rc 0) | yes (rc 0) |
| cbsoak TTFT p50 / p99, ITL p50 / p99 | 405 / 7386 ms, 5.48 / 5.64 ms | 421 / 8301 ms, 5.47 / 5.68 ms |
| decode gauge median under load | 181.0 tok/s | 181.7 tok/s |

Together 26.5 min of mixed load and 1571 requests. **Deviation from the brief:** it asked for 15 min of load in one window on the
#59 head with `=1`. That window (1) reached 12.4 min of load (13.4 min with the sentinels and references), because I sized the stage
budget to leave room for the pre and post generates. Window 2 ran 14.0 min of load (15.0 min lane) on the flip binary's default
path, the same kernels and ids. A strict single-window 15-min soak on the #59 head would need about 18 min of downtime; it is
not run unless the user wants it.

## 2. What was run

### Window 1 (fleetq #3301, radiance down 19:08:44 to 19:22:44, 14 min 0 s)

- Binary: H1's `~/intarsiaml-h1-head/h1-intarsia`, built from `11563b5`. PR head `5c0b9f57` differs from it only in docs and
  scripts (`git diff --stat 11563b5 5c0b9f57 -- crates kernels Cargo.toml Cargo.lock recipes` is empty).
- Stages (`scripts/hs-hip-fusion/stages-soak.sh`):
  1. CLI `generate` 256 greedy ids, 35B tp=2 short and 4k, `INTARSIA_TP_FUSION=1`: identical to H1's reference (2/2).
  2. `intarsia serve` in container `hs-serve`, both GPUs, `INTARSIA_TP_FUSION=1`, default `--max-batch 1`, `--max-seq 8192`,
     `--mem-budget 29`; `tools/hs_soak.py`.
  3. After the stop: the same two generates again: identical to the reference (2/2).
- Fusion reached the serve process: the start sentinel read 182 to 186 tok/s (H1's unfused graphs number is 140). The driver
  aborts below 160.

### Window 2 (fleetq #3306, radiance down 19:25:42 to 19:41:54, 16 min 12 s)

- Binary: `~/intarsiaml-hs-flip/hs-intarsia`, built from `a733652` (the flip; stamp `code-tree=a173b61d1c81`; later commits on
  the branch touch only scripts and docs). Build and `cargo test --workspace --exclude intarsia-backend-hip --no-fail-fast`
  ran in CPU slots (#3303, #3305): 522 passed, 0 failed, including the new unit test. The HIP GPU tests did not run, as in
  H1's `build.sh`; none ran while radiance served.
- Identity checks, 256 greedy ids (`stages-flip.sh`):

| check | result |
|---|---|
| 35B tp=2 short and 4k: default vs explicit `INTARSIA_TP_FUSION=1` | identical (2/2) |
| 35B tp=2 short and 4k: default vs H1's fusion-on reference | identical (2/2) |
| 35B tp=2 short and 4k: `INTARSIA_TP_FUSION=0` vs H1's graphs-only reference | identical (2/2): the opt-out restores the old path |
| 8B tp=2 short: default vs H1's fusion-on reference; `=0` vs graphs-only reference | identical (2/2) |
| 35B tp=1 `plan show --json` vs H1's window-7 head | identical |
| 35B tp=1 short ids vs H1's window-7 head | identical |
| decode tok/s, 35B tp=2 short / 4k | default 184.5 / 179.2, `=1` 183.2 / 178.7, `=0` 139.9 / 137.8 |
| decode tok/s, 8B tp=2 short | default 63.4, `=0` 60.4 (+5 %) |

  The default run is fused: its speed matches `=1`, not `=0`.
- Then the serve soak on the default path: no `INTARSIA_TP_FUSION` in the container environment. Start sentinel 184 to 187
  tok/s.

### Driver (`tools/hs_soak.py`, on `tools/p2prime_soak.py`)

- Start and end sentinels: flush the KV prefix, then a greedy 256-token chat request, thinking off, token ids returned
  (3 short, 2 at 4420 prompt tokens). Decode tok/s is the server's `last_decode_tokens_per_second` gauge. The criteria compare
  ids start vs end and against H1's reference, and the median tok/s start vs end.
- Between them: a solo reference, the replays and `cbsoak` together until the deadline, a drain with the `/metrics` invariants
  (running = waiting = 0, requests_total = sum finished, nothing rejected, engine not failed), a second solo reference, then
  `cbidentity compare` of every replay and of the second reference against the first.
- Memory: VRAM used of both cards (`/sys/class/drm/card{0,1}/device/mem_info_vram_used`, 32 GB cards) and the serve RSS every
  15 s. Pass is a post-warmup (150 s) VRAM spread of at most 256 MiB per GPU and an RSS rise of at most 10 %.
- The driver, the soak and the stage budget were sized for one window of at most 20 min; the stage has a hard `timeout`.

## 3. The change

`crates/intarsia-cli/src/common.rs`: `tp_fusion(env, backend, tp)` replaces the inline env test.

| `INTARSIA_TP_FUSION` | result |
|---|---|
| unset | on for `hip` with `--tp 2`; off otherwise |
| `0` | off |
| anything else | on (as before; a backend without `fusion_under_tp` refuses at load) |

- The library default (`LoadOptions::tp_fusion`) stays off. Only the CLI default changes.
- tp=1 never reaches the fused path (`tp_fusion` acts only at world > 1). Metal, CPU and CUDA are unchanged. HIP at tp > 2 stays
  opt-in: only the 2-way shard shapes were soaked and gated.
- Unit test `common::tests::tp_fusion_default_on_for_hip_tp2_only_and_env_overrides`.

## 4. Decisions and open points

1. **Merge with G1c (`feat/tp2-fusion-default`): three files conflict, not one.** It rewrites the same `tp_fusion` line in
   `crates/intarsia-cli/src/common.rs` into a 2-argument `tp_fusion(env, backend)` with `None => backend == "cuda"`, and rewrites the
   `tp_fusion` doc comments in `crates/intarsia-core/src/backend.rs` and `crates/intarsia-core/src/model.rs` that this branch also
   edits. Its unit test calls the 2-argument function and asserts `tp_fusion(None, "cuda")` is true. The resolved tree needs ONE
   3-argument function with `None => backend == "cuda" || (backend == "hip" && tp == 2)`, both unit tests (G1c's with the extra
   argument, this one), and doc comments that name both backends. A clean-looking textual resolve that keeps one side drops the
   other backend's default. G1c's comment that HIP refuses an explicit request is stale once #59 is in (H1 noted the same).
2. **This PR is stacked on #59** (base `lane-h1/tp2-hip-graphs`). If #59 merges first, retarget to main.
3. **Not covered.** The HIP serve cannot batch (`--max-batch 1` only), so "concurrency 8" is eight clients queuing for one
   decoder, not batched decode. One model (35B; 8B for ids only). HIP tp=4 does not exist on this box.
4. **Still open from H1:** the change-vs-base reference-noise gate for HIP tp=2 (fails at 1.66x short, 1.31x long for tp=2 vs tp=1,
   equally for a correct tp=1 kernel variant). Fusion is bitwise to graphs-only on every gate, so the default flip does not move
   that number. Unchanged here.

## 5. Downtime (radiance, `raw/downtime.log`)

| window | fleetq | down | work |
|---|---|---|---|
| soak | #3301 | 19:08:44 to 19:22:44 (14 min 0 s) | PR head: pre ids, serve soak, post ids |
| flip | #3306 | 19:25:42 to 19:41:54 (16 min 12 s) | flip branch: default-path ids, tp=1 unchanged, default-path serve soak |
| **total** | | **30 min 12 s** | lane cap 60 min before 00:00; each window under 20 min; radiance answered 10 s after each start |

Off-window CPU slots (no downtime): #3303, #3305 (build and tests of the flip tree).

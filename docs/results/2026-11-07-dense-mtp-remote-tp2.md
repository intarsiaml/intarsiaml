# Dense 27B MTP at tp2 over remote ranks (lane d2r)

Coordinator go-ahead 2026-11-07 03:30 EST (user: 'keep building overnight'): 27B dense MTP at tp2 over remote ranks, opt-in, vLLM stays up, no default flips; merge on passing gates; wrap up by 06:30.

## Status

**Done, opt-in; PR opened, merges on rule (a) + gates + 6 SUCCESS checks** (measured 2026-11-07 03:54 to 04:07 EST on spark1 + spark2 with both vLLMs up). Rule (a) PASS, (a2) PASS,
rule (b) PASS (recommendation: use `--spec mtp --spec-k 2` for the 27B at tp2 across the Sparks; no default flips, the default is the user's call).

## Why it was refused

`load_engine_group` refused `--spec mtp` for a `Group::Remote` whenever `dims.moe.is_none()`, and `serve_remote_rank` (the `tp-worker`)
refused it for a dense model. The reason (git blame: the dense refusal predates mt2's `6ba33d00f`, which narrowed it to dense heads; comments in `load.rs`): the dense head of lane dfh is *sharded* like the
trunk (q/k/v/gate/up by rows, o/down by columns) and every rank runs it in lockstep with rank 0 (`Model::mtp_draft_tp`, threads and channels,
the head's own all-reduces), which a rank in another process (`tp_remote`, commands over TCP) does not take part in. Nothing in the head itself
needs in-process ranks; lane mt2 already made the *whole-on-rank-0* head work over remote ranks (rank 0 drafts alone, no collective, the
remote rank only runs verify passes and rollbacks, wire protocol v3).

## Change

* `LoadOptions::mtp_whole` (new, default off): a dense MTP head runs whole on rank 0 under tensor parallelism, exactly as an MoE head always does
  (`mtp_whole = opts.mtp && world > 1 && (moe || opts.mtp_whole)` in `Model::load_shard`): unsharded against the global dims, passes capped at
  `chunk / world` rows, no head on the other ranks. The engine loader sets it for `Group::Remote`; `serve_remote_rank` sets it, so rank 0 and the
  worker agree (the handshake fingerprint compares the load options). In-process groups keep the lane-dfh lockstep head unless a caller sets the flag.
* The dense FFN joins the whole-head switch: `ffn_act` reads `adims()` (global intermediate) and `ffn_block` runs a plain `down` projection without
  the reduce inside a whole-head pass (`attn_out_proj` already did, lane mt2).
* `INTARSIA_MTP_QUANT` (lane c27) now also applies to a whole dense head (rank 0 holds the whole head, so the format applies as at tp1).
* The two refusals (`load_engine_group`, `serve_remote_rank`) are removed. Opt-in only: `--spec mtp` has to be asked for; `mtp_head_usable` and the
  recipe defaults are untouched, and the default path (no `--spec`, remote group) loads exactly what it did.

## Pre-registered rules (written 2026-11-07 03:36 EST, before any measurement; not edited after data)

Setup: Qwen3.8-27B-Quark-AWQ-MXFP4 (W4A16), CUDA, recipe defaults (fp8 lm_head, 65536-row draft head). tp2 = rank 0 on spark1 (`generate --tp-remote`) +
rank 1 on spark2 (`tp-worker`), NCCL over CX-7, `--max-seq 8192` (20480 at 16k) `--mem-budget 24` per rank; tp1 on spark1 `--mem-budget 45`.
vLLM stays up on both Sparks. One fresh process per visit, `generate --chat --no-think --temperature 0 --ignore-eos --show-ids`, 192 tokens for speed,
corpus prompts of lane c27 (`head -c 3300 | 14035 | 54000` of `tools/prompts/corpus_docs.txt`). Binary `dev` = this branch.

* **(a) ids, gates the merge.** 256 greedy ids (`scripts/cmx/ids-prompts` short-0 and 4k-1) from dev tp2 `--spec mtp --spec-k 2` must be identical to dev
  tp2 `--spec off`, both prompts, every token. Also recorded: acceptance and tokens per verify pass.
* **(a2) head diagnostic, recorded, added 03:48 EST before any measurement (advisor review).** The ids of (a) cannot see a bad head (every emitted
  token is a verify row's argmax), so the drafts are judged by acceptance: the same two prompts also run tp1 MTP k2 on dev. tp2 MTP acceptance
  should be within 3 points of tp1 MTP (lane c27 #8988 reference on the same binary family: 74.5 % short-0, 70.3 % 4k-1). If it is far below, the
  whole head is broken at tp2 on CUDA (TP fusion, the dense FFN under the whole-head switch) and the branch does not merge even with ids equal.
* **(b) speed, decides the recommendation only (stays opt-in either way).** Decode tok/s at 4k, mean of an ABAB (round 1 tp1m, tp2off, tp2m; round 2
  reversed): tp2 MTP k2 must be **>= 1.30x tp2 `--spec off` and >= 1.20x tp1 MTP k2**. Both hold: recommend; else not. 1k and 16k are reported, not gating.
  Reference (lane c27, #336): tp2 off 23.67 / 23.27 tok/s at 1k / 4k, tp1 MTP k2 31.0 / 29.1 / 25.1 at 1k / 4k / 16k.
* Not measured: batching (a remote tp2 group is single-sequence), sampled requests (single-rank speculation by design).

## CPU tests (`crates/intarsia-models/tests/tp_mtp.rs`, the tiny dense hybrid with a dense MTP head; endor, fleetq #9016 on the merged HEAD)

| test | what | result |
|---|---|---|
| `dense_mtp_head_whole_on_rank0_is_bitwise_the_tp1_head` | head logits from the same hidden rows, t = 1..4, f32 and bf16: tp2 whole on rank 0 vs tp1; a pass over the `chunk / world` row cap is refused | bitwise equal |
| `dense_mtp_whole_head_tp2_drafts_equal_tp1_drafts` | prefill + catch-up + chained drafts (k 3), 3 prompts, draft vocab 0 and 40: tp1 vs in-process whole (replicated and vocab-parallel lm_head) vs remote rank | equal |
| `dense_mtp_whole_head_tp2_greedy_ids_equal_spec_off` | engine, tp2 `--spec mtp` vs `--spec off`, in-process and remote rank, k 1/2/3 own drafts, k 2/3 scripted (every acceptance pattern, rollback over the connection), 256 tokens + max_tokens 1/4/7, a sampled request on the remote group | equal; rank 1 served passes |

Also on the merged HEAD (`a5832a4`): `cargo test -p intarsia-core` 198 passed; the mtp / tp suites (`mtp`, `tp`, `tp_mtp` 12, `tp_mxfp4`, `tp_hybrid`,
`tp_remote_acks`, `tp_vocab`, `tp_arnorm`, `tp_batch`) all ok; `docs_env_vars` ok; `cargo test --workspace --no-run` and `cargo build --workspace` ok.

## Rule (a): greedy ids on the Sparks (fleetq #9018, 03:54 to 03:57 EST; raw `raw/w1/`)

`scripts/d2r/xg.sh` `STEPS="ids speed"`: `generate` (rank 0 on spark1 `--tp-remote`, `tp-worker` on spark2), 256 greedy ids per prompt.

| comparison | short-0 | 4k-1 | verdict |
|---|---|---|---|
| dev tp2 `--spec mtp --spec-k 2` vs dev tp2 `--spec off` | identical (md5 b594fdcc7955) | identical (md5 56eefb4d9f43) | **PASS** |

**(a2) head diagnostic** (the ids cannot see a bad head): acceptance of tp2 MTP equals tp1 MTP on the same text. short-0: tp1m 74.5 % (102 verify
passes, 2.49 tokens per pass), tp2m 74.5 % (102 passes, 2.49): the same drafts, pass for pass. 4k-1: tp1m 70.3 % (106 passes), tp2m 71.4 % (105 passes; the
trunk's two-way sums move a few rows by rounding, so a different token is accepted once). Within the 3 points pre-registered: **PASS**. Decode on these
single runs: short-0 tp1m 29.8, tp2 off 24.0, tp2m 41.0 tok/s; 4k-1 28.0 / 23.5 / 38.9.

## Rule (b): speed on the Sparks (fleetq #9018 round 1 and #9019 round 2, reversed; `--quiet gpu,cpu`, vLLM up; raw `raw/w1/`, `raw/w2/`)

`scripts/d2r/xg.sh` `STEPS=speed`, 192 tokens per visit, a fresh process group per visit; round 1 order tp1m, tp2off, tp2m, round 2 reversed (ABAB). Two
jobs, not one: the first (#9018) lost its round 2 to a macOS `tac` bug in the script (fixed, `ROUND_FROM=2` for the second job; same binary, same flags).
Cells: decode tok/s, mean of the two rounds (round 1 / round 2). `scripts/d2r/table.py` over both `steps.log`.

| arm | decode 1k | 4k | 16k | prefill 4k | prefill 16k |
|---|---|---|---|---|---|
| tp1m: tp1 `--spec mtp --spec-k 2` (spark1 only) | 31.77 (31.8 / 31.7) | 28.88 (28.9 / 28.9) | 25.12 (25.1 / 25.1) | 803 | 817 |
| tp2off: tp2 `--spec off` | 24.02 (24.0 / 24.0) | 23.92 (23.9 / 24.0) | 22.50 (22.4 / 22.6) | 1094 | 1179 |
| **tp2m: tp2 `--spec mtp --spec-k 2` (this lane)** | **41.77** (41.9 / 41.7) | **39.72** (39.8 / 39.6) | **36.13** (36.1 / 36.1) | 1072 | 1164 |
| tp2m / tp2off | 1.738x | **1.661x** | 1.606x | 0.98x | 0.99x |
| tp2m / tp1m | 1.315x | **1.375x** | 1.438x | 1.33x | 1.42x |

* **Rule (b), as pre-registered (4k, mean of the ABAB): PASS.** tp2m / tp2off = 1.661 (>= 1.30) and tp2m / tp1m = 1.375 (>= 1.20), both rounds individually
  (round 1: 1.668 / 1.378; round 2: 1.653 / 1.372). Spread between rounds under 1 % in every cell.
* Acceptance (mean of the visits): 4k and 16k tp2m = tp1m (74.5 % / 74.3 %, 2.48 / 2.49 tokens per pass). At 1k the arms generate different text (tp1 and tp2
  round differently), so tp1m 83.2 % vs tp2m 78.4 % is not a head difference (short-0 and 4k-1 above agree pass for pass).
* tp2 MTP decodes 1.6 to 1.7x tp2 plain across context, where tp1 MTP gains about 1.9x over tp1 plain at 4k (lane c27: 14.8 plain on 4k-1): the tp2 verify pass is eager (the
  captured spec step is single-rank, `spec_graph`) and carries the trunk's all-reduces over the wire, so the gain per verify pass is smaller than at tp1,
  but the tp2 trunk is 1.58x faster per pass, which the MTP arm keeps. A tp2 captured verify is follow-up work (lane gt2 measured it for in-process peers only).
* Prefill is unchanged by MTP at tp2 (the head's catch-up over a prompt is KV-only) and tp2 prefills 1.3 to 1.4x faster than tp1.
* Memory: `avail` stayed 83 GB on spark1 and 76 GB on spark2 in the measurement window; the watcher never fired.

## Gates

| gate | result | fleetq |
|---|---|---|
| CUDA plan JSON (35B, 8B, 27B tp1; 27B and 35B tp2 `--spec off`, both ranks on device 0) base (origin/main `e8621e9`) vs dev (`a5832a4`) | byte-identical for 35B, 8B, 27B tp1; the two tp2 plans differ only in the two `free_mem` lines (as in lane mt2) | #9017 |
| `kernels test --quick` (dev, CUDA) | exit 0, 0 FAIL lines (106 kernels, 954 cases) | #9017 |
| Metal no-op (`scripts/dm/noop.sh`; plans 8B / 27B / 35B x default, `--spec off`, `--spec mtp`; kernels list and test) | all SAME (the 8B `--spec mtp` plan is empty on both sides: no head); kernels test 51 kernels, 1136 cases exit 0 both sides | #9014 |
| CPU: core, mtp / tp suites, docs_env_vars, workspace build | all pass (above) | #9016 |
| HIP | not run (tatooine left alone tonight); the in-process dense lockstep head of lane dfh is unchanged (`mtp_whole` is off there) | |

Metal note: the first Metal no-op run (#9012) used binaries built without `--features metal` and compared empty files; it is discarded, #9014 is the gate.

## Left undone / uncertain

* Default flips: none. The default for tp2 across the Sparks stays `--spec off`; turning MTP on there is the user's call (this result supports it: 1.66x at 4k).
* Quantised head (`INTARSIA_MTP_QUANT=fp8`) on the whole dense head at tp2 was enabled but not measured here (c27 measured +4 % at tp1).
* A captured verify pass at tp2 across processes, a vocab-parallel head for a remote group, batching, sampled speculation: not attempted.
* The in-process dense head (HIP tp2, lane dfh) still uses the lockstep sharded head; `LoadOptions::mtp_whole` could be set there to compare, not done.

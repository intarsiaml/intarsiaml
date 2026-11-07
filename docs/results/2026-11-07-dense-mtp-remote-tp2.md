# Dense 27B MTP at tp2 over remote ranks (lane d2r)

Coordinator go-ahead 2026-11-07 03:30 EST (user: 'keep building overnight'): 27B dense MTP at tp2 over remote ranks, opt-in, vLLM stays up, no default flips; merge on passing gates; wrap up by 06:30.

## Status

In progress (written 2026-11-07 03:36 EST). Code and CPU tests done; rules below pre-registered before any Spark measurement.

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

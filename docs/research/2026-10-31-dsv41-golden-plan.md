# DSv4.1 Flash goldens from MiaAI-Lab's TensorFold server: plan (2026-10-31)

Goal: top-k log-prob goldens (`intarsia-golden-topk-v1`, `docs/accuracy.md`) and a reference-noise budget from the Mia 2x-Spark
server (TensorFold `fbbc971` + recipe patches, EXL3 2.9 bpw), for phase 0 and the phase-3 gate ("KL within 2x reference noise",
`docs/plans/2026-10-31-dsv41-unpark.md`). Tool: `tools/dsv41/golden_tf.py` (modes `probe`, `topk`, `tokens`, `compare`,
`compare-topk`). Source paths below are in the patched tree `.scratch/tf/zig/src` (equal to `fbbc971` + patches 0001-0038 is
UNVERIFIED; the recipe dir on spark1 is not a git checkout). Recipe: `spark1:~/code/mia-dsv41-2xspark`.

## i. What the server returns today

| Question | Answer | Source |
|---|---|---|
| `logprobs: true` / `top_logprobs` | HTTP 400 "logprobs are not supported by this model or backend"; `top_logprobs` 0..20 needs `logprobs: true`; an int `logprobs` (vLLM style) gets "logprobs must be a boolean or null" | `server/fields.zig:276-287` |
| choices | always `"logprobs": null` | `server/openai.zig:273,385` |
| `echo`, `prompt_logprobs` | not read anywhere (silently ignored) | grep of `server/` |
| Responses API | refuses `top_logprobs` | `server/responses*.zig` |
| `/v1/decisions` | scores 2..10 answer levels through an engine hook; no dsv41 hook found; not a log-prob source (UNVERIFIED that it is unset) | `server/decisions.zig:1` |
| generated token ids | `"return_token_ids": true` puts `tensorfold.token_ids` in the reply | `server/openai.zig:143,302`, `server/chat.zig:571-574` |
| prompt as ids | `/v1/completions` takes `prompt` as an int list | `server/openai.zig:179-192` |
| `/tokenize` | returns `tokens`, `count`, and `max_model_len` (the window) | `server/tokens.zig:17-66` |
| determinism knobs | `temperature <= 0` is greedy argmax; `"draft": false` per request, `--no-drafts` per server | recipe `start.sh` |
| cached tokens | `usage.prompt_tokens_details.cached_tokens` | `server/openai.zig:291` |
| auth | `/health` is open; `/tokenize`, `/v1/*` are gated when the recipe runs `API_KEY=1` (our `local.sh` does) | `server/auth.zig:230-256` |

Why the engine cannot answer even if the server allowed it: prefill runs the head for ONE row per chunk, the chunk's last
(`prompt.head`, `families/dsv41/prompt.zig:1729`, called once per chunk at `model.zig:1232`; `chunk_rows = 2048`, `model.zig:37`),
keeps `head_g` = fp32 logits of that row for the 129,280 vocabulary (`prompt.zig:266-267`), and only the final chunk's row is
read on the host (`promptLogitsHost`, `model.zig:1345`, used at `lanes.zig:840`). Decode rounds do copy R full rows of fp32
logits to the host (`roundLogits`, `model.zig:1335-1340`) but the greedy path uses the device argmax (`roundArgmax`). The
sampler is host-side top-k + margin (`sampling.zig`). The other tools do not help: `tf-dsv41-check` is a no-GPU weight-layout
check (`check_main.zig:1-3`); `tf-dsv41-layer` / `tf-dsv41-lanes` check recorded requests and only look at the prompt's last
row and decode rows (`layer_main.zig` header; `lanes.zig:841` prints a SHA-256 of the last prompt row's logits when
`logits_sha` is on, useful as a determinism fingerprint but not as a golden).

So today the server gives token ids only. `tokens` + `compare` give run-to-run / config-to-config divergence (first
divergence, leading top-1 match). They are a proxy, not a KL.

## ii. Engine patch proposal: `zig/patches/0039-logprobs.patch`

The series ends at 0038. The Dockerfile applies `zig/patches/*.patch` in order and tags the image from them
(`zig/patches/README.md`); `scripts/prepare.sh` rebuilds (about 6 min per Spark, two Zig builds, line 78). Ship it as two
steps so the cheap one lands first.

**0039a, generation-time log-probs (small; unlocks `topk --strategy continuation`)**
- `server/fields.zig` `probabilityOptions`: accept `logprobs` (bool or int 0..20) and `top_logprobs`; return the chosen K instead of refusing; keep `n == 1`.
- `server/chat.zig` `Input`/`Reply`: carry `logprobs_k`; per generated token store `(id, logprob, top-K ids, top-K logprobs)`.
- `server/openai.zig:273,385`: emit `choices[0].logprobs` (completions: `tokens`, `token_logprobs`, `top_logprobs`; chat: `content[]`). With `return_tokens_as_token_ids` use `token_id:N` keys so ids are unambiguous.
- `families/dsv41/lanes.zig` (around :840 for the first token, and the decode round loop): when the stream asked, read the row with `promptLogitsHost` / `roundLogits` and compute log-softmax in fp64 on the host (`lse = max + log(sum(exp(x - max)))`, then top-K) next to the existing `draw`. Cost: one DtoH of 517 KB per generated token (fp32 x 129,280), top-K on one core, well under a millisecond-scale decode step.
- With drafts on, the verify rows already hold per-row logits; use `"draft": false` for goldens so each token is one plain round.
- Rank 1 needs nothing (logits are already gathered to rank 0, `head_g` is the rank-order gather).

**0039b, `prompt_logprobs` for every prompt position (the teacher-forced prefill gate)**
- Request field `prompt_logprobs: K` (vLLM name) and/or `echo` + `logprobs`; answer layout as `tools/golden_vllm.py` expects: a list of n+1 entries, first `null`, entry j = top-K of the distribution that predicts token j (dict `"id": {"logprob", "rank"}`).
- New `prompt.headAll(e, ch, row0, rows)` beside `prompt.head`: `collapseNorm` over `rows` rows (the call passes `1` today, so a row-count form exists; whether its row stride fits `ch.h` is UNVERIFIED), then `mmRows` with `rows` instead of 1 into a `[rows, cols]` fp32 buffer, then the gather. Do it in sub-blocks of 128 rows: one full-vocab block is 128 x 129,280 x 4 B = 66 MB (33 MB per rank); a whole 2048-row chunk would be 1.06 GB, too much with ~4 GiB free. `ch.head_l`/`head_g` (`prompt.zig:455-457`) grow to the block size; the 2D layout (`e.two`, `t.quarters`) needs the same block form.
- Call it from `model.zig:fillChunk` after the layers (line 1232) when the stream asked, once per sub-block, and copy each block to the host. Log-softmax and top-K on the host over all 129,280 columns (reuse the `sampling.zig` top-K), threaded; do NOT ship raw logits (64K rows x 517 KB = 34 GB).
- Rank 1 runs the same `headAll` (the all-gather is collective), so the flag has to reach it in the chunk descriptor on the wire (`lanes.zig` `self.wire`); how that message is versioned is UNVERIFIED.
- Prompt cache: rows that were restored from a snapshot have no logits, so a `prompt_logprobs` request must bypass or invalidate the prompt cache (also `--prompt-cache-gib 0` for goldens). Prefill-interleave pieces (`TF_DS_PREFILL_INTERLEAVE_MS`, `TF_FILL_CHUNKS`) must keep the block copy inside the piece.
- Cost estimate (UNVERIFIED, no measurement): head GEMM 2 x 5,120 x 129,280 = 1.32 GFLOP per token, 87 TFLOP for 64K tokens, split over two ranks, a few seconds at a realistic tens of TFLOP/s; host log-softmax + top-K of 64K x 129,280 = 8.5 G elements, 15-45 s on 8-16 threads and overlappable with the next chunk. Baseline prefill is 2,008-2,062 tok/s (4K-64K, `docs/measurements.md:31`), i.e. ~32 s for 64K. Expect a 64K golden request to take about 1-1.5 min instead of 32 s.
- Cheap interim inside 0039b, no new GEMM: export the chunk-end row that `head` already computes for every 2048-token chunk (`model.zig:1232`): 32 rows for a 64K prompt, 1 for a short one. Too sparse for the prefill gate on short prompts, fine as a long-context sanity check.

**Indexer selected-set Jaccard (requirement c): not available.** The selection `out` of `topkIndices` (prefill, `prompt.zig:1300`) and the decode path (`round.zig:995-999`, `candFast`/`scoreKeys`) stays on the device; the server and the lane tools never return it. Patch: env `TF_DS_SEL_DUMP=DIR`, which copies the int32 top-512 indices of selected (layer, row) pairs to the host after the indexer (with `TF_DS_IDX_SPLIT` they already sit in the all-gather buffer) and writes `layer, row, indices` to a per-rank file. Dump a sample (every 64th row, a few layers, long prompts only; the candidate-block / top-512 path engages past about 16K compressed positions) because every row is 2 KB per layer. Jaccard = |A and B| / |A or B| per pair, computed offline between configurations. Needs the 0039 image rebuild too.

## iii. One takedown window: exact procedure

Facts used: fleetq status of 2026-10-31 20:55 shows `spark1-vllm` (roles flight, guidance, capcom, 34 GB) and `spark2-vllm` (retro, 38 GB); job #7531 (staging) was still running on both Sparks. Past windows #7529 and #7530 ("Mia DSv4.1 EXL3 TP2 ... vLLM restored after") are the template; their driver is `.worktrees/spark-disk/run-mia.sh` (earlyoom check, `docker stop -t 60 missioncntrl-guidance` on both, `PULL=0 ./start.sh`, restore in a `trap`, `fleetq mark-up`). `.worktrees/spark-disk/RUN-MIA.md` lists the rules.

**Pre-checks (coordinator)**
1. `fleetq status` (nothing else on spark1/spark2); `fleetq when --host spark1,spark2 --mem spark1=116,spark2=116 --exclusive --expect 90` for the start time; book it: `fleetq reserve --host spark1,spark2 --takes-down spark1-vllm,spark2-vllm --mem spark1=116,spark2=116 --exclusive --at HH:MM --for 100m --label "DSv4.1 Mia goldens"`; claim within 15 min (`fleetq wait <id>`).
2. Memory: 98.9 GiB weights + 12.2 GiB engine reserve per Spark; start needs `MIN_AVAIL_GIB=112`. After load only 4.6 GiB (spark1) / 5.4 GiB (spark2) are free at `CONTEXT=65536` (`mia-start2.log`), the bench dipped to 3.8 / 3.9 GiB (`mem2.log`), `memguard` kills the rank at 2.5 GiB held 4 s (hard 1 GiB). That is below the DGX Spark "10 GB for the OS" rule and below earlyoom's default 6 % (~7.3 GiB), so it needs the user's explicit waiver again (#7529 had one) and earlyoom at `-m 3,2` set by the user (`run-mia.sh` refuses to start otherwise). Someone must be present to power-cycle if a Spark hangs. I cannot grant any of this; the coordinator asks the user.
3. Window size: our `local.sh` has `CONTEXT=65536 PARALLEL=1 API_KEY=1`. A 64K prompt plus continuation does not fit, so the window needs `CONTEXT=131072` (KV for the window is ~0.1-0.25 GiB more per rank, `docs/memory-and-safety.md:40`; the 12.2 GiB reserve is fixed). A 64K prefill's workspace on a Spark with ~4 GiB free is UNVERIFIED (a 479K prompt on a head with 6 GiB of other software dipped to 2.67 GiB, `docs/measurements.md:55`); hence the order below: short sets, then 32K, then 64K, one output file per stage.
4. Prefix/prompt cache: TensorFold has no `cache_salt`. The scoring request re-sends prompt + continuation, which would hit rank 0's prompt cache and read cached KV instead of a cold prefill. Start with `./start.sh --prompt-cache-gib 0` (extra args go to rank 0's `serve` and win, `start.sh` header; whether the recipe's flag check accepts it is UNVERIFIED, try `DRY_RUN=1` first). `golden_tf.py` records `cached_tokens` per prompt in the metadata and warns when it is non-zero. No `--spill-gib` (default 0).
5. Provenance to read before the window (read-only): `docker image inspect -f '{{index .Config.Labels "tf.patches"}}' tensorfold-dsv41:exl3-tp2-bed94e70a019` on spark1 (patch hash; the image tag and `TF_DS_ENGINE_ID=tensorfold@<ref>+<hash>` identify the build), recipe commit (`RUN-MIA.md` says `0e7d421`), checkpoint `45ee5d0f`, Engram `engram-1bae0e10`, `TF_REF` `fbbc97132cd0...`. After start: `docker inspect -f '{{.Config.Env}}' dsv41-1017-exl3-tf` on both Sparks. Put all of it in `--note`.

**Driver (a script, run by `fleetq run` so the slot is held and restore is in a `trap`; copy `run-mia.sh`)**
```
fleetq run --host spark1,spark2 --takes-down spark1-vllm,spark2-vllm --mem spark1=116,spark2=116 --exclusive \
  --expect 90 --label "DSv4.1 Mia goldens: tokens x4 configs, vLLM restored after" -- ./run-golden-window.sh
```
Inside: (1) earlyoom `-m 3,2` check on both; (2) `docker stop -t 60 missioncntrl-guidance` on both Sparks; (3) `watch-mem.sh` in the background; (4) per configuration C: `ssh spark1 'cd ~/code/mia-dsv41-2xspark && <ENV> PULL=0 ./start.sh --prompt-cache-gib 0'` (about 3.5 min: load 178 s, "Server answered after 185 s"), poll `curl http://100.120.88.104:8888/health`, run the collection, `ssh spark1 ./stop.sh`; (5) in the `trap` always: `./stop.sh`, `docker start missioncntrl-guidance` on both, wait for "Application startup complete" (up to 10 min), `fleetq mark-up spark1-vllm spark2-vllm`.

**Order of runs** (each run: short sets, then 32K, then 64K; the server is the endor-visible `http://100.120.88.104:8888/v1`, key from `$STATE_DIR/api-key` via `--api-key-file`, path UNVERIFIED; the commands run on endor inside the slot)

| Run | Server start env (`start.sh`) | Why |
|---|---|---|
| A1 | `CONTEXT=131072 PARALLEL=1 API_KEY=1` (our `local.sh`), default switches | the reference |
| A2 | same server, no restart | run-to-run determinism on one instance |
| A3 (optional, if time) | restart, same env | determinism across restarts and graph capture |
| B | `GRAPHS=0 SIDE=0 PREFETCH=0 ENGRAM_AIO=0` | eager path, no side stream |
| C | `RDMA=0 TF_DS_IDX_SPLIT=1 PREFILL_INTERLEAVE_MS=0 TF_FILL_CHUNKS=4 TF_FILL_FIRST=4` | NCCL decode exchange instead of RoCE proxy (`config.sh`), indexer split on every chunk, prompt pass whole / in pieces |

Env names are those of `scripts/config.sh` and `docs/configuration.md:28-29`; `TF_DS_IDX_SPLIT` is documented bit-identical, so C partly is a control. The prefill chunk size is a compile-time constant (`chunk_rows = 2048`, `model.zig:37`), not a legitimate runtime variant here; the chunk plan can change with the prompt-cache cut points (`server/chunk_plan.zig`), UNVERIFIED whether `--prompt-cache-gib 0` shifts it, so keep it identical in all runs.

Commands per run X (on endor, inside the slot):
```
G="uv run --with numpy --with safetensors python tools/dsv41/golden_tf.py"
$G probe --base-url $URL --api-key-file $KEY                       # once per start; today: REFUSED
$G tokens --out golden/dsv41/tok-X-short.st  --base-url $URL --api-key-file $KEY --note "$NOTE"
$G tokens --out golden/dsv41/tok-X-32k.st    --prompts --long-prompts 32768 ...
$G tokens --out golden/dsv41/tok-X-64k.st    --prompts --long-prompts 65536 ...
# after patch 0039b only (replace tokens by topk; configs after A1 re-score A1's ids):
$G topk --out golden/dsv41/top20-A1-short.st ... ; $G topk --out ...-B-short.st --ids-from ...-A1-short.st
$G compare tok-A1-short.st tok-A2-short.st --expect-identical ; $G compare tok-A1-... tok-B-... ; $G compare-topk ...
```
Include `golden_4k.jsonl` / `golden_12k.jsonl` in `--prompts` for the prefix-reuse gate if time allows (they carry `reuse_after_message`; `golden_tf.py` records `reuse_split`).

**Time estimate.** Prefill 2,008-2,062 tok/s: 32K ~16 s, 64K ~32 s; decode with drafts off assumed ~40 tok/s (the scope doc's 42-46 tok/s bound; 55-61 tok/s with drafts, `measurements.md:38-39`). One `tokens` run: 10 short prompts x ~2 s + 32K (16 s + 16 tok) + 64K (32 s + 16 tok) is about 2 min. With the 0039 patch a `topk` run does two prefills per prompt (continuation, then scoring with the head on every row): about 6 min. Start/stop: ~5 min per configuration. Today (tokens only): A1 + A2 + B + C = 4 starts-equivalents (A2 shares A1's start): 3 x (5 + 2) + 2 + restore (vLLM back up ~5-10 min) is about 40 min; reserve 60 min. With topk: about 3 x (5 + 6) + 6 + 10, reserve 90 min.

**Restore and verify (in the `trap`, also on failure)**: `ssh spark1 'cd ~/code/mia-dsv41-2xspark && ./stop.sh'` (stops both ranks, saves logs); `docker ps` shows no `dsv41-1017-exl3-tf` on either Spark; `docker start missioncntrl-guidance` on both; `curl :8000/v1/models` on both until it lists the model; `fleetq mark-up spark1-vllm spark2-vllm`; `fleetq status` shows both `up`, roles flight/guidance/capcom/retro healthy, no `not_restored`; user restores earlyoom to its normal flags. If a Spark stops answering SSH, only a power cycle helps (`fleetq/deploy/LOCAL.md`), then `docker start missioncntrl-guidance` by hand.

**Copy off the boxes** (the goldens already live on endor because the client runs there): `~/.cache/dsv41-tensorfold/logs/` on spark1 (memguard.log, saved rank logs), `mem*.log` from `watch-mem.sh`, both ranks' `docker logs` (or the `LOG_DIR` copies `stop.sh` makes), the output of step 5 above. Commit goldens (small: tokens ~0.4 MB each; topk ~17 MB for the default set + two long prompts) per repo rules, logs under `docs/results/`.

**What to record in the golden metadata** (the tool does it; `--note` carries the rest): server URL, served name, `max_model_len`, `/health`, `/v1/models` and any `/metrics` info lines (`server_meta`), strategy, `cached_tokens` per prompt, sampling (temperature 0, `draft: false`, `ignore_eos`), prompt files and long prompt lengths, skipped prompts, plus in `--note`: image tag + `tf.patches` hash, `TF_REF`, recipe commit, checkpoint and Engram revisions, `start.sh` env of the run (name of run A1/B/C), earlyoom flags, KV dtype (not exposed; recipe docs say compressed FP4 KV, so record "unknown").

## iv. Noise protocol

1. A1 vs A2 (same instance) and A1 vs A3 (after restart): `compare --expect-identical`, later `compare-topk`. Mean KL exactly 0 and top-1 100 % means that configuration is bit-deterministic, so "reference noise" cannot be taken from it.
2. A1 vs B, A1 vs C: the config-to-config spread. `compare-topk` prints `max(measured symmetric KL, 1e-3)` as the budget (the coordinator's absolute floor), then the gate uses 2x that. KL here is on A's top-20 support with B floored at its 20th log-prob (a bound, as `tools/golden_cmp.py`); fp rounding of reported log-probs is not renormalised.
3. Long-prompt rows are the ones that matter for the indexer's top-512 path; report them separately (the tool prints one line per prompt).

## v. Decision table

| | Today (unpatched server) | After 0039a | After 0039b |
|---|---|---|---|
| Token ids, greedy continuation (`tokens`) | yes, all prompts incl. 32K/64K | yes | yes |
| Config-to-config proxy | first divergence, leading top-1 match (`compare`) | same | same |
| Decode-window log-probs (`topk --strategy continuation`, rows only for the `continue` tokens) | no (HTTP 400) | yes, ~64 teacher-forced positions per prompt; 16 on the long ones; NOT valid for `intarsia accuracy` (padded rows, `rows_valid`) | yes |
| Prefill rows, full golden (`prompt_logprobs`) | no | no | yes, every position; usable with `intarsia accuracy --golden` |
| KL noise budget with the 1e-3 floor | no, only divergence statistics | decode rows only | prefill and decode rows |
| Indexer selected-set Jaccard | no | no | no; needs `TF_DS_SEL_DUMP` (ii) |
| Engine rebuild needed | no | yes (image, ~6 min per Spark, new window) | yes |
| Phase-3 gate input | token agreement as a smoke test only | decode KL | full gate |

Recommendation: spend the first window on `tokens` (A1, A2, B, C) so determinism and the divergence statistics are known; build 0039a+b meanwhile and use a second window for `topk`. Run the same prompt files and long prompts in both so the sets line up.

UNVERIFIED summary: patched-tree equality with the pin; `--prompt-cache-gib 0` accepted by `start.sh`; API key path; 64K prefill workspace margin on ~4 GiB free; `collapseNorm` row stride; rank-1 flag propagation; all patch cost estimates; whether `/metrics` shows a build/patch line (the tool just records matching lines); the `docker image inspect` label.

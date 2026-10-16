# Metal continuous batching: M3 gates (speculation under batching)

Date 2026-10-16. Branch `metal-cb-m3gates` off `spike/metal` 66feda6 (no engine commit on top; the branch holds tools, this doc and raw results only; nothing under `crates/` changed; it contains nothing from `metal-cb-m2close` or any other unreviewed engine branch). Machine endor (Apple M5 Max), Qwen3-8B BF16, `intarsia serve --backend metal --max-seq 8192 --max-batch 8 --profile-every 0`, binary `intarsia-66feda6` (sha256 prefix `917f31e09649bd6f`, `cargo build --release --features metal -p intarsia-cli` at 66feda6, 17.6 s). Gate source: `docs/research/continuous-batching-design.md` section 9, row M3, and section 6 for the policy.

Arms: **off** = `--spec off`; **on** = `--spec ngram` (k = 4, `--spec-off-at` 4, the defaults). Every comparison is paired and interleaved: one exclusive fleetq slot, a fresh server per arm and round, arm order alternating by round, a warm-up pass of every scenario shape before each measured pass, `macmon` at 100 ms for the whole slot. Raw: `docs/results/raw/2026-10-16-metal-cb-m3-gates/` (per directory: `slot.log`, per-arm `server-*.log`, `metrics-*.txt`, `macmon.jsonl`, `gpu-clock.md`, `summary.md`, result JSON).

## 1. Result per gate

| gate | verdict | headline |
|---|---|---|
| (a) B = 1 equals G1 | **PASS** | grounded **3.55x** (3.70 / 3.53 / 3.55 per round; G1 3.3x-3.4x); open-ended **1.009x** (0.992 / 1.009 / 1.010; limit 0.99x) |
| (b) B >= 4 within 2 % of spec off | **PASS on medians and means, not resolvable per pair** | localeval aggregate_decode_tok_s, on/off: B = 4 median 1.004 (3 rounds) and 0.997 (5 rounds, cooled); B = 8 median 1.006 and 0.985. Single pairs scatter 0.95 to 1.02 (clock drift, section 3), so ±2 % cannot be shown per pair |
| (c) mixed 1 grounded + 3 open-ended | **PASS on medians (about 0.5 % real cost); the per-pair 2 % bound cannot be resolved** | mixed-once on/off medians 0.991 / 0.994 / 0.997 over three independent slots, mixed-loop 0.991 / 0.995 / 1.000; two single pairs are below 0.98 (0.972, 0.959). The `on` arm is not the same code path as `off`: it runs 2 to 3 verify passes per run in mixed-once (ramp) and 40 to 42 in mixed-loop (B = 3 gaps), so a small real cost exists; the closest A/A arm scatters 0.963 to 1.029 per pair (section 4) |
| (d) identity with spec on | **PASS, cited not re-run** | gate B in `docs/results/2026-10-16-metal-cb-gates-reuse-spec.md` section 3 (240 of 240 requests identical at B = 2, 3, 4, 6, 8, token ids compared) and soak3 Part B in `/opt/code/intarsiaml-cb-soak3/docs/results/2026-10-16-metal-cb-soak3.md` section 3 (268 requests per B at B = 2, 3, 8, 0 mismatches in 799 comparable comparisons, negative control gives exactly the 3 planted ones) |

**Not covered by any M3 gate, but a regression the gates would not catch: at B = 2 and B = 3 `--spec ngram` is 10 to 17 % slower than spec off on open-ended text** (section 5), because a verify pass still runs on its own slot. On grounded text the same configuration is 1.28x to 1.86x faster. The deliverable that removes this (verify groups inside the batched step) is missing (section 2).

`--spec-off-at` sweep at B = 4 (section 6): the default 4 is right. 3 is identical to 4 at B = 4; 5 (speculation on at B = 4) costs 20 to 27 % on open-ended and mixed load and gains nothing on grounded.

## 2. M3 deliverables in code (at 66feda6)

| deliverable | status | where |
|---|---|---|
| row-group step input `(slot, pos0, q_len)` | **missing** | The step input is still one token per sequence: `Row { slot, token, mode }` `crates/intarsia-core/src/scheduler.rs:56`, `BatchModel::decode_step(&[Row])` `scheduler.rs:91`, `SlotModel::decode_step` `crates/intarsia-core/src/engine/batch.rs:134`, `Model::forward_batch(states, ids, modes)` with one id per state `crates/intarsia-core/src/model/batch.rs:121`. No `q_len` anywhere in that path |
| per-sequence drafter state | **present** | `Seq` fields `spec_on`, `spec_k`, `spec_paused`, `pending`, `plain_next`, `ctx` (drafter lookup context), `gate: SpecGate` `crates/intarsia-core/src/engine.rs:832-842`, built in `Seq::build` `engine.rs:892-920`; drafter `spec::propose_min` called from `Seq::plan_step` `engine.rs:1122-1135`; back-off gate `SpecGate` `crates/intarsia-core/src/spec.rs:167-212` (`allow` 188, `record` 197), fed by `Seq::verify` `engine.rs:1142`. The scheduler hands each sequence the on/off gate `spec_allowed` (`scheduler.rs:707`, `BatchSeq::prepare` `scheduler.rs:137`, `SeqAdapter::prepare` `engine/batch.rs:238-246`) |
| spare-row allocation by acceptance | **missing** | No shared row budget (`R_step_max` appears nowhere in `crates/`). The scheduler's only speculation policy is the batch-size switch `b < spec_off_at` (`scheduler.rs:707`). The nearest piece is the per-request back-off `SpecGate` (`spec.rs:167`: after `fails` = 6 passes without an accepted draft it skips 2, 4, .. 32 steps); it reacts to acceptance but does not allocate rows between sequences, and its probe cost is what hurts at B = 2, 3 (section 5) |
| `Phase::Batch` rows include verify groups | **missing** | `Phase::Batch` `crates/intarsia-core/src/plan.rs:60` is `t` sequences of one row each (probes `plan.rs:218-219`, `pos0 + 1` at `plan.rs:228`); `Phase::Verify` (`plan.rs:212-213`) is a separate single-sequence phase. At run time a verifying sequence runs `spec_verify` (`engine.rs:749`, `Model::forward_verify` `crates/intarsia-core/src/model.rs:778`) on its own slot inside `Seq::prepare` (`engine.rs:1177`) before the batched step, i.e. a second full weight pass per verifying sequence per iteration |

What exists is the M3 *scaffolding* (gate, per-sequence state, identity-checked single-slot verify). The throughput part of M3 (batched verify) is not built; gates (a) to (c) pass only because B = 1 needs no batching and B >= 4 turns speculation off.

## 3. Gate (b): aggregate throughput at B >= 4

`localeval speed --sizes 1024 --gen 512 --reps 3 --concurrency ...` (localeval 0.3.0, `/opt/code/deepeval-decode-agg`, the branch that has `aggregate_decode_tok_s`; random-word filler prompts with a nonce first, forced 512 tokens, ctx 986 prompt tokens). `aggregate_decode_tok_s` = completion tokens / wall time (includes prefill and queueing); `overlap_decode_tok_s` = decode tokens inside the window where all streams decode. Table cells are median (per round).

**Run 1: fleetq #2210, `--concurrency 1,2,4,8`, 3 rounds** (`localeval/summary.md`, busy GPU clock 1329 to 1414 MHz):

| streams | off | on | on/off per round | median |
|---|---|---|---|---|
| 1 | 34.8 (34.4 / 34.8 / 34.8) | 40.1 (40.1 / 43.1 / 39.3) | 1.166 / 1.239 / 1.129 | 1.166 |
| 2 | 66.0 (65.5 / 66.0 / 66.2) | 57.5 (57.5 / 58.6 / 55.9) | 0.878 / 0.888 / 0.844 | 0.878 |
| 4 | 114.3 (114.3 / 113.2 / 118.0) | 114.2 (114.8 / 114.2 / 110.6) | 1.004 / 1.009 / 0.937 | **1.004** |
| 8 | 160.7 (159.3 / 160.7 / 165.7) | 161.6 (161.9 / 161.6 / 151.9) | 1.016 / 1.006 / 0.917 | **1.006** |

Round 3 of this run: the `on` server ran at 1329 MHz busy GPU clock against 1414 MHz for `off` (thermal; `localeval/gpu-clock.md`), and its B = 4 / 8 pairs (0.937, 0.917) track that 6 % clock gap, so they are clock artefacts, not speculation cost.

**Run 2: fleetq #2224, `--concurrency 4,8`, 3 rounds, `on` first** (`localeval-b48/summary.md`): the GPU throttled in rounds 2 and 3 (busy clock 1138 to 1214 MHz against 1388 to 1418 in round 1, max 98 C), pairs 0.978 / 0.895 / 0.973 (B = 4) and 0.958 / 0.845 / 0.998 (B = 8). Not usable as evidence either way; kept in the raw directory. It is why Run 3 adds a cool-down.

**Run 3: fleetq #2228, `--concurrency 4,8`, 5 rounds, 45 s idle before every server start** (`localeval-b48-cool/summary.md`, busy clock 1362 to 1463 MHz, per-pair clock gap <= 3.5 %):

| streams | off | on | on/off per round | median | mean |
|---|---|---|---|---|---|
| 4 aggregate | 120.0 (115.1 / 120.0 / 120.6 / 119.3 / 121.3) | 118.9 (115.7 / 115.5 / 121.1 / 118.9 / 120.1) | 1.005 / 0.963 / 1.004 / 0.997 / 0.990 | **0.997** | 0.992 |
| 8 aggregate | 163.1 (155.9 / 163.1 / 163.1 / 164.9 / 161.7) | 160.6 (157.5 / 155.4 / 160.6 / 162.3 / 162.3) | 1.010 / 0.953 / 0.985 / 0.984 / 1.004 | **0.985** | 0.987 |
| 4 overlap decode | 130.4 | 130.1 | 1.014 / 0.975 / 1.002 / 1.002 / 0.995 | 1.002 | |
| 8 overlap decode | 186.3 | 184.7 | 1.021 / 0.954 / 1.000 / 0.986 / 1.003 | 1.000 | |

Client-side cross-check with the same scenario shape (4 open-ended streams, `open-n4`, 256 tokens each): fleetq #2211 3 rounds on/off 0.984 / 0.978 / 0.997 and #2223 4 rounds 0.986 / 0.999 / 0.991 / 0.983 (median 0.989).

Reading: the median and mean ratios at B = 4 and B = 8 are within 2 % of spec off in every run with a clean clock (0.985 to 1.006), so the gate holds as an aggregate statement. Individual pairs scatter by 3 to 5 % (round 2 of Run 3 has `on` 4 to 5 % under `off` at equal clocks, and no cause in the counters), so a strict per-pair 2 % bound cannot be shown with 3 to 5 rounds. There is a small real cost: `on` sits 0.5 to 1.5 % under `off` in most B = 4 pairs, and the counters show why. While 4 or 8 requests arrive and prefill one after another, the first sequences decode at B = 1 to 3 where speculation is allowed. Per scenario at B = 4 the `on` server ran 1 to 3 verify passes (`b4-repeat/m3-on-*.json`, `counters_delta`) against 258 batched steps, each an extra full pass of about 28 ms. The A/A control agrees: `offat3` (also spec off at B = 4, but no probes during the B <= 2 ramp) is level with `off` (offat3/off median 0.999 / 1.005 / 1.006) while `on`/`off` is 0.989 / 0.997 / 0.995 (`b4-repeat/summary.md`, 4 rounds).

## 4. Gate (c): mixed load, 1 grounded + 3 open-ended

Client `tools/cbm3.py` (streams `/v1/chat/completions`, greedy, thinking off, G1's prompts, see section 7). `mixed-once`: the 4 streams start together, 256 tokens each (open-ended with `ignore_eos`); aggregate = tokens / wall. `mixed-loop`: 3 open-ended streams of 640 tokens plus a client that sends grounded requests back to back (B between 3 and 4); aggregate = tokens of finished requests / wall until the open streams end.

| slot | scenario | off | on | on/off per round | median |
|---|---|---|---|---|---|
| #2209, 3 rounds | mixed-once | 128.5 | 128.0 | 0.972 / 0.991 / 0.999 | 0.991 |
| #2209 | mixed-loop | 120.5 | 120.2 | 1.000 / 1.014 / 0.959 | 1.000 |
| #2211 sweep, 3 rounds | mixed-once | 129.9 | 129.1 | 0.988 / 0.994 / 0.999 | 0.994 |
| #2211 | mixed-loop | 123.7 | 122.7 | 0.983 / 0.991 / 0.998 | 0.991 |
| #2223, 4 rounds | mixed-once | 127.5 | 127.2 | 0.987 / 1.004 / 1.003 / 0.991 | 0.997 |
| #2223 | mixed-loop | 121.2 | 120.9 | 0.993 / 0.994 / 0.995 / 1.001 | 0.995 |

Per-stream: open-ended streams decode 32.2 to 33.0 tok/s with spec off and 32.0 to 32.7 with spec on; the grounded stream in `mixed-once` takes 7.8 to 8.3 s in both arms. Conclusion: the gate (never below spec off by more than 2 %) is a per-pair bound. It holds on medians in all three slots, but two single pairs (#2209 mixed-once round 1 0.972, mixed-loop round 3 0.959) are below it and the arms are not the same code path (the `on` arm runs the verify passes counted below), so I do not call those two noise. The noise floor is wide: `offat3`, which only speculates during the B <= 2 ramp and is the closest A/A arm, scatters 0.963 to 1.029 per pair against `off` (`offat-sweep`, `b4-repeat`, section 6). The tightest evidence is the 4-round repeat (#2223): mixed-loop on/off 0.993 / 0.994 / 0.995 / 1.001, consistent and about 0.5 % under, which I read as a small real cost of the verify passes at B = 3 and during the ramp. So (c) is a pass on medians with a cost of about 0.5 to 1 %, and a strict per-pair 2 % bound cannot be shown at ±3 % pair noise. Speculation delivers no gain here either: with 4 streams decoding it is off by design, so the grounded stream gets no speed-up. In `mixed-loop` the grounded requests run 40 to 42 verify passes per run (74 % acceptance, `b4-repeat/m3-on-*.json`) in the B = 3 gaps and the net effect is 0.995.

## 5. Gate (a) and the B = 1 to 3 picture

Gate (a), fleetq #2209, 3 rounds, server `--max-batch 8`, one request at a time, 3 reps per round per arm, 256 tokens, G1's two prompts (`tools/prompts/m3_grounded.txt`, `m3_open.txt` = `docs/results/raw/2026-10-15-metal-spec-g1/prompts/{grounded,open}.txt` on `metal-spec-ngram`), `--chat --no-think` equivalent; decode tok/s = (completion tokens - 1) / (last chunk - first chunk), median of 3 reps:

| prompt | off | on | on/off per round | G1 (docs/results/2026-10-15-metal-spec-g1.md section 5) |
|---|---|---|---|---|
| grounded (371 tokens) | 35.2 (34.6 / 35.3 / 35.2) | 125.2 (127.7 / 124.7 / 125.2) | 3.695 / 3.531 / 3.551 | 35.0 to 114.7 / 120.9, 3.3x to 3.4x |
| open-ended | 35.6 (35.5 / 35.6 / 35.6) | 35.8 (35.8 / 35.4 / 35.9) | 1.009 / 0.992 / 1.010 | 0.99x to 1.01x |

Individual reps of the open-ended prompt scatter by ±3 % (diag run, `diag-b23`: 0.968 and 1.035). Server-side counters of the diag run: spec on, solo open-ended, 53 to 58 verify passes of 211 to 231 proposed tokens, 10 to 17 accepted.

B = 2 and 3, N streams of the same prompt (each with its own nonce line), aggregate tok/s, 3 rounds (the table is fleetq #2226, `offat2-b23/summary.md`; #2209 `client/summary.md` agrees within about 2 %, for example grounded B = 2 67.7 off / 123.1 on, ratio 1.818):

| scenario | off | on (default `--spec-off-at` 4) | on/off per round | `--spec-off-at 2` / off |
|---|---|---|---|---|
| grounded, B = 2 | 67.3 | 125.2 | 1.855 / 1.856 / 1.865 | 0.993 |
| grounded, B = 3 | 95.4 | 122.7 | 1.283 / 1.308 / 1.281 | 0.998 |
| open-ended, B = 2 | 69.0 | 60.9 | 0.875 / 0.893 / 0.883 | 1.004 |
| open-ended, B = 3 | 100.6 | 83.5 | 0.818 / 0.836 / 0.830 | 0.996 |
| filler (localeval), B = 2 | 66.0 | 57.5 | 0.878 / 0.888 / 0.844 | |

**Why it loses.** `diag-b23` (fleetq #2221, `/metrics` deltas per scenario): open-ended B = 2, 32 to 42 verify passes (128 to 168 drafts proposed, 6 to 7 accepted) on top of 251 to 253 batched decode steps; B = 3, 48 to 62 verify passes (12 to 16 accepted) on top of 255 to 257 steps. A verify pass runs alone on its slot (section 2, last row), so every pass the gate lets through is a whole extra weight pass that the batched step does not absorb: 32 / 253 = 12.6 % more passes against a measured loss of 10 to 12 % at B = 2, 48 / 255 = 19 % against 16 to 17 % at B = 3. The back-off gate only closes after 6 consecutive misses and then probes again at 2, 4, .. 32 steps, which is cheap inside one sequence (B = 1: 1.009x) and expensive when each probe costs a pass for everyone. On grounded text the same serial verify still wins (accepted 378 of 512 drafts at B = 2, 74 %) because one verify pass emits ~4 tokens.

Options (not implemented, for the coordinator): (1) build the missing deliverable, verify groups inside the batched step, so a probe costs `m` extra rows (1.03x per pass up to 6 rows by the design doc) instead of a pass; (2) raise the gate's cost when B >= 2 (for example no probes after the first failures while other sequences decode), which keeps the grounded gain and removes the open-ended loss; (3) `--spec-off-at 2` (speculation only for a lone sequence) removes the loss today (ratios 0.99 to 1.00 above) and gives up the grounded gains of 1.28x to 1.86x at B = 2, 3. None changes B >= 4 behaviour.

## 6. `--spec-off-at` sweep at B = 4

Fleetq #2211, 4 arms x 3 rounds, aggregate tok/s, median (per round); ratios are against `off` (`offat-sweep/summary.md`). `offatN` = `--spec ngram --spec-off-at N` (speculation offered while fewer than N sequences decode, so at B = 4: offat3 off, offat4 = `on` = default off, offat5 on).

| scenario (B = 4) | off | offat3 | **offat4 (default)** | offat5 | offat3/off | offat4/off | offat5/off |
|---|---|---|---|---|---|---|---|
| open-ended x4 | 132.1 | 129.2 | 129.2 | 106.0 | 0.978 | 0.984 | **0.799** |
| grounded x4 | 124.2 | 121.5 | 122.1 | 123.1 | 0.970 | 0.983 | 0.983 |
| mixed-once (1 grounded + 3 open) | 129.9 | 126.7 | 129.1 | 95.2 | 0.983 | 0.994 | **0.729** |
| mixed-loop | 123.7 | 120.3 | 122.7 | 100.1 | 0.986 | 0.991 | **0.809** |

At B = 4 offat3 and offat4 run the same code (speculation off for all 4 decoding sequences; they differ only in the B <= 3 ramp phase), so their spread of 0.5 to 2 % is the noise floor of this setup. `offat5` (speculation on at B = 4) loses 20 to 27 % on open-ended and mixed load (verify passes run serially per sequence, open-stream decode falls from 32.5 to 23.8 tok/s in `mixed-once`) and breaks even on grounded x4 (0.983, no gain). **The default 4 is right with serial verify**; no tested workload gains from speculation at B = 4. 3 versus 4 only matters at B = 3, where 4 gives grounded 1.28x and open-ended 0.83x (section 5). A different value should wait for batched verify.

## 7. Method, tools, fleetq jobs

* Tools (this branch): `tools/cbm3.py` (client: gate (a), gate (c), N-stream scenarios, `/metrics` spec counters before and after every scenario, `analyze` with paired ratios), `tools/cbm3_slot.sh` (one exclusive slot: fresh server per arm and round, order alternating, macmon at 100 ms, `client` or `localeval` mode, `COOLDOWN`, `LE_CONC`, `SERVER_ENV`), `tools/cbm3_localeval_table.py` (gate (b) table from localeval results), `tools/cbm3_gpu.py` (busy GPU clock per arm run from macmon), prompts `tools/prompts/m3_grounded.txt`, `m3_open.txt`. Gate (b) uses localeval as the local-model-bench skill requires; gates (a) and (c) and the sweep need grounded text and mixed concurrency, which localeval does not offer, hence the client.
* Prompts: grounded = G1's input-grounded Rust edit (371 prompt tokens); open-ended = G1's 300-word story prompt; both with a nonce line first so no request shares a prefix; open-ended streams use `ignore_eos` so all run 256 tokens.
* Jobs (all `fleetq run --host endor --exclusive --mem 24`, each under 45 min, each ended with `no intarsia serve left`): #2205 smoke (output not kept; first look at the client, GPU was at a low clock), #2209 gates (a)(c) client, #2210 localeval 1,2,4,8 (its slot script hit a zsh parse error at the very end because I edited it while it ran, after all arms had finished; no data affected), #2211 sweep, #2221 B = 2, 3 diagnosis, #2223 B = 4 repeat with A/A arm, #2224 localeval 4,8 (throttled), #2226 off-at 2 vs 4 at B = 2, 3, #2228 localeval 4,8 with cool-down. Busy GPU clock means per slot: 1469 (#2209), 1382 (#2210), 1577 (#2211), 1466 (#2221), 1584 (#2223), 1286 (#2224), 1575 (#2226), 1425 (#2228) MHz: compare only inside a slot.
* Localeval run ids and unanimis records (also in `localeval-run-ids.md`):

| slot | arm | round | localeval run id (`/opt/code/deepeval-decode-agg/results/`) | unanimis record |
|---|---|---|---|---|
| #2210 | off | 1 | `20261016-155604-adhoc-m3gates-off-r1` | `f7e0b960-34b7-45f3-bb95-6270a07c96a8` |
| #2210 | on | 1 | `20261016-155901-adhoc-m3gates-on-r1` | `6d3792a8-d9e0-4d44-a3a1-970f2c0b736a` |
| #2210 | off | 2 | `20261016-160438-adhoc-m3gates-off-r2` | `48ca09c1-ef86-4d0b-ac66-21eb57ee036b` |
| #2210 | on | 2 | `20261016-160150-adhoc-m3gates-on-r2` | `27727963-8ed7-4986-b5fb-01cd07fb8b0c` |
| #2210 | off | 3 | `20261016-160734-adhoc-m3gates-off-r3` | `3b339653-d8fe-4820-8d77-49bc6713a4e1` |
| #2210 | on | 3 | `20261016-161033-adhoc-m3gates-on-r3` | `b3ab092a-1707-41dc-ab64-4d38ecc4237a` |
| #2224 | off | 1 | `20261016-172453-adhoc-m3gates-off-r1` | `0d56d669-74f9-4bc6-b4e6-8601e5d48714` |
| #2224 | on | 1 | `20261016-172234-adhoc-m3gates-on-r1` | `2b4001cf-5cd2-49cb-9ef6-fbe7259aedb9` |
| #2224 | off | 2 | `20261016-172719-adhoc-m3gates-off-r2` | `668cf35c-e58b-4258-a820-6cfba84720a1` |
| #2224 | on | 2 | `20261016-172945-adhoc-m3gates-on-r2` | `d9dbdb7e-8c56-4ae2-a672-93a331e44145` |
| #2224 | off | 3 | `20261016-173436-adhoc-m3gates-off-r3` | `663fb347-bc9b-46e3-a6c6-16bca2d8ad34` |
| #2224 | on | 3 | `20261016-173211-adhoc-m3gates-on-r3` | `09ad6741-5dbb-46d2-be20-5d24d7401589` |
| #2228 | off | 1 | `20261016-180049-adhoc-m3gates-off-r1` | `8ebddfe0-031c-464b-9aee-c9934a55ab63` |
| #2228 | on | 1 | `20261016-180401-adhoc-m3gates-on-r1` | `7d76fd3b-7626-4567-b2be-37070897f8f2` |
| #2228 | off | 2 | `20261016-181009-adhoc-m3gates-off-r2` | `caa7ce72-0e95-4e4b-ba0d-c13f28ea4109` |
| #2228 | on | 2 | `20261016-180706-adhoc-m3gates-on-r2` | `d504a038-d0ef-481c-a8ea-9491e5bd232c` |
| #2228 | off | 3 | `20261016-181316-adhoc-m3gates-off-r3` | `a6385cee-66f8-4d0c-b769-a162b6b22ccd` |
| #2228 | on | 3 | `20261016-181623-adhoc-m3gates-on-r3` | `e8c78b32-e05c-4ec1-a9fc-526fd2b19daa` |
| #2228 | off | 4 | `20261016-182224-adhoc-m3gates-off-r4` | `9485dd59-8ada-4c43-9aad-3a423c3c1e0a` |
| #2228 | on | 4 | `20261016-181925-adhoc-m3gates-on-r4` | `cb038186-9aa7-4e6c-b008-f146fa77f111` |
| #2228 | off | 5 | `20261016-182532-adhoc-m3gates-off-r5` | `4644bf2d-abe5-4557-a119-68e5d979c957` |
| #2228 | on | 5 | `20261016-182839-adhoc-m3gates-on-r5` | `4f911d38-4b46-4902-8c20-05c088b5a3a4` |

* Limits: 3 to 5 rounds per arm, pair noise ±3 to 5 %, so a 2 % bound is judged on medians and means; thermal throttling hit two localeval slots (reported above, not dropped); the sweep did not test `--spec-off-at` at B = 3 directly (offat3 is off at B = 3; B = 3 is covered by `on` against `offat2`); localeval's random-word filler drafts well at N = 1 (spec-on 1.17x there), which is a property of that prompt, not a G1 number.
* Build and checks: `cargo build --release --features metal -p intarsia-cli` at 66feda6 passed. No Rust code was changed, so `cargo test` was not rerun; the tools were exercised by the slots above and `counters()` against a fake `/metrics` server.

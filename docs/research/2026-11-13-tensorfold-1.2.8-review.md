# TensorFold 1.2.4–1.2.8: what intarsiaml should port

Date: 2026-11-13. Read-only review (Fable) of TensorFold `v1.2.3..09a9eb331` (Release 1.2.8; 1,918 commits, 2026-11-05 to 2026-11-13) against intarsiaml origin/main. Nothing was built or run; gains are TensorFold's own numbers unless marked ours. File:line references are as read on 2026-11-13.

**Summary.** Nothing in the range addresses our two hardest open items (the NVFP4 logit shift vs vLLM, the ITL tail). TensorFold also runs NVFP4 experts weight-only (`zig/src/cli/cuda_precision.zig:38`), as we do. Useful finds: server parity, GB10 copy safety, and two gaps that keep MTP idle in the production shape (store on, sampled traffic).

## Ranked candidates

| # | Candidate | TensorFold | Ours today | Target / gain | Effort / risk |
|---|---|---|---|---|---|
| 1 | MTP drafting that survives a prefix-store restore | `zig/src/families/qwen3_5/cuda_mtp_absorb.zig:22-28` (head context ring; restart after a restore gap); exercised by #2214 (TP2 drafts keep prompt states: 28.5K history replies 1.5 s vs 21.2 s), #2447 | Store restores main-model state only; drafting requests bypass it (`engine.rs:1391-1397`); batched engine refuses store + spec (`engine/batch.rs:896-900`); LOG 2026-10-23 12:10, 2026-10-26 17:33 | Spark 35B: MTP in `--max-batch 2 --prefix-store on`. Ours MTP k2 +2.4/+4.8/+9.0% decode at 128/1k/4k (LOG 2026-10-23 01:50) | 3-5 days; greedy ids exact (drafts verified); needs store-on MTP-on vs off ids gate |
| 2 | Sampled requests draft in the batched engine | #2455 `18f556378` (`.../hip/engine/mtp.zig`): keep a draft only where it equals the target's keyed draw; +4-12% tok/s at T=1.0 on gfx1151 | Batched verify greedy-only (`scheduler.rs:191-236`; `2026-11-12-ab35b-cutover-followup.md:154`); the accept-iff-draw-equals-draft rule exists for DSv4.1 (LOG 2026-11-05 20:00) and the single engine | Spark 35B + HIP 27B; the 35B default is T=1.0, so default traffic never drafts | 2-3 days; seeded output identical to spec-off; no kernel change |
| 3 | Tool calls written just before `</think>` returned as calls | #2313 `84d242a96` (`zig/src/server/reply_tool_tail.zig`, `reply_text.zig`, `think_tool_tail_test.zig`) | `ReasoningParser` puts everything before `</think>` in reasoning (`crates/intarsia-server/src/structured.rs:1011-1065`) | Spark 35B API parity for agent traffic with thinking on | ½-1 day; no numerics; check vLLM's behaviour first |
| 4 | Pinned bounce buffer for D2H on GB10 | #2103 `a3962b45f` (`zig/src/cuda/download_stage.zig`, `memory.zig:203`), #2289: pageable sync D2H returned success with stale pages (10/60 runs, driver 580.178.04) | `kernels/cuda/ik_runtime.cu:93-97` async D2H into pageable memory then sync; no pinned path on CUDA | Spark correctness (store captures). TF cost: 303 KB row 10.6 -> 21.0 µs; 64 MiB 1.35 -> 4.10 ms | 4-8 h; bitwise; exposure of our path/driver unverified (restore-acc 200/200 identical) |
| 5 | Paged KV pool (`--max-batch` as a cap, not a reservation) | #2011 (`8d8e4a67e`, `829a61dcf`, `733e4144b`), `TF_QWEN_KV_PAGES=256`, opt-in | Not present (LOG 2026-10-22 fx7); `--max-batch 16` ≈ 16K per slot (`docs/tuning.md:447`) | Capacity at max-batch 4-16 with long contexts | 1-2 weeks; attention kernel changes; highest risk |
| 6 | Conversation-aware prefix-store eviction | #2699 `e5f75e8be` (`zig/src/core/prompt_cache_policy.zig`, `prompt_cache_lifetime.zig`): 16 agents on GB10 (27B), median follow-up TTFT 90-114 s -> 28 s | Plain LRU (`prefix_store.rs:1256-1384`), no protection for a queued request's entry | Spark above 2 streams or tight host budget | 1-2 days; no numerics |
| 7 | Direct-read weight load | #2282 `3883583ac`: 28-31% faster cold load on GB10, +0.6 GB host during load | mmap (`crates/intarsia-models/src/safetensors.rs:71`); `hostmem.rs:1-11` notes page-cache distortion on GB10 | Restart time | 1-2 days; same bytes |
| 8 | Cancel by request id | #3193 `38a339c5e` (`zig/src/server/cancel_ids.zig`): `POST /v1/requests/{id}/cancel` | Disconnect-only (`crates/intarsia-server/src/api.rs:55-58,222-281`) | Extra, not vLLM parity | ½ day |
| 9 | Tool call cut by `max_tokens` returned as far as written | #2511 `b10fda39f` (`zig/src/server/tool_json_cut.zig`) | `openai.rs:320` keeps `finish_reason: length`; unterminated-call output untraced | Low; differs from vLLM | ½ day |
| 10 | DeepSeek-V4.1 server parity (for when the port is unparked) | #2193 (encoder-recorded prompts), #2201 (DSML calls stream from the invoke name), #2182 (NVMe session restore: 319K tokens in 1.2-1.4 s vs 155-178 s) | No DSML / encoder-parity code found in `dsv41.rs` or the server | DSv4.1 | 2-4 days; untestable now (local checkpoints deleted) |

## Looked at, not worth it

- Router register tiles (#2537, 9.9-14.1% cold prompt on our model and box): fixes TensorFold's scalar router; ours goes through the bf16 matmul kernels (`kernels/cuda/matmul.cu:88-100,171`). Cheap check: confirm the router is not hot in our prefill profile.
- One-row rollback copy skip (#2933), 4/8-row windows (#2934): TensorFold-specific structures.
- Request starts (#2176), reasoning-stripped resume (#2723): we have both (`prefix_store.rs` `boundary_tokens`, `header_end_tokens`).
- `--decode-share` cap in the prompt pass (#2802): we have `INTARSIA_PREFILL_PAUSE_ROWS` 768.
- Last-layer skip (#2229, 1-3% TTFT): we have it opt-in (`INTARSIA_PREFILL_SKIP_DEAD`); a default flip is a decision, not a port.
- Intermediate checkpoint ladder (#1953): our store keeps grid entries (`grid_tokens`).
- Native FP4 decode instruction (#2238): needs nvcc 13.2 (we build on 13.0.88); 4-5% of FP4 prompt projections.
- HIP half-value-head DeltaNet (#2582), tail loads (#2247), gfx1151 tiles: gfx1151-only, tied to their kernels.
- HIP DFlash2 drafter (#2726): we have it (lane dfh).
- Metal items (#2851, #2435, #2436, #2507, #3013): 27B- or M5-specific.
- Device-memory prompt states (#2113): our store has a device tier.
- Thinking-off refusal (#2170), tokenizer memo (#2509), server hardening, Responses WebSocket: out of scope or negligible.
- Benchmark-client ITL fix (#2625): same problem as localeval's ITL p99; belongs in localeval.

## Recommended first (and status)

1. Candidate 3: started 2026-11-13 as lane `think-tool` (vLLM behaviour checked first).
2. Candidate 4: started 2026-11-13 as lane `gb10-d2h` (check exposure first; fix only if exposed).
3. Candidates 1 + 2 as one design lane: the largest decode lever found; MTP actually runs in the production shape (store on, sampled traffic). Not started.

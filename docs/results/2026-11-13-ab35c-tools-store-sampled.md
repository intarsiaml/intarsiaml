# ab35c: tool calls on the qwen3_xml build, prefix-store stall, sampled long context (2026-11-12/13, spark2)

One spark2 window (vLLM down 23:16:20 to 23:38:06, 21.8 min, fleetq `--takes-down retro`), Qwen3.6-35B-A3B-NVFP4, same harness as ab35b (`docs/results/2026-11-12-ab35b-cutover-followup.md`). Follows up cutover blockers 1, 3 and 5 of that doc. Scripts: `scripts/ab35c/` (window, lib, toolsmoke with `--thinking default`); raw: `docs/results/2026-11-13-ab35c-tools-store-sampled/raw/`.

Builds: PR #351 at `672b17393` (qwen3_xml tool parser, reasoning split by default; merged as `0e9dd325b`), and PR #352 at `541a156ea` (prefix-store settle moved after `Done`, checksum on the writer thread).

## Blocker 1: tool-result turns. Passes

| arm | thinking | get_weather | add_numbers | create_event | json_schema |
|---|---|---|---|---|---|
| C-default (#351, `--max-batch 2`) | off | 0/5 (model answers without the tool) | 5/5 call, 5/5 receipt, stream 2/2 + 2/2 | 5/5, 5/5, 2/2, 2/2 | 5/5 |
| C-default | default (no flag, stock client) | 2/2, 2/2, 1/1, 1/1 | 2/2, 2/2, 1/1, 1/1 | 2/2, 2/2, 1/1, 1/1 | 2/2 |
| A-vllm (vLLM qwen3_xml, before the window) | default | 2/2, 2/2, 1/1, 1/1 | 2/2, 2/2, 1/1, 1/1 | 2/2, 2/2, 1/1, 1/1 | 2/2 |

- Every receipt turn after a call succeeds (ab35b: 28 of 28 HTTP 400 on a build before #349).
- `get_weather` with thinking off: the model writes "The current weather in Paris is 15°C." without a call. vLLM did the same on this prompt in ab35b (1/5), so it is the prompt, not the engine.
- Thinking default: reasoning is split (no `</think>` in content), calls parse, receipts pass; equal to vLLM on the same script. Small n (2 + 1 stream per tool).

## Blocker 3: prefix-store decode stall. Fixed (PR #352)

Interleaved replay (8 conversations x 6 turns, seed 0), `--prefix-store on`, `INTARSIA_PREFIX_STORE_DEBUG=1`:

| arm | TTFT sum | decode tok/s median | settle | wall |
|---|---|---|---|---|
| C-store (settle between token 1 and 2) | 58.0 s | 43.4 | 102 settles, 72.0 s | 230.3 s |
| C-storefix (#352: settle after Done) | 122.3 s | 75.3 (turn 1 81.4, turn 6 68.2) | 49 after-done settles, 70.0 s; prefill-start 0 | 225.4 s |
| ab35b B-store sequential (reference) | | 68 to 79 | | |

- The stall is the settle: about 1.5 s per request inside the decode window. Moving it after `Done` restores decode to the sequential range, but the replay sends the next request at once, so the cost lands in that request's TTFT (sum 58 -> 122 s, wall unchanged). A relocation, not a fix.
- Where the time goes: 1476 demotes, 11.8 GiB, **download 70.9 s, alloc 0.4 s**: about 170 MB/s device-to-host, with the GPU idle (after-done arm: 68.9 s for 11.6 GiB). `prefix_store_cuda` showed one 32 MiB object at 227 ms into a fresh zero-mapped `Vec` and 1 ms into a reused, written one. Reading: on GB10 a download into untouched host pages pays a fault per page. Candidate fix on #352: pre-fault the destination (gate pending). Restores (upload) run at GB/s on the same box.
- `prefix_store_cuda` on #352 (`541a156ea`): restored vs cold (store on) bitwise true.

### Window 2 (vLLM down 23:49:32 to 00:00:32, 11 min): pre-faulted host buffers (#352 at `368346b24`)

| arm | TTFT sum | decode tok/s median | wall | D2H demotes |
|---|---|---|---|---|
| C-store (main, window 1) | 58.0 s | 43.4 | 230.3 s | 11.8 GiB in 70.9 s (170 MB/s) |
| C-storefix (`541a156ea`, window 1) | 122.3 s | 75.3 | 225.4 s | 11.6 GiB in 68.9 s |
| **C-storepf** (`368346b24`, `--prefix-store on`) | **60.5 s** | **75.1** (turn 1 81.4, turn 6 68.3) | **164.0 s** | 11.6 GiB in 2.7 s (4.4 GB/s) |
| C-storepf-dev (`device_mb=8192,host_mb=2048`) | 56.8 s | 74.7 | 160.2 s | |

- Pre-faulting the destination (`host_buf`: one write per 4 KiB page before the download) takes the demote from 170 MB/s to 4.4 GB/s; `prefix_store_cuda` on that build: demotes 465 MB/s -> 11.7 GB/s, 2 passed, restored vs cold bitwise true.
- Same TTFT as main with decode at the sequential rate: the interleaved stall is gone. Remaining settle 6.8 s over 98 settles (page faults, now in the alloc column).
- Device tier: a little better again (TTFT 56.8 s); a recommendation candidate, not a default change (the device tier is not counted in `--mem-budget`; min MemAvailable 73.3 GB).

## Blocker 5: sampled decode at long context

Temperature 0.7, top_p 0.95, 256 tokens, 3 reps, `scripts/ab35b/sampled.py`:

| arm | 4k | 32k | 64k |
|---|---|---|---|
| A vLLM (ab35b) | 75.1 | 69.2 | 62.7 |
| B batched bf16 KV (ab35b) | 80.6 | 65.9 | 54.8 |
| **C-fp8 batched, `--kv-dtype fp8`** | | **70.9** | **58.3** median (62.8, 58.3, 58.2) |
| B-single, sampled MTP `match` (ab35b) | 76.8 | 68.5 | 52.3 |
| **C-single-off, `INTARSIA_SPEC_SAMPLED=0`** | **81.0** | **66.2** | **55.1** |

- fp8 KV closes 32k (+2 % vs vLLM) and half of 64k (-7 %, was -12.6 %). 64k is bimodal: the first request on the fresh server ran at 62.8 (vLLM parity), the next two at 58.2 to 58.3; not explained.
- Sampled MTP on the single engine gives no gain against its own spec-off control (different builds: ab35b vs ab35c). Porting it to the batched engine as is would not help.
- The `--max-batch 2` banner says "sampled requests: speculation Match" although the batched engine never drafts on sampled requests (misleading banner).

## Not covered

- HIP tool-call gate (the change is server-side; not run). Endor Metal no-op skipped for #351 (server parsing + recipe `tool_parser` only).
- DSv4.1's thinking default after #351 (`think-tags`; changes only if its prompt ends in `<think>\n`): not checked.

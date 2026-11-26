# Inkling-2-Mini Metal prefill: is there a per-chunk host gap? (lane ink-pfgap2, 2026-11-25)

Status (2026-11-25 20:30): **no per-chunk host gap on current main; nothing to fix, no code change.** Between prefill chunks the GPU idles
0.3 ms per chunk boundary (0.04 % of a 4k or 16k prefill). The only GPU idle of any size is a one-time 30-60 ms stall at the start of a request
that follows several seconds of GPU idle, and it is a GPU-side start latency (the command buffer is already committed), not host work.
Branch `ink/pfgap2` (base `072dd0863`, busy prefill chunk 2048 default); this doc and the raw files only. Not pushed.

## Question

`2026-11-21-inkling2-prefill.md` measured "4 chunks in 4014 ms, GPU busy 94 %, idle 232 ms (~58 ms per chunk host gap)" and listed "batch the
prefill command buffers" as the next speed item. That trace is from 11-21, before the later prefill merges and before chunk 2048 became the busy
default. A host clock cannot answer this (every non-final prefill step returns in 3.6-3.9 ms because each chunk is queued asynchronously, the last
step absorbs the GPU wait), so the measurement is GPU-side: `INTARSIA_METAL_PROFILE=cb:<file>` records `GPUStartTime` / `GPUEndTime` of every command
buffer (plus the host's encode-open / commit times on the same clock), and gaps between records are GPU idle.

## Method

fleetq #11910, endor, exclusive slot, tree `072dd0863` (clean), binary built by #11909 (`cargo build --release --features metal -p intarsia-cli`).
One `intarsia serve --backend metal --spec off --max-seq 20480` process, default recipe and chunk (2048), no `INTARSIA_*` variables, prefix store off.
`POST /v1/completions` with `max_tokens 1`, greedy, every prompt unique (distinct corpus slice and a `Request N.` prefix; `cached_tokens` 0 in all).
Prompts: 4.1-4.3k tokens (3 chunks: 2048 + 2048 + rest) and 15.9-16.0k tokens (8 chunks). Request times are on the trace's clock
(`CLOCK_UPTIME_RAW`), so each request's command buffers are selected exactly. Per size: back-to-back requests (0 s idle before; the client turnaround
is ~60 ms) and requests after a 5 s idle. A warm-up 4k and 16k request ran first. Files: `docs/results/2026-11-25-inkling2-pfgap2/` (`gap.sh`,
`analyze.py`, `requests.tsv`, `analysis.txt`; the 2.5 MB `cb.jsonl` stays in the worktree's `.scratch/gap/`).

Columns of `analysis.txt`: GPU span = first command buffer start to the last prefill command buffer end of the request (up to the first-token head);
busy = sum of command buffer durations; idle = span - busy, split into lead (before the first compute buffer), chunk boundaries (before the
embedding buffer that opens each later chunk), tail (before the head), other.

## Result

| req | prompt tokens | idle before | GPU span ms | busy ms | idle ms | idle % | lead ms | chunk boundaries ms (n, max) | other ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4216 | 0 s | 1544.7 | 1539.6 | 5.1 | 0.33 | 0.0 | 0.6 (2, 0.3) | 4.5 |
| 3 | 4243 | 0 s | 1605.4 | 1601.2 | 4.3 | 0.27 | 0.0 | 0.7 (2, 0.3) | 3.6 |
| 1 | 4138 | 2 s | 1507.2 | 1439.9 | 67.2 | 4.46 | 58.5 | 0.6 (2, 0.3) | 8.1 |
| 4 | 4290 | 5 s | 1550.1 | 1482.2 | 68.0 | 4.38 | 58.2 | 0.6 (2, 0.3) | 9.2 |
| 5 | 4272 | 5 s | 1482.9 | 1443.2 | 39.7 | 2.68 | 33.4 | 0.7 (2, 0.4) | 5.6 |
| 6 | 4284 | 5 s | 1497.3 | 1426.3 | 71.0 | 4.74 | 61.1 | 0.6 (2, 0.3) | 9.3 |
| 7 | 15976 | 0 s | 5980.4 | 5974.0 | 6.4 | 0.11 | 0.0 | 2.3 (7, 0.3) | 4.1 |
| 8 | 15947 | 0 s | 6411.5 | 6404.6 | 7.0 | 0.11 | 0.0 | 2.2 (7, 0.4) | 4.7 |
| 9 | 15935 | 5 s | 5863.0 | 5824.8 | 38.1 | 0.65 | 29.6 | 2.3 (7, 0.4) | 6.2 |
| 10 | 15914 | 5 s | 7658.6 | 7613.6 | 45.0 | 0.59 | 36.3 | 2.2 (7, 0.4) | 6.5 |

- **Chunk boundary: 0.3 ms (max 0.4) per boundary**, 0.6 ms per 4k prefill, 2.2-2.3 ms per 16k prefill, 0.04 % of the span. The host is ahead of the GPU the whole
  time: the next chunk's buffers are committed while the previous chunk is still running (`hc` before the previous buffer's end). The 11-21 "58 ms per chunk" is gone
  (it was probably the pre-merge schedule; this lane did not bisect which merge removed it).
- **Total idle, back-to-back requests:** 0.27-0.33 % (4k), 0.11 % (16k). Under the 2 % rule at both lengths.
- **After a few seconds of GPU idle:** 2.7-4.7 % at 4k, 0.6-0.65 % at 16k. All of it but 5-9 ms is one stall of 30-61 ms in six of six such requests
  and none of four back-to-back ones: the first compute buffer (`moe_experts_situ_bf16`, 128 dispatches) starts 30-60 ms after the 4-dispatch embedding buffer
  before it ends, although `commit` of that buffer returned 7-10 ms *before* the embedding buffer ended (`hc - prev_e` is -7 to -10 ms in the trace). The host has nothing
  left to do in that window, so it is GPU-side start latency after idle (clock / power state or per-buffer residency work; not told apart here). The first embedding
  buffer runs 4.8-6.5 ms against 0.3 ms for later chunks, which points the same way.
- It does not cost wall time overall: the after-idle 4k spans (1483-1550 ms) are *shorter* than the back-to-back ones (1545-1605 ms), because the busy time is 5-10 % lower
  after idle (1426-1482 ms against 1540-1601 ms; a cooler GPU or higher clock is the likely reason, not measured here). A fix for the stall would trade against that.
- Other idle, once per request: 3.2-3.5 ms before the head buffers after the last chunk, and 0.8-6 ms before the first embedding buffer. Outside the prefill, per request, the
  host spends 10-24 ms before the first buffer starts and 15-21 ms after the last buffer ends (tokenise / admit / read back / HTTP), consistent with the 20-48 ms
  client-minus-engine difference in `2026-11-21-inkling2-mtp.md`.

## Earlier lane on the same question (11-22, `ink-pf-gap`)

`.worktrees/ink-pf-gap/.scratch/` (gap.sh ... gap4.sh, fleetq #11686 #11691 #11699 #11723) holds cb traces on tree `8c7bb7cb1`; no write-up reached main and no branch has one
(`git log --all --grep=gap` finds only unrelated docs). Re-reading its traces with this lane's classifier agrees with the table above: chunk boundaries 0.3-0.4 ms, and the same
one-time 35-65 ms stall before the first compute buffer of a request after idle, in all three keep-warm modes (default residency set 35 / 53 / 64 ms, `INTARSIA_METAL_KEEPWARM=0`
39 / 51 / 66 ms, `blit` 57 / 62 / 65 ms), so the stall is not the residency keep-warm choice. Idle between requests in those traces (55-58 ms between one request's last buffer and the
next one's first) is client / HTTP turnaround, not prefill.

## Decision

Stopped after step 1 under the lane's rule: the per-chunk host gap does not exist, so there is nothing for command-buffer batching to remove, and no code changed. Batching would
at best recover 0.3 ms x (chunks - 1). "Batch the prefill command buffers" should come off the next-ideas list of `2026-11-21-inkling2-prefill.md` (item 2).

Open, not pursued (outside the prefill command-buffer and encode path, and not host work): the 30-60 ms first-buffer start latency after idle, worth 2-4 % of a cold 4k TTFT and
0.5 % of a 16k one. A direction if someone wants it: a small compute buffer committed when the request is admitted, so the GPU leaves its low state while the host tokenises
(10-24 ms available). Whether that helps depends on the cause (a dummy kernel does not ramp a clock by itself), and the 5-10 % lower busy time after idle suggests the gain could be partly
paid back. Measure with `gap.sh` (idle 0 / 1 / 5 s arms) before building anything.

## Commands

```
fleetq run --host endor --load cpu --expect 25 -- .scratch/build.sh                      # #11909, build of 072dd0863
fleetq run --host endor --exclusive --expect 20 -- .scratch/gap.sh                       # #11910, 4 min 34 s
python3 analyze.py cb.jsonl requests.tsv                                                 # offline, any machine
```

# graceful-cut: a stream cut by the shutdown drain deadline, 35B on spark1

Lane `graceful-cut` (PR #482). Re-runs the one case of [`2026-11-21-drill2.md`](2026-11-21-drill2.md) that ended as a silent truncation (SIGTERM to the supervisor while generations outlast the 30 s grace), once, with the drain deadline (`INTARSIA_DRAIN_S`, default 28 s for a supervised child) in the binary.

Binary `drill-4bb837eb674b` (the PR's head commit, built on spark1, fleetq #10498, `--load cpu`, 6 min), same config as drill2: `intarsia serve --backend cuda --model <nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a> --port 8191 --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store on`, no environment variables. Drill job fleetq #10500 (`--mem 52 --expect 12`, 3 min). Same two 12,000-token streams as drill2 (`scripts/drill/longgen.py`: one through the OpenAI Python SDK, default `max_retries=2`, one raw httpx SSE reader), SIGTERM to the supervisor about 6 s into the generations. Production vLLM on spark1 was never touched (its GPU process, 26,768 MiB, was unchanged in every snapshot). One repeat only (drill2 ran three). The orchestrator was `scripts/drill/drill2.sh` with the repeat list cut to R1, the baseline verify dropped and the output directory moved (not committed).

## Result

| | drill2 (before) | this run (drain deadline) |
|---|---|---|
| streams ended after SIGTERM | 29.9 s, by the supervisor's SIGKILL | 28.2 s, by the server |
| child gone / supervisor gone after SIGTERM | 30.03 to 30.14 s / 30.7 to 30.8 s | 28.08 s / 28.76 s |
| supervisor log | no line for the SIGKILL | no SIGKILL (none was needed) |
| raw SSE reader | HTTP 200, then `RemoteProtocolError` (incomplete chunked read), no `finish_reason`, no `[DONE]`, no error event | HTTP 200, 3,715 content chunks, final chunk `finish_reason` `abort`, error event `{"message": "the server is shutting down and could not finish this request in time; retry it on a running server", "type": "server_error", "param": null, "code": 503}`, `[DONE]`, clean end of the body |
| OpenAI SDK | `APIConnectionError: Connection error.` | 3,712 content chunks, a chunk with `finish_reason` `abort`, then `APIError: the server is shutting down and could not finish this request in time; retry it on a running server` |
| port 8191 free after SIGTERM | 0.09 s | 0.091 s |
| processes of ours left | none | none |

The streams were at about 3,715 tokens (31 %) when ended. `serve-R1.log` has `[server] drain deadline (28 s after the stop signal): ending the requests still running`. Memory after the exit back to the pre-launch level (`used` 42,449 MB at the end against 43,062 before launch), nothing of ours left on the GPU, `/dev/shm` unchanged. Lowest `MemAvailable` 52.3 GB (the 15 GB kill line never came near). The cancel reached the engine: both child and supervisor exited 0.7 s apart with no SIGKILL, so the sequences and their KV were released and the process exited well inside the grace.

Not measured: an idle SIGTERM and the restart check on this binary (the CPU tests `crates/intarsia-cli/tests/serve_drain.rs` cover the idle exit without a SIGKILL); the group-signal case (systemd control-group stop), where both processes get SIGTERM and the child's drain starts at its own signal.

Raw: `2026-11-21-graceful-cut/raw/` (`long-R1.jsonl`, `ctl-R1.log`, `serve-R1.log`, `drill.log`, `mem-R1.txt`, `ctl-cleanup.log`). At the end no slot is held, no server of ours runs, `~/intarsiaml-drill` on spark1 is removed.

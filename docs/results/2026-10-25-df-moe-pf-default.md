# CUDA MoE prefill L2-prefetch kernel (`INTARSIA_CUDA_MOE_W4A16_PF`): default-on

Lane df, 2026-10-25. User decision 2026-10-25 00:06 EDT: "turn the MoE prefill speed-up on by default and use 8192-token prefill chunks on CUDA".
This is the first half (the second is the chunk-8192 package). The kernel and its evidence are #129
(`docs/results/2026-10-23-mp-cuda-moe-prefill.md`): the gate/up w4a16 routed-expert prefill schedule with L2 prefetch, for chunks of at most
2048 tokens (64 pairs per expert at 256 experts, top-8), bitwise the same output as the old kernel. Speed is not re-measured here: #129's
same-window numbers stand (prefill +7 % at 1k, ~+1 % at 4k and 16k, chunk 2048).

Change: `pf_enabled` in `kernels/cuda/moe_w4a16.cu` is on unless `INTARSIA_CUDA_MOE_W4A16_PF=0`; `docs/tuning.md` row and a `docs/decisions/LOG.md` row.

## Gates (spark2, `--quiet gpu,cpu`, lane-private golden dir `~/intarsia-decfuse/df-golden`, base = origin/main `73ac1bbf` build, dev = this branch)

| gate | result | fleetq / raw |
|---|---|---|
| greedy ids, base (knob off, #129 merge) vs default build: 256 ids after a ~1.2k-token prompt and 64 ids after a ~4k-token prompt, Qwen3-8B and 35B | all four **identical** (35B 256 sha 2dd7fe93, 64 sha 5efcc225; 8B 256 sha e2c61289, 64 sha fcafb118) | #4965, `raw/dfA-noop/` |
| `plan show` 8B (json) and 35B, base vs default build | byte-identical (8758 B, 3505 B): the plan does not name the knob | #4965 |
| acc-short vs the committed CUDA baseline (`--pre-change c37a8d3`, stamped build) | PASS: prefill mean KL 6.82514e-3 (baseline 6.82514e-3, limit 7.84891e-3), decode 6.98641e-3 (6.98641e-3, 8.03437e-3): identical to main | #4977, `raw/dfA-acc3/` |
| 3 KL goldens (short, long, 4k) at chunk 2048, knob unset | golden gate PASS x3; 21-way KL (scored rows) 5.408e-3 / 1.417e-2 / 1.214e-2, the same as #129's knob-on run; the full tables equal #129's control on main's build (`mpgA0`) line for line except the ref-noise sidecar path | #4966, `raw/dfA-acc/` |
| prefix-reuse gate at chunk 2048 with the CPU continue oracles | PASS (5 of 5 judged) | #4967, `raw/dfA-gb/` |
| `batch_cuda_real` (35B and 8B) | 3 passed, 0 failed, FAILED count 0 | #4967 |
| `batch_cuda_real --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine` | 1 passed, 0 failed | #4967 |
| backend-cuda `moe_w4a16` (bitwise PF 0/1 x BM auto/32/64), `quantmoe`, `moe_nvfp4`; `docs_env_vars` | 4 / 13 / 5 passed, 2 passed | #4968 |
| `intarsia kernels test --quick`, knob unset and `=0` | 77 kernels, 531 cases, 0 FAIL, both | #4968, `raw/mp-kt-*.txt` |

The two binaries are the same code except the default (`INTARSIA_BUILD_STAMP` head 58602d3, code-tree 4a60e99bfa39). Opt-out: `INTARSIA_CUDA_MOE_W4A16_PF=0`.

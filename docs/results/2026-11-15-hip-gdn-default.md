# HIP: chunked GDN delta rule default-on (lane gdn-default)

Status: 2026-11-15, user decision ("Yes, flip GDN"). `gdn_delta_rule_bf16_chunk` (lane gdn-chunk, #376,
docs/results/2026-11-14-hip-gdn-chunk.md) is now the HIP default for delta-rule calls of T >= 64 at dk = dv = 128;
`INTARSIA_GDN_CHUNK=0` is the kill switch (back to v3 / v2, main's bits before this change). Decode, verify rows, calls
under 64 tokens and the fused conv + delta are unchanged.

## What the user accepted

The change-vs-base gate fails, by a reference-noise-sized margin (gdn-chunk section 4): prefill mean KL21 3.455e-3 vs the
limit 3.161e-3 (short golden, 1.09x) and 2.791e-2 vs 2.218e-2 (long golden, 1.26x). The cause is the bf16 WMMA operands, not
a tunable. In exchange: HIP 35B prefill +9.7 / +13.7 / +13.0 % at 1k / 4k / 16k (tp1), +17.4 % at 4k (tp2); the delta rule
88.1 -> 33.1 ms per 4420-token prefill.

## Checks on the flipped tree (tatooine, 2x R9700, head=4379dd0 code-tree=a5d1066ac581 dirty=no; fleetq #9366-#9368)

| check | result |
|---|---|
| HIP crate unit tests (kernel selection now picks the chunked kernel at m >= 64), Mac and tatooine | PASS (30 / 30) |
| HIP GPU test suite (`cargo test --release -p intarsia-backend-hip --tests`, benches / timing skipped) | PASS, 0 failed |
| tiny-model packed / macro / MTP prefill == solo on HIP, default and `INTARSIA_GDN_CHUNK_MIN=1` (every segment chunked) | PASS |
| real 35B: `real_35b_macro_prefill_is_bitwise_the_plain_schedule`, batched macro prefill == local engine, batched long prompts == local engine | PASS (3 / 3) |
| `kernels test --backend hip --quick` | PASS; chunked kernel 4 cases ADMITTED (max rel 3.76e-3, BF16_TOL) |
| `qualify` (tp=2 included) with the four 35B HIP baselines removed | MISSING-BASELINE as intended; acc-short floor prefill top-1 98.47 % KL 6.00e-3, decode 98.75 % KL 1.11e-2; acc-oracle floor KL 1.04e-2 (limit 2e-2) |
| `qualify` again with the re-recorded baselines installed | **PASS** (quick 6 pass; full 7 pass, 0 fail) |
| `acc-4k` (prefix reuse), chunk on | FAIL: restore / aligned cases "nothing was reused (no restorable state at or below the split)"; all three continue cases pass (two oracle-relative) |
| `acc-4k` control, `INTARSIA_GDN_CHUNK=0`, same build | FAIL the same way (restore / aligned), and also `long_code_4k continue` (streams diverge at token 15). Not caused by this change: HIP has no restorable hybrid state at those splits on main either |

Packed prefill: the delta rule runs once per sequence on its own state (`crates/intarsia-core/src/model/gdn_block.rs`
`gdn_block_segs`), so the chunked kernel never sees a multi-sequence call; each segment picks its kernel by its own length.

## Re-recorded baselines (from this build)

`golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip.{baseline,envelope}.json`,
`golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip{,-tp2}.selfgolden.safetensors` (the method of
`scripts/fx1-oracle/hip-window2.sh` B-D; `scripts/gdn-default/stages.sh` steps rec / chk).

**Not re-recorded: the four `qwen3.8-27b-mxfp4-*.hip.*` baselines.** The 27B has dk = dv = 128, so its HIP prefill numerics
move too, and its HIP acc-short will read against stale baselines until they are re-recorded. The 27B checkpoint is not on
tatooine now; re-record when it is back.

Open, not measured: the 27B on HIP with the flip; served speed on main after the flip (the gdn-chunk numbers were measured
with the knob on the lane tree).

Raw: `2026-11-15-hip-gdn-default/raw/` (window = all steps, control = acc-4k knob off). Scripts: `scripts/gdn-default/`.

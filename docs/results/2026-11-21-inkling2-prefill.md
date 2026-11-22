# Inkling-2-Mini prefill on Metal (lane ink-prefill, endor M5 Max, 2026-11-21)

Branch `ink/prefill` from 1dacac3a8 (ink/metal). Not pushed or merged. Every GPU run had a fleetq slot (timing `--quiet gpu,cpu`).

## Briefs received

1. 2026-11-21 (coordinator, via SendMessage): **both go-aheads approved; keep the order MoE experts first, the looped blend
   second, rel_mode third.**
   - **rel_mode conditions:** (1) `ink.rel_mode` defaults to 0 and mode 0 stays exactly today's contract; (2) a CPU reference
     test asserts the mode-1 pair equals the mode-0 pair bitwise; (3) Metal admission sweep cases for mode 1 at prefill shapes
     (T 17, 512, 2048; SWA and full visits; tau past 8192); (4) the graph picks mode 1 only when the backend admits both calls in
     mode 1, the CUDA lanes need nothing; (5) rerun G3 short/long/4k with mode 1 active on Metal and update the op-list entry
     in the spec doc on this branch.
   - **Blend:** this lane owns the prefill path (T > 16), in its own file plus the one-line hook in `launch_mix`, after
     ink/m-mix lands in ink/metal. m-mix keeps decode (T <= 16). No edits to m-mix's kernel files beyond the hook.
   - Report prefill tok/s at 2k / 8k / 12k next to oMLX's ~1300 when available.
2. 2026-11-21 (coordinator, via SendMessage): **ink/metal is integrated and runs end to end** (tip a88db4620: m-attn, m-moe,
   m-mix up to d99d2c0fc, ink/skel; G1-G3 pass on Metal). Its baseline: 5k-token prefill 1219 tok/s at chunk 256, 2221 tok/s
   at chunk 2048 (~920 ms per 2048 chunk vs this lane's ~700 ms kernel sum).
   - Merge ink/metal a88db4620 into ink/prefill now (unblocks the blend hook).
   - Measure end-to-end prefill at 2k / 8k / 12k, chunk 2048, on the merged tip before and after each kernel switched on (bench
     command in the ink-metal results doc).
   - Report the 920 vs 700 ms split: per-chunk host gap, router / routing, whatever else is missing. Prefill host gaps belong to
     this lane (ink-hostgap is decode only).
   - One line on 7.3 vs 11.9 ms per MoE visit (m-moe's setup: weights cycled from DRAM, union = 128 experts); no more time on it.

## Status

In progress; tables below are filled as measurements land.

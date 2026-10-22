### 3.1 The prefill gap on the 8B is mostly the default chunk size (opt-in arm `intarsia4k`)

`intarsia4k` is `serve --prefill-chunk 4096`, **opt-in, not a default, not counted in any rank**. On the 8B it lifts prefill from 2.4k to 4.2k tok/s at 4k (1.7-1.9x the default build) and puts
intarsia within 7-10 % of vLLM and SGLang (4k: 4170 against vLLM 4603 and SGLang 4305, bands disjoint from vLLM's; 16k: 3583 against 3837 and 3830); decode is unchanged. On the 35B the same flag gives +5 % at 4k and 16k (section 2).
Chunk size changes the numerics and the prefix-reuse behaviour (round 7 reported the opt-in 4096 build failing `chat_reuse_4k`; LOG 2026-10-21 11:23 kept it opt-in), so this lane only reports it and changes no default. The `--mem-budget 32` used for the
other intarsia 8B visits leaves a 11800-token context with a 4096 chunk, so `intarsia4k` visits p34-p36 ran with `--mem-budget 40` to cover 16k; p31-p33 have no 16k row.
The first intarsia visit (p1) also ran with `--mem-budget 24` (context 11800, no 16k row); its 128-4k rows are in the table.

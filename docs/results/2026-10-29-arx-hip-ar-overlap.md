# Lane ARX: the wide all-reduce on the 35B, and the GEMM that pushes its partial (HIP tp2, 2026-10-29)

Branch `lane-arx/hip-ar-overlap` (main `f9fabeb13`, merged up to `200be3f05`). tatooine, 2x R9700 (gfx1201), tp=2, intarsiaml's own
container with Radiance stopped (one fleetq job per window holding both cards). Scripts: `scripts/arx/` (copies of PF27b's driver and
micro sections plus the 35B model mod, `ids2`, `tests2`, `push`, `tap`, `probe`). Raw: `docs/results/2026-10-29-arx-hip-ar-overlap/raw/`.
Speed is localeval (`/opt/code/deepeval`, 3 reps, 128 forced tokens, median of the reps; the tables give the median over the rounds).

## Phase 1: `INTARSIA_HIP_AR_WIDE` (#232) on Qwen3.6-35B-A3B NVFP4

#232 turned the wide all-reduce (> 64 KiB, 64 blocks) on for every HIP tp2 model and measured only the 27B. W1 and W2 ran the 35B with
`INTARSIA_HIP_AR_WIDE=0` (the old path) against the default, 5 interleaved rounds each (3 MTP-default, 2 `--spec off`):

| 35B tp2 | WIDE=0 | default (WIDE) | change |
|---|---:|---:|---:|
| prefill 1k (tok/s) | 8703 | 8826 | +1.4 % |
| prefill 4k | 12155 | 12576 | +3.5 % |
| prefill 16k | 11340 | 11754 | +3.7 % |
| decode 1k (tok/s) | 191.3 | 191.2 | -0.05 % |
| decode 4k | 189.9 | 189.7 | -0.1 % |
| decode 16k (not asked; see note) | 181.2 | 178.7 | -1.4 % (best round -0.3 %) |
| decode at 4 / 8 / 16 streams (1k, mb16) | 290.2 / 370.3 / 394.5 | 288.3 / 371.7 / 390.3 | -0.7 / +0.4 / -1.1 % (one round) |

Correctness, both paths: greedy ids identical WIDE=0 vs default for short-0, 4k-0, 16k-3, each spec off and the 35B's default spec (6 / 6,
256 tokens; `raw/w1/gates-11.txt`); served texts 18 / 18 identical with the default spec and 18 / 18 with spec off; `tests/comm.rs`
13 / 13 and `tests/ar_norm.rs` 4 / 4 under both settings, both negative controls (LL without the tag check, the wide path without its
wait) still catch the missing handshake.

**Verdict: WIDE stays on for the 35B.** Prefill gains at every length, ids and texts are identical. Decode at 1k / 4k is within 0.1 %.
The 16k decode median is 1.4 % lower, but decode never runs the wide path: a decode or MTP-verify all-reduce of the 35B is at most
16 tokens x 2048 x bf16 = 64 KiB, the LL limit. The WIDE=0 rounds scatter more (179.1-193.4 vs 178.6-179.6), and their best round is
0.3 % off. So no code change; a 16k decode re-check is cheap if the coordinator wants one.

### PF27b's one-off max-batch 16 divergence

27B on main, `--max-batch 16`, stream A (200 greedy tokens) with a 4k-0 prefill admitted beside it (`scripts/pf27/mixids.py`, cold and
warm pair per server), 5 fresh servers with the default prefix reuse and 5 with `INTARSIA_PREFIX_REUSE=exact`: **20 / 20 pairs gave
PF27b's majority text** (A `915f1c0953`, B `dd42cb5b78`; `raw/w1/mixids-*`). Not reproduced. (The coordinator then asked for lane PRB's
probe on HIP: HIP had BI8's prefix-reuse bug for dense models, fixed in PR #234, `docs/results/2026-10-29-arx-hip-prefix-reuse.md`; the
27B's recurrent state never takes that cut, so #234 does not explain PF27b's case either.)

## Phase 2: the row-parallel GEMM pushes its partial (`INTARSIA_HIP_AR_PUSH=1`, opt-in)

### Attribution (27B 4k prefill, 3858 tokens, rocprofv3 kernel trace, rank 0)

The 256 row-parallel W4A8 GEMMs per 4k prefill (attention o_proj, GDN out_proj, FFN down; `a8_direct_kernel`) take 129.5 ms, and the
wide all-reduce after each takes 207.4 ms (median 822 us, about 25 GB/s): the all-reduce is serial after its GEMM on the one hardware
queue (`GPU_MAX_HW_QUEUES=4` is not an option: about -20 % decode on the R9700).

### Design

- `Comm::prepush(partial)` (core, default no-op): the model calls it right before a row-parallel GEMM whose all-reduce follows
  (`proj_then_reduce`, only on an eager pass on stream 0, never while a plan is recorded or a graph captured, never with a TP test hook).
- HIP `P2pComm::prepush` arms the next `matmul_mxfp4_a8_mma` launch whose output is exactly that partial (bf16, > 64 KiB, <= one 20 MiB
  wide slot) with the peer's scratch slot of the next wide launch (a host mirror of the wide launch counter gives its parity).
- The GEMM (`a8_direct_kernel<..., PUSH>`, `kernels/hip/mxfp4_a8_gemm.hip`) runs as a persistent grid (as many blocks as are resident,
  walking the tiles bx-fastest). It stores every C word as before and also into the peer's slot in fragment order (`kernels/hip/ar_push.h`:
  each wave's 32 lanes write 512 contiguous bytes). Each tile's words are kept in registers and pushed spread over the k stages of the
  block's next tile; the last tile pushes in its epilogue; one system fence per thread at the end.
- The all-reduce then launches `ar2p_kernel` (`kernels/hip/allreduce.hip`): the wide path's counters, flags and slot parity, no push,
  wait for the peer's flag, add the peer's words mapped back to row-major. Bitwise `ar2_kernel<0>`'s own + peer sum. A launch that did not
  push (staged kernel at M <= 64, N % 8 != 0, too large) leaves the arm spent and the all-reduce pushes itself.

What did not work, on the way (trace, rank 0, GEMM + its all-reduce):

| variant | GEMM ms | AR ms | 4k prefill tok/s |
|---|---:|---:|---:|
| main (push off) | 129.5 | 207.4 | 4280-4353 |
| push in each epilogue, normal grid (m2) | 339.4 | 24.6 | 4167 |
| persistent grid, one fence (m4) | 265.9 | 23.6 | 4505 |
| words pushed during the next tile, read back from `out` (m8) | 238.3 | 22.8 | 4662, **wrong** at some M |
| words pushed during the next tile from registers (m14, this PR) | 235.6 | 23.1 | 4665-4707 |

The read-back variant passed the first bitwise test (9 shapes) and the 4k texts, but served 16k-0 and 16k-5 changed. A det-tap of the
served run put the first difference in the last prefill chunk (1144 rows), layer 0. A new M-sweep test (27 Ms, K 3072) then failed at
M = 511, 767, 1023, 1144, 1147, 1599, 2047. A histogram of the wrong elements matched exactly the partial tiles pushed one tile late
(32 / 48 / 64 / 72 of 80 feature tiles for grids of 96 and 32 blocks): the reload of the thread's own `out` stores did not reliably see
them. Keeping the words in registers (8 uint4 per thread, statically indexed) fixed every M.

### Tests (`crates/intarsia-backend-hip/tests/ar_push.rs`, two R9700s)

- `pushed_all_reduce_is_bitwise_the_plain_pair`: 9 shapes (27B prefill chunks 2048 / 1810, the cases the push must decline); the push
  group without `prepush` is the plain pair; with it, the push is taken exactly where expected, both ranks equal, bitwise the plain pair.
- `pushed_all_reduce_is_bitwise_the_plain_pair_over_m`: 27 Ms (65-2048, every tile cutover and served 16k remainder) at N 5120, K 3072.
- `pushed_all_reduce_survives_a_delayed_rank`: 30 rounds, one rank delayed before a third of its GEMMs, interleaved with plain wide
  all-reduces on the same buffers: 0 wrong.
- negative control `pushed_all_reduce_without_the_wait_is_caught_by_the_stress_test` (drain 9: the reduce without its wait): 94 / 102
  wrong results in the runs, so the stress test sees a missing handshake.

### A/B (W3, served, localeval; main = `200be3f05`, push = this branch with `INTARSIA_HIP_AR_PUSH=1`)

Prefill tok/s, median over the rounds (MTP: 3 interleaved rounds per arm; spec off: 1 round per arm):

| 27B tp2 | main | push | change |
|---|---:|---:|---:|
| prefill 1k, MTP | 3968 | 4206 | +6.0 % |
| prefill 4k, MTP | 4313 | 4725 | +9.6 % |
| prefill 16k, MTP | 4000 | 4261 | +6.5 % |
| prefill 1k, spec off | 3990 | 4216 | +5.7 % |
| prefill 4k, spec off | 4355 | 4756 | +9.2 % |
| prefill 16k, spec off | 3978 | 4292 | +7.9 % |
| decode 1k / 4k / 16k, spec off | 55.8 / 54.9 / 52.0 | 55.8 / 54.9 / 52.0 | 0 |
| decode 1k / 4k / 16k, MTP | 92.9 / 86.7 / 77.4 | 90.5 / 90.6 / 81.1 | -2.6 / +4.5 / +4.8 % (noise, see below) |

Notes:
- The 4k prompt was 4082-4086 tokens in five of the six runs (2048 + ~2036) and 4107 tokens in one run per arm (2048 + 2048 + 11).
  Like for like: about 4083 tokens +8.9 % (main 4358 / 4313, push 4725 / 4725), 4107 tokens +1.9 % (4205 vs 4285, one sample each). The
  CLI (3858 tokens, `raw/m14`): 886 -> 820 ms, +8.1 %. No prompt length was slower.
- Decode never pushes: every decode / MTP-verify partial is <= 4 x 5120 x bf16 = 40 KiB, below the 64 KiB wide limit, so `prepush`
  arms nothing. The cost is a few branches and two uncontended mutex checks per row-parallel projection. Spec-off decode is identical;
  MTP decode scatters ±5 % between rounds in both arms (main 79.5-95.3, push 84.7-97.5) with identical texts, so these runs cannot show
  0.5 %.
- Served texts (`scripts/pf27/texts.py`, 18 prompts x 256 greedy tokens): **18 / 18 identical** push vs main with MTP and **18 / 18** with
  spec off (`raw/w3/texts-*`). The dev binary with the knob off also gives main's 18 / 18 (`raw/w2/texts-4-*`).

### Gates

| gate | result |
|---|---|
| greedy ids, 18 prompts x (spec off, MTP), 256 tokens, push vs push off | **36 / 36 identical** (`raw/w3/gates-13/ids-*`) |
| served texts vs main (MTP, spec off) | **18 / 18** and **18 / 18** |
| GEMM hashes (`tests/mxfp4_a8.rs` `a8_mma_hashes`, main vs branch) | **48 / 48 identical** |
| `--max-batch 16`, a prefill admitted beside a decode (`mixids`, 4k-0 / 16k-3 / 4k-3) | identical to main (A `915f1c0953`; B `dd42cb5b78` / `e74f590855` / `537fb01934`) |
| push tests incl. the M sweep and the delayed-rank stress; negative control | 4 / 4 pass; the control finds 94-102 wrong results |
| decode within 0.5 % | spec off: identical. MTP: not measurable (±5 % between rounds in both arms); decode never pushes |
| prefill +5 % at 4k | **met**: +9.6 % (MTP), +9.2 % (spec off); 1k +6.0 %, 16k +6.5 % |
| no prompt length slower | met (1k, 4k, 16k, MTP and spec off) |
| 120-process soak byte-identical | **met** (ARX2, below): 120 / 120 taps byte-identical to each other and to main's |
| Mac `cargo test --workspace --release` | 205 suites, 1090 passed, 0 failed (also after merging main `ebf8192a0`, #235); after merging `bfbee3cb6` (#237): 206 suites, 1096 passed, 0 failed |
| merged tree on tatooine (m15 #6795) | HIP build ok; `tests/ar_push.rs` 4 / 4; 4k CLI prefill 886 -> 813 ms with push (+9.0 %) |

**Verdict (ARX2 below): still opt-in (`INTARSIA_HIP_AR_PUSH=1`, default off), draft PR.** The soak and the 27B gates now pass; the default-on rule also asks for
35B and spec-off decode medians within 0.5 % and prefill not slower anywhere, which the GPU budget could not resolve (see Phase 3).

## Phase 3 (ARX2, 2026-10-29): soak, 27B MTP A/B with a paired nonce, 35B, the 4107-token prompt

Merged `origin/main` (`69fc170ed`, #234 / #235 / #237 / #238) into the branch (LOG.md conflict only: both rows kept in time order); Mac
`cargo test --workspace --release`: 206 suites, 1098 passed, 0 failed (fleetq #6841). Tatooine tree `~/intarsiaml-arx2`; main = `arx-base-cli`
(`origin/main`), push = `arx-dev-cli` + `INTARSIA_HIP_AR_PUSH=1`. Scripts: `scripts/arx/` (now for the `-arx2` tree), new `leval.py` (localeval
with a seeded request nonce: both arms see the same prompts in the same order, so MTP's text-dependent acceptance cancels; before, the uuid nonce
made MTP decode scatter +-5 % between rounds), `p4107.py`, `speedtab2.py`. Raw: `raw/x2w` (window 1), `raw/x2w2` (window 2), `raw/x2c1`, `raw/x2c2`,
tables `raw/x2-speedtab.txt`.

**Soak (window 1, `raw/x2w/gates-1`):** 120 fresh tp2 processes of the dev CLI with `INTARSIA_HIP_AR_PUSH=1`, prompt `4k-0` (2610 tokens), MTP k2, 16
tokens, `INTARSIA_DET_TAP`: 120 / 120 rc 0, 1 distinct id line, **120 / 120 taps byte-identical**, equal to main's (3 base processes, tap md5
`c006426589a15146614139e4dca54ffb`, the same as PF27b's soak). Push was taken: prefill of the 2610 tokens 917-939 ms (median 929) with push vs 979-982 ms
on main (-5.4 %).

**27B MTP A/B** (5 interleaved rounds per arm, order alternating; the first main round ran in window 1, the other 9 runs in window 2; median of the
per-run medians, [min-max] over the 5 runs):

| 27B tp2, MTP | main | push | change |
|---|---:|---:|---:|
| prefill 1k (tok/s) | 3983 [3378-3993] | 4242 [3738-4258] | +6.5 % |
| prefill 4k | 4374 [4330-4384] | 4741 [4450-4748] | +8.4 % |
| prefill 16k | 3969 [3940-3976] | 4246 [4220-4248] | +7.0 % |
| decode 1k | 93.0 [93.0-97.5] | 93.3 [93.0-95.3] | +0.3 % |
| decode 4k | 88.3 [86.7-88.3] | 88.4 [88.3-90.0] | +0.1 % |
| decode 16k | 75.1 [74.9-78.2] | 75.0 [74.9-79.1] | -0.1 % |

With the paired nonce MTP decode is steady (rounds 93.0 / 88.3 / 75.0 in both arms except round 4, where both arms read high, 95.3-97.5 / 79 / 78). The
decode medians are within 0.3 %. Prefill is faster at every size in every round (the slowest push round, round 4's 1k at 3738, is still above main's
3378 in the same round; main's low rounds are a clock/ordering effect, not push). Served texts (256 greedy tokens, `short-0`, `4k-0`, `16k-3`):
identical push vs main with MTP: `f01c64c1b9ce`, `a36fcaedc82c`, `dec43e3a9546` (the earlier 18 / 18 and 36 / 36 runs were on the pre-merge head).

**The 4107-token prompt** (localeval's 4k filler at 182 sentences = 4106-4108 tokens, chunks 2048 + 2048 + ~11; 181 sentences = 4084 tokens), 3 requests
per arm in each of 3 rounds (`raw/x2w*/p4107-*`), prefill tok/s, median of the 3 round medians: 182 sentences main 4135 (4120 / 4135 / 4141), push 4465
(4465 / 4457 / 4568) = **+8.0 %**; 181 sentences main 4322, push 4698 = **+8.7 %**. So the earlier +1.9 % was one noisy sample: the 4107-token prompt
gains as much as the 4084-token one (the tiny third chunk costs both arms about 3 %).

**35B tp2 (NVFP4)** (`raw/x2c1`, `raw/x2c2`; 3 interleaved rounds per arm for the default spec, 1 pair spec off): the push is a no-op here (the 35B
does not use `matmul_mxfp4_a8_mma`, the only GEMM that arms the push), so the differences below are noise:

| 35B tp2 | main | push | change |
|---|---:|---:|---:|
| prefill 1k / 4k / 16k, default spec | 8849 / 12388 / 11718 | 8791 / 12493 / 11718 | -0.7 / +0.8 / 0.0 % |
| decode 1k / 4k / 16k, default spec | 190.9 / 188.3 / 179.4 | 191.3 / 187.3 / 179.0 | +0.2 / -0.6 / -0.2 % |
| prefill 1k / 4k / 16k, spec off (1 pair) | 9022 / 12646 / 11437 | 8931 / 12523 / 11773 | -1.0 / -1.0 / +2.9 % |
| decode 1k / 4k / 16k, spec off (1 pair) | 191.7 / 180.9 / 190.8 | 190.8 / 188.8 / 178.9 | -0.5 / +4.4 / -6.3 % |

The 35B's own decode reps scatter 175-192 tok/s (peaks to 242) in both arms, so 0.5 % is not resolvable on it with these rounds. Served texts
(256 greedy tokens, `short-0`, `4k-0`, `16k-3`): identical push vs main with the default spec and with spec off: `bb1f3acae6de`, `f71dc2314b31`,
`690670ba1f7e`.

**Default-on rule:**

| gate | result |
|---|---|
| soak 120 / 120 identical | **met** |
| ids identical, 27B and 35B | met on the 3 served prompts per model and mode run here (27B MTP, 35B default spec and spec off); the 27B spec-off texts were not re-run on the merged head (18 / 18 on the pre-merge head) |
| decode medians within 0.5 %, MTP and spec off, both models | 27B MTP **met** (+0.3 / +0.1 / -0.1 %); 27B spec off **not re-measured** on the merged head (W3: identical, push never runs in decode); 35B default spec -0.6 % at 4k (noise: the push does not run on the 35B), 35B spec off one pair at +-4 % scatter: **not resolved** |
| prefill not slower anywhere, both models | 27B **met**; 35B -0.7 % at 1k (default spec) and -1.0 % (spec off, one pair): noise on a no-op, but **not strictly met** |

**Decision: stays opt-in, draft.** The strict rule fails on the 35B (noise on a path the push does not take) and the 27B spec-off decode was not
re-run within the GPU cap (window 1 lost about 10 minutes: four items were built with the zsh expansion `$P:speed`, which is a history modifier, so
the server of item 5 got a binary named `arx-dev-clieed` and waited out its health timeout; the window was stopped at 04:24 and window 2 ran the
remaining 27B items). If the coordinator accepts "the 35B and spec-off decode cannot differ because the push never runs there", the flip is one line
(`INTARSIA_HIP_AR_PUSH` default 1 in `comm.rs:317`, `0` opts out, tuning.md row, scoreboard 27B prefill/TTFT rows) and the numbers above (27B prefill
+6.5 / +8.4 / +7.0 %) are the ones for the scoreboard.

## Coordinator no-op check: #237 (row-invariant CUDA prefill) on HIP

Main `bfbee3cb6` (#237) vs `200be3f05`, both built on tatooine (fleetq #6812, both cards, `scripts/arx/noop-237.sh`):
- `plan show --json`: Qwen3-8B BF16 (one card) 8343 / 8343 bytes and Qwen3.6-35B-A3B NVFP4 tp2 13541 / 13541 bytes, both byte-identical
- 35B tp2 greedy ids, spec off, 256 tokens: short-0 and 4k-0 identical (`raw/noop-237/`)
- this branch merged with `bfbee3cb6` (m16 #6817): HIP build ok, `tests/ar_push.rs` 4 / 4, 4k CLI prefill 886 -> 813 ms with push, 27B
  greedy ids push vs off identical for 4k-0 and 16k-0 (spec off and default)

## GPU jobs

Both cards in one fleetq job, Radiance down and restored (`curl tatooine:8080/v1/models` answered after every job). Windows:

- W1 #6711 (23:49-00:09, 35B A/B, ids, comm tests, mixids repro). Exit 2 came from `drive.sh`, which was edited on the Mac while it ran,
  after the last item; Radiance was restored by the guard.
- W2 #6762 (01:02-01:29, 27B push A/B, read-back variant). A first W2 attempt #6759 was stopped after 2 min (its item list was split
  wrongly by zsh); Radiance was restored by #6760.
- W3 #6784 (01:55-02:29, register variant: A/B, texts, ids, hashes, mixids).
- ARX2: window 1 #6849 (04:03-04:24, soak 120 + the first 27B round; stopped by SIGTERM, Radiance answered after 20 s), window 2 #6857 (04:25-04:42, 27B MTP rounds,
  p4107), check 1 #6860 (04:43-04:44, 35B default spec round 1), check 2 (04:45-04:48, 35B spec off, rounds 2-3). Builds #6843 (dev CLI + `ar_push`) and
  #6842 (main CLI), Mac tests #6841. Radiance answered on :8080 after every job.

Short checks (< 3 min each, both cards): m1 #6725, m2 #6726, m3 #6732, m4 #6733, m5 #6735, m6 #6738, m7 #6743, m8 #6749, m9 #6766,
m10 #6771, m11 #6773, m12 #6775, m13 #6777, m14 #6779, m15 #6795 (merged tree), #6812 (#237 no-op), m16 #6817 (merged with #237). Builds and Mac tests ran as `--load cpu` jobs.

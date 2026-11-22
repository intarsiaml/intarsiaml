# Inkling-2-Mini speculative decoding with its DSpark draft (lane ink-mtp, 2026-11-21)

Status: in progress. Branch `ink/mtp` (not pushed, not merged).

## Briefs received

- 2026-11-21 (coordinator, SendMessage): Fable's decode review ranks this lane #2 (`.worktrees/ink-fable/decode-profile-review.md`,
  item 2, sections 3.5 and 4; read section 4 before designing the verify path).
  1. Shape: short confidence-gated drafts, 2-3 verify rows; the drafter rows share the verify's unembed launch.
  2. Estimated saving 3.0-4.4 ms of the real 15.76 ms step at acceptance 0.6-0.7, about 6 ms if consecutive tokens share experts.
  3. #10743 decides it: report acceptance by draft position and the expert-union size for 2 and 3 consecutive tokens
     (union over the 8-of-128 per layer) as soon as available.
  4. Device-side accept / verify readback is ink-hostgap's job: send interface needs to the coordinator, who relays.
  5. Measure Metal decode with `--profile-every 0` (real greedy about 63.4 tok/s; the default profiler adds about 4 ms/token).

- 2026-11-21 (coordinator, SendMessage): endor's queue is about 5 h deep. Move the real-model CPU token-identity job (#10807) to
  spark2's CPU (released on endor, resubmitted there); keep the Metal decode tok/s job (#10806) on endor. Every other CPU-only
  job of this lane goes to a Spark.
- 2026-11-21 (coordinator, SendMessage): priorities in order. (1) Fold the drafter's base-logit read into the verify's unembed
  (Fable §4.3; a separate 922 MB read is about 1.6 ms a cycle). (2) Move the tap mixes and the top-64 Markov head to the device;
  send the interface needs (what the device hands back per cycle: accepted count, next ids, draft confidences) after the
  per-cycle profile. (3) Tune the policy on a mixed set (greedy golden + prose + chat turns), report tokens per pass and real
  tok/s per policy; the default must never be slower than spec off on prose; the p1 >= 0.6 gate is the right shape.
  (4) Sampling: confirm whether temperature > 0 uses exact speculative rejection sampling; if not, keep spec greedy-only and
  say so. Speculation stays opt-in until #10806's timed numbers; default-on is the coordinator's call. If drafter changes land
  before #10806 starts, edit its script rather than queue a new job (within its --expect).
- 2026-11-21 (coordinator, SendMessage): §4.3 agreed circular for the anchor; the separate MXFP4 draft head is the fix, its
  precision a measured choice in #10806. Ownership: this lane's drafter device ops ((c) top-k ids / values + p1, the tap mixes)
  take discriminants 92-95; ink-hostgap owns (a) device argmax over the R verify rows with the accepted length and (d) the single
  per-cycle readback. Send the interface after the #10806 profile (relayed to hostgap). Remove the 7 x V head download + host
  top-64 sync first ((c) on device). Greedy-only DSpark is fine: say so in the doc and the serve / CLI `--spec` help.
- 2026-11-21 (coordinator, SendMessage): endor's quiet-GPU line is about 9 h. Move the token-identity checks out of #10806 into
  a default-GPU-load job with a short --expect (backfills); keep #10806 for tok/s plus the DSPARK_PROFILE run. ink-metal fixed
  the rows_exact pin-list test at a0751f302: merge ink/metal.
- 2026-11-21 (coordinator, relay from ink-hostgap; their design note docs/results/2026-11-21-inkling2-hostgap.md on ink/hostgap,
  6855e486a, nothing built): a DSpark cycle has 3 host syncs, S1 ingest (taps down, host mix, upload per layer), S2 draft head
  (base-logit download, host top-64, host Markov + gate), S3 verify (argmax readback, accept). hostgap can provide H1 (one 40 B
  per-cycle record [n_acc, bonus, k_eff, drafts[7]]) and H2 (encode cycle c+1 ahead, discard on reject). Removing the syncs needs
  this lane's M1 (device ingest mix), M2 (device Markov head + gate), M3 (verify always at kmax+1 rows + an accept op), M4
  (device positions for Inkling's ops). Ask: report per-cycle host ms split S1 / S2 / S3 from INTARSIA_DSPARK_PROFILE=1 (a short
  default-load job if #10806 lacks it); under about 1 ms per cycle in total drops M1-M4. And: may hostgap add
  INTARSIA_HOST_PROFILE hooks (profiling lines only) in the spec loop; name the file to keep out of.

## 1. Acceptance (Python oracle, #10743 on spark2's CPU)

`tools/inkling2_dspark_ref.py` (modes `taps`, `draft`, `analyze`, `tiny`) runs the target oracle (`inkling2_ref.py`, W4A16,
fp32 math) over fixed token streams. It saves the four DSpark taps per position (the partial entering visits 5, 23, 41, and the
final-norm output), the target argmax and the routing. It then drafts at every anchor exactly as vLLM's decode step does: context
K/V from the taps of positions < t, block = anchor + 7 masks, f_rnn from position t-1, 7 sequential Markov steps.

Streams: vLLM's greedy outputs on the 12 greedy golden prompts (2564 anchors on the target's greedy path) and the two human prose
texts of the long golden (875 anchors). A draft prefix counts as accepted when every draft equals the target's argmax and the text
stays on the target's path. Off-path anchors (where vLLM's W4A4 near-ties differ from the W4A16 oracle) are reported separately.
Raw output: `docs/results/2026-11-21-inkling2-mtp/analysis.{md,json}`.

P(at least j drafts accepted):

| set | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy golden | 0.722 | 0.521 | 0.384 | 0.293 | 0.230 | 0.189 | 0.152 |
| prose | 0.408 | 0.158 | 0.067 | 0.030 | 0.009 | 0.002 | 0.000 |
| all on-path | 0.642 | 0.428 | 0.303 | 0.226 | 0.174 | 0.141 | 0.113 |

On greedy, the acceptance of position j given j-1 is 0.72, 0.72, 0.74, 0.76, 0.79, 0.82, 0.80. Calibration of the position-1
top-1 probability (greedy, bucket: accept rate): <0.3: 0.17, 0.3-0.5: 0.29, 0.5-0.7: 0.43, 0.7-0.8: 0.53, 0.8-0.9: 0.62,
0.9-0.95: 0.75, >=0.95: 0.95 (1382 of 2564 anchors).

Expert union per visit over R consecutive rows (real routing, 8 of 128 per visit):

| R | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVFP4 visits | 8.0 | 12.5 | 16.0 | 18.9 | 21.4 | 23.7 | 25.7 | 27.6 |
| MXFP8 visits | 8.0 | 12.6 | 16.3 | 19.2 | 21.7 | 23.9 | 25.9 | 27.7 |
| independent routing | 8.0 | 15.5 | 22.5 | 29.1 | 35.3 | 41.1 | 46.5 | 51.6 |

Consecutive tokens share experts strongly: at R = 2 the second row adds 4.5 new experts, not 7.5.

Tokens per verify step (accepted + 1) and modelled speedup. The cost model is the m-moe expert timings at the real union, the
m-attn attention per row, the dense bytes at 480 GB/s plus 1.5 ms fixed (14.4 ms plain step), and the native drafter (block 0.52
GB, base logits 0.92 GB, fold 0.1 GB per step). "Cheap" is a 0.5 GB drafter (perf-strategy §6). These are modelled numbers:
the measured ones are in section 3.

| policy (greedy) | mean k | tokens / step | speedup | cheap drafter |
|---|---:|---:|---:|---:|
| fixed k=1 | 1.00 | 1.722 | 1.23 | 1.38 |
| fixed k=2 | 2.00 | 2.242 | 1.46 | 1.64 |
| fixed k=3 | 3.00 | 2.626 | 1.59 | 1.78 |
| gate p1 >= 0.6, kmax 2 | 1.42 | 2.121 | 1.46 | 1.64 |
| gate p1 >= 0.6, kmax 3 | 1.94 | 2.460 | 1.62 | 1.83 |
| gate p1 >= 0.8, kmax 3 | 1.54 | 2.298 | 1.57 | 1.77 |
| gate p1 >= 0.6, kmax 4 | 2.37 | 2.720 | 1.74 | 1.96 |
| gate p1 >= 0.6, kmax 7 | 3.29 | 3.223 | 1.95 | 2.21 |

On prose every policy is 0.99-1.09x modelled (fixed k=3: 0.99; gate 0.6 kmax 3: 1.07). The gate keeps prose from going below 1.

## 2. The Rust drafter

`--spec dflash --spec-draft <Inkling model dir>`: the loader (`intarsia-models/src/load.rs:open_dspark`) sees `mtp_config` +
`mtp.safetensors` and loads the draft as a DFlash-family head with `DflashConfig::dspark`. The engine's DFlash verify, accept
and rollback path is used unchanged. The Inkling refusal of drafters now admits only this draft (MTP and other DFlash drafts are
still refused). Code: `crates/intarsia-core/src/model/dspark.rs`.

- Taps: `ink_pass` copies the partial entering visits 5, 23, 41 and the final-norm output into the DFlash staging buffers
  (`model/inkling2.rs`). This is vLLM's `aux_hidden_states`: `stack[num_blocks]` at those visits, none of which is a pass
  restart, plus the output-norm hidden.
- Context ingest (after every forward, the accepted rows only): taps downloaded, the four-tap softmax(fuse_weight) mix +
  dflare RMSNorm on the host (bf16-rounded), uploaded, `k_proj_ctx` / `v_proj_ctx`, `k_norm_ctx` + neox RoPE (theta 1e6),
  appended to per-layer draft caches. `f_rnn` rows go to a 64-row host ring.
- Block: `[embed_norm(embed(anchor)), mask_embed x 7]` at positions P..P+7. Pre-norm layers with q/k norm + RoPE, block K/V
  appended after the context. Each row is one `AttnDecode` at P+7 (every block row sees all 8 block keys: bidirectional);
  layers 0 and 2 use a window of 1024 + 8 keys. SwiGLU MLP, final norm.
- Head: base logits of the mask rows through the draft's own MXFP4 copy of the target unembed (`INTARSIA_DSPARK_HEAD`),
  device top-64 per row (`Backend::topk_rows`, new on Metal, `kernels/metal/topk.metal`). The rnn_proj hidden / f_rnn parts
  run as two small GEMVs. The rank-256 recurrence runs on the host with vLLM's bf16 rounding points. For each step, the Markov
  bias `fold(latent)` is evaluated on the 64 candidates only. fold = unembed . hidden_out, folded at load: 200058 x 256,
  4.7 s on endor's CPU.
- Gate: stop at the first step whose top-1 probability (softcapped logits over the candidates) is below
  `INTARSIA_DSPARK_TAU` = 0.6. An empty draft is a plain step.
- Opt-ins: `INTARSIA_DSPARK_CAND=0` (exact full-vocab head), `INTARSIA_DSPARK_LAYERS=fp8`, `INTARSIA_DSPARK_DEVICE_TOPK=0`.
- Greedy only: `--spec dflash` never speculates on sampled requests (`engine.rs`, sampled speculation is `SpecMode::Mtp`
  only), so a temperature > 0 request takes plain steps and its distribution is unchanged. The CLI / serve `--spec-draft`
  help says so.

Fable §4.3 (drafter rows sharing the verify's unembed) is circular for this draft. The block's row 0 is the embedding of the
target's argmax at the last accepted verify row, and that argmax exists only after the verify's unembed; attention spreads
row 0 into every mask row from layer 0 on. The separate MXFP4 draft head cuts the read from 0.92 to 0.24 GB per cycle
instead. Emitted tokens are still the bf16 verify argmax.

Tests:
- `crates/intarsia-models/tests/inkling2_dspark.rs`: tiny target + random 3-layer DSpark, `golden/inkling2-tiny-dspark`,
  written by `tools/inkling2_dspark_ref.py tiny`. Greedy ids equal spec off at k 1 / 3 / 7 and max_tokens 1 / 2 / 5 / 40;
  scripted partly-right drafts give every accepted length. Passes on cpu and Metal.
- `crates/intarsia-backend-metal/tests/topk.rs`: `topk_rows` equals `sampler::topk_row` (ties, -inf, NaN, V 200058,
  f32 / bf16).
- Real-model token identity: section 3.

RESULTS_PLACEHOLDER

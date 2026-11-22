# Inkling-2-Mini batched decode (lane ink-batch, 2026-11-21)

Batched decode of 2..=16 sequences in one step (`Model::forward_batch`) for Inkling-2-Mini, CPU reference and Metal.
Branch `ink/batch` from `ink/metal` 1dacac3a8.

## Briefs received

- 2026-11-22 (main): endor checked 06:33-06:40: no tensorfold process, kev idle, no thermal / performance warnings, AC high
  power, no other model server; swap nearly full (21.07 / 21.5 GB), which fits the system page-ins but not our slowness
  (0 major faults, slow in low-page-in runs too). (1) Let #11297 finish; its numbers are labelled "box-slow, not speed".
  (2) Control in the same slot if --expect covers it, else a fresh quiet slot: #11261's exact binary, args and input
  type, solo + B16, 1 rep. If it reproduces ~61-65 tok/s solo and 80-88 ms B16, the box is fine and the slowdown is
  #11297's config: find which. If slow too, the box is slow: release and requeue #11297's speed half later. (3) #11309
  (correctness) stays queued either way; hold #11306's timing until the control answers, and if the box is slow requeue
  #11306 rather than run it. Record the control result and this rule here.

- 2026-11-22 (main): batched-regression check is mine. (1) Wait for #11266's synthetic batch_step x3 on c918e9fda: B16
  within 5% of 59.96 ms = no regression, #11261 a bad slot, stop. (2) If confirmed: runtime switches in one binary on the
  final tree, random prompts, #11181 protocol, one quiet slot, pages warm, order alternated, 8546d143f binary as reference
  arm; suspects (a) fp8 screened greedy head, (b) dfuse controller switches at B>1, (c) hostgap host pool / argmax_v4t
  (INTARSIA_HOST_THREADS=1, INTARSIA_METAL_ARGMAX_V4T=0); env names from docs_env_vars. (3) Only if no switch explains it,
  bisect the merges in one slot. Report B8/B16 per arm and the culprit with its switch; fix/revert is main's call. May fold
  into #11297 if long enough, else queue right after. Later from ink-metal: the combined gate on 7e5e6010e (pre-pack12,
  not quiet) logged B 2/5/16 at 23.5 / 27.7 / 74.9 ms, so B16 matches #11261 and any regression predates pack12 (fp8,
  prefill, dfuse or hostgap); consider a B16-only path (screened head, per-row controller loop). ink-metal sends #11266's
  B1/8/16 (and its pack12 on/off arm) when it lands.

- 2026-11-22 (main, priority, CPU only): ink-mtp suspects batched-decode sampler readbacks skip the final logit softcap
  (model.rs batch-head top-k path ~2594-2612: LastTopK vals and LastLogits rows; model/batch.rs; softcap only in head_logits
  and graph_decode_step). (1) Test first: batched rows, top-k values and full logits, bitwise == one-sequence decode's capped
  logits after the same prefill, tiny Inkling cap 30 (shape: ink/mtp-softcap 36cbb9536 tests/inkling2_verify_softcap.rs).
  (2) If it fails, fix the same way, sharing Model::softcap_step from that branch (cherry-pick, no duplicate); batch tests +
  G1 on CPU (bespin or spark2). (3) Report: reproduced yes/no with max |d|, fix commit, exposed serve paths and models.
  rsync -a keeps mtimes: use cp or touch.

- 2026-11-22 (main): the steady auto B16 drop across reps (87.7 / 84.4 / 80.4 ms) looks like warm-up: expert weights may be
  mmap-backed and real text (~63 experts/call vs ~28) pages in never-touched or evicted experts. Check whether weight
  buffers are mmap'd and resident before timing (page-in / fault counts, vm_stat pageins before / after each rep). If so,
  rerun the paired comparison next quiet slot: one full untimed real-text warm rep across all B, then 3 paired reps; report
  pairs within each rep and the mean paired difference. Default-policy decision waits; auto stays.

- 2026-11-22 (main): main killed an orphaned `intarsia serve --port 8131` from this worktree (pids 85506/85510, ppid 1, 11 GB
  RSS, up since ~02:42 EST, no fleetq job) inside slot #11257. (1) Serve job scripts must kill the whole process group
  (set -m or setsid, `kill -- -$PGID` in an EXIT trap) and check the port is closed before exiting. (2) Note next to every
  quiet endor result since ~02:42 (incl. #11181) that it ran with that idle 11 GB server resident; re-measure only if
  borderline.

- 2026-11-22 (main): stop accepted; (a)+(b) approved. (a) batch_step and batch_prof take real-text prompts: fixed,
  deterministic tokenized corpus slice, a different slice per row, same slices every run; record expert union per MoE call
  (mean and max per B); keep random-id mode behind a flag. (b) same quiet job: INTARSIA_INK_MOE=auto vs union-fused at
  B 1/8/16, median of 3, greedy md5 per stream equal across arms. Build and test the bench change now (bench-only). Queue the
  quiet job only after ink-metal's combined gate #11206 passes, on the final ink/metal tree, ~40 min. After it: pick the B16
  candidate from a trace at the realistic union (same rules: >= 6 ms real build, 3-6 report, < 3 drop). If union-fused wins
  at B>1 bitwise, propose flipping Auto with the numbers.

- 2026-11-22 (main): MoE fixed-launch candidate approved. Step 1 profile: queue after ink-metal's combined gate passes, on the
  final ink/metal tree; one quiet slot (~30 min), kernel trace B 2/4/8/12/16, MoE time vs union size. Decide on the real
  (untraced) B16 fixed part: >= 6 ms build fused MoE launches (gate+up and down per layer over union experts in one grid,
  gather folded into down, per-row order unchanged); 3-6 ms report to main first; < 3 ms drop and report in_proj with its
  estimate. Before editing check file overlap with ink-prefill (MoE NVFP4 prefill, #11187) and ink-dfuse (ink_route preload,
  #11190); if the same MoE dispatch files, tell main before writing code. Build gate: rows bitwise vs pre-change tree at
  B 1/8/16 (greedy md5 per stream), Metal suite green, admission OK, quiet B 1/8/16 median of 3 showing the gain. Two
  attempts without gain: stop and name the broken assumption. conc-plain-1k prefill interleave: not mine yet, raise after.

- 2026-11-22 (main): #11181 accepted; NE 3 costs nothing batched, RB16 stays off, B16 slowdown closed as not reproducible;
  skip the paired pre-merge vs merged run. (1) Send ink-metal (af43898d63e5b5190) the exact B=1/8/16 batched-step command for
  its combined post-merge gate (that number becomes the reference). (2) Propose the next batched-decode candidate for B=16
  (~60 ms/step, ~267 tok/s), with a per-kernel breakdown of the B16 step if available and the estimated gain.

- 2026-11-22 (main): move #11174 (NE/RB speed A/B) behind ink-metal's NE fix gate #11179 (fix 7eca1b012, NE 3 fallback at
  D 2304): release it, requeue the same script, flags and log, update the monitor.

- 2026-11-22 (main): bisect names m-mix's NE 4 from D 2048 (25ff7864a) as the G2 decode culprit; NE=3 restores base KL;
  BLEND_RB and InkMixP cleared for accuracy; ink-metal is fixing. Do not requeue the five-arm b8reg; queue ONE trimmed quiet
  job on the merged tip: tip default | tip INTARSIA_INK_MIX_NE=3 | tip NE=3 + BLEND_RB=16, B=1/8/16, 3 reps. Report #11104's
  rtvar correctness when it finishes, then the trimmed table.

- 2026-11-21 (main, after the design note): design approved, proceed. (1) Memory: count the full-attention KV of the 8
  full visits per sequence at max context as well as the rings; `enable_batch` (and the server) must cap B so everything
  fits and refuse cleanly rather than over-allocate. (2) Solo path: single-stream decode tok/s and G2 short / long
  unchanged before vs after the merge (bitwise logits on the short set preferred). (3) MTP later: one row per sequence now;
  do not design out several rows of one sequence (draft verify across streams), do not build it; one line on what it needs.
  (4) Throughput through localeval (local-model-bench) at 1/4/8/16 streams under `fleetq --quiet gpu,cpu`, oMLX's 94 tok/s
  at 16 streams in the same table.
- 2026-11-21 (coordinator): Metal `generate` defaults to `--profile-every 16` (every 16th step waits on 419 command buffers,
  about 4 ms/token of fake gap; real single-stream greedy is about 15.76 ms = 63.4 tok/s). Use `--profile-every 0` for every
  engine-side decode timing, the solo no-regression check included; localeval through serve is unaffected.
- 2026-11-21 (coordinator): endor's quiet-GPU line is 14+ jobs (about 9 h). Correctness runs (#10810 mixed-context
  batched == solo, the G2-bitwise part of #10795) go as default-gpu-load jobs with short honest `--expect` (they share the box
  and backfill); only tok/s (#10773, the tok/s of #10795, #10811) stays `--quiet gpu,cpu`. ink-metal fixed the rows_exact pin
  list at a0751f302: merge ink/metal.
- 2026-11-21 (coordinator, after 173.7 tok/s at 16): (1) do the 9..16-row single-read work now: dense row GEMVs (in_proj, wo,
  unembed) read the weights once for all rows instead of two 8-row chunks, then the experts read each expert's weights once
  per step over the rows routed to it; target 16 streams >= 230 tok/s, every row still bitwise its solo step (the tiny tests
  and #10844's mixed-context check are the gate); new kernels opt-in by row count first; if a 16-row form cannot stay
  bitwise with the 8-row one, ask before changing the contract. (2) Confirm from code that serve / the engine route a single
  live sequence through the solo path, not a B = 1 batched step (B = 1 batch_step 17.5 ms vs 15.86 ms solo); if not, make
  them. (3) Keep #10844 / #10845 / #10811 queued; tiny / cpu tests for new work on spark2 (`--load cpu --mem 30`, copies in
  ~/intarsiaml-ink-batch); endor only for Metal admission and timing.
- 2026-11-21 (coordinator): `intarsia serve` on Metal refused Inkling (require_admitted: no admitted moe_experts kernel; the
  plan-shape case builder dropped `attrs.ink`, so the SiTU kernels were never admitted). Fixed in ink-metal 761a1561b
  (quant/cases.rs, test `situ_experts_case_keeps_the_ink_attrs`). Merge ink/metal at or after 761a1561b now; #10811's script
  must build from that tip at job start, not use a prebuilt binary.
- 2026-11-21 (coordinator, for the 8-vs-16 profile): ink-m-mix measured InkInProj at the real shape: 1 row 69.8 us, 8 rows
  83.3 us, 16 rows 168 us (241 GB/s, arithmetic-limited; the three-Matmul form is no faster at 16 rows, 260 GB/s), about
  +4 ms per step from 8 to 16 rows over 48 visits. Their idea 3, a matrix-unit GEMV for 9..16 rows bitwise equal to the 1-row
  kernel, is my call after the profile: build it only if InkInProj is the biggest share and it can stay bitwise per row.
  ink-metal is merging ink/m-mix 47570539e (InkInProj up to 8 rows on the GEMV instead of NAX): re-merge ink/metal before the
  next batch timing.
- 2026-11-21 (coordinator): ink/metal now has m-mix 47570539e (merge 606fd36e4, tip f55bfe8b7), no default flips: re-merge
  before the opt-in A/B starts.
- 2026-11-21 (coordinator): when #10845 shows solo tok/s base vs branch within noise (<= 1 % slower, or faster), ink/batch at
  that tip merges into ink/metal (ink-metal does the merge); report #10845 with the tip hash as soon as it lands. A solo
  regression above 1 %: find it before anything else.
- 2026-11-21 (coordinator): hypothesis for the ~17 ms not in GPU time: host softcap 0.67 ms per 200058-wide row (ink-hostgap)
  plus a host full-vocab pass ~0.2 ms+, per row, if the batched step downloads B logits rows and softcaps / argmaxes them on the
  host: 16 x ~1 ms. (1) Check from code what happens per row on the host in a greedy batched step. (2) If so, greedy rows on
  the device (the solo path's softcap + argmax per row, one small readback of B ids). (3) Sampled rows keep host logits; the
  per-row softcap on the host pool waits for hostgap's a2f2b87b9. Report before building; record this brief.
- 2026-11-21 (coordinator, rule for all Inkling lanes): this session and every worktree run ON endor; a local cargo
  build / check / test or shader compile loads endor's CPU and skews every quiet timing job there (a possible cause of noisy
  A/Bs). No local builds or tests on endor outside a fleetq slot. macOS compile checks: bespin
  (`fleetq run --host bespin --load cpu --mem 6 --expect <min>`, a `git archive` copy in ~/intarsiaml-ink-batch, toolchain in
  ~/.cargo/bin; no GPU timing or real model there). CPU tests: spark2. GPU work: endor fleetq jobs. Reading, git and editing
  locally are fine.
  Lane audit: every build and test of this lane ran inside an endor fleetq slot except one `cargo check -p intarsia-core`
  at 21:1x, before any timing job of this lane. Every quiet timing job queued now (#10900 A/B, solo A/B, #10811) builds inside
  its own slot before timing, so nothing outside a slot overlaps it.
- 2026-11-21 (coordinator): in ink-metal #10890 `intarsia bench speed` failed at its calibration call ("no tokens received")
  against the Metal serve, while chat and localeval streaming worked; ink-metal is fixing it. #10946 must not end on that
  failure: merge their fix before #10946 builds (hash to follow) or skip the bench step. Done: #10946 now runs localeval
  `conc-plain-1k` first, then the bench as a non-fatal step with a 30 min watchdog. Solo baseline from #10890: at
  `--max-batch 1` `conc-plain-1k` is serialized, 50.8-51.6 tok/s aggregate at 1 / 4 / 16 streams; the `--max-batch 16` run is
  compared against it.
- 2026-11-22 (coordinator): the `bench speed` calibration fix is ink/metal b9dd0d6b4 (crates/intarsia-bench/src/speed.rs only;
  Inkling's first generated token is a block marker the text stream drops, so a max_tokens 1 probe returned usage only and
  counted as "no tokens received"). Cherry-pick it before #10946 builds; keep the non-fatal bench step. Done: cherry-picked as
  6b5ff520f before #10946 started (it builds the tip at job start); the bench step stays non-fatal.
- 2026-11-22 (coordinator, after #10914): ink-metal merges ink/batch bbc0bec00 into ink/metal after #10901, right after
  ink/m-mix 52b337aa9 (both touch the mix shader), and messages the merge hash. (1) Get the 0.6 % back: from that ink/metal
  tip make the row-table branch compile-time (Metal function constant or a separate pipeline variant), so the solo path
  compiles without MX_ROWTAB; shader / pipeline only (ink-dfuse owns the inkmix.rs row kernels for fusion: touch no more
  than needed). (2) Correctness: rows-exact and batched == solo at B 2 / 5 / 16, run on endor in a default-load slot; macOS
  compile checks on bespin; no local builds. (3) Timing: if the two-pass curve job has not started, fold a solo ABAB
  (merged tip vs variant, 6 reps, --profile-every 0) into it; else one short quiet job. (4) Merge criterion: solo >= the
  merged tip and batched output identical; report the hash.
- 2026-11-22 (ink-metal, coordinator): ink/m-mix merged into ink/metal at 331238a4e, ink/batch bbc0bec00 at 45887198b, tip
  8546d143f (docs only); no textual conflicts. m-mix changed `mix_geom` (NE 4 from D 2048) and `BLEND_RB` 16 -> 4, so the
  row-table mix path now runs at NE 4 (md5s differ from before: reduction order). Merge gate #11048 (default load) reruns
  tiny B 2 / 5 / 16, rows_single, rows_exact, real-model batch_step mixed contexts, G2 / G3. Branch the compile-time
  MX_ROWTAB variant from 8546d143f; its solo ABAB's base arm is 8546d143f (not f55bfe8b7).
- 2026-11-22 (coordinator): pid 53955 on 8099 was an orphan of ink-dfuse's finished #10882; killed (with parent 53944),
  8099 free. Keep the port-pick + owner check; no other change.
- 2026-11-22 (coordinator, relaying ink-dfuse): ink-dfuse's hoist switch (INTARSIA_INK_MIX_HOIST) picked x vs ring from the
  row index instead of the in-call index + xoff under row tables; fixed on ink/dfuse ac2371c81 (contiguous path unchanged).
  Run the row-table rows-exact tests and tiny B 2 / 5 / 16 batched == solo with INTARSIA_INK_MIX_HOIST=1 against ink/dfuse
  ac2371c81 in a default-load slot (folded into #11056 if not started, else after ink/dfuse merges into ink/metal); report
  pass / fail to the coordinator and ink-dfuse (a194e917d8f92b542).
- 2026-11-22 (coordinator, on the B 8 / 16 regression A/B): the same m-mix merge (NE 4, BLEND_RB) is the main suspect in
  ink-metal's decode-mode G2 regression on the merged tip (teacher-forced decode rows 2.6e-3 -> 1.1e-2, drift growing row to
  row: looks like 1-row decode writing bad state); their bisect #11142 runs NE=3 / BLEND_RB=16 arms. Send the five-arm B 8 / 16
  table to ink-metal (af43898d63e5b5190) and the coordinator. If ink-metal reverts NE 4 for accuracy: rerun B 8 / 16 on the
  fixed tip first (a batched-only override may be moot). If accuracy is fine at NE 4 and only batched speed needs NE 3 /
  RB 16: approved, a batched-only override in mix_geom / blend_geom, written by ink-dfuse (a194e917d8f92b542, owns the
  inkmix.rs row kernels now; m-mix lane closed): send dfuse the exact lines, solo defaults stay. Solo rtvar: inconclusive,
  stays unmerged; retry in a later quiet slot only if cheap (0.6 % at best).
- 2026-11-22 (coordinator, relaying ink-hostgap): a side worktree whose target/ was cloned (cp -cR, APFS clone) keeps the
  origin's mtimes, so cargo can keep old crates as fresh and build stale code. If a target was cloned: touch every tracked
  file under crates/ before each build (or cargo clean -p intarsia-core -p intarsia-backend-metal), and each arm's log must
  show those crates compiling; for the HOIST=1 check, confirm the hoisted kernel dispatched.
  Done: `cp -cR` here preserves mtimes (checked). Every side target of this lane was cloned (ink-batch/target from ink-metal
  at the start; base-, rtvar-, dfuse-target from it). The queued five-arm A/B and #11104 now touch all tracked files under
  crates/, kernels/, recipes/ of each tree before its build and grep the `Compiling intarsia-*` lines; #11104's dfuse tree
  carries an uncommitted scratch-only `eprintln` that names the first hoisted mix dispatch with and without a row table
  (tests run with --nocapture). Exposure of past results: ink-batch/target was cloned before this lane's first build, from
  a newer ink-metal tip, so the very first tiny runs (before the 21:5x ink/metal merge) may have mixed in ink-metal code
  (they were all rerun after the merge); #11057's rtvar build (its target cloned at job start, 15 s build) is treated as
  unverified: the A/B's arm C rebuilds it from touched sources.
- 2026-11-22 (coordinator, build hazard 2 from pack12 #11066): `cargo test -p intarsia-cli --test <x>` rebuilds
  target/<profile>/intarsia without `--features metal` over the backend binary; later CLI steps fail ("built without the
  Metal backend") and md5 of two empty outputs reads SAME. Rules: build the backend binary last (or tests in their own
  --target-dir); count md5 / self-golden matches only when both runs exit 0 with non-empty output. Reply only if affected.
  Checked: no script of this lane runs `cargo test -p intarsia-cli`; intarsia-models has no metal feature (the backend is a
  plain dependency), so its test builds do not change batch_step. Past matches all had non-empty output and exit 0 (G2:
  EXIT 0 x 4, tensors compared one by one; generate outputs 1025 / 946 bytes). Not affected. Hardened anyway: #11104 compares
  digests only when both runs produced all three, and the timing job's text comparison requires non-empty files.
- 2026-11-22 (coordinator): (1) when #11104 starts, the coordinator runs ink-metal's G2 bisect inside its slot
  (FLEETQ_JOB=11104, ~25 min, 20 GB, correctness only): do not release #11104 early; extra GPU load during it is expected
  (bitwise / correctness tests only). (2) The five-arm regression hunt #11164 hunts the same suspects (m-mix NE 4 / BLEND_RB
  16 -> 4, InkMixP 112 -> 144): when the bisect lands (~04:15) read its table in
  /opt/code/intarsiaml/.worktrees/ink-metal/.scratch/bisect_job.log; if it names the culprit, cut #11164 to the arms it does
  not settle for batched B 8 / 16, or cancel it.

## 1. Design note (contract delta, written before any kernel)

### 1.1 Shape of a batched step

A batched step of B sequences is ONE pass of the visit table over B rows (row r = sequence r's next token), exactly the
loop of a solo B-row pass (`Model::ink_pass`), with two differences:

* every row-local op (`Embedding`, `RmsNorm`, `InkInProj`, `Matmul` wo, the router `Matmul` / `Logits`, `MoeRoute`,
  `MoeExperts` / `MoeExpertsMxfp8`, `AttnResMix`, `LoopBlend`) runs once over the B rows under `exact_rows` (the row-exact
  multi-row kernels that are bitwise the 1-row kernel per row: the property solo 2..=16-row passes already rely on,
  `ink_rows_exact`). The workspace planes are `[B, D]` exactly as for a B-row chunk. **MoE union grouping across
  sequences needs nothing new**: `MoeExperts` sees B rows and does not know or care which sequence a row is from.
* every op that touches per-sequence state, `SconvRes` (69), `InkKvPrologue` (70), `RelSinkAttn` (71), `ConvMix` (74) and
  `InkAttnDecode` (75), runs in a new **row-table form**: one call over the B rows, row r being the 1-row (T = 1) call
  on sequence r's own state at sequence r's own position.

### 1.2 Contract delta (appended, optional; the existing calls are unchanged bit for bit)

The row-table form reuses the existing batched-decode mechanism (`OpCall::rows: Option<RowTab>`, R3a; today defined for
`GdnConv`, `GdnDeltaRule`, `KvAppend`, `AttnDecode`, `QkNormRopeKv`) and extends it to ops 69, 70, 71, 74, 75:

* `call.rows = Some(RowTab { ptrs: u8 [B, stride, 8], rows: B, stride, index, byte_off: 0 })` and `call.pos = Some(i32 [B])`
  (per-row positions; required with a row table). A row-table call has `T = B` rows in every row-indexed tensor
  (`[T, ..]` activations, and row r of each plane of `[N, T, D]` / `[S, T, D]` sources, messages and the `[S, T, D]` mix).
* **Semantics.** Row r of the call is bitwise the T = 1 call at position `pos[r]` (the device-position form of the solo
  step) whose row-indexed tensors are row r of the call's and whose state tensors are sequence r's. No row reads or writes
  another row's state; there are no "in-call" history rows (each row's history is its own ring / cache).
* **State addressing.** The state tensors of the call (the rings and caches, in their output order) describe row 0's
  buffers and are shape / dtype / offset carriers: the j-th state tensor of row r lives at
  `tab[r, index + j] + carrier.offset` (`carrier.offset` = the tensor's byte offset in row 0's buffer; every sequence's
  slots have the same layout). State tensors per op: `SconvRes` / `ConvMix`: the ring (j = 0). `InkKvPrologue`: k ring,
  v ring (j = 0, 1). `RelSinkAttn`: k cache, v cache (j = 0, 1). `InkAttnDecode`: k cache, v cache, k ring, v ring
  (j = 0..4).
* **Row table entries** (`Model::upload_row_table`): per slot two entries, as for `AttnKv` (k, v): an Inkling-2 `Ring`
  slot (`RingKind::InkKv` / `InkConv`) gets `(buffer address, buffer address)`, so both planes of a ring are reached by
  the one entry plus their carrier offsets, and `index = 2 * slot` everywhere. `InkAttnDecode` needs two slots
  (KV, then the K/V conv ring) behind one `index`: the loader guarantees `kvconv_slot == kv_slot + 1` (asserted at load),
  so entries `2*kv .. 2*kv + 4` are k cache, v cache, k ring, v ring. DeepSeek-V4.1's ring kinds and shared slots keep
  their "unsupported" error.
* **Checks** (`OpCall::check`, `inkling2::ops::check`): a row table on 69/70/71/74/75 needs `pos` `i32 [B]`,
  `rows == T`, `index + n_state <= stride`; `InkAttnDecode` keeps its 1..=16 rows. A linear cache's `pos0 + T <= Cr`
  check is the device-position one (skipped, as today with `pos`).
* **Registration.** A row-table kernel is registered `.row_table()` (selected only for calls with a table) and named
  `<base>_rows` so it takes its base kernel's admission (`Registry::effective_admission`). The plan's batch check counts
  a row-table call as B of the solo step's calls (`OpCall::row_weight`), so 69/70/71/74/75 join `plan::PER_ROW_OPS`.
* **Fused vs pair.** Chosen per the 1-row rule (`ink_attn_fused_rule(1, .., ctx)`) over the batch's largest context;
  both forms give the same bits, only time moves.

### 1.3 The 144 per-sequence slots in a batch

Nothing moves: every sequence keeps its own `SequenceState` (48 visits x (KV, K/V conv ring, h conv ring) = 144 slots).
The host writes one row table per step (`u8 [B, 2 * 144, 8]`, 36 KiB at B = 16) and the position vector; a captured
graph of the step can then replay with any set of sequences. Memory: the rings are per sequence and fixed size (about
170 MB per sequence on the real model: 40 SWA rings of 1040 x 1024 x 2 planes bf16, plus 48 x 2 conv rings), held once in
the plan's workspace today; `enable_batch` reserves ring bytes x (B - 1) out of the KV budget like recurrent state, and
shrinks the per-sequence full-attention KV capacity accordingly.

### 1.4 Model / scheduler / engine

* `run_layers(Seqs::Batch)` on an Inkling-2 model runs the visit loop with B rows; the stateful call sites go through one
  helper that, in a batch, swaps in row 0's carriers, the row table and the position vector (no fork of `ink_pass`).
* `enable_batch` accepts `Block::Ink` when every row-table Inkling-2 op the step runs has a registered `row_table` kernel
  (cpu: the decomposition reference below; Metal: the new `_rows` kernels); `row_tables_wanted` gets an Inkling-2 clause.
* Head parity: the solo step softcaps logits on the host (`softcap_logits`); the batched step's `LastLogits` rows and
  device top-k values get the same treatment (argmax is unaffected, the cap is monotone).
* A failed batched pass resets every Inkling-2 state of the batch (half-written rings), as `forward` does for one.
* Several rows of one sequence (batched speculative verify across streams, later): the row-table form would take a
  segment table (row -> (sequence, k-th row of its run)) and the in-call history rule of the solo T-row call within each
  run; nothing in the entry layout or positions above prevents it. Not built.
* Memory (main's note 1): `enable_batch` reserves rings x (B - 1) and divides the remaining KV budget by B x the
  full-visit bytes per token, refusing (`Error::Memory`) when fewer than 16 tokens per sequence would remain, as today.
* Engine / scheduler: nothing Inkling-specific (`engine/batch.rs` drives `forward_batch`); graphs of the batched step stay
  off for Inkling-2 (the solo graph path is not there either).

### 1.5 CPU reference of the row-table form

The cpu backend registers `cpu_f32_<op>_rows` (and `ref_<op>_rows`) for 69/70/71/74/75: the call is decomposed into B
T = 1 sub-calls (row r's slices gathered per plane into temporaries, row r's state tensors as views of the addressed
buffers, `pos0 = pos[r]`), each run by the ordinary kernel, results scattered back. Bitwise the solo row by construction;
it is the reference the Metal `_rows` kernels are tested against. `device_address` on cpu returns the buffer's host
address (owned buffers only); `supports_row_tables` stays false for every other model.

## 2. What was built

| Part | Where | What |
|---|---|---|
| Contract | `crates/intarsia-core/src/inkling2/ops.rs` ("Row tables"), `ops.rs` (`OpCall::check`), `plan.rs` (`PER_ROW_OPS`) | `ROW_TABLE_OPS`, `row_roles` (Row / Planes / State(j) / Fixed per tensor), `row_states`; a row-table call needs `pos` i32 `[B]`, `rows == T`, `index + states <= stride`; the five ops count as `B` solo calls in the plan's batch check. Calls without a table: unchanged code path. |
| Model | `model.rs` (`run_layers` Ink arm), `model/inkling2.rs` (`InkBatch`, `sx` / `rowtab` helpers), `model/batch.rs` | `Seqs::Batch` runs the visit loop over B rows; the 5 stateful call sites go through one helper that adds the row table + positions in a batch (identical call otherwise). `enable_batch` accepts `Block::Ink`, reserves rings x (B - 1); row-table entries `(buf, buf)` for Ink rings; softcap on batched logits / top-k values (solo parity); a failed batched pass resets every state of the batch; batched graphs off for Ink. A batch takes fused `InkAttnDecode` when the backend has no row-table pair (both give the same bits). |
| cpu | `crates/intarsia-backend-cpu/src/rowtab.rs`, `mem.rs` (`Raw` view, `address`), `lib.rs` | `ref_<op>_rows` / `cpu_f32_<op>_rows`: the call run as B one-row sub-calls on the addressed buffers (the reference of the form); registered after the base kernels (admission takes an op's first reference kernel). |
| Metal | `kernels/metal/inkling2_attn.metal` (`ink_attn_dec<GP, RT>`, `ink_attn_comb<RT>`), `inkling2_mix.metal` (`MX_ROWTAB`), `inkattn.rs` (`launch_ink_decode_rows`), `inkmix.rs` (`conv_mix_bf16_rows`), `dense.rs` (`dispatch_ind`) | Templated RT flag on the existing kernels: row r reads its caches / rings through the table (address in the params, resident via the indirect set), its own position, its own T = 1 scratch block; per-row split choice unchanged (grid over the most splits, the combine returns for a row with one split). The non-RT instantiations compile to the same code as before. |
| Recipe | `recipes/inkling2.toml` | `max_batch = 16` (the cap of `--max-batch`; without the flag one sequence at a time, as before). |
| Bench | `crates/intarsia-models/examples/batch_step.rs` | model name from the directory; `--ctx-spread N` (sequence i at ctx + i N). Asserts batched tokens == solo tokens every step. |

## 3. Tests

`crates/intarsia-models/tests/inkling2_batch.rs`, tiny fixture (window 8, rel extent 12, 2-layer blocks, 3 passes), cpu and
`INTARSIA_INKLING2_BACKEND=metal`; logits compared bitwise (softcapped) against the solo decode step of each sequence:

| Test | What | cpu | Metal | Metal, `INTARSIA_INK_SPLIT_KEYS=4` |
|---|---|---|---|---|
| `batch_equals_solo_2_5_16` | 2, 5, 16 sequences at lengths 1..64 (before / at / past the window, ring wrapped many times), 6 steps | pass | pass | pass |
| `batch_argmax_equals_solo` | device argmax rows | pass | pass | pass |
| `batch_membership_changes` | one finishes mid-batch, rejoin, shuffled row order, single-row step | pass | pass | pass |
| `batch_rollback_per_sequence` | two sequences cut back by length (3, 2) while the third continues | pass | pass | pass |
| `batch_failed_pass_resets_every_sequence` | injected failure in a row-table `ConvMix`: every sequence reset to 0, then prefill + batched decode bitwise again | pass | pass | pass |
| `batched_engine_equals_solo_engine` | `BatchedEngine` (4 slots, 9 requests at once, one seeded-sampled) token for token = `LocalEngine`; batched passes ran | pass | pass | pass |

`INTARSIA_INK_SPLIT_KEYS=4` forces several splits per row at tiny contexts, so the row-table combine and rows with different
split counts in one launch are exercised.

Other models (same branch): `intarsia-core` 232 + doc tests, `intarsia-backend-cpu` (admission quick sweep included),
`intarsia-models` `batch`, `batch_recurrent` (15, cpu), `batch_metal`, `inkling2` (8, cpu and Metal), `intarsia-backend-metal`
lib (kernel-list golden gains `ink_attn_decode_bf16_rows` only; `conv_mix_bf16_rows` is an Inkling op, outside the golden) and
its `inkling2_attn` (1), `inkmix` (3), `row_tables` (8), `rows_exact` (10 of 11) tests: pass (#10815). The 11th,
`rows_exact_registrations_are_pinned`, fails identically on the base a88db4620 (#10818): ink/m-mix's
`ink_in_proj_bf16_gemv_rows` is `rows_exact` but not in the pin list; not touched here (integration lane). `plan show --model ~/models/Qwen3-8B --backend metal`:
stdout and stderr byte-identical to a88db4620 (#10785).

## 4. Speed (Metal, endor, real checkpoint)

`batch_step` (`crates/intarsia-models/examples/batch_step.rs`), fleetq #10773 `--quiet gpu,cpu --mem 25`, checkpoint b85c3cfa
(W4A16, bf16), every sequence at context 1024 (+64 decoded), 4 rounds x 16 steps, solo and batched arms alternating;
`profile_every: 0`. Each step asserts the batched tokens equal the solo tokens (held for every B).

| streams | B solo steps ms | 1 batched step ms | speedup | solo agg tok/s | batched agg tok/s | per-stream tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.30 | 17.53 | 0.99x | 57.8 | 57.0 | 57.0 |
| 2 | 36.83 | 23.29 | 1.58x | 54.3 | 85.9 | 43.0 |
| 4 | 71.49 | 32.55 | 2.20x | 56.0 | 122.9 | 30.7 |
| 8 | 146.64 | 58.38 | 2.51x | 54.6 | 137.0 | 17.1 |
| 16 | 292.58 | 92.13 | 3.18x | 54.7 | **173.7** | 10.9 |

oMLX on endor at 16 streams: 94 tok/s aggregate; Fable's estimate for this engine: about 250 (perf-strategy section 7).
Step time grows 5.8 ms from 1 to 2 rows, then about 2.3 ms per row to 8 and 4.2 ms per row from 8 to 16: the 8 -> 16 step
is the one that does not amortise (perf-strategy section 7: the chunked row GEMVs read the weights twice above 8 rows, and
the expert union grows). Engine-path numbers through `intarsia serve --max-batch 16` (`bench speed`, localeval
`conc-plain-1k`): #10811, queued.

## 5. Single-read 9..16-row dense GEMVs (opt-in)

Above 8 rows the row-exact bf16 GEMV (`matmul_bf16_gemv_rows`: Inkling-2's wo) and the logits GEMV (`logits_bf16_gemv_rows`: the
unembed, the router) ran as two chunk launches (`row_chunks`), so their weights came from DRAM twice. The MoE experts were
already read once per step for all rows (`Policy::Auto`: union gate+up and union down above one row) and InkInProj already
takes 1..=16 rows in one dispatch, so these two are the only re-reads.

`<base>_mr8t<T>` (kernels/metal/gemv_multirow.metal, T = 1..8, families 16-byte / 8-byte bf16 and f32-out logits): one launch of
8 + T rows, as an 8-row pass and a T-row pass inside each threadgroup (simdgroup for logits) on the same weight row, the
weights read from DRAM once. Each (weight row, activation row) accumulator is the mirrored 1-row kernel's (same pass helpers,
lane mapping and reduction order). Opt-in `INTARSIA_METAL_ROWS_SINGLE=1` (default off; only 9..=16-row calls of those two
kernels; `with_rows_single` test hook). G-B: `crates/intarsia-backend-metal/tests/rows_single.rs`, 9..=16 rows, wo shape on
both load families, a K off the 16-byte block, logits at N 8195 and 128: every row bitwise the one-row launch and the two-chunk
launch (#10877, pass); `rows_exact` 11 / 11 and the backend lib tests 18 / 18 in the same run.

Expected gain: about 0.9 GB of re-read per step each for wo (48 visits) and the unembed, about 3 ms at 16 rows. A/B on the
batched step: queued (quiet slot).

## 6. Real-checkpoint correctness (#10844, default gpu load)

* **Solo path unchanged**: `intarsia accuracy --backend metal --max-seq 4096 --write-self-golden` on the G2 oracle goldens,
  base a88db4620 vs this branch (bf6eac934 + docs): every tensor of the self-goldens bitwise identical (short 48 / 48,
  long 60 / 60; the files differ only in the `built_unix` / `self_meta` metadata). G2 short KL21 2.380e-3 (gate 4e-3), the
  integration lane's figure; both arms exit 0 on short and long.
* **Batched == solo on the real model at mixed positions**: `batch_step --ctx 600 --ctx-spread 150` (sequence i at
  600 + 150 i: 600..2850, so the 1040-row SWA rings of the later sequences have wrapped while the early ones have not, and the
  full visits read up to 2.9k positions), B = 2, 5, 16, 8 steps: batched tokens == solo tokens every step (asserted).
  Its timings were taken under a shared (non-quiet) slot and are not reported as speed.
* Tiny tests on the same build: cpu 6 / 6, Metal 6 / 6; `rows_exact_registrations_are_pinned` pass after the ink/metal fix.

## 7. Engine path (serve)

First attempt #10811 (22:26) exited 1: `serve --max-batch 16` refused at load, `moe_experts` without an admitted kernel at the
batch phase's shapes (`Batch t=10..`). It ran before the merge that brought ink/metal 761a1561b (the plan-shape SiTU case,
merged at 22:27:45) and on the old script that used a prebuilt binary: the known failure, not a batch-specific one. Log kept
in `.scratch/serve_batch-10811.log` / `serve-10811-prefix/`. Rerun queued with the script that builds the tip at job start; a
default-load admission probe (`serve --max-batch 16` on the tip until `/v1/models` answers, one chat request) runs first.

## 8. Where the 8 -> 16 step goes (#10885, kernel-mode trace, default load: shares only)

`batch_prof --ctx 1024`, `INTARSIA_METAL_PROFILE=kernel` (one command buffer per launch, so absolute times are inflated and
launch-count changes are over-weighted), cut by step marks, median over 10 steps; GPU ms per step:

| op | B 8 | B 16 | delta | launches at 16 |
|---|---:|---:|---:|---:|
| matmul_bf16_gemv_rows (wo, two chunks above 8) | 2.87 | 6.50 | +3.63 | 96 |
| ink_in_proj_bf16_gemv_rows | 8.19 | 11.80 | +3.61 | 48 |
| moe_experts_situ_bf16_rows (gate+up / down / gather) | 29.05 | 33.91 | +4.86 | 132 |
| ink_attn_decode_bf16_rows | 10.59 | 12.08 | +1.48 | 48 |
| logits_bf16_gemv_rows (unembed) | 3.47 | 4.63 | +1.16 | 2 |
| logits_bf16_gemv_rows (router, two chunks above 8) | 1.06 | 1.83 | +0.76 | 96 |
| moe_experts_mxfp8 (the bf16 / mxfp8 visits) | 3.06 | 4.37 | +1.31 | 12 |
| everything else | 9.73 | 9.02 | -0.71 | |
| **sum** | **67.0** | **84.1** | **+17.1** | |

Summed GPU time grows 17 ms while the real step (`batch_step`, quiet) grows 34 ms (58.4 -> 92.1). Before choosing a kernel
lever the command-buffer timeline (GPU busy vs wall per step, B 2 / 8 / 16) is measured, to tell real GPU growth from host
gaps (queued). The coordinator's InkInProj figure (+4 ms) matches this table; it is one of three similar shares, not the
largest (the MoE situ path is).

## 9. A/B of `INTARSIA_METAL_ROWS_SINGLE` (#10900, quiet, tip 76cedfc93)

`batch_step --ctx 1024 --rounds 2 --steps 16 --batches 8,16`, arms alternating off / on, 2 reps; batched step ms
(aggregate tok/s). Batched tokens == solo tokens in every arm (asserted, exit 0), so the one-launch form is also exact on the
real model.

| rep | B 8 off | B 8 on | B 16 off | B 16 on |
|---|---:|---:|---:|---:|
| 1 | 39.18 (204.2) | 38.03 (210.4) | 71.20 (224.7) | 70.33 (227.5) |
| 2 | 39.47 (202.7) | 38.21 (209.4) | 78.45 (204.0) | 78.28 (204.4) |

The flag cannot change a B = 8 step (it only acts on 9..16-row calls), yet B 8 "on" is 1.1-1.3 ms faster in both reps: that is
the run-order bias of this A/B (on always runs second). The B 16 difference (0.9 and 0.2 ms) is inside it. **No measurable
gain; the flag stays opt-in and off.** (Expected about 3 ms; the re-read is mostly served from the SLC, it seems.)

**Bigger finding: the whole curve moved.** Against #10773 (section 4, ink/metal 1dacac3a8-era code), the same harness at this
tip gives solo 61 tok/s (was 57.8), B 8 39.2 ms (was 58.4), B 16 71.2 ms = **224.7 tok/s** (was 92.1 ms = 173.7) in rep 1;
rep 2 of B 16 is 78 ms (204 tok/s), a 10 % spread between reps of the same setting. The likely cause is the merged ink/metal
work since then, chiefly m-mix 47570539e (InkInProj up to 8 rows on the GEMV instead of the matrix units; the 8-row InkInProj
was 8.2 ms per step in #10885); unrelated local CPU load on endor during #10773 (the coordinator's build rule) may also have
inflated it. A full 1 / 2 / 4 / 8 / 16 curve at the tip, two passes, is queued to replace section 4's table.

## 10. Solo path speed: ink/metal f55bfe8b7 vs ink/batch 177cfd2f5 (#10914, quiet)

`intarsia generate --backend metal --max-seq 4096 --max-tokens 256 --temperature 0 --profile-every 0`, both built in the slot,
6 reps per arm alternating order (base/new, new/base, ...); output text identical.

| arm | mean tok/s | median | min | max |
|---|---:|---:|---:|---:|
| base ink/metal f55bfe8b7 | 64.13 | 64.09 | 63.92 | 64.40 |
| ink/batch 177cfd2f5 | 63.73 | 63.64 | 63.47 | 64.11 |

Branch -0.62 % (mean), -0.70 % (median): inside the <= 1 % merge criterion, but consistent (the branch is slower in 5 of 6
pairs), about 0.1 ms per token. Suspects, all in code the solo step runs: the `MX_ROWTAB` runtime branch and the now-mutable
ring pointer in the mix kernel's conv loop, the 32 bytes added to `InkP` / `InkMixP` (set_bytes), and the per-call
`row-table` checks in `ink_pass` / `OpCall::check`. Not chased here (inside the criterion); a compile-time row-table variant of
the mix shader would remove the first two if it matters.

## 11. Host gap of the batched step (#10956, default load: shares only) and the serve probe (#10947)

`batch_prof --ctx 1024`, `INTARSIA_METAL_PROFILE=cb` (command-buffer timeline; the trace costs time itself, so the wall
times are above the untraced run beside each), GPU busy = union of the command buffers inside each step, median of 12:

| B | wall ms | GPU busy ms | host gap ms | command buffers / step |
|---:|---:|---:|---:|---:|
| 2 | 30.91 | 30.45 | 0.46 | 7 |
| 8 | 43.19 | 42.54 | 0.65 | 7 |
| 16 | 83.05 | 82.22 | 0.83 | 7 |

The batched step is GPU-bound: under 1 ms of host gap at every B. So the 17 ms that the kernel-mode trace (#10885) did not
account for between B 8 and B 16 is GPU time the per-launch trace under-counts (one command buffer per launch changes
overlap and cache state), not host work; the coordinator's host-softcap hypothesis is out on this evidence too (greedy rows
never come back as logits, section 8 note). The levers above 8 rows are GPU kernels.

Serve probe #10947 at ink/batch 38e487a5d: `intarsia serve --backend metal --max-batch 16 --max-seq 8192` starts in 62 s
("continuous batching: up to 16 sequences (--max-batch), busy prefill chunk 256") and answers a chat request; the admission
refusal of #10811 is gone with ink/metal 761a1561b.

## 12. #10946 is invalid: it measured another lane's server

#10946's own `intarsia serve` loaded, then failed to bind (`bind 127.0.0.1:8099: Address already in use`, supervisor exit
78). Port 8099 was held by ink-dfuse's `intarsia serve ... --max-batch 1 --port 8099` (pid 53955, running since 23:07), so
the script's readiness probe answered and localeval `conc-plain-1k` (50.8 / 52.1 / 52.6 tok/s at 1 / 4 / 16 streams,
deepeval results/20261122-004010-adhoc-endor-intarsia-metal-ink-batch, unanimis 9be1ea95) measured that `--max-batch 1`
server, not this lane's: **not a `--max-batch 16` result.** (It matches #10890's serialized 50.8-51.6.) Peak RSS in the log
is of the failed start. Fix: the script now takes a free port from 8131 up, and after readiness checks that the listener is
this job's binary (and still the same pid before the bench); rerun queued. The serve probe #10947 (port 8097, "listening on
http://127.0.0.1:8097" in its own log) was its own server and stands.

## 13. Compile-time row-table mix variant (ink/batch-rtvar) timing (#11057, quiet)

Variant: ink/batch-rtvar fc76eb318 (+ 2b740dee7, `batch_step --digest`, example only) on the merged ink/metal tip 8546d143f.
`ink_mix_row<NE, NS, QB, RT>`: the solo instantiations (`ink_mix_row_e*`, RT = false) carry no row-table code, the batched
calls take `ink_mix_row_rt_e*`; host: `mix_row_shader(.., rt)` and the two dispatch sites only. bespin compile check: Rust
compiles; bespin's Metal toolchain cannot build the shaders (empty metallib, `attention.metal` fails there), so the shader
compile is checked on endor (#11104).

**Solo ABAB** (6 reps each, alternating, 256 tokens, `--profile-every 0`; output text identical):

| arm | mean tok/s | median | min | max |
|---|---:|---:|---:|---:|
| merged tip 8546d143f | 62.81 | 62.77 | 61.60 | 63.59 |
| rtvar 2b740dee7 | 62.56 | 62.81 | 61.31 | 63.14 |

Inconclusive: median +0.06 %, mean -0.4 %, and this run's rep spread (3 %, both arms) is several times the 0.6 % being
recovered (#10914's spread was 0.8 %). The whole box was slower in this slot (both arms about 2 % under #10914).

**Curve on rtvar** (2 passes, ctx 1024, batched tokens == solo tokens every step):

| B | pass 1 step ms | agg tok/s | pass 2 step ms | agg tok/s |
|---:|---:|---:|---:|---:|
| 1 | 16.94 | 59.0 | 16.42 | 60.9 |
| 2 | 22.12 | 90.4 | 21.34 | 93.7 |
| 4 | 28.80 | 138.9 | 27.89 | 143.4 |
| 8 | 48.88 | 163.7 | 46.26 | 172.9 |
| 16 | 79.10 | 202.3 | 80.79 | 198.1 |

B 8 is 18-25 % slower than #10900 at ink/batch 76cedfc93 (39.2 ms) and B 16 about 12 % (71.2 ms in #10900 rep 1; rep 2
was 78), while solo moved only about 2-4 %. The batched step got slower between ink/batch 76cedfc93 and the merged tip (or
with the variant). Candidates: m-mix's new defaults (mix_geom NE 4 from D 2048, `BLEND_RB` 16 -> 4) applied to the row-table
mix path, other ink/metal changes in the merge (25 files), or rtvar itself. Quiet A/B queued: ink/batch vs 8546d143f vs
8546d143f with `INTARSIA_INK_MIX_NE=3` / `INTARSIA_INK_BLEND_RB=16` vs rtvar, B 1 / 8 / 16, forward then reversed order.

## 14. Engine path: `intarsia serve --max-batch 16` (#11093, quiet, ink/batch f64368dbb)

Own server checked (port 8131, listener = this job's binary, pid 85510). ink/batch here is the pre-m-mix-merge code (the
#10900 curve: B 16 71-78 ms in batch_step).

**localeval `conc-plain-1k`** (closed loop, ~1k-token prompts, ~254 generated tokens per request, temperature 0, 3 reps per
stream count; deepeval results/20261122-024220-adhoc-endor-intarsia-metal-ink-batch/result.json):

| streams | aggregate decode tok/s (median; reps) | per-stream | TTFT s | `--max-batch 1` (#10946, ink-dfuse server) | oMLX |
|---:|---:|---:|---:|---:|---:|
| 1 | 46.5 (46.5, 46.5, 50.8) | 46.5 | 0.90 | 50.8 | |
| 4 | 84.6 (84.6, 84.6, 84.7) | 21.2 | 1.23 | 52.1 | |
| 16 | **117.7** (117.7, 118.1, 113.6) | 7.4 | 1.68 | 52.6 | 94 |

16 streams: 2.2x the serialized server, 1.25x oMLX. **`intarsia bench speed`** (calibration fix in; 1024-token prompts,
128 generated): single stream decode 60.2-61.3 tok/s (prefill 1.3-1.45k tok/s); concurrency rows (every stream prefills its
prompt each request, TTFT 13.4 s at 16 when 16 prompts arrive together): 2: 73.6, 4: 83.0, 8: 91.6, 16: 93.3 tok/s aggregate.

Why the serve path sits well under batch_step's 200+ at 16: the workload is mostly prefill. At 16 streams localeval ran 80
requests (80 x ~1k = ~82k prompt tokens) in 172 s; at the measured ~1.4k tok/s prefill that is ~55 s of prefill, which the
engine interleaves with decode steps (busy prefill chunk 256), so decode runs at roughly 20.3k tokens / ~117 s = ~173 tok/s
when it runs, and the batch is short of 16 rows while new requests prefill. The next levers on this cell are prefill speed and
how prefill shares steps with a full decode batch, not the batched decode step. Peak RSS not recorded (the `time -l` wrapper
was killed with the server; the script should signal the server pid, not the wrapper). Job exit 1 is that kill, not a failure.

**Orphaned server (found 2026-11-22 by main).** This job's server outlived it: `kill $SP` hit `/usr/bin/time`, not the
`intarsia serve` child (pids 85506/85510, port 8131, 11 GB RSS), which stayed up idle from ~02:42 EST until main killed it
in slot #11257 (~05:00). The serve scripts now start the server in its own process group (`.scratch/serve_lib.sh`:
`set -m`, EXIT trap TERMs the server, then the group, then KILL, then checks the port is closed and the group is empty;
tested with a dummy server under `time -l`).

## 15. Open at hand-off (2026-11-22)

- #11104 (rtvar correctness + ink/dfuse HOIST=1 row-table tests; ink-metal's G2 bisect inside the slot) still running: `.scratch/rtvar_correct.log`.
- #11164 (five-arm B1/8/16 regression A/B) cancelled before it started; requeue with `.scratch/b8reg.sh` (log `.scratch/b8reg.log`).
  The post-merge batched slowdown (B=8 ~39 -> 46-49 ms, B=16 ~71 -> 79-81 ms) is unexplained.
- 230 tok/s at 16 streams (batch_step) not confirmed: 224.7 / 204 pre-merge, 198-202 post-merge.
- `plan show` Qwen3-8B Metal byte-identical last checked at a88db4620 (#10785), before the ink/metal merge.

## 16. rtvar correctness (#11104, `.scratch/rtvar_correct.log`)

`ink/batch-rtvar` 2b740dee7 vs merged base 8546d143f: Metal lib / inkmix / mxfp4_gemv / rows_exact tests 18+3+5+11 pass
(4 ignored); tiny CPU 8+6 pass; tiny Metal 8+6 pass; real model, mixed contexts, batched-token digests at B 2/5/16
363b4c051e0fbbff / b6691f473fb73a82 / 6c18980dd1c6861f on both: BATCHED TOKENS IDENTICAL base == rtvar.

ink/dfuse ac2371c81 in the same job (#11104, exit 0): row-table tests on Metal with INTARSIA_INK_MIX_HOIST=1 and =0 both
pass (tiny inkling2_batch 6/6, inkmix 3/3, tiny state 8/8), and HOIST=1 with forced splits (INTARSIA_INK_SPLIT_KEYS=4) 6/6.

## 17. NE / BLEND_RB speed A/B on the merged tip (#11181, quiet, `.scratch/nerb-11181.log`)

Merged tip 8546d143f (code = ink/metal 1d0892eb7, before the NE fix 7eca1b012), `batch_step --ctx 1024 --rounds 2 --steps 16`,
arms in order fwd / rev / fwd. Batched step ms per rep, median, and aggregate tok/s at the median:

| B | tip (NE 4, RB 4) | NE=3 | NE=3 + BLEND_RB=16 |
|---:|---|---|---|
| 1 | 16.40 / 16.43 / 16.36: **16.40** (61.0) | 16.31 / 16.34 / 16.33: **16.33** (61.2) | 16.29 / 16.41 / 16.62: **16.41** (60.9) |
| 8 | 34.93 / 35.11 / 34.93: **34.93** (229.0) | 35.09 / 34.98 / 34.78: **34.98** (228.7) | 35.04 / 35.14 / 35.27: **35.14** (227.6) |
| 16 | 59.96 / 59.96 / 60.61: **59.96** (266.9) | 59.90 / 59.75 / 59.77: **59.77** (267.7) | 60.39 / 60.41 / 60.50: **60.41** (264.9) |

- NE 4 vs NE 3 costs nothing measurable at B 1/8/16 (within 0.3%); BLEND_RB 16 is ~0.6-0.8% slower at B 8/16, not faster.
  Neither is the batched speed cost, so the NE fix (7eca1b012, NE 3 at D 2304) should not slow the batched step.
- The 71 -> 80 ms B=16 slowdown does not reproduce here: the merged tip runs B 16 at ~60 ms (266.9 tok/s), B 8 at ~35 ms,
  faster than every earlier figure (pre-merge #10900 71 ms; merged #11057 79-81 ms). This job ran without a pre-merge arm, so
  it cannot say what changed; the earlier figures came from other jobs (load, run order, the rtvar build or flags may differ).
  Before trusting the 230 target as met (266.9 here), a paired pre-merge vs merged run in one quiet job is still needed.

- Residency caveat: #11181 (and #11104, correctness only) ran while the orphaned idle `intarsia serve` from #11093 held
  11 GB on endor (section 14). An idle server takes little GPU, and the three arms ran interleaved under the same condition,
  so the A/B conclusion stands; the absolute B 16 figure (59.96 ms) is not borderline against any decision, so no re-measure.

## 18. Batched baseline handed to ink-metal; next B=16 candidate (proposal, 2026-11-22)

**Baseline command** (sent to ink-metal af43898d63e5b5190 for its combined post-merge gate; quiet slot, median of 3 reps;
reference = column 3 batched step ms and column 6 aggregate tok/s):
`cargo build --release -p intarsia-models --example batch_step` then
`INTARSIA_METAL_TEST_MODEL=<Inkling-2-Mini snapshot> ./target/release/examples/batch_step --ctx 1024 --rounds 2 --steps 16 --batches 1,8,16`.
Reference from #11181: B1 16.40 ms / 61.0, B8 34.93 / 229.0, B16 59.96 / 266.9 tok/s.

**B=16 step breakdown.** The only per-kernel trace is #10885 (section 8: pre-merge tree, kernel-mode trace, shares only, traced
sum 84.1 ms). Scaled to today's 60 ms step, assuming the shares held (the m-mix merge changed the mix kernels, which sit in
"everything else"):

| op | share at B 16 | ~ms of 60 | B 8 -> 16 growth (traced) |
|---|---:|---:|---:|
| MoE experts bf16 rows (gate+up / down / gather, 132 launches) | 40% | 24 | +4.9 ms (+17%) |
| ink_attn_decode_bf16_rows | 14% | 8.6 | +1.5 |
| ink_in_proj_bf16_gemv_rows | 14% | 8.4 | +3.6 |
| wo bf16 GEMV rows (two chunks above 8) | 8% | 4.6 | +3.6 |
| unembed + router logits | 8% | 4.6 | +1.9 |
| MoE mxfp8 visits | 5% | 3.1 | +1.3 |
| everything else | 11% | 6.4 | -0.7 |

**Candidate: the fixed part of the MoE expert launches at B 16.** Top-8 of 128 gives an expected union of ~52 experts per layer
at B 8 and ~82 at B 16 (+59%), but traced MoE time grew only 29.1 -> 33.9 ms (+17%). A two-point fit t = a + b * union gives
b ~0.16 ms per union expert per step (~6.7 us per expert per layer, near weight-read speed) and a ~21 ms traced (~15 ms of the
real step) that does not scale with the experts touched: per-launch tail and occupancy over 132 launches with few rows per
expert, plus the gather. Removing it (fewer, fuller launches: gate+up and down per layer over all union experts in one grid,
gather folded into down, per-row reduction order unchanged so rows stay bitwise) is the largest lever left.
Estimate: 3-7 ms per step, so B 16 at 53-57 ms, ~280-300 tok/s (+5-13%).
The fit has only two points from the old tree, so step 1 is one quiet profile job on the final tree: kernel trace at
B 2/4/8/12/16, MoE time against union size. It confirms or kills the fixed part before any kernel is written.
Second candidate: ink_in_proj rows (+3.6 ms from B 8 to 16; gain ~2 ms if it scales like wo would with a single read).
The single-read 16-row GEMV already failed for wo/logits (#10900), so this is ranked lower.

For the localeval `conc-plain-1k` cell (117.7 tok/s at 16), the bigger lever is still prefill and how prefill chunks share
steps with a full decode batch (section 14), not the decode step.

## 19. The MoE candidate's premise does not hold (checked before queuing the profile)

Two facts from the code and the existing #10885 trace (`.scratch/prof/k{8,16}.jsonl`, whose MoE labels carry the distinct
experts `n` and the pairs `k` of each call, `label_moe_routing`):

1. **The proposed launch structure already exists.** `inkmoe.rs` `launch_ink_experts` at T <= 16 already runs gate+up
   union-grouped over all pairs in one grid, then union down, then the gather (3 launches per MoE call, 44 calls per step =
   the 132). Down fused with the k-order token sum (`ink_moe_dn_f`, one launch, no gather) also exists:
   `INTARSIA_INK_MOE=union-fused` (or `fused`), all policies bitwise. `Auto` picks fused only at T = 1. So the code change I
   proposed is mostly an env flag, not a new kernel.
2. **In batch_step / batch_prof the rows share experts far more than real traffic would.** Distinct experts per MoE call:
   B 8 mean 20.9 (median 21, max 29), B 16 mean 23.3 (median 23, max 36). For independent routing, top-8 of 128 would give
   ~52 and ~82. Doubling the rows added only ~2.4 experts per call. The bench prompts are uniform-random token ids, and the
   rows evidently route to almost the same experts (likely they decode the same few tokens). So the +17% MoE growth from
   B 8 to 16 is per-pair work at a nearly constant union. My two-point "fixed part" fit assumed union 52 -> 82 and is void.
   It also means the 266.9 tok/s B 16 figure is optimistic for real prompts, where the union and the MoE weight reads per
   step would be several times larger.

Not queued: the step-1 profile as specified (MoE time vs union size) cannot vary the union with this bench. Reported to main.

## 20. Real-text bench prompts: functional test (#11209, default load next to ink-metal's gate #11206: no speed claims)

ink/batch ff9f65bf9 (bench-only: dcff9e8c4 + ff9f65bf9). Corpus `crates/intarsia-models/examples/data/bench_corpus.txt`
(158 KB frozen snapshot of repo prose, a license, Rust source; 49 909 Inkling-2 tokens, 16 slices of up to 3 119).
Log `.scratch/benchtest-11209.log`, ctx 512, 1 round x 8 steps.

- Deterministic: two text runs give identical digests at B 1 / 4 / 16 (all 16 per-row digests equal; B 16 combined
  889a7108a816c248).
- Rows decode differently on text: 16 distinct per-row digests at B 16. Random ids: at B 16 three rows share one digest
  (14a607d087669a18) and two rows repeat B 1 / B 4 rows' streams: the rows collapse onto the same tokens.
- Expert union per MoE call at B 16 (kernel trace, 44 calls per step): text mean 63.5 (median 65, max 83) vs random 27.7
  (median 27, max 52); independent routing would give ~82. So real text gives ~2.3x the union of the old bench.
- Timings in this shared slot (indicative only): B 16 batched step text 74.8 / 85.6 ms (214 / 187 tok/s) vs random 66.7 ms
  (240 tok/s); B 1 and B 4 equal within noise. The quiet job after #11206 measures it properly.

## 21. Real-text MoE job (#11261, quiet, tree e4ebcfe77 = ink/metal 7e5e6010e + bench; `.scratch/realmoe-11261.log`)

**auto vs union-fused, real text, batched step ms per rep (order fwd / rev / fwd):**

| B | rep 1 auto / uf | rep 2 uf / auto | rep 3 auto / uf | paired diff uf - auto (ms) |
|---:|---|---|---|---|
| 1 | 15.48 / 15.67 | 15.30 / 15.44 | 15.34 / 15.45 | +0.19, -0.14, +0.11 |
| 8 | 53.30 / 51.32 | 53.14 / 51.09 | 47.40 / 51.50 | -1.98, +2.05, +4.10 |
| 16 | 87.72 / 88.29 | 84.28 / 84.37 | 80.41 / 91.28 | +0.57, -0.09, +10.87 |

Per-row digests identical across both arms and all reps (B 1 67843d1dbf0249f5, B 8 74478685fe7299e9, B 16
846a505a4055b0ba): the policies are bitwise as documented. No verdict: auto drifts down rep by rep (87.7 / 84.4 / 80.4 at
B 16), consistent with mmap'd expert weights being paged in on first use (the bench never enables the keep-warm residency
set; real text touches ~63 experts per call). Rerun with an untimed warm rep and page-in counts: #11297. auto stays.

Random-prompt continuity run in the same slot: B 1 15.35 / B 8 42.79 / B 16 74.84 ms (65.1 / 187.0 / 213.8 tok/s), vs
#11181 on 8546d143f 16.40 / 34.93 / 59.96. ink-metal's combined gate on 7e5e6010e (not quiet) logged B 16 74.9 ms, so a B 16
regression may predate pack12; check pending (#11266, then runtime switches).

**Expert union per MoE call on real text (kernel trace, 44 calls per step):**

| B | union mean / median / max | independent routing | MoE GPU ms/step (traced) | all GPU ms/step (traced) |
|---:|---|---:|---:|---:|
| 2 | 14.2 / 15 / 16 | 16 | 8.50 | 21.91 |
| 4 | 24.8 / 25 / 31 | 29 | 18.22 | 38.82 |
| 8 | 41.0 / 42.5 / 55 | 52 | 28.09 | 53.14 |
| 12 | 53.5 / 55 / 71 | 69 | 33.04 | 62.97 |
| 16 | 62.5 / 64 / 83 | 82 | 32.60 | 61.97 |

**MoE candidate: dropped (< 3 ms rule).** MoE time scales with the union: from B 2 to 12, ~0.62 traced ms per extra expert
in the union per step (8.5 -> 33.0 ms as the union goes 14 -> 54), with an intercept of ~0. There is no fixed launch part
to remove; at B 16 MoE is flat (32.6 ms at union 62.5), below that line. Real text's MoE is weight-read work over the
experts actually touched. Fewer launches would not reduce it.

**B 16 per-op on real text (traced, ms/step, B 8 -> B 16):** MoE situ gate+up 15.93 -> 18.23, down 8.65 -> 10.27, gather
0.34 -> 0.27; ink_in_proj 6.36 -> 8.76; ink_attn_decode 8.49 -> 9.78; unembed 2.69 -> 3.62; wo / dense GEMV 2.10 -> 2.70;
MoE mxfp8 3.16 -> 3.82; conv_mix 4.15 -> 3.52; route 0.93 -> 0.73. The traced sum (62 ms) is below the untimed step
(80-88 ms); the per-launch trace under-counts, as in section 11.

**in_proj option (reported per the rule):** ink_in_proj is the next-largest share that grows with B: +2.40 traced ms from
B 8 to 16, about 14% of the traced step. A rows kernel that reads its weights once for all 16 rows could at best recover
that growth: ~2.4 traced, ~3 ms real, so B 16 ~81-85 instead of 84-88 ms (~+3%). The same single-read idea gave nothing for
wo / logits (#10900), so I would not build it without a microbench showing the 8 -> 16 growth is weight reads.

## 22. Batched-decode softcap check (main's priority brief, ink-mtp's suspicion)

**Batched decode applies the cap.** `forward_batch`'s head (`model/batch.rs` ~496-510, since 869be5b87) runs
`softcap_logits` on both the top-k values (`LastTopK`; the cap is monotone, so the device's ids and order stand) and the
full logits rows (`LastLogits`, and top-k rows that fall back to logits on a NaN). The uncapped readbacks ink-mtp found are
in the **verify** paths, not in batched decode: `model.rs` `VerifyRead::Logits` / `VerifyRead::TopK` (~2490-2510),
`forward_verify_batch_run`'s rows (~2694-2716) and `spec_graph.rs` (~425-447). ink/mtp-softcap 36cbb9536 caps those with
`Model::softcap_step`; per its commit, sampled requests do not speculate on Inkling today. No fix is needed on ink/batch.

**Test** (70446e0e7, `inkling2_batch_sampler_rows_are_softcapped`): tiny Inkling-2 (`final_logit_softcapping` 30),
2 / 5 / 16 sequences at mixed positions (before, at and past the SWA window), 3 steps, modes mixed per row (logits, top-k 8,
top-k 40). Every logits row must be bitwise the solo capped logits; every top-k row's values must be bitwise the solo capped
logits of its ids and the k largest of them. Device top-k rows are requested only where the backend has them (Metal; the cpu
backend has no `supports_topk`, so on cpu only logits rows run).
- cpu (bespin #11310): `inkling2_batch` 7/7 pass, max |d| = 0; G1 (`g1a_f32_vs_f64`, `g1b_teacher_forced_bf16`) 2/2 pass.
  Mutation (batch.rs without its two `softcap_logits` calls): the new test FAILS at the logits assertion, so it detects a
  missing cap. (spark2 cannot build this test file: its dev-dependency `intarsia-backend-metal` is macOS-only.)
- Metal tiny (#11309, default load): Metal has no `supports_topk`, so its sampled batched rows are full logits rows (the host
  sampler takes the top-k): the test passes with max |d| = 0, and the mutation copy FAILS. Device top-k is a cpu-backend
  feature (`CpuBackend::arc_topk`) on this model; cuda / hip also have it.
- Device top-k rows on cpu: the test first ran with the plain cpu backend (no top-k); fixed in cd372d2a7 (the softcap test
  wraps `CpuBackend::arc_topk` in the `Failing` test backend, which now forwards `supports_topk` / `topk_rows`). bespin
  #11335-ish run: "device top-k rows tested", 7/7 pass, max |d| = 0, G1 2/2. Mutation with only the top-k softcap removed:
  FAILS at "row 1 (LastTopK(8)): top-k values are not the solo capped logits of their ids". Both row kinds are covered.
- Remote-test hazard hit once: `git archive` sets file mtimes to the commit time, older than a previous run's mutated build,
  so cargo reused a stale mutated `intarsia-core` (every test failed). The bespin script now touches every extracted file.

## 23. #11297: warm rep + paired auto vs union-fused on real text, BOX-SLOW (not speed); control rule

`.scratch/realmoe2-11297.log`, tree e4ebcfe77, quiet slot. The whole box ran ~50% slow: solo decode 45-52 tok/s (61-65 in
#11261, same binary source and args), batched B 16 120-130 ms (80-88). System page-ins per run 5k-57k pages while our
process had 0 major faults and ~1.7 M page reclaims every run (each run is a fresh process, so the warm rep cannot warm the
next run's mappings; reclaims are minor faults on resident pages). Main found no other model server, kev idle, no thermal
warnings, swap 21.07 / 21.5 GB. **Labelled box-slow, not speed.**

| B | rep 1 auto / uf | rep 2 uf / auto | rep 3 auto / uf |
|---:|---|---|---|
| 8 | 64.03 / 74.04 | 62.55 / 66.73 | 69.63 / 64.94 |
| 16 | 130.31 / 128.32 | 127.82 / 125.91 | 123.61 / 123.22 |

(warm rep, untimed: B 8 67.27, B 16 120.92 ms). Paired diff uf - auto at B 16: -2.0, +1.9, -0.4 ms (mean -0.2); at B 8:
+10.0, -4.2, -4.7 (mean +0.4). Digests identical across arms. Even in a slow box there is no policy difference beyond
noise; no flip proposal from this.

**Control rule (main):** a control with #11261's binary (e4ebcfe77 build, kept at `.scratch/ctl/batch_step`), args and
real-text input, solo + B 16, 1 rep. Solo >= 58 tok/s and B 16 <= 92 ms means the box is fine and #11297's config is the
cause (find which). Otherwise the box is slow: requeue #11297's speed half and the toggles later. Combined job
`.scratch/ctl_then_toggle.sh`: the control first, the regression toggles only if the box is fine (#11306 and the separate
control job were released so that the toggles cannot run first). #11309 (tiny Metal softcap, correctness) stays queued.

Control vs #11261, side by side (main asked that the two be visibly identical):

| | #11261 arms (`.scratch/realmoe.sh`) | control (`.scratch/ctl.sh`) |
|---|---|---|
| binary | `.scratch/base-target/.../batch_step` built from e4ebcfe77 | the same tree's build, copied to `.scratch/ctl/batch_step` |
| input | real-text prompts (the bench default since dcff9e8c4; no `--prompts` flag) | real-text prompts (same default, no flag) |
| args | `--ctx 1024 --rounds 2 --steps 16 --batches 1,8,16 --digest` | `--ctx 1024 --rounds 2 --steps 16 --batches 1,16 --digest` |
| env | `INTARSIA_METAL_TEST_MODEL` only (auto arm) | `INTARSIA_METAL_TEST_MODEL` only |

The only difference is the batch list (no B 8). #11297 differed from both by `/usr/bin/time -l`, `INTARSIA_INK_PACK12=0`
(no such switch on e4ebcfe77) and stdout to a file.

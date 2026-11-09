# DSpine tap-replay test: is the acceptance gap a port bug?

**Verdict: no port bug found.** Replaying the exact taps our engine produced through Tencent's MLX reference DSpine (dspine_mlx at commit 6450af7) gives the same drafts on 98.0-98.5% of rows (95.0-95.2% of passes fully identical), and the same acceptance: tokens per pass 3.318 ours vs 3.316 reference (fp8 head), 3.392 vs 3.389 (bf16 head). The residual differences are at the level of the reference's own MLX GPU-vs-CPU noise. The gap to DFlash2 therefore comes from the model (or the MXFP4 target), not from the port, with the caveat that the replay cannot see tap errors (see Limits).

## Setup
- Engine: intarsia generate, Metal, Qwen3.8-27B-Quark-AWQ-MXFP4, `--spec dflash --spec-draft .../dspine-taps-work/model --spec-k 7`, greedy, `--chat --no-think`, 256 tokens forced, first case of each of the 7 files in scripts/dfh/cats (ho-chat, ho-code, ho-prose, ho-recall, rcp-code, rcp-prose, rcp-recall) plus scripts/hip-decode/prompt4k.txt (4420 prompt tokens, 256 generated). Dump via INTARSIA_DSPINE_DUMP + INTARSIA_DSPINE_DUMP_LAYERS=1 (tools/dspine/dump_cats.sh).
- Two runs: default (fp8 lm_head, `head: fp8` in meta.json) and INTARSIA_LMHEAD_QUANT=off (bf16 head). Each replayed with `--head auto`, i.e. the reference uses the same head format the engine ran (fp8 emulated as e4m3 with one f32 scale, or the checkpoint's bf16), so the head is not a confounder.
- Replay: tools/dspine/replay_taps.py, reference run on the full 16-row block per pass from the dumped taps, `--noise cpu --noise-passes 24`. Acceptance is scored against the target's actual greedy tokens (ids of the run itself), capped at k=7; t/p = 1 + accepted.
- Fleetq jobs on endor: #9107 (dump, both heads), #9108 (replay, both heads).
- Row numbers in the first-divergence column: 1 = first draft token of the pass .. 7; passes with no divergence are not listed.

## Results, default head (fp8)
| prompt | passes | engine t/p (its own counter) | ours t/p (replay of dump) | reference t/p | row agree | passes fully identical | first diverging row 1..7 (n passes) |
|---|---|---|---|---|---|---|---|
| ho-chat-chat-sleep | 95 | 2.68 | 2.684 | 2.684 | 0.995 | 0.968 | [0, 0, 0, 0, 0, 2, 1] (3) |
| ho-code-code-csv | 41 | 6.22 | 6.220 | 6.220 | 1.000 | 1.000 | [0, 0, 0, 0, 0, 0, 0] (0) |
| ho-prose-prose-bakery | 114 | 2.23 | 2.228 | 2.228 | 0.970 | 0.886 | [0, 0, 2, 1, 2, 3, 5] (13) |
| ho-recall-recall-tls | 70 | 3.64 | 3.643 | 3.629 | 0.990 | 0.957 | [1, 1, 0, 0, 1, 0, 0] (3) |
| long4k | 60 | 4.25 | 4.250 | 4.250 | 0.995 | 0.983 | [0, 0, 0, 0, 0, 1, 0] (1) |
| rcp-code-code-lru-cache | 49 | 5.20 | 5.204 | 5.204 | 0.991 | 0.959 | [0, 0, 0, 0, 0, 2, 0] (2) |
| rcp-prose-prose-lighthouse | 120 | 2.12 | 2.117 | 2.117 | 0.986 | 0.950 | [1, 1, 1, 1, 2, 0, 0] (6) |
| rcp-recall-recall-tcp-handshake | 65 | 3.91 | 3.908 | 3.908 | 0.971 | 0.954 | [0, 0, 2, 1, 0, 0, 0] (3) |
| **ALL** | 614 | | 3.318 | 3.316 | 0.985 | 0.950 | [2, 2, 5, 3, 5, 8, 6] (31) |

Noise floor (24 evenly spread passes per prompt, 192 passes): reference GPU vs reference CPU row agreement 0.970, passes identical 0.922; ours vs reference on the same passes: row agreement 0.992, identical 0.969; t/p on these passes ours 3.635 / ref GPU 3.630 / ref CPU 3.661. Candidate (top-16) overlap 0.988, argmax agreement 0.984. Layer out rel-L2 median (layers 1..5) [0.58, 0.68, 0.76, 0.81, 0.81] %, max [0.86, 1.01, 1.44, 1.57, 1.85] %; reference GPU vs CPU median [0.82, 1.0, 1.09, 1.16, 1.08] %.

## Results, INTARSIA_LMHEAD_QUANT=off (bf16 head)
| prompt | passes | engine t/p (its own counter) | ours t/p (replay of dump) | reference t/p | row agree | passes fully identical | first diverging row 1..7 (n passes) |
|---|---|---|---|---|---|---|---|
| ho-chat-chat-sleep | 94 | 2.70 | 2.702 | 2.702 | 0.995 | 0.989 | [0, 0, 0, 0, 1, 0, 0] (1) |
| ho-code-code-csv | 41 | 6.22 | 6.220 | 6.220 | 1.000 | 1.000 | [0, 0, 0, 0, 0, 0, 0] (0) |
| ho-prose-prose-bakery | 115 | 2.22 | 2.217 | 2.217 | 0.961 | 0.904 | [0, 3, 1, 1, 3, 1, 2] (11) |
| ho-recall-recall-tls | 66 | 3.86 | 3.864 | 3.864 | 0.991 | 0.985 | [0, 0, 0, 1, 0, 0, 0] (1) |
| long4k | 59 | 4.32 | 4.322 | 4.305 | 0.963 | 0.932 | [1, 0, 0, 1, 1, 0, 1] (4) |
| rcp-code-code-lru-cache | 50 | 5.10 | 5.100 | 5.100 | 1.000 | 1.000 | [0, 0, 0, 0, 0, 0, 0] (0) |
| rcp-prose-prose-lighthouse | 105 | 2.42 | 2.419 | 2.419 | 0.982 | 0.943 | [0, 0, 2, 1, 0, 1, 2] (6) |
| rcp-recall-recall-tcp-handshake | 69 | 3.61 | 3.609 | 3.594 | 0.964 | 0.913 | [0, 0, 1, 2, 2, 1, 0] (6) |
| **ALL** | 599 | | 3.392 | 3.389 | 0.980 | 0.952 | [1, 3, 4, 6, 7, 3, 5] (29) |

Noise floor (24 evenly spread passes per prompt, 192 passes): reference GPU vs reference CPU row agreement 0.976, passes identical 0.922; ours vs reference on the same passes: row agreement 0.982, identical 0.953; t/p on these passes ours 3.724 / ref GPU 3.719 / ref CPU 3.719. Candidate (top-16) overlap 0.988, argmax agreement 0.987. Layer out rel-L2 median (layers 1..5) [0.58, 0.68, 0.76, 0.81, 0.82] %, max [0.91, 1.02, 1.17, 1.33, 1.69] %; reference GPU vs CPU median [0.8, 0.97, 1.09, 1.14, 1.1] %.

## Reading
- Sanity check of the dump: the replay's t/p for our drafts equals the engine's own `spec:` counter on every prompt in both runs (to the printed 2 decimals).
- Agreement: 1.000 row agreement on the code prompts and 0.961-0.995 elsewhere; the worst prompt is ho-prose-prose-bakery (0.97 / 0.96, 13 / 11 diverged passes of ~115). Diverging passes are spread over rows 1-7 with no concentration on a particular row, and the accepted-length histograms are identical to within one pass (default: ours 173/129/92/61/38/27/18/76 for 0..7 accepted, reference 174/128/92/61/38/27/18/76).
- Tiny net differences in t/p come from the few passes where an argmax flips on a near tie; they go both ways (long4k and recall-tls/tcp-handshake favour ours by 0.01-0.02, the aggregate by 0.002-0.003).
- Reference vs itself: MLX GPU vs CPU on the same passes agrees on only 97.0% (fp8) / 97.6% (bf16) of rows and 92.2% of passes identically; ours-vs-reference on those passes agrees on 99.2% / 98.2% of rows. Our per-layer rel-L2 (0.6-0.8% median, max 2%) is the same size as the reference's own GPU-vs-CPU (0.8-1.2% median). So ours is closer to the reference GPU than the reference's CPU pipeline is, i.e. no stage diverges beyond bf16 noise; no first-diverging layer or bridge to localize (no file:line suspect from this test).
- The engine's tokens per pass on Metal reproduce the earlier HIP sweep gap (default head: chat 2.68, prose 2.23 / 2.12, recall 3.64 / 3.91 against DFlash2 k7 chat 3.08, prose 2.51-2.58, recall 3.93-4.10), so the gap is not backend- or bridge-kernel-specific.
- fp8 head vs bf16 head: per-prompt t/p moves by -0.30..+0.30 (chat 2.68 vs 2.70, prose 2.23 vs 2.22 and 2.12 vs 2.42, recall 3.64 vs 3.86 and 3.91 vs 3.61, code csv 6.22 both, lru 5.20 vs 5.10, long4k 4.25 vs 4.32; aggregate 3.318 vs 3.392). The sign is mixed and each cell is one 256-token prompt whose token stream also changes (the target's own head is switched too), so this does not show a consistent fp8 effect; it cannot be the whole gap.

## Limits
- The replay feeds our own taps, so it cannot catch a wrong tap index/tap point, wrong layer numbering or a systematic error in the target's hidden states (for example MXFP4 target numerics): both drafts would see the same bad input. By code reading the engine taps the residual after 0-based layer li (model.rs, around line 3629), matching the reference hook on layers[lid] for [1,16,31,46,61], but that is read, not run. A numerical comparison against an MLX target run (or a bf16 target) was not done; if the gap has to be pursued further, that is the next test, together with checking how Tencent's own acceptance numbers were measured (target precision, prompts, k).
- 8 prompts, 256 tokens each (599-614 passes), single sequence, rank 0 only; the category-level numbers have wide error bars (a handful of passes decide a few hundredths of t/p).
- DFlash2 numbers quoted above are from the earlier sweep table in the task brief, not re-run here.

## Artifacts
- Per-prompt and aggregate replay json, run.log, noise data: /opt/code/intarsiaml/.worktrees/dspine-taps-work/out/default/ and /opt/code/intarsiaml/.worktrees/dspine-taps-work/out/bf16/ (ALL.json holds the summary); logs /opt/code/intarsiaml/.worktrees/dspine-taps-work/out/replay-default.log and /opt/code/intarsiaml/.worktrees/dspine-taps-work/out/replay-bf16.log.
- Engine dumps (about 2.4 GB, taps/passes/layers/index per prompt): /opt/code/intarsiaml/.worktrees/dspine-taps-work/runs/sw-default/<prompt>/dump and /opt/code/intarsiaml/.worktrees/dspine-taps-work/runs/sw-bf16/<prompt>/dump; engine logs in the same folders (gen.err, steps.log).
- Tooling (uncommitted, worktree dspine-taps): crates/intarsia-core/src/model/dspine/dump.rs, tools/dspine/replay_taps.py, tools/dspine/dump_cats.sh; sweep driver scripts /opt/code/intarsiaml/.worktrees/dspine-taps-work/run-sweep.sh and /opt/code/intarsiaml/.worktrees/dspine-taps-work/run-replay.sh.

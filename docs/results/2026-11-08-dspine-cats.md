# DSpine-b16 category sweep, 27B HIP tp=1 (lane dspine, 2026-11-08)

Main `8532f57` (DSpine merged in #343), tatooine card 0, Quark 27B MXFP4, `--mem-budget 29`, `scripts/dt1/cat-window.sh` with DSpine arms `dsp<k>`
added, the 7 category files of `2026-11-06-27b-dflash-hip-tp1.md` (held-out `ho-*` and recipe `rcp-*`), 512 forced tokens (`min_tokens`), greedy.
fleetq slot held both R9700s with radiance down 00:32:05-01:24:54. Raw: `2026-11-08-dspine-cats/raw/w1` (server logs, spec counters, targets);
localeval runs `/opt/code/deepeval/results/20261108-*-dt1-w1-*`.

Two rounds, arm order reversed in round 2. Round 2 lost `mtp2` (all) and `dfl7` (ho-code, rcp-prose, rcp-recall): localeval commit `9c66af2`
(another session, "--thinking on|off") landed mid-window and the harness failed at argparse ("conflicting option string: --thinking") before any
request. Not an engine failure. Where both rounds exist they agree within 0.6 % (DSpine k3 and k7 within 0.5 %), so round-1-only cells are kept.

## Decode tok/s (mean over cases; tokens per pass from the spec counters, round 1)

| category (prompt tok) | off | MTP k2 | DFlash2 k7 | DSpine k3 | DSpine k7 | best |
|---|---:|---:|---:|---:|---:|---|
| ho-chat (46) | 35.9 | **64.2**, 2.25 | 61.5, 3.08 | 57.4, 2.43 | 53.5, 2.72 | MTP |
| ho-code (45) | 35.7 | 79.4, 2.79 | 98.3, 4.82 | 79.8, 3.35 | **100.3**, 5.03 | DSpine k7 (+2 %) |
| ho-prose (43) | 35.8 | **59.3**, 2.08 | 49.6, 2.51 | 47.8, 2.03 | 41.1, 2.11 | MTP |
| ho-recall (39) | 36.0 | 69.8, 2.44 | **77.9**, 3.93 | 69.5, 2.93 | 73.2, 3.75 | DFlash2 |
| rcp-code (47) | 35.8 | 78.1, 2.73 | 104.2, 5.15 | 82.6, 3.47 | **105.8**, 5.35 | DSpine k7 (+2 %) |
| rcp-prose (39) | 35.8 | **60.0**, 2.11 | 51.0, 2.58 | 50.4, 2.15 | 43.6, 2.25 | MTP |
| rcp-recall (47) | 35.8 | 70.4, 2.46 | **81.8**, 4.10 | 68.9, 2.91 | 71.9, 3.66 | DFlash2 |
| **geomean** | 35.8 | 68.3 | **72.1** | 63.9 | 65.8 | DFlash2 |

off, MTP k2 and DFlash2 k7 reproduce the "now" columns of `2026-11-06-27b-dflash-hip-tp1.md` within 3 %.

## Reading

* On these short prompts (39-47 tokens) DSpine is not the best drafter: DFlash2 k7 leads the geomean by 9.6 % over DSpine k7 and 12.8 % over
  DSpine k3 (the default). DSpine k7 wins only code, by 2 %.
* The gap is acceptance, not pass cost. At the same k=7, DSpine accepts less than DFlash2 on chat (2.72 vs 3.08 t/p), prose (2.11-2.25 vs
  2.51-2.58) and recall (3.66-3.75 vs 3.93-4.10), and slightly more on code (5.03-5.35 vs 4.82-5.15). Its pass is about 1 ms longer.
* Tencent reports DSpine above DFlash2 in acceptance. Our port's parity gap (rel-L2 2e-2 against a 2e-3 target, `2026-11-07-dspine.md`) and the
  unrun real-weight parity are the first suspects; prompt length is the second (DSpine's one measured win was a 4420-token prompt).
* No default changes. DSpine stays opt-in.

## Next

1. Acceptance before speed: real-weight parity against Tencent's MLX reference on these prompts (per-row draft agreement, not only rel-L2), and
   f32 fc partial sums. If our drafts differ from the reference's, that is the lever, worth more than every pass-cost lever in
   `docs/plans/2026-11-08-dspine-speed-tp2.md` combined.
2. A long-prompt category set (2k-8k tokens) before concluding DSpine has a niche: the only DSpine win so far is one 4k prompt.

# asa: `--spec auto` policy from the MTP acceptance, Qwen3.8-27B MXFP4 tp=2 (HIP), 2026-10-28

Lane `lane-asa/auto-priors` (PR #223), on `origin/main` `ff31c2081` (after ASP #219). tatooine, 2x R9700, `serve --tp 2 --devices 0,1 --max-batch 1`, localeval `--judge none`, 512 forced tokens, seven category files `scripts/dfh/cats/`, driver `scripts/asa/cat-window.sh` (copy of `scripts/asp/`). Raw: `docs/results/2026-10-28-asa-auto-priors/raw/` (w1 pass traces, w2 stopped early, w3 final). Replay: `tools/asa/replay.py`. Status: PR; **`--spec auto` stays opt-in, the 27B default stays `--spec mtp --spec-k 2`.**

## 1. Why the old policy lost on prose

The ASP policy scored each drafter by its own EWMA of tokens per pass over ms per pass. DFlash tokens per pass swing between 0 and 8, so its estimate is noisy; a lucky pass (probe weight 0.5) tipped the lead and prose spent ~30% of passes on DFlash (passes there cost 20% more per token). Switching itself is almost free: a fresh pass costs +0.8 ms (to MTP) / +0.3 ms (to DFlash) over a normal one (w1 trace).

## 2. Offline replay (step 1)

Window w1 logged every pass (`INTARSIA_SPEC_AUTO=pass`: drafter, tokens, ms, first-after-switch) for auto, forced-MTP and forced-DFlash, 2 rounds, 36 requests. Greedy output is the same in every mode, so a pure run gives, for any generated position, the pass a drafter would take there (nearest pass start; the covering pass is length-biased). The replay reproduces the logged old-policy auto within ~1-2 points per category (prose -3.9 / -6.1 vs real -4.3 / -6.2 % from the best pure mode). `tools/asa/replay.py` (Python) and the ignored test `auto_policy_replays_window_logs` (Rust) give identical numbers for the shipped policy.

Findings from the traces:
- Priors on the old score (tried: seeds, priors, dwell, slower EWMA, lower probe weight) get at best -2.5 % from the best in some category; the per-pass noise is the limit.
- **The MTP acceptance is a free covariate.** Per 96-token block, MTP tokens per pass >= 2.6 meant DFlash was faster in 55 of 57 blocks, < 2.4 meant it lost in 78 of 90.

Shipped policy: stay on MTP; try DFlash when MTP's tokens per pass (EWMA 0.06) reach `theta` 2.5 (after 4 passes); enter DFlash with a prior of 1.5x MTP's rate; leave when DFlash's rate falls under MTP's last measured rate / 1.1 (after 16 passes); while on DFlash, one MTP probe pass every 8 passes (weight >= 0.3) refreshes the covariate. 2 MTP seed passes, priors (2.2 tok, 25.5 ms) / (3.6 tok, 38.0 ms) worth 2 passes. Deterministic. Knobs for A/B: `INTARSIA_SPEC_AUTO=theta=2.6,...` (docs/tuning.md). Replay (sim, % vs best pure mode, ho-chat..rcp-recall): old -0.0 -1.9 -3.9 -1.2 -5.4 -6.1 -2.2; shipped -0.4 +2.1 0.0 +3.1 -0.2 -0.2 +0.4. Sensitivity: theta 2.4 or alpha 0.1 cost prose 1.5-3 %; theta 2.6 is on the plateau (arm `autob`).

## 3. GPU confirmation (w3)

Pooled decode tok/s, 3 interleaved rounds for mtp2, dfl7, auto (autob theta 2.6: 2 rounds), ids gate first. Round-to-round spread 0.1-1.1 % (`raw/w3/cat-table.txt`).

| category | mtp k2 | dflash k7 | auto | autob | auto/mtp2 (3 rounds) | auto vs best |
|---|---:|---:|---:|---:|---|---:|
| ho-chat | **90.3** | 79.8 | 89.3 | 89.0 | 0.980 0.990 0.989 | -1.1% |
| ho-code | 108.3 | **139.5** | 135.4 | 132.5 | 1.249 1.252 1.266 | -2.9% |
| ho-prose | 80.3 | 65.5 | 80.4 | 80.2 | 1.003 1.001 0.999 | 0.0% |
| ho-recall | 95.9 | 95.9 | 96.2 | 97.2 | 0.998 1.008 1.002 | +0.3% |
| rcp-code | 108.1 | **139.4** | 133.9 | 134.4 | 1.235 1.241 1.237 | -3.9% |
| rcp-prose | 83.3 | 68.4 | 83.0 | 83.1 | 0.995 0.998 0.997 | -0.4% |
| rcp-recall | 98.8 | **109.3** | 108.2 | 105.6 | 1.095 1.098 1.096 | -1.0% |

Old auto (ASP, other window): ho-prose 75.4, rcp-prose 77.9 (-6%): prose is fixed (parity with MTP k2). Code stayed ~2-3 points under DFlash (old: -1.6 / -2.9 %; now -2.9 / -3.9 %: the new policy leaves DFlash more readily and spends the MTP probes), chat -1.1 % under MTP (old -0.9 %).

## 4. Gates

- Greedy ids `--spec auto` equal `--spec off`, tp=2, 18 prompts (6 short, 6 ~4k, 6 ~16k), 256 tokens: **18 / 18** (`raw/w3/ids/steps.log`).
- Policy unit tests (`cargo test -p intarsia-core --lib spec::`, 34) and the CPU forced-switch test `spec_auto_greedy_ids_equal_spec_off_with_forced_switches_and_the_policy` (`dflash2_cpu`, 8 tests) pass.
- Mac `cargo test --workspace --release`: 1081 passed, 0 failed at the final head.

## 5. Default decision: no

Rule: auto >= mtp k2 in every category and within 3 % of the best. Misses: ho-chat -1.1 % vs MTP k2 (consistent in all 3 rounds, 0.980-0.990), rcp-code -3.9 % and ho-code -2.9 % vs DFlash. The 27B tp=2 default stays `--spec mtp --spec-k 2`; the recipe/model-dir lookup for the DFlash2 checkpoint was not added (nothing uses it while auto is opt-in; `--spec auto --spec-draft DIR` unchanged).

Not done / next: chat (an auto-mode floor of ~0.7 % remains even with auto forced to MTP: lazy DFlash stash, catch-up); code: a faster return to DFlash after an MTP probe (sim says the probe cadence 6-8 helps), a per-position token-rate predictor instead of a threshold. Three windows used (w2 stopped after round 1 once the covariate policy superseded the first, score-based candidate; its data stays in `raw/w2/`).

## 6. Fleet

Windows (radiance restored each time, `raw/downtime.log`): #6519 w1, #6543 w2 (stopped by SIGTERM), #6555 w3; build #6513, #6539, #6551 (the `ls` in `build.sh` exited 2 after a good build; fixed in 7c990fc6f). Cleaned: `~/intarsiaml-asa`, docker volumes `intarsia-cargo-asa-dev(-hiptest)`.

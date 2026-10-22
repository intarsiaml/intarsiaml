# fx2: gate-definition history in the verdicts (review finding 2, requirement 3)

Nothing here changes a verdict, an exit code or a threshold. Gates that were redefined on 2026-10-21 now also report the
metric and the verdict under the definition they replaced.

## What each output carries

Every record has `gate`, `changed_on`, `log_ref` (LOG.md rows by date, time and lead words), `metrics`, `old_def` and
`current_def` (each: `definition`, `metric_name`, `metric`, `limit`, `verdict`). Code: `crates/intarsia-cli/src/cmd/accuracy/gatedefs.rs`.

| Gate | Old definition | Current definition | Where it is written |
|---|---|---|---|
| `kl_floor` (35b-long scored rows, 10-21 10:43, plus the reference-noise term, 11:23) | KL over ALL rows against the static class floor alone | KL over SCORED rows against `max(static floor, factor x ref noise)` | `accuracy --write-verdict` (`verdicts[].definition_changes`); baseline JSON `modes[].overall.gate_definitions`, which `qualify` copies into its verdict JSON; one `gate-definition:` stdout line per mode |
| `self_vs_base.*` (`--self-golden --self-vs-ref-noise`, 10-21 11:23 / 13:23 / 13:58) | every self-golden budget gated | mean KL vs `factor x noise` and a per-prompt check gate; largest position KL, flips, tail and the default worst-prompt budget are reported, `NOT_GATED` | same three places, only in that mode; one record per changed check plus `self_vs_base.overall` |
| `metal_decode_floor[<size>]` (10-21 12:25 and the clock-normalised row) | raw decode of the best arm >= 85 % of the flat weights-only floor (bw / weights = 40.56 tok/s), no admission | normalised decode of the best valid arm >= 85 % of the context-aware floor; `STATE-INVALID` with no valid arm | `scripts/metal-speed-gate/gate.py` prints `gate-definition:` lines and writes `<dir>/gate-verdict.json` |

`qualify.rs` is not touched (fx1's file): it already copies `modes[].overall` from the baseline JSON.

Human line: `gate-definition: kl_floor [prefill]: old def: FAIL 4.360e-2 (...) | current def: PASS 1.420e-2 (...) [changed 2026-10-21; LOG ...]`.
The prefix and wording avoid the substrings `qualify.rs` and `phase1b-acceptance.sh` parse (`golden gate:`, ` floor [`, `... verdict`, `self-golden =>`, `SELF-STATS`).

## Checks

* CPU unit tests (fleetq #3833, endor): `cargo test -p intarsia-cli` 146 passed, 1 ignored, 0 failed (5 new in `gatedefs::tests`:
  35b-long all-rows FAIL / scored-rows PASS with both metrics, same-rows and NaN, noise-bound pass, `--self-vs-ref-noise` ungated checks with both verdicts, no change no record);
  `cargo test --workspace` all `test result: ok`, no failures (log tail in `raw/`).
* Metal gate tests: `python3 -m unittest scripts/metal-speed-gate/test_gate.py` 11 passed (4 new: rescued run old FAIL / current PASS, STATE-INVALID run, healthy run, real regression).
* Real run, Qwen3-8B BF16, Metal, endor, `accuracy --mode both --decode-positions 16 --max-seq 4096 --golden qwen3-8b-fp32.safetensors`,
  origin/main `659f770c` binary against this branch's binary, same slot:
  * verdict JSON: identical after deleting `definition_changes` (`jq -S`); baseline `modes`: identical after deleting `gate_definitions`;
    stdout identical except the two `gate-definition:` lines and the output file names.
  * the new fields are populated: prefill all rows 2.832e-4 (old def PASS) vs scored rows 2.817e-4 (current PASS); decode 3.313e-4 vs 3.070e-4. Both limits 1e-3 (static, `full-vocab KL`).
  * `golden gate: PASS` on both binaries.

## Not covered, and caveats

* The plain Qwen3-8B golden run never enters self mode, so `self_vs_base.*` is exercised by its unit test only, not by a real run.
* Neither the 8B run nor any unit test reproduces the 35b-long FAIL itself (4.36e-2 against 3.0e-2); the unit test uses those figures from LOG.md.
* The 35b-short advisory baseline (ADR-0008) is judged in `scripts/phase1b-acceptance.sh` and is not given a record.
* The `kl_floor` old definition is "all rows against the static floor alone", the 2026-10-20 definition. The reference-noise term is folded into the same record rather than shown as a separate step.
* The gate-change policy is only a PROPOSED row in `docs/decisions/LOG.md`; no code enforces it or the 30-day retention.

fleetq jobs: #3819 (tests failed to compile: a missing struct field, fixed; the origin/main accuracy run in it is the "before"), #3833 (tests, branch build, "after" run). #3806 cancelled while queued.

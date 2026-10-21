# Lane MG: clock-normalised Metal 4k decode gate (2026-10-21)

## Headline
- The Metal speed gate now divides decode by a machine-state factor before the 85 % context-aware floor. The factor is the GB/s of a fixed reference workload, `scripts/metal-speed-gate/refbw.py`, run around each arm in the same slot, over a stored baseline (`scripts/metal-speed-gate/ref-baseline.json`).
- Raw, reference, factor and normalised values are all printed (`raw/gate-after.txt`).
- User decision 2026-10-21: the 4k decode gate is clock-normalised (`docs/decisions/LOG.md`).
- One real run on endor, fleetq #3302 (exclusive, 24 GB), no role taken down, no downtime.

## What changed
- `refbw.py`: streams four distinct 256 MiB bf16 matrices (1 GiB, larger than the SoC cache) through a matrix-vector product and prints the median GB/s of 7 repetitions as a JSON line. It uses mlx and no intarsia code, so it moves with the machine (GPU clock, DRAM clock, CPU contention) and not with the code under test. About 1 s per call.
- `sameslot.sh`: calls `refbw.py` before and after each arm's localeval and writes `<label>-ref.jsonl`.
- `gate.py`: factor = arm reference GB/s (median of its lines) / baseline GB/s; normalised decode = raw / factor; the floor is judged on the normalised value. A factor outside 0.70 to 1.30 is STATE-INVALID (exit 3). An arm with no reference file keeps the old GPU 1300 MHz / CPU 0.30 admission, so older result dirs still run. The same-slot ratio gates are untouched and stay on raw numbers.
- Unit tests: `python3 -m unittest scripts/metal-speed-gate/test_gate.py` (7 tests, pass): the factor and normalise math, the factor window, the reference median, a slow state rescued (raw 83.3 % of the floor, factor 0.90, normalised 92.6 %), a real regression on a healthy machine still failing, an out-of-window factor giving STATE-INVALID, and the legacy path.

## The run
Binary: `intarsia` built from main `ab4ff9ba` with `--features metal` (fleetq #3294, md5 `5d9ea87f`). Raw: `raw/` (localeval json, macmon, reference lines per arm, server logs, `versions.txt`).

Before (`raw/gate-before.txt`: the gate as it was merged, run with the baseline unset): every intarsia arm admitted (GPU 1338 / 1328 MHz, CPU 0.08 / 0.12), 4k decode 34.91 tok/s = 89.6 % of the 38.96 tok/s context-aware floor, VERDICT PASS.

After (`raw/gate-after.txt`, same dir, baseline 554.2 GB/s):

| arm | GPU MHz | CPU | ref GB/s | factor | 4k raw tok/s (% floor) | 4k normalised tok/s (% floor) |
|---|---:|---:|---:|---:|---:|---:|
| intarsia-1 | 1338 | 0.08 | 557.0 | 1.005 | 34.91 (89.6 %) | 34.74 (89.1 %) |
| intarsia-2 | 1328 | 0.12 | 549.7 | 0.992 | 34.84 (89.4 %) | 35.13 (90.2 %) |
| mlx-lm | 1342 | 0.12 | 533.9 | 0.963 | 32.1 (82.3 %) | 33.3 (85.4 %) |
| llamacpp | 1214 | 0.12 | 554.4 | 1.000 | 33.1 (84.9 %) | 33.1 (84.9 %) |

Floor gate at 4k: best valid arm intarsia-2, raw 34.84, factor 0.992, normalised 35.13 tok/s, 90.2 % of the floor: ok. All four sizes pass; same-slot ratios 1.05 to 1.12 over mlx-lm and 1.05 to 1.07 over llama.cpp. VERDICT PASS.

## Limits, read before trusting the number
- The machine was in a good state (about 1330 MHz, CPU 0.1), so the factors are near 1.0 and the normalisation changed the 4k figure by 0.5 to 1 %. This run shows the output format and that the reference is stable (4 samples 547 to 558 GB/s for the intarsia arms), not a rescue of a degraded run. The rescue is shown only on synthetic data in the unit tests.
- **The baseline is circular for this run:** it is the median of the four reference samples of the two admitted intarsia arms in this same run (554.2 GB/s). It is a first measurement in a good state, not a long-term one. Re-baseline only after a deliberate machine change.
- Not measured: how well the reference tracks decode in a degraded state. G3 found decode follows the GPU clock (r = +0.94) because achievable DRAM bandwidth falls with the clock, which is the mechanism the reference reads, but a run under CPU load (reference and decode together) has not been taken. mlx-lm's first reference sample was 509 GB/s against 555 to 559 for the rest, so single samples can swing by about 8 %; the median of two per arm is used. Follow-up: run the gate on a loaded endor and check that the normalised decode stays within about 1 % of the clean value.
- `refbw.py` needs the mlx-lm uv tool environment on endor (`/Users/josef/.local/share/uv/tools/mlx-lm`), which the gate already needs for the mlx-lm arm.

## Decisions (user, 2026-10-21; rows in `docs/decisions/LOG.md`)
1. CUDA FP8-E4M3 prefill stays opt-in; lane F2 is diagnose-only for the 3/32 `chat_reuse_4k` failures.
2. The Metal 4k decode gate is clock-normalised (this change).
3. Reasoning parsing (R2, PR #57) stays opt-in; the open R2 row is closed.
4. HIP tp=2 fusion (`INTARSIA_TP_FUSION=1`, PR #59): soak first, then default-on if the soak passes.
5. HIP tp=2 need not pass `--self-vs-ref-noise` against tp=1; the `--ref-noise` gate against the vLLM golden stays required (`docs/accuracy.md`).

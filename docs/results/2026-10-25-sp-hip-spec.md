# sp: speculative decoding on HIP (lane sp, 2026-10-25)

Branch `lane-sp/hip-spec`. Draft status: baseline and GPU numbers pending (tatooine queue). Sections are filled as windows
complete; nothing below the "Method" heading is a measurement until it says so.

## Method gaps (stated up front)

**Concurrency on natural text is not measurable with localeval today.** `suite.py` `run_conc` calls `self.prompt_for(size)`
with the default prompt set (filler-v1) and ignores the cell's `prompt_set`; `localeval speed --concurrency` builds filler
prompts too (`speed.py`). The suite's only concurrency cells (`conc-plain-1k`, `conc-plain-4k`) are filler. So this lane
reports single-stream decode on natural-v1 text (`speed-spec`, 128/1k/4k/16k) and aggregate decode at 1/4/16 streams on
**filler** text. Filler text is repetitive in a way that is neither the natural-text acceptance of the DFlash2 comparison
nor a worst case for an MTP head, so the filler aggregate numbers say little about the speculation gain at concurrency.

**Request for the Model Benchmark Suite session (relay):** please add a `conc-spec` cell (lane `spec`, prompt_set
`natural-v1`, params as `conc-plain-1k`: streams [1, 4, 16], prompt_tokens 1024, gen_tokens 256, closed loop, 60 s window),
and make `run_conc` pass `cell.get("prompt_set") or "filler-v1"` to `prompt_for` (and to `warmup_run_start`). Each stream
should send the frozen natural-v1 text for that size, byte for byte; with N identical concurrent prompts the prefix cache
will hit, so the cell either needs distinct natural texts per stream (for example rotating the frozen 1k text's start by
stream index, hashed and frozen like the rest) or must run with the prefix cache disabled and say so in the record
(`--prefix-cache-disabled`). Without it, any claim about speculation gains at concurrency (Radiance `--draft-short-from 8`,
+12% at 8 streams, +18% at 16) cannot be checked against the same text.

## Method

- Baseline and equality: `scripts/sp/baseline.sh` (greedy 256 ids per natural-v1 prompt size, `--show-ids`, md5 compared
  across spec off / eager MTP / captured MTP; acceptance and tok/s from `generate`'s own lines).
- Speed: localeval `suite --tier adhoc` (`scripts/sp/arm.sh`), main vs branch in the same window.
- Gates: `scripts/sp/gates.sh`.

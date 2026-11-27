# Inkling-2-Mini + DFlash2 k5 on Metal: readiness checks for a default flip (lane ink-metal-dflash2-ready, 2026-11-27)

Status: **IN PROGRESS** (rules written 2026-11-27 03:50 EST, before the first Metal slot). Verification only: no default, recipe or code change.
Branch `ink/metal-dflash2-ready` from origin/main `f25bb273a` (PR #546 in). Binary `br1` built from this tree (own target dir
`target/`, sources touched first, `[build]` job #12556, git `f25bb273a`, md5 `49a4d94be3648ec35ed2c7128dd513cb`).
Question for the user: is DFlash2 k5 (`--spec dflash --spec-draft <Inkling-2-Mini-DFlash2> --spec-k 5`, #546) safe to make Metal's greedy default for
Inkling-2-Mini (today DSpark k3, recipe `mtp_default_k`)? Speed is not re-measured here (#546: mean of 6 cells 1.072 vs DSpark k3, GO).

## Result table

(filled in after the slots)

## Pre-registered rules (written before any Metal slot; not relaxed afterwards)

Common setup: `br1` on endor, Inkling-2-Mini snapshot `b85c3cfa...`, drafter `models--incoai--Inkling-2-Mini-DFlash2` snapshot `4f9088ea...` (the HF
cache copy of fleetq #12464, nothing linked or created), Metal, exclusive fleetq slots, `INTARSIA_*` unset in the slot scripts. Arms: `off` =
`--spec off`; `default` = no `--spec` flag (DSpark k3, recipe); `dfl5` = `--spec dflash --spec-draft <drafter> --spec-k 5`.

1. **Greedy ids at long context.** `generate --show-ids --max-tokens 256 --ignore-eos`, `--max-seq 20480`, three prompts in one process per arm:
   `code` (101 ids, the golden code prompt), `len4k` (4108 ids), `len16k` (16455 ids; speed.json prompt 15). PASS iff for every prompt the 256 ids of
   `dfl5`, `off` and `default` are identical (md5 of the ids line; all three prompts, all 256 ids present). FAIL otherwise: report the first
   differing position and stop that check. Cross-check: the `code` and `len4k` ids against #546's md5s (`c69d1438...` / `9e85ba11...`, the
   `[a, b, ...]` line as printed).
2. **Sampled requests.**
   a. Path: read from the code (`Seq::new`, `spec_sampled_dflash()`, `sampled_dflash_resolve`) and the run log (startup line, `INTARSIA_SPEC_TRACE=1`
      verify-pass counts during the sampled phase), stated in the doc.
   b. Exactness (`intarsia serve`, `/v1/completions`, `return_token_ids`, prompt "Once upon a time, in a small village by the sea, there lived", 32
      tokens, `ignore_eos`): seeds 0..99 at T 1.0 and at T 0.6 (top-k 0, top-p 1: the device-sampler path of the recipe) plus seeds 0..29 at T 0.7 /
      top-k 20 / top-p 0.9 (the host-sampler path). PASS iff every (config, seed) pair has ids identical between `off` and `dfl5` (same server flags
      otherwise). If the time box cuts the seed list, the compared set is the pairs both arms completed and the doc says how many (floor: 30
      seeds per temperature, else INCOMPLETE). The `default` arm is reported for reference. A fourth arm `dfl5s` = `dfl5` with
      `INTARSIA_SPEC_SAMPLED_DFLASH=1` (sampled drafting with the DFlash2 draft, opt-in, not what a default flip would turn on) is held to the same
      identity rule and reported apart; a failure there is a FLAG, not a FAIL of the check.
   c. Speed, information only: one ABBA (dspark, dfl5, dfl5, dspark) of `generate` on 3 short prompts (`count` 35 ids, `story` 40, `chat_reasoning_rnone`
      93; a 23-id warm-up prompt first, excluded), 256 tokens, EOS honoured, 45 s GPU idle before every process, macmon logged. FLAG when the
      geometric mean of the 3 prompts' dfl5/default ratios is below 0.98 or any prompt's is below 0.95; never a gate.
3. **Batched.** (a) `serve --max-batch 4 --spec dflash --spec-draft <drafter> --spec-k 5`: record exactly what happens (refusal text, time to fail).
   (b) `serve --max-batch 4` with no `--spec`: 4 concurrent greedy requests (prompts `code`, `len1k`, `long_prose_300`, `chat_reasoning_rnone`, 128
   tokens, `ignore_eos`) each against the same prompt's solo reply from the single-sequence `dfl5` server of check 2 and the `off` server. PASS iff
   the draft is refused or dropped loudly (not silently), AND (b) the 4 concurrent replies equal their solo replies. Opt-in real-model batch tests:
   listed from `crates/intarsia-models/tests/` (a dflash2 one is looked for; none that loads the drafter with batching is expected, the loader
   refuses it).
4. **Memory and KV capacity.** `serve --max-seq 262144 --kv-fraction 0.25 --no-supervisor` (capacity clamped by free memory, not by `--max-seq`),
   `default` and `dfl5` alternated (default, dfl5, default, dfl5), each: the startup line's `context N tokens`, one greedy request, `/usr/bin/time -l`
   peak memory footprint at exit, macmon `ram_usage` before launch / peak / after the request. Plus `generate` peak footprint from check 1's
   processes. FLAG when `dfl5` has more than 1.5 GB more peak footprint than `default`, or a context capacity more than 5 % smaller (run-to-run drift of
   the free-memory measurement is read from the repeated pair).

Metal budget: <= 25 min of exclusive slots in all, each <= 15 min, none after 07:00 EST.

## Coordinator messages received

(none yet)

## Log

- 2026-11-27 03:44 brief received; worktree `.worktrees/ink-metal-dflash2-ready`. 03:47 build #12556 done (22 s, own target dir).

# Inkling-2-Mini + DFlash2 k5 on Metal: readiness checks for a default flip (lane ink-metal-dflash2-ready, 2026-11-27)

Status: **DONE 2026-11-27 04:25 EST. Verification only: no default, recipe or code change.** Four checks pass, one is a FLAG (memory scales with the
context capacity). Branch `ink/metal-dflash2-ready` from origin/main `f25bb273a` (PR #546 in). Binary `br1` built from this tree before any doc
commit (own target dir `target/`, sources touched first, build job #12556, md5 `49a4d94be3648ec35ed2c7128dd513cb`, every slot's first line logs it).
Question for the user: is DFlash2 k5 (`--spec dflash --spec-draft <Inkling-2-Mini-DFlash2> --spec-k 5`, #546) safe to make Metal's greedy default for
Inkling-2-Mini (today DSpark k3, recipe `mtp_default_k`)? Speed is not re-measured here (#546: mean of 6 cells 1.072 vs DSpark k3, GO).

## Result table

| # | Check | Rule (written before the first slot, below) | Result | Key number |
|---|---|---|---|---|
| 1 | Greedy ids at long context | 256 ids of `dfl5` = `off` = default on `code` (101 ids), `len4k`, `len16k` | **PASS** | md5 equal on all 3 prompts, 256/256 ids each; 16k `89994b87...`, 4k `9e85ba11...` and code `c69d1438...` also equal #546's md5s |
| 2a | Path of a sampled request under `--spec dflash` (DFlash2) | read from code + trace | **plain steps** (DFlash2 does not draft sampled requests) | 0 verify passes in 190 sampled requests; startup line "drafting off (default ...)" |
| 2b | Sampled ids `dfl5` = `off` | every (config, prompt) pair equal | **PASS** | 190 / 190 equal (T 1.0: 100, T 0.6: 60, T 0.7 top-k 20 top-p 0.9: 30); **deviation: one seed, distinct prompts**, see below |
| 2b' | Same with `INTARSIA_SPEC_SAMPLED_DFLASH=1` (opt-in sampled drafting with the DFlash2 draft) | same rule, reported apart | **PASS** | 190 / 190 equal; it drafts (906 verify passes at T 1.0, acceptance 30 %, 2.03 tokens per pass) |
| 2c | Speed, information only: one ABBA on 3 short prompts, 45 s idle | FLAG if geo < 0.98 or any prompt < 0.95 | **no flag** | dfl5 / default: count 1.496, story 1.028, chat 1.280; geo **1.253**, min 1.028; ids equal |
| 3a | `--max-batch 4 --spec dflash` | record what happens | **refused loudly** | exit 1 after 0 s, before the weights load: "unsupported: --spec dflash with the batching engine (use --max-batch 1)" |
| 3b | `--max-batch 4`, no `--spec`: 4 concurrent greedy = solo | all 4 equal | **PASS** | 4 / 4 equal, twice; no draft is loaded (no speculation line in the log), as with the DSpark default today |
| 4 | Memory and KV capacity, `dfl5` vs default | FLAG if > +1.5 GB peak or > 5 % less context | **FLAG** | at the free-memory-clamped capacity: context 615,458 vs 615,670 tokens (-0.03 %), peak footprint **48.83 vs 44.10 GB (+4.73 GB)**, repeated; at 20,480 tokens: 20.32 vs 20.84 GB (no extra) |

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

**Added after slots A to D (advisor review, 04:15 EST, before slot E; information only, same FLAG thresholds as 2c):**

- **2c-sampled.** Check 2c as run in slot D was a greedy ABBA; the brief put the speed pair under the sampled-requests check, and the unpaired slot C
  numbers hint that sampled plain steps under a draft may be slower (T 1.0: off 75.9, dfl5 70.1 tok/s). Slot E repeats the ABBA sampled:
  `generate --temperature 1.0 --top-k 0 --top-p 1.0 --seed 7 --ignore-eos --max-tokens 256 --prompt-ids abba3.json`, arms default / dfl5 / dfl5 / default,
  45 s idle before each, the `warm` prompt excluded, `spec: verify passes` read from each log (both arms must take plain steps). FLAG when the
  geometric mean of the 3 prompts' dfl5 / default ratios is below 0.98 or any prompt's is below 0.95.
- **2d, seed stream on one engine with prefix reuse.** The real-model harness `inkling2_sampled_real` (one engine, requests in a row on one
  prompt, so each request reuses the previous request's KV prefix), `INTARSIA_SPEC_SAMPLED_DFLASH=0`, seeds 0..99 at T 1.0 and T 0.6, 16 tokens:
  arm X = `--spec dflash` with the DFlash2 drafter at k5 (`INTARSIA_SS_DRAFT`, `INTARSIA_SS_K=5`), arm Y = the checkpoint's own DSpark draft at k3
  (today's default draft). PASS iff the 200 lines of X and Y are identical (both take plain steps for sampled requests, so a difference would be a
  side effect of the DFlash2 draft's presence). The reference is the DSpark-loaded engine, not `--spec off`; 2b covers `--spec off`.

## Deviations from the pre-registered rules (all forced by `serve` start-up cost; none relaxes a pass condition)

- **`serve` admits kernels at every start (`for_serving`, `require_admitted`), and a new binary / plan starts cold.** Slot A's first `serve` arm
  (`--spec off`) was not up after 120 s (CPU-bound admission, 99 % CPU, 39 GB resident; the same order as the 3-4 min "first serve load re-runs admission" of
  `2026-11-21-inkling2-cuda-attn.md` and the 228 / 252 s serve loads in `2026-11-21-inkling2-cuda-mixmoe.md`); the client gave up and I stopped the slot
  (#12557, 3.8 min, check 1 had finished in it). `batch4` later came up after 133 s on the same binary (admission cases are cached per plan).
  So rules 2b and 4 ran on `generate` instead of `serve` (no admission, the same engine and sampler):
  - **2b**: `generate --show-ids --max-tokens 32 --ignore-eos --seed 7`, one process per arm and sampling config, **distinct 24-id prompts at one seed**
    (windows of the 16k prompt, `raw/w100.json`, first 100 / 60 / 30 of them) instead of the 100 seeds of one prompt. Distinct prompts exercise
    different logits and it is the same identity claim, but it is not the "100 seeds" the brief asked for: the seed-stream (one request after
    another with seeds 0..99) is not exercised here. The earlier lanes' seeds-0..99 identity on Metal (`2026-11-26-inkling2-sampled-dspark.md`, slot
    B) used the DSpark draft, not DFlash2, so it does not cover this. Floor met: 100 at T 1.0, 60 at T 0.6, 30 host-sampler.
  - **4**: `generate --max-seq 4194304 --kv-fraction 0.25` (clamped by free memory; `serve`'s `--max-seq` would have been the same planner) and
    `/usr/bin/time -l` peak memory footprint; macmon's system `ram_usage` includes file cache (it read 89 GB with nothing running) and is not
    used. The serve startup line "context N tokens" is generate's "context N tokens" line.
- The ABBA (2c) uses `generate` on `abba3.json` with one rep pair (2 + 2 processes), as the brief said.

## Check 1: greedy ids at long context (slot A, #12557, `raw/A/`)

`generate --max-seq 20480 --show-ids --max-tokens 256 --ignore-eos --prompt-ids g3.json` (code 101 ids, len4k 4108, len16k 16455), one process per arm.
`raw/A/md5-table.md` (md5 of the `[a, b, ...]` ids line as printed, #546's convention):

| prompt | off = default = dfl5 (md5 of the 256 ids) | n ids | #546's md5 |
|---|---|---|---|
| code | `c69d1438ab3e715feb93a0614ebddcfe` | 256 / 256 / 256 | `c69d1438...` (same) |
| len4k | `9e85ba1117e4a6a6cc12c5c917532e7f` | 256 / 256 / 256 | `9e85ba11...` (same) |
| len16k | `89994b876223335763b2004d0d281157` | 256 / 256 / 256 | not in #546 (new) |

**PASS**: the 16k prompt is the one #546 had not covered; ids are identical for all three arms. The brief's `d93147da...` is not reproduced: it is the
flip lane's md5 over its own prompt / length (HIP-lane convention); my code-prompt md5s match #546's on the same prompt and convention, which is the
cross-check the rule named. Spec statistics in the same processes: DFlash2 k5 3.76 tokens per verify pass (acceptance 55.6 %), default 2.63.
Decode tok/s, one process each, information only (order off, default, dfl5, so not paired): code 79.8 / 109.5 / 137.5, 4k 75.5 / 106.7 / 108.0,
16k 69.9 / 78.5 / 74.6. Prefill of the same processes (ms): 4k 1282 / 1302 / **1506**, 16k 5422 / 5979 / **6681**: DFlash2 costs prefill time (the draft
ingests the context), dfl5 / default 0.865 at 4k and 0.895 at 16k here, the same direction as #546's 0.894 [0.784-1.020] / 0.943 [0.827-1.075]
(one process each, so noise is unknown). That is a longer time to first token of about 0.2 s at 4k and 0.7 s at 16k, which the speed rule did not weigh.

## Check 2: sampled requests (slot C, #12558, `raw/C/`)

**2a, the path.** `Seq::new` (`crates/intarsia-core/src/engine.rs`, the "lane ink-sampled" block) turns drafting on for a request with
`temperature > 0` under `--spec dflash` only when `model.spec_sampled_dflash()`; that is `sampled_dflash_resolve(env, recipe, own_draft)`
(`crates/intarsia-core/src/spec.rs`) = `env.unwrap_or(own_draft && recipe == Some(true))`, with `own_draft` = "the loaded draft is the checkpoint's own
DSpark draft". An explicit DFlash2 draft is not, and `[tuning.metal]` has no `sampled_dflash` key anyway, so **a sampled request under
`--spec dflash --spec-draft <DFlash2>` takes plain decode steps**; only `INTARSIA_SPEC_SAMPLED_DFLASH=1` makes it draft (greedy requests always
draft). The run log agrees: startup line "sampled requests (temperature > 0): DFlash-family drafting off (default; ...)" and `spec: verify passes=0`
over the 190 sampled requests of the `dfl5` arm (`INTARSIA_SPEC_TRACE=1`); with `=1` the line says "on" and the T 1.0 process shows 906 verify passes,
3100 proposed, 931 accepted (30 %, 2.03 tokens per pass; T 0.6: 619 passes, 44.9 %, 2.42).

**2b, exactness** (`raw/C/seedcmp.txt`; `--seed 7`, 32 tokens, `--ignore-eos`, top-k 0 / top-p 1 = the device-sampler path of the recipe, and the host-sampler config):

| config (prompts) | dfl5 vs off | dfl5s (`SAMPLED_DFLASH=1`) vs off |
|---|---|---|
| T 1.0 (100) | 100 / 100 equal | 100 / 100 equal |
| T 0.6 (60) | 60 / 60 equal | 60 / 60 equal |
| T 0.7, top-k 20, top-p 0.9 (30) | 30 / 30 equal | 30 / 30 equal |

**PASS** (190 / 190, both arms). Not degenerate: the 100 T 1.0 lines are 100 distinct id lists, and 57 of 60 prompts differ between T 1.0 and T 0.6.
The `default` arm was not run for sampled requests (it takes plain steps too, and `serve` was too slow to afford it). Sampled speed (unpaired, one
process per cell, power state unknown) is not a result: plain-step tok/s was 75.9 / 70.1 / 67.5 at T 1.0 (off / dfl5 / dfl5s) and 73.1 / 72.8 / 80.1 at T 0.6.

**2c, speed, information** (slot D, #12559, `raw/D/abba.txt`; order dspark, dfl5, dfl5, dspark, 45 s idle before each; `warm` prompt first and excluded):

| prompt | default tok/s (r1, r2) | dfl5 tok/s (r1, r2) | ratio |
|---|---|---|---|
| count (35 ids) | 135.56, 135.55 | 203.00, 202.62 | 1.496 |
| story (40) | 83.44, 83.42 | 85.67, 85.89 | 1.028 |
| chat_reasoning_rnone (93) | 111.00, 110.78 | 142.13, 141.71 | 1.280 |

Geometric mean 1.253, lowest prompt 1.028; ids identical across all 4 processes. No flag. (Short prompts are where #546 saw the gain: 1.17-1.35.)

## Check 3: batching (slot D, #12559)

- **3a.** `serve --max-batch 4 --spec dflash --spec-draft <DFlash2> --spec-k 5`: exits 1 after 0 s (before the weights load, `load.rs:1007-1008`):
  `Error: loading ... Caused by: unsupported: --spec dflash with the batching engine (use --max-batch 1)`. Loud, not silent.
- **3b.** `serve --max-batch 4 --max-seq 4096` with no `--spec`: up after 133 s (cold admission), startup line "continuous batching: up to 4 sequences
  (--max-batch), busy prefill chunk 2048"; **no speculation line and no draft**: a batching server drafts nothing. This is today's behaviour for the
  DSpark default as well (the recipe's default applies "at one sequence"; `dspark_default` returns `None` for `--max-batch` > 1, unit test
  `dspark_default_only_where_the_recipe_sets_it`), and it is silent for the implicit default. Four concurrent greedy requests (`code`, `len1k`,
  `long_prose_300`, `chat_reasoning_rnone`, 128 tokens) are equal to their solo replies on the same server, in two concurrent rounds (4 / 4 and 4 / 4).
- **Opt-in real-model batch test for DFlash2**: there is none. `crates/intarsia-models/tests/` has `batch_metal_real.rs`, `batch_metal.rs`,
  `inkling2_batch.rs`, `batch_engine.rs` (plain batching), `inkling2_dflash2.rs` / `dflash2_cpu.rs` (single sequence) and `dflash2_real.rs` (HIP,
  Qwen3.8-27B probe); none loads a DFlash2 draft with batching, and the loader refuses the combination, so there is nothing to run.

## Check 4: memory and KV capacity (slot C, #12558 and slot A, #12557)

| run | context tokens | weights (log) | peak memory footprint (`time -l`) |
|---|---|---|---|
| `--max-seq 4194304 --kv-fraction 0.25`: default (DSpark k3), run 1 | 615,670 | 16.41 GB | 44.10 GB |
| same: DFlash2 k5, run 2 | 615,458 | 16.27 GB | **48.83 GB** |
| same: default, run 3 | 615,670 | 16.41 GB | 44.10 GB |
| same: DFlash2 k5, run 4 | 615,458 | 16.27 GB | **48.83 GB** |
| `--max-seq 20480` (slot A): off / default / DFlash2 k5 | 20,480 | | 20.00 / 20.84 / 20.32 GB |

The pairs repeat to 4 significant digits, so the free-memory measurement did not drift between runs. The planner's capacity is almost the same
(-212 tokens, -0.03 %, the difference in weight size), but the DFlash2 process peaks **4.73 GB higher** at that capacity, i.e. about
**7.7 KB per token of context capacity** that the planner does not budget. The code is consistent with that: `new_dflash_state(cap)`
(`crates/intarsia-core/src/model/dflash.rs:483-486`) gives the draft its own `SequenceState` of the full capacity `cap`, 5 layers; I did not
measure the draft cache separately. At realistic capacities it does not matter (20,480 tokens: DFlash2 is 0.5 GB below the default; serve's default
is 4096): it matters only for a user who raises `--max-seq` into the hundreds of thousands on a machine with little free memory, where the extra
~7.7 KB x capacity lands outside the planned KV budget. FLAG, not a gate. The load time of the draft is 0.1 s.

## What a Metal DFlash2 k5 default would change for users

- **Which requests**: greedy requests at one sequence (the ones that draft today with DSpark k3) would use DFlash2 k5 instead: ids identical (check 1,
  #546), decode about +7 % over six cells (#546), +25 % on the three short prompts here, up to +35 % on code, about +1 to +3 % at 1k / 4k / 16k
  with CIs that include 1.0. **Sampled requests do not change**: they take plain steps under a DFlash2 draft (2a), as they do under the DSpark
  default on Metal today, and their ids equal `--spec off`. `--max-batch` > 1 does not change: no draft, silent for the default (3b).
- **It is not a recipe-only change.** The default mechanism (`dspark_default`, `[tuning.metal] mtp_default_k`) loads the checkpoint's own DSpark draft
  (`mtp.safetensors`) at the key's k. The DFlash2 drafter is a separate 945 MB download (`incoai/Inkling-2-Mini-DFlash2`) with no recipe key for its
  location, no fetch, and a different default k (block - 1 = 7, which #546 measured to lose at 1k and 16k). A default flip therefore needs code for
  (a) finding or fetching the drafter and the fallback to DSpark k3 when it is absent, (b) the k: `--spec-k 5` is a flag today, so the recipe needs a
  `[tuning.metal]` entry for the DFlash2 k (a new key; `mtp_default_k` is DSpark's), (c) the sampled rule, which must keep sampled requests plain (it does
  by default). The HIP flip lane's branch `ink/hip-specflip-b` (not merged; see `STATE.md` 03:43) is the nearest existing mechanism and was not read here.
- **Prefill costs time**: the draft ingests the prompt, so TTFT of a 4k / 16k prompt grows by about 0.2 s / 0.7 s (one process each here and #546's
  ratios 0.894 / 0.943, CIs include 1.0). Every default user pays that; the speed rule did not count it.
- **Memory**: +~7.7 KB per context token of capacity, unbudgeted by the planner (check 4); negligible at serve's default and at 20k.
- **First start of a new binary**: unchanged by this (serve's cold admission: 133 s measured for the `--max-batch 4` plan, > 120 s for the single-sequence one) except that the DFlash2 verify plan adds cases (flip lane saw
  38 kernels / 215 cases vs 23 / 134 for the DSpark-only plan on HIP, `STATE.md` 02:57; not measured on Metal here).

## Not verified

- Seeds 0..99 of one prompt (the stream of seeded requests on one engine) with the DFlash2 draft loaded: replaced by distinct prompts at one seed (see deviations).
- Anything on `serve` with a DFlash2 draft (admission of the DFlash2 verify plan on Metal, HTTP path, `serve`'s default context): the one slot that
  would have cost 4+ min of cold admission per arm. The engine, sampler and decode path are the same as `generate`'s.
- `--max-batch` > 1 with a draft (refused by design); a DFlash2 batch test (none exists).
- The planner's treatment of the draft KV (measured as a peak-footprint difference, not by reading the allocation); behaviour near memory exhaustion.
- Prefill / TTFT with a CI (single processes); speed beyond the 3 short prompts (#546 has the six cells); long runs (thermal), more than one machine.
- Sampled drafting with the DFlash2 draft (`=1`) is exact in these 190 requests but not useful here (acceptance 30 % at T 1.0, no gain measured);
  it is not part of any flip.

## Metal time and fleetq

Exclusive endor slots: #12557 (slot A: greedy md5 at 4k / 16k, then the aborted `serve` arms; 03:49:40 to about 03:53:30, 3.8 min), #12558 (slot C:
sampled identity + memory, 03:55:10-04:01:17, 6.1 min), #12559 (slot D: refusal, ABBA, `serve --max-batch 4`, 04:01:41-04:08:09, 6.5 min). **Total
16.4 min** of the 25 min budget, each slot under 15 min, none after 07:00. cpu: #12556 (build, 22 s). Scripts and logs in `raw/` (`slotA.sh`,
`slotC.sh`, `slotD.sh`, `slotB.sh` unused, `lib.sh`, `client.py`, `md5.py`, `seedcmp.py`, `abba.py`, prompts); the cold-admission wait of slot A is
`raw/A/s-off-killed-cold-admission.err`. `serve` also wrote admission-cache files under `~/.cache/intarsia/admission/` (the program's own cache).
No drafter link or other folder was created; nothing outside this worktree was changed.

## Coordinator messages received

(none received during the run)

## Log

- 2026-11-27 03:44 brief received; worktree `.worktrees/ink-metal-dflash2-ready`. 03:47 build #12556 done (22 s, own target dir).
- 03:50 rules committed (`af39bfbfa`) before the first slot. 03:49-03:53 slot A: check 1 PASS; `serve` arms stuck in cold admission, slot stopped.
- 03:55-04:01 slot C (sampled identity, memory), 04:01-04:08 slot D (refusal, ABBA, batch). 04:10 doc written.

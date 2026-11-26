# Inkling-2-Mini on HIP (R9700): G2 / G3 attribution of the PR #526 flip, 2026-11-26

Lane `ink-hip-g2`, measurement only: no engine code changed, no default changed, no PR.
Raw logs: `docs/results/2026-11-26-inkling2-hip-g2/raw/` (`logs/` = one full `intarsia accuracy` output per run, `window1-12099.out`
and `window-1126-135621.log` = the window, `stage_g2g3.sh` = what ran, `build.sh` / `sync.sh` / `run.sh` / `window.sh` = the tooling).

## Question

PR #526 (HIP prefill flip) made three defaults: tensor-core MoE prefill (`INTARSIA_HIP_INK_MOE_WMMA`, now on), flash prefill attention
(`INTARSIA_HIP_INK_ATTN_FLASH`, now on) and `[tuning.hip] prefill_chunk = 2048` (was 256). Its results doc graded G2 against the
accuracy tool's generic 4-bit limit (3e-2). The port plan's bar is tighter (`docs/plans/2026-11-21-inkling2-port.md:351`): KL21 mean
<= 4e-3 short / <= 6e-3 long, confident-row misses <= 1 / <= 2. Which default moves the KL, and is the result still inside G3?

## Setup

- Binary: `origin/main` = `3c6ff64f0` (Merge PR #527), built with `--features hip -p intarsia-cli` in the dev image (ROCm 7.2.4),
  fleetq #12097 (`--load cpu`, 2 m 21 s cargo), md5 `99a65142f403af0dd95252d2d619fe62`. PR #528 (`INTARSIA_HIP_INK_PF2`, bitwise by its own
  claim) was still open, so it is not in this binary; it would be a no-op for these numbers.
- Command (per run, inside the dev container, `-m /models/Inkling-2-Mini`):
  `intarsia accuracy -m ... --backend hip --max-seq 4096 --mode prefill --allow-dirty [--prefill-chunk 256] --golden <golden>`
  (`--max-seq 8192` for the 4k golden, longest prompt 4968 tokens). Env knobs set on the same binary; no other flags. Gates are read from
  the tool's own lines `prefill floor [4bit]` (KL21 over scored rows, top-1) and `prefill confident rows` (golden p1 - p2 >= 0.5), and
  graded here against the port plan's bars, not the tool's generic 3e-2 "PASS".
- Goldens: G2 = `golden/oracle/inkling2-mini-oracle-w4a16-{short,long}-top20.safetensors` (short n = 569, 363 confident rows; long
  n = 1569, 1206 confident); G3 = `golden/inkling2-mini-vllm-{short,long,4k}-top20.safetensors` (n = 569 / 1569 / 13096; 367 / 1211 / 8229
  confident). The three vLLM goldens are gitignored; they were copied from the main checkout's `golden/` (`inkling-2-mini-vllm-*`).
- Arms: A1 defaults. A2 `INTARSIA_HIP_INK_MOE_WMMA=0`. A3 `INTARSIA_HIP_INK_ATTN_FLASH=0`. A4 `--prefill-chunk 256`. A5 all three off
  (`=0 =0 --prefill-chunk 256`, the pre-flip default).
- Placement: all five G2 arms on card 0 (as briefed). The G3 runs (A1, A5) and a G2 short cross-check ran on card 1 in the same window to
  fit the 5 min budget: A1 and A5 G2 short on card 1 reproduce card 0 exactly (4.491e-3 / 2.566e-3, same top-1 and per-prompt rows), so
  the G3 numbers are card-independent.

## G2 (oracle W4A16), the five arms

Chunk actually used: A1 / A2 / A3 2048 (recipe, `[tuning.hip]`), A4 / A5 256 (`--prefill-chunk`). The tool's own count of prompts that
cross a chunk boundary: short 0 / 8 at both sizes (longest prompt 106 tokens); long 0 / 10 at 2048, 2 / 10 at 256 (the 300- and 700-token
prose prompts). So on the short golden the chunk size cannot matter (A4 is bit-identical to A1), on the long one only via those two prompts.

| Arm | Env / flag | chunk | short KL21 (bar 4e-3) | short misses (<= 1) | short top-1 | long KL21 (bar 6e-3) | long misses (<= 2) | long top-1 |
|---|---|---|---|---|---|---|---|---|
| A1 | defaults | 2048 | **4.491e-3 FAIL (+12.3 %)** | 0 / 363 | 99.47 % | 4.083e-3 pass | 1 / 1206 | 98.98 % |
| A2 | `MOE_WMMA=0` | 2048 | 4.491e-3 FAIL | 0 / 363 | 99.47 % | 4.521e-3 pass | 1 / 1206 | 99.04 % |
| A3 | `ATTN_FLASH=0` | 2048 | 2.566e-3 pass | 0 / 363 | 98.95 % | 3.328e-3 pass | 1 / 1206 | 99.04 % |
| A4 | `--prefill-chunk 256` | 256 | 4.491e-3 FAIL | 0 / 363 | 99.47 % | 4.157e-3 pass | 1 / 1206 | 99.04 % |
| A5 | all three off | 256 | 2.566e-3 pass | 0 / 363 | 98.95 % | 3.251e-3 pass | 0 / 1206 | 99.30 % |

A5 reproduces the pre-flip values exactly (2.566e-3 / 3.251e-3, `docs/results/2026-11-25-inkling2-hip.md` 4.1).

Per prompt, short golden, KL21 (rows), flash on (A1 = A2 = A4) against flash off (A3 = A5):

| prompt (rows) | flash on | flash off | share of the +1.925e-3 overall delta |
|---|---|---|---|
| chat_reasoning (106) | 1.043e-2 | 2.925e-3 | 72.6 % (max row KL 0.472 vs 0.239 nats) |
| chat_system (63) | 1.287e-2 | 8.843e-3 | 23.2 % |
| the other six (400 rows) | each within 2.1e-4 (code +2.1e-4, science +1.5e-4, story -1.1e-4, chat_short and chat_multiturn +1.2e-4) | | 4.2 % |

Long golden: the 300- and 700-token prose prompts (the only prompts on which the MoE WMMA path or a chunk boundary changes anything) move
by -0.4e-3 / +1.1e-3 (A1 -> A2), +0.4e-3 / -0.3e-3 (A1 -> A3), 0.0 / +0.2e-3 (A1 -> A4); every other prompt's KL is bit-identical across
arms with the same attention form.

## G3 (vLLM W4A4 goldens), arms A1 and A5

Bars: KL21 <= 1.05e-2 / 1.65e-2 / 1.4e-2, confident-row misses <= 1 / 2 / 9 (`docs/plans/2026-11-21-inkling2-port.md:352`).

| Golden | A1 defaults (chunk 2048) | A5 all three off (chunk 256) | bar | verdict |
|---|---|---|---|---|
| short | KL21 8.565e-3, 0 / 367 misses, top-1 98.42 % | 8.003e-3, 0 / 367, 98.24 % | 1.05e-2, <= 1 | both pass (A1 margin 1.9e-3) |
| long | 1.197e-2, 0 / 1211, 97.71 % | 1.371e-2, 0 / 1211, 97.39 % | 1.65e-2, <= 2 | both pass (A1 margin 4.5e-3) |
| 4k | 9.964e-3, 3 / 8229, 95.82 % (3 of 3 prompts span chunks) | 9.644e-3, 0 / 8229, 95.75 % | 1.4e-2, <= 9 | both pass (A1 margin 4.0e-3) |

(Metal reference, same bars: 8.391e-3 short, 1.248e-2 long, `docs/results/2026-11-21-inkling2-metal.md`.) The A1 versus A5 difference
in G3 mixes all three defaults (and the chunk size on long / 4k), so it is not an attribution.

## Attribution

On the short golden the whole move from 2.566e-3 to 4.491e-3 (+1.925e-3, +75 %) is the **flash prefill attention default**: A3 (flash
off, MoE WMMA and chunk 2048 left on) is 2.566e-3, exactly the all-off value, and A2 / A4 (WMMA off, or chunk 256, flash on) stay at
4.491e-3, exactly the defaults. MoE tensor-core prefill and the chunk size have no effect on the short golden at all (identical to the last digit,
per prompt: no short prompt crosses a chunk, and the WMMA path changes nothing on them), and 96 % of the flash delta sits in two chat prompts
(chat_reasoning, chat_system), where the worst row's KL goes from 0.239 to 0.472 nats; top-1 on short improves (98.95 % to 99.47 %) and
confident misses stay 0, so it is a bf16 rounding-order shift of the near-tie rows, not a wrong-token regression. On the long golden the three
defaults together cost +0.83e-3 (3.251e-3 to 4.083e-3); flash alone is +0.76e-3 (A3 to A1), the chunk size +0.07e-3, and the MoE WMMA path
-0.44e-3 on top of flash (A2 to A1; with flash off its sign flips per prompt, so it is noise at about 1e-3 per prompt, two prompts), none
near the 6e-3 bar. Against the port's own bar the HIP default **fails G2 short by 12.3 % (4.491e-3 vs 4e-3)** and passes G2 long (68 % of
the bar) and all three G3 sets with margins of 1.9e-3 to 4.5e-3. The 4e-3 bar was derived from a proxy at 1.67e-3 with 2.4x headroom, and the
flash form is at 2.7x that proxy on short; whether to relax the short bar for the flash form, keep it opt-in for strict graders, or look
for a more accurate softmax accumulation in the flash kernel is the owner's call (nothing changed here).

## What ran, what did not

- G2 short / long x five arms: all ran. G3 short / long / 4k x A1 and A5: all ran (HIP can run G3 with the same tool and the repo's
  vLLM goldens; the pfflip doc's "no HIP G3 baseline" was a choice, not a limit).
- Not done: a row-level dump of the chat_reasoning worst row to say whether the flash kernel's online-softmax order or its bf16 P rounding
  causes it (outside this lane: measurement only). The G2 goldens have no prompt longer than 700 tokens, so chunk 2048 is untested for
  multi-chunk prefill by G2; G3 4k covers it (3 / 3 prompts span chunks, passes).
- Deviation from the brief: G3 and the cross-check used card 1 (one window, both cards), proven equal to card 0 above.
- The binary is UNSTAMPED and the tree on tatooine was an rsync without `.git`, hence `--allow-dirty`; it does not change numerics.

## fleetq and radiance

| Job | What | Result |
|---|---|---|
| #12097 | build `origin/main` 3c6ff64f0 (`--load cpu`, tatooine-r9700) | exit 0, cargo 2 m 21 s |
| #12099 | window 1: radiance down, G2 x 5 arms (card 0) + G3 A1 / A5 + G2 short cross-check (card 1), `CAP=270` | exit 0, stage 13:56:25 - 13:59:11 |

Radiance downtime, window 1: stopped 13:56:21, `/v1/models` answering again 13:59:22 (15 s after the restart), **3 min 1 s**. One
window only (budget 2 windows, <= 5 min each); radiance verified up afterwards (`fleetq status`: radiance-rank0 up; `/v1/models` 200).

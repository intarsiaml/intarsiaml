# Inkling-2-Mini on HIP: combined speed on final main (lane ink-hip-combined, 2026-11-26)

Branch `ink/hip-combined` (worktree `.worktrees/ink-hip-combined`), base main `178d87ece` (#536 merged, #537 merged). Docs only.

## Result

Inkling-2-Mini (W4A16), one R9700 (tatooine), greedy, graphs on. "Before today" and "now" are the same 4 + 4 processes, ABBA on both
cards, 256 tokens, in two windows; the ratio is now / before today with a 95 % CI over the 4 process pairs.

| tok/s | start (11-25) | before today (`80719483a`) | now (`178d87ece`) | now / before today, 95 % CI |
|---|---:|---:|---:|---|
| spec-default decode, short (12 golden prompts) | 62.5 (a) | 104.31 | 109.57 | **1.052** [1.047-1.056] |
| spec-default decode, 1k | - | 96.45 | 102.01 | **1.059** [1.054-1.064] |
| spec-default decode, 4k | 60.5 (a) | 110.02 | 117.12 | **1.066** [1.062-1.070] |
| spec-default decode, 16k | - | 84.51 | 91.01 | **1.078** [1.074-1.081] |
| plain decode (`--spec off`), short (4 golden prompts) | 62.5 (a) | 77.22 | 83.69 | **1.084** [1.082-1.087] |
| plain decode, 4k | 60.5 (a) | 74.34 | 80.35 | **1.081** [1.080-1.082] |
| plain decode, 16k | - | 70.06 | 75.31 | **1.075** [1.074-1.077] |
| prefill, 4k | 524 (b) | 3454 | 3448 | 1.001 [0.992-1.011] |
| prefill, 16k | 578.7 (c) | 3264 | 3256 | 1.000 [0.993-1.008] |

Sources of the start column (11-25). It is single runs of other prompts, so read it as the level, not as a paired ratio:
- (a) `docs/results/2026-11-25-inkling2-hip.md` section 4.2 ("Speed (first numbers, correctness forms)", fleetq #11945): `generate
  --spec off`, 256 tokens, two runs each. Short = a 31-token prompt, 4k = a 4228-token prompt. HIP had no speculative decode on
  11-25 (DSpark on HIP is #527, 11-26), so the default decode then was plain decode. 1k and 16k decode were not measured.
- (b) Same section, the 4228-token prompt at the default chunk then (256): 524 tok/s (547-553 at chunk 1024-2048).
- (c) `docs/results/2026-11-25-inkling2-hip-moe.md`, "S2 evaluation in the model" (window 9, fleetq #11990): the 16915-token
  prompt at chunk 2048 with WMMA and flash off (the defaults then): 578.7 tok/s.

Today's four PRs (#533 P12, #534 k5, #536 verify rows, #537 PF2 tiny-shape guard) give **+5.2 to +7.8 % spec-default decode and
+7.5 to +8.4 % plain decode**, with prefill unchanged. Greedy output is unchanged:
- all 12 md5 runs are equal (section C2);
- the speed processes' ids are identical between old and new for all 16 + 7 prompts at 256 tokens.

Radiance: 2 windows, **7:15** of the brief's 8:00 (C1 3:39, C2 3:36).

## Briefs received

1. 2026-11-26 ~17:25 EST (coordinator), verbatim in substance: #536 is merged into main as `178d87ece`, #537 is merged too. One
   last brief: a combined speed measurement on the final main. Record this brief in a new results doc first
   (`docs/results/2026-11-26-inkling2-hip-combined.md`, new branch `ink/hip-combined` from origin/main, worktree
   `.worktrees/ink-hip-combined`).
   1. Build on tatooine (fleetq, `--load cpu`): the CLI at origin/main `178d87ece`, and the CLI at `80719483a` (main before today's
      #533 / #534 / #536 / #537). Reuse a binary already built for `80719483a` if it is that exact commit.
   2. One window (at most 5 min radiance down; split into two windows under 5 min each if needed, at most 8 min in total). 4 + 4
      processes ABBA on both cards, same protocol as before, 256 tokens. Arms: new = `178d87ece` defaults, old = `80719483a` defaults.
      Measure spec-default decode at short / 1k / 4k / 16k (new defaults: k5, P12, vrow); plain `--spec off` decode at short / 4k /
      16k; prefill at 4k and 16k. If time runs short, drop plain 16k first and then plain short.
   3. The same window: greedy md5 of new against old at short / 4k, 64 / 256 tokens, `--spec off` and spec default. They must be
      equal; report any difference at once.
   4. Write the results doc. Lead with one table "start (11-25) / before today (`80719483a`) / now (`178d87ece`)", the start column
      from the 11-25 numbers already in docs/results with a source line each. Ratios with 95 % CIs. One docs PR with this doc only,
      no LOG.md row. Do not merge.
   5. Report the table and the md5 result, then list every leftover (worktrees, tatooine folder and volume); the coordinator removes
      them.
   Rules as before: fleetq for every job, work folders only under `.worktrees/`, real checks.

## Binaries

- **old** = `intarsia-main2`, main `80719483a` exactly (built in lane ink-hip-verify's CPU job #12153, md5
  `2f41908f6b79136e7a59d0f5cbe1d92a`; reused).
- **new** = `intarsia-comb`, main `178d87ece` (tatooine CPU job #12252 from a copy of this worktree, md5
  `1212314c1f79be0b94b93d63a17ea899`). Its processes log `--spec-k 5` (recipe `[tuning.hip] mtp_default_k`), P12 for the unembed
  and in_proj + wo, and `matmul_p12_logits_gemv_rows` in the verify kernels.

Both arms run with their defaults: no `--spec`, `--recipe`, TAU / CUT or other env (plain arms add only `--spec off`).

## C1: spec-default decode and prefill (4 + 4 processes, 256 tokens)

fleetq #12253, radiance down 18:14:31-18:18:10 EST (**3:39**), both cards (`raw/stage_c1.sh`, `raw/c1/`). `generate --prompt-ids
speed.json` (a warm-up + the 12 short golden prompts + 1021 / 4108 / 16455 ids), 256 tokens, greedy, EOS honoured, graphs on, max-seq
20480. Order per card ABBA: card 0 old new new old, card 1 new old old new. Ratio per (prompt, process) to the old process of the same
card and rep, geometric mean per class, 95 % CI: t over the 4 process pairs [bootstrap over processes x prompts for short]
(`raw/speed.py`, `raw/c1/speed-spec.txt`). Prefill from the same processes: tok/s of prompt 14 (4108 ids) and 15 (16455 ids), ratio
per pair, t-interval (`raw/prefill.py`, `raw/c1/prefill.txt`).

| spec-default decode, tok/s (median of 4) | short (12) | 1k | 4k | 16k |
|---|---:|---:|---:|---:|
| old, `80719483a` (k3) | 104.31 | 96.45 | 110.02 | 84.51 |
| new, `178d87ece` (k5, P12, vrow) | 109.57 | 102.01 | 117.12 | 91.01 |
| **new / old, 95 % CI** | **1.052** [1.047-1.056] | **1.059** [1.054-1.064] | **1.066** [1.062-1.070] | **1.078** [1.074-1.081] |

| prefill, tok/s (median of 4) | old | new | new / old, 95 % CI |
|---|---:|---:|---|
| 4k (4108 ids) | 3454 | 3448 | 1.001 [0.992-1.011] |
| 16k (16455 ids) | 3264 | 3256 | 1.000 [0.993-1.008] |

- **Output:** the ids of all 16 prompts are identical in all 8 processes (old and new, 256 tokens, EOS honoured).
- **Spec counters, per process:** old 1286 verify passes, 3138 proposed, 1951 accepted (62.2 %, 2.52 tokens a pass); new 1182 / 3840 /
  2069 (53.9 %, 2.75 tokens a pass). Every process of an arm has the same counters.
- Prefill is unchanged within the CI at 4k and 16k. #533's LOG row keeps the prefill GEMMs on bf16 (P12 runs at 1..=4 rows for
  in_proj / wo and 1..=8 for the head).

## C2: greedy md5 and plain decode (4 + 4 processes, 256 tokens)

fleetq #12255, radiance down 18:29:00-18:32:36 EST (**3:36**), both cards (`raw/stage_c2.sh`, `raw/c2/`).

**Greedy md5.** `generate --ignore-eos`, the text before `finish=`, prompts `scripts/cmx/ids-prompts/short-0.txt` / `4k-0.txt`
(as lane ink-hip-verify's window 4 and lane ink-hip-vrow's window 8). Card 0 ran short, card 1 ran 4k. New ran all 8 cells. Old ran
the four 64-token cells again in this window. Old's 256-token values are from the same binary in lane ink-hip-verify's window 4
(`docs/results/2026-11-26-inkling2-hip-verify.md`).

| prompt, tokens | new spec default (k5) | new `--spec off` | old spec default (k3) | old `--spec off` |
|---|---|---|---|---|
| short, 64 | `04cdb28a6849728e8aa5be39bf65c07e` | `04cdb28a…` | `04cdb28a…` (this window) | `04cdb28a…` (this window) |
| short, 256 | `9a27c21b6fdc4ea7700439c818e698d8` | `9a27c21b…` | `9a27c21b…` (window 4) | `9a27c21b…` (window 4) |
| 4k, 64 | `2252566a5c54cf52cb564bd214c9af2b` | `2252566a…` | `2252566a…` (this window) | `2252566a…` (this window) |
| 4k, 256 | `9675bbb95adabf53ce09c1f77a1d34b4` | `9675bbb9…` | `9675bbb9…` (window 4) | `9675bbb9…` (window 4) |

**All equal, no difference.**

**Plain decode.** `generate --spec off --prompt-ids plain-s4-4k-16k.json`, 256 tokens, EOS honoured, ABBA as in C1. The prompt file
is a cut of speed.json: the warm-up, the first 4 of the 12 short golden prompts, and the 4108- and 16455-id prompts. All 12 short
prompts at 256 tokens did not fit the 8-minute budget next to the md5 runs, so plain short has 4 prompts, not 12; nothing was dropped.
Ratio and CI as in C1 (`raw/c2/speed-plain.txt`).

| plain decode, tok/s (median of 4) | short (4) | 4k | 16k |
|---|---:|---:|---:|
| old, `80719483a` | 77.22 | 74.34 | 70.06 |
| new, `178d87ece` (P12) | 83.69 | 80.35 | 75.31 |
| **new / old, 95 % CI** | **1.084** [1.082-1.087] | **1.081** [1.080-1.082] | **1.075** [1.074-1.077] |

- **Output:** ids identical in all 8 processes.
- **Prefill from these processes** (`raw/c2/prefill.txt`): 4k 3474 -> 3470 tok/s, 1.002 [0.992-1.012]; 16k 3281 -> 3272, 1.000
  [0.992-1.009]. This agrees with C1.
- **Where the plain gain comes from:** only #533 (P12) changes the 1-row decode. #534 and #536 change only verify passes, and #536
  keeps the 1-row launches as on main. #537 changes a prefill-chunk condition (`rel_pass`, `dr <= e`) that Inkling-2-Mini's shapes
  (dr 16, E 1024) do not reach.

## Windows, jobs, radiance

| window | fleetq | what | radiance down (EST) | down |
|---|---|---|---|---:|
| C1 | #12253 | spec-default decode 4 + 4, prefill | 18:14:31-18:18:10 | 3:39 |
| C2 | #12255 | md5 (12 runs), plain decode 4 + 4 | 18:29:00-18:32:36 | 3:36 |
| | | | **total** | **7:15** of 8:00 |

Radiance answered 15 s after each restart (`/v1/models` checked by `raw/window.sh`). Build: tatooine CPU job #12252 (`comb`,
`raw/build-cli-comb-12252.log`). Window logs: `raw/window-c1-12253.log`, `raw/window-c2-12255.log`.

# DeepSeek-V4.1 prefill chunk 2048 against 512 on multi-fact long documents, with a rounding-noise control (set 2)

Date 2026-11-03/04 (overnight window). Branch `dsv41/chunk32k-2` (on top of 3b362e301). Follows [2026-11-03-dsv41-chunk32k-quality.md](2026-11-03-dsv41-chunk32k-quality.md) (set 1: 52 single-fact and two-hop items, 38/52 vs 39/52, 0 / 1 discordant, half the set saturated). Question here: **does chunk 2048 lose accuracy on questions that need facts from several distant blocks, and is any shift larger than what a pure rounding-order change gives?**

**Answer.** 180 items (150 of 31K tokens, 30 of 64K), same binary and window, greedy, thinking off, three arms paired by item:

| arm | correct | accuracy | 32K | 64K |
|---|---|---|---|---|
| chunk 512 (ab0) | 133/180 | 0.739 (Wilson 0.670 to 0.798) | 113/150 | 20/30 |
| chunk 2048 (ab1) | 134/180 | 0.744 (0.676 to 0.803) | 113/150 | 21/30 |
| chunk 512, `HC_FUSED=0` (ab2, noise control) | 125/180 | 0.694 (0.624 to 0.757) | 108/150 | 17/30 |

- **2048 minus 512: +0.6 points (+1 item), Newcombe paired 95% CI -3.5 to +4.6 points. The largest loss the CI still allows is 3.5 points (6.2 of 180 items).** 6 items right only at 512, 7 only at 2048, McNemar p = 1.0.
- **The noise control moved more than the chunk change did:** `HC_FUSED=0` minus 512 is -4.4 points (-8 items), CI -8.8 to -0.1, 12 items right only at 512 and 4 only at the control, McNemar p = 0.077. A change of rounding order alone gives a swing the size of the whole CI of the chunk change.
- **Is the 2048 shift larger than pure rounding noise? No.** Items whose correctness differs from the 512 arm: 13 for 2048, 16 for the control (sign test on who moves, p = 0.61). Items with a different final answer: 31 and 30 (p = 1.0). Items with different reply text: 103 and 107 of 180 (p = 0.64). The two perturbed arms also differ from each other on 15 items. By every measure the 2048 arm is as far from 512 as a rounding-only toggle is, and no further.
- 2048 prefill is **24 % faster**: mean time to first token 40.2 s against 52.7 s at 31K, 84.8 s against 111.4 s at 64K (control 52.3 / 110.3).
- **Recommendation: on this evidence flipping the default to 2048 is supported on quality grounds; it stays the user's decision.** Details and caveats at the end. Nothing was changed in any default.

## What was run

| | |
|---|---|
| fleetq jobs | **#8183** main window (`fleetq run --at 20:00 ... --expect 640`), 20:00:00 to 04:53:36, spark1 + spark2 as one job; pilots **#8184** (difficulty at chunk 512 and 64K memory check at chunk 2048) and **#8191** (answer format and harder levels). vLLM was taken down by each job and restored by its script; state after the main window is in "State at the end" |
| binary | `~/intarsia-dsv41-prefill/bin/intarsia`, STAMP `a60292329 dirty=no 1102-1850`, both Sparks, not rebuilt; `INTARSIA_DSV41_CAP=66000` for every arm (CAP does not change numerics, set 1) |
| arms | ab0 `INTARSIA_DSV41_CHUNK=512`; ab1 `INTARSIA_DSV41_CHUNK=2048`; ab2 `INTARSIA_DSV41_CHUNK=512 INTARSIA_DSV41_HC_FUSED=0` (separate HcMixes + HcPre + RmsNorm, rounding order only), run last. One fresh start of both ranks per arm, same shard order in each; ab0 20:00 to 23:13, ab1 23:13 to 01:42, ab2 01:42 to 04:53 |
| scoring | localeval per shard (`uv run localeval quality --dataset <shard> --judge none --max-tokens 400 --thinking off --url http://100.120.88.104:8899/v1 --model dsv41-intarsia`, 27 runs, each logged to unanimis by localeval), inside the job's own slot (`FLEETQ_JOB` set). "Contains expected": the reply contains `FINAL: <answer>` (case-insensitive, whitespace collapsed). The analysis also parses the FINAL line exactly: the exact-parse count equals the localeval count in all three arms (133 / 134 / 125) |
| launcher | `raw/launch-main.sh` + `raw/run-full-chunk32k2.sh` (= `scripts/dsv41-tp2/run-full.sh` plus a step `ldata_shards`, `STOP_EPOCH` deadline and `LSHARDS_AB<i>`); stop time 06:30 was never needed |

Memory (lowest MemAvailable, GiB, spark1 / spark2): ab0 7.3 / 8.1, **ab1 5.6 / 6.3**, ab2 8.2 / 7.6; memfine's 5 GiB line never fired (0 KILLED; `raw/mem.log.gz`). The 64K prompts did not lower the minimum below that of the 31K prompts in any arm (the pilot's 64K prompts at chunk 2048 gave 6.6 / 6.6); the chunk-2048 arm is the one that came within 0.6 GiB of the line. The brief's two rules (10 GB kill, 5 GiB line) conflict for this engine, whose idle MemAvailable after load is 8.9 to 10 GiB; the 5 GiB line was used as instructed.

## The question set (frozen before the first request of the main window)

Generator `raw/gen_items2.py` (fixed seeds; texts from git commit 47af18777 and the DSv4.1 tokenizer), spec `raw/spec-final.json`, manifest `raw/items.manifest.json` (kind, level, depths of every needed fact, span, token count, expected answer, prompt sha256), hashes `raw/items.sha256.json`. A second generator run gives the same hash.

- **sha256 of the full set** (180 rows): `0b4aed88a388a7331c48431b116c7cd8a82b17f4981f8fe649025b88a4c68e8b`; 9 shards (25 items each at 31K, 10 at 64K), hashes in the json. Frozen in commit 14958bcea (the first freeze, 423c80910 with sha 319c0e8d..., was replaced after pilot 2 and before any request of the final set).
- Every item needs **several facts from different places** of a ~31K (30.3K to 31.2K tokens) or ~64K (64.0K to 64.5K) document. The reply may show a few lines of working (thinking stays off), then ends with `FINAL: <answer>`; the expected string is `FINAL: <answer>`, so there is no quoted-constant artifact (set 1) and no brackets (the model dropped them in the pilot). `max_tokens` 400.
- Prompts differ per item (unique first line, different haystack slice); the engine reported `cached_tokens` 0 on all 540 calls; prompt token counts are identical across arms for every item.
- Expected answers are derived from the generator's structure **and** re-derived from the finished document text by an independent parser (`solve` in the generator); entity words must not occur in the filler prose.

| kind | items (31K / 64K) | levels | what the model has to combine |
|---|---|---|---|
| `hop2` | 30 / 6 | L3 | alias -> release -> rollback token, the two facts at least 60 % of the document apart, 10 decoy alias/token pairs |
| `hop3` | 20 / 4 | L3 | alias -> release -> owning team -> team token (3 facts, about 22 % or more apart), 12 decoy chains |
| `agg_sum` | 50 / 10 | L3, L4 | total units of all storage shards of one cluster: 8 (L3) or 12 (L4) facts spread over the document, 30 / 40 decoy shards of other clusters |
| `agg_max` | 30 / 6 | L3 | which release of a line recorded the most failed tests: 8 facts, 30 decoys |
| `repo_sum` | 20 / 4 | L4 | sum of 6 integer constants named by file, each in a different Rust file of a document made of this repo's sources |

The needed facts of an item span 0.36 to 0.98 of the document (mean 0.75 to 0.81 by kind), so the answer depends on blocks far apart.

**Difficulty gate (aim for chunk 512 at about 60 to 85 %, checked on chunk 512 only).** Two pilots, with items from other seeds (`raw/pilot/`): pilot 1 (33 items, 3 levels per kind) gave 25/30 at 31K (L1 10/10, L2 8/10, L3 7/10) and 2/3 at 64K, but every reply lacked the bracket the format asked for, so localeval scored 0/33 and the replies were re-scored offline; pilot 2 (21 harder items, new format) gave 15/21 with localeval and showed 4 replies cut at 160 tokens (so 400 in the window), hop and repo items at 12/12 and `agg_sum` at 1/6. The set was therefore weighted to `agg_sum` and dropped the easy levels. **Result at chunk 512: 133/180 = 73.9 % (31K 75.3 %, 64K 66.7 %), inside the band.** But it is not uniform: `hop2` 35/36, `hop3` 23/24, `repo_sum` 24/24 and `agg_max` 31/36 are close to the ceiling, `agg_sum` 20/60. Of the 180 items **119 are right in all three arms, 39 wrong in all three, and only 22 differ between arms** (`agg_sum` 10, `agg_max` 7, `hop3` 3, `hop2` 2). The informative part of the set is the `agg_sum` and `agg_max` items; see "What n allows".

## Results

Source: `raw/analysis.md` (from `raw/analyze2.py`; replies `raw/replies-ab{0,1,2}.jsonl`).

| arm | correct | exact-parse correct | no FINAL line | cut at 400 tokens | call errors | cached tokens | mean TTFT 31K / 64K (s) |
|---|---|---|---|---|---|---|---|
| 512 | 133 | 133 | 1 | 1 | 0 | 0 | 52.7 / 111.4 |
| 2048 | 134 | 134 | 0 | 0 | 0 | 0 | 40.2 / 84.8 |
| 512 + HC_FUSED=0 | 125 | 125 | 1 | 1 | 0 | 0 | 52.3 / 110.3 |

(The one cut reply is `agg_sum-L4-32k-000`, wrong in all three arms; in the 2048 arm it ended in time and was also wrong.)

**Paired comparisons with chunk 512 as reference** (difference = arm minus 512; Newcombe 1998 method 10 for paired proportions, exact McNemar):

| comparison | both right | only 512 right | only other right | both wrong | McNemar p | difference | 95 % CI | largest loss allowed |
|---|---|---|---|---|---|---|---|---|
| 2048 vs 512 | 127 | 6 | 7 | 40 | 1.000 | +0.006 (+1 item) | -0.035 to +0.046 | 3.5 points (6.2 items) |
| HC_FUSED=0 vs 512 | 121 | 12 | 4 | 43 | 0.077 | -0.044 (-8 items) | -0.088 to -0.001 | 8.8 points (15.9 items) |
| 2048 vs HC_FUSED=0 | | 3 (only HC0 right) | 12 (only 2048 right) | | 0.035 | +0.050 | +0.008 to +0.092 | |

**By item kind** (right at 512 / at the other arm; only-512 / only-other / both wrong; final answer differs from 512):

| kind | items | 2048: correct | only 512 / only 2048 / both wrong | final differs | HC0: correct | only 512 / only HC0 / both wrong | final differs |
|---|---|---|---|---|---|---|---|
| hop2 | 36 | 35 (512: 35) | 1 / 1 / 0 | 2 | 35 | 1 / 1 / 0 | 2 |
| hop3 | 24 | 23 (23) | 1 / 1 / 0 | 2 | 22 | 1 / 0 / 1 | 1 |
| agg_sum | 60 | 20 (20) | 3 / 3 / 37 | 24 | 16 | 6 / 2 / 38 | 22 |
| agg_max | 36 | 32 (31) | 1 / 2 / 3 | 3 | 28 | 4 / 1 / 4 | 5 |
| repo_sum | 24 | 24 (24) | 0 / 0 / 0 | 0 | 24 | 0 / 0 / 0 | 0 |

**How far each perturbation moves the answers (the noise question).** Pairwise over all 180 items:

| pair | correctness differs | final answer differs | reply text differs |
|---|---|---|---|
| 512 vs 2048 | 13 | 31 | 103 |
| 512 vs 512 + HC_FUSED=0 | 16 | 30 | 107 |
| 2048 vs 512 + HC_FUSED=0 | 15 | 31 | 106 |

Reply text is byte-identical across all three arms for 54 items. Sign tests on "which item moves in one perturbed arm and not the other" (`raw/analysis.md`, last section): correctness p = 0.61, final answer p = 1.0, reply text p = 0.64.

**Other cuts (2048 vs 512; HC0 vs 512 in `raw/analysis.md`).** By size: 31K 113 / 113 (4 only-512, 4 only-2048), 64K 20 / 21 (2, 3); HC0: 31K 108, 64K 17. By level: L3 97 / 99 (3 / 5), L4 36 / 35 (3 / 2). By span of the needed facts: span < 0.7: 32 / 33 (1 / 2); 0.7 to 0.85: 62 / 65 (2 / 5); at least 0.85: 39 / 36 (3 / 0); the only bin where 512 is ahead is the widest spread, 3 items against 0, which is the kind of pattern the indexer-gate worry predicts but is 3 items (p = 0.25 on those discordants), and the control shows the same direction there (4 / 0). Everything here is a tally of a few items, not a rate.

## What n allows

- **n = 180 pairs, 22 of which are not fixed by the model alone.** The CI of 2048 minus 512 is -3.5 to +4.6 points; its half-width (about 4 points) equals the swing the pure rounding toggle produced (-4.4). So this run can say "2048 is not worse than rounding noise in these items" but not "2048 loses nothing": a true loss of 2 to 3 points is inside the interval. To halve the half-width the set would have to be about four times larger (about 700 items of this mix); more of the same `hop2` / `repo_sum` items would not help, as they sit at the ceiling.
- **The noise control is one draw.** The `HC_FUSED=0` arm happened to land 4.4 points below 512 (p = 0.077) and 5.0 points below 2048 (p = 0.035). That is not a finding about `HC_FUSED=0`: the 512 and 2048 arms are equally one draw each, and the three arms disagree pairwise on 13 to 16 items. It shows that a rounding-only change can move this set's score by several points, i.e. the set's resolution is about that size. It also means 2048's +0.6 should not be read as a gain.
- `agg_sum` is arithmetic over 8 to 12 retrieved numbers; a miss of one shard or a slip in the sum counts as wrong, so part of its variability is arithmetic, not retrieval. Still, it needs the facts from 8 to 12 places spread over the document, the case the Jaccard gate worried about.
- Tested only: greedy decoding, 31K and 64K prompts, retrieval and aggregation questions on repo prose, rank 0 and rank 1 together. Not tested: sampled decoding, 128K, free-form generation quality, other languages. No f32 reference exists, so this compares two bf16 layouts through their answers; it does not say which is closer to the truth.

## Recommendation

**Flipping the default to 2048 is supported on quality grounds, by two independent sets and a noise control; whether to do it is the user's call, and nothing was changed.** What speaks for it: 0 / 1 discordant pairs in set 1 (n = 52) and 6 / 7 here (n = 180), no sign of a loss in any kind, size or span class beyond a few items either way, an accuracy difference of +0.6 points whose CI (-3.5 to +4.6) lies within the -4.4 points a rounding-only toggle gave, the same count of changed answers as the toggle (13 against 16 changed outcomes, 31 against 30 changed final answers), the earlier KL study (position 32767: 0.016 to 0.018 nats) and MMLU 0.879 at 2048, against a gain of 24 % in prefill time. What speaks against flipping now: the CI cannot exclude a true loss of about 3 points on hard multi-fact items, and the run says nothing about sampled decoding or contexts beyond 64K. If the user wants to close that gap first, the useful next steps are a larger `agg_sum` / `agg_max`-type set (about 700 items for a 2-point half-width) and a second rounding-only control with another seed of the same toggle, to see how much the 512 arm itself wanders. If the user prefers to wait, keeping `INTARSIA_DSV41_CHUNK=2048` opt-in loses only the 24 % for people who do not set it.

## State at the end

Job #8183 finished (exit 0) at 04:53:36; its script restored vLLM, and at 04:57 `fleetq status` shows spark1-vllm and spark2-vllm up with job #8183 done. No fleetq slot is held. The three arms ran to completion (180 / 180 / 180 items), so the 06:30 stop was not needed.

## Not done / deviations

- The two pilots ran before the window (jobs #8184, #8191) with items from other seeds and are not part of the results; pilot 1 used the bracket format that failed, as described.
- Set 1's first freeze (423c80910) was replaced by 14958bcea after pilot 2 and before any request of the final set; both commits are in the branch history.
- The localeval scores use "Contains expected" (substring); the exact parse agrees on every item, so no wrong answer extended a right one.
- The two kill-line rules of the brief conflict (10 GB against 5 GiB); the 5 GiB line was used. Chunk 2048 came within 0.6 GiB of it.
- Not pushed; the work folder `.worktrees/chunk32k-2/.scratch` holds the full question files (about 9 MB) and run logs and was left in place.

## Reproduce

```
uv run --no-project --with tokenizers python docs/results/2026-11-04-dsv41-chunk32k-quality-2/raw/gen_items2.py \
   --spec docs/results/2026-11-04-dsv41-chunk32k-quality-2/raw/spec-final.json --out .scratch/final        # set sha256 as above
fleetq run --host spark1,spark2 --takes-down spark1-vllm,spark2-vllm --quiet gpu,cpu --mem spark1=105,spark2=105 --expect 640 --at 20:00 \
   --label "dsv41 chunk32k-2: ..." -- .scratch/launch-main.sh          # raw/launch-main.sh, raw/run-full-chunk32k2.sh
python3 docs/results/2026-11-04-dsv41-chunk32k-quality-2/raw/analyze2.py --arms ab0=512 ab1=2048 ab2=512-hc0 --since 20261103-2000 \
   --manifest docs/results/2026-11-04-dsv41-chunk32k-quality-2/raw/items.manifest.json --out docs/results/2026-11-04-dsv41-chunk32k-quality-2/raw
```

Files in `raw/`: generator, specs (pilot, pilot2, final), manifest, hashes, launchers, run script, `analyze2.py`, `analysis.md`, per-item replies of the three arms, `ab*-steps.log`, `win-steps.log`, `le-logs/` (27 localeval logs with unanimis record ids), `ab*-spec.txt`, `fleetq-job-8183.log`, `mem.log.gz`, and `pilot/` (summaries, replies, logs of both pilots).

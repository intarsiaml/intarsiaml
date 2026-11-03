# DeepSeek-V4.1 prefill chunk 2048 against 512 on 31K-token documents: task accuracy (substitute check 2 of the chunkdep doc)

Date 2026-11-03. Branch `dsv41/chunk32k` (from origin/main 47af18777). Follows [2026-11-02-dsv41-chunkdep.md](2026-11-02-dsv41-chunkdep.md) ("Why there is no f32 reference", substitute check 2): chunk 2048 saves about 24 % prefill but stays opt-in because it failed the indexer gate at 32K (Jaccard 0.76 against 0.98) and no f32 reference exists. This run asks the task-level question instead: **does chunk 2048 answer long-document questions worse than chunk 512?**

**Answer on this question set: no loss was seen, and the data cannot show there is none.** 52 items of about 31K tokens, same binary and window, greedy, thinking off. Chunk 512 got 38/52 (0.731), chunk 2048 got 39/52 (0.750). Paired: 38 items right in both, 13 wrong in both, **0 items right at 512 and wrong at 2048, 1 item right at 2048 and wrong at 512** (exact McNemar p = 1.0). Difference (2048 minus 512) +1.9 percentage points, Newcombe 95 % CI -2.8 to +7.0 points. **The largest loss the CI still allows is 2.8 points (1.5 items of 52).** 48 of 52 reply texts are byte-identical between the arms. Half of the set (the 24 single-fact needle items) is at 100 % in both arms, so it cannot separate anything; the evidence comes from the 28 harder items. Recommendation at the end: keep chunk 2048 opt-in, get more and harder items before any default change.

## What was run

| | |
|---|---|
| fleetq job | **#8174** (`fleetq run --host spark1,spark2 --takes-down spark1-vllm,spark2-vllm --quiet gpu,cpu --mem spark1=105,spark2=105 --expect 125`), spark1 + spark2 as one job, started 14:44, finished 16:06, vLLM back up on both Sparks at 16:09:29 (`fleetq status`: spark1-vllm and spark2-vllm up) |
| binary | `~/intarsia-dsv41-prefill/bin/intarsia` on both Sparks, STAMP `a60292329 dirty=no 1102-1850` (the pre-RDMA-merge build; numerics unaffected by the merge per the golden 640/640), not rebuilt |
| launcher | `scripts/dsv41-tp2/run-full.sh` TP=2 (rank 0 spark1 serve, rank 1 spark2 tp-worker), as `raw/run-full-chunk32k.sh` = run-full.sh plus a step `ldata_shards` (shards through localeval, early read of the first shard, `$OUT0/abort` / `continue` decision file) and `CUR_CFG` in the localeval recipe. Wrapper `raw/launch.sh`: `CAP=33000 STEPS=ldata_shards LDATA_TOKENS=48 MAX_MIN=105 EARLYOOM_OK="3,2 2,1" AB="INTARSIA_DSV41_CHUNK=512|INTARSIA_DSV41_CHUNK=2048"`, items `.scratch/items.{pilot,rest1,rest2}.jsonl` |
| scoring | localeval (`/opt/code/deepeval`, `uv run localeval quality --dataset <shard> --judge none --max-tokens 48 --thinking off --url http://100.120.88.104:8899/v1 --model dsv41-intarsia`, temperature 0 by localeval's default), "Contains expected" check: the expected string appears in the reply (case-insensitive, whitespace collapsed). No judge model. Run through the job's own slot (`FLEETQ_JOB` set, so localeval took no second slot) |
| arms | **ab0 = `INTARSIA_DSV41_CHUNK=512`, ab8 = `INTARSIA_DSV41_CHUNK=2048`** (not ab1: my gate-wait loop in the wrapper reused the shell variable `i` of the arm loop, so the second arm was numbered ab8; harmless, the configs, directories and localeval labels are consistently ab0 / ab8). The engine's own startup line confirms the chunk: `workspace 0.302 GiB (chunk 512)` and `workspace 1.128 GiB (chunk 2048)` |
| localeval runs | ab0: pilot `20261103-144442-adhoc-dsv41-chunk32k-ab0-pilot`, rest1 `...145539...`, rest2 `...151303...`; ab8: `...153105...-ab8-pilot`, `...153910...-ab8-rest1`, `...155233...-ab8-rest2` (unanimis records ddabfb42, 3d9fc6c6, 538b5aee; 053636ec, a8fb85cf, 2d4f5c64) |

Memory: the lowest MemAvailable over the window was 6.4 GiB (spark1) and 6.3 GiB (spark2), at chunk 2048 (7.2 / 8.2 at chunk 512), as in the earlier 32K windows (6.7 / 6.8); memfine's 5 GiB line never fired (`raw/mem.log.gz`, 0 KILLED). Note for the record: the brief's rule "kill below 10 GB available" cannot apply to this engine, whose idle MemAvailable after load is already 9.9 GiB (the weights alone are 98.9 GiB of the 121); I used memfine's 5 GiB line, which is what the earlier windows used.

## The question set (frozen before any model run)

Generator `raw/gen_items.py` (fixed seeds; every source text is read from git commit 47af18777 and the DSv4.1 tokenizer `.worktrees/dsv41-official/mia/tokenizer.json`), manifest `raw/items.manifest.json`, hashes `raw/items.sha256`; frozen in commit c8e7d1656 before the first request, and re-run after the window with identical output.

- sha256 of the full set `items.jsonl` (52 rows): `7f2ee04cf51f2a1af1a8b288300681d22454e047303b926e75a37c0c91f532ff`
- shards (the same rows, run in this order): pilot (items 0-11) `a646fb62...`, rest1 (12-31) `3da9787c...`, rest2 (32-51) `958d6349...`
- every prompt is its own document of 30.3K to 31.2K tokens (engine-reported `prompt_tokens`; identical per item in both arms; CAP was 33000, `max_tokens` 48), starts with a unique line `Run id chunk32k-v1-NN-<kind>-NN.`, and uses a different haystack slice. Prefix reuse: the engine reuses a common prefix with the previous request (`dsv41_engine.rs`, "prefix reuse"); it has no off switch, but the first differing token is within the first line, and every call in both arms reported `cached_tokens` 0. So every prompt was prefilled from the start in the chunk layout under test. Nonce flags were not used (they differ per call and would unpair the arms).
- four kinds, answers are short and checked mechanically:

| kind | items | what |
|---|---|---|
| `needle` | 12 | one fact "the rollback token of the X release was set to CODE" inserted into repo prose, depth 4 to 96 % |
| `needle_dist` | 12 | twelve such facts (different names and codes) in one document, ask for one; target depth 4 to 96 % |
| `needle_hop` | 12 | two hops: "the release nicknamed A is tracked as the X release" and X's token, plus 3 decoy aliases and 6 decoy tokens |
| `qa_const` | 16 | 30K tokens of this repo's Rust sources; value of a `const NAME` in a named file (15 of 16 have a same-named const with another value in a second included file) |

Prose comes from the repo's own result docs, so nothing can be memorised. Depths: each kind has its depths evenly spread over 4 to 96 % and shuffled; the `-dNN` in an item name is the measured depth of the answer-bearing text (for `qa_const` the start of the asked file).

**Difficulty gate (set before the window: chunk 512 outside 40 to 97 % makes the set uninformative).** The first 12 items (3 per kind) were run and read at chunk 512 before the second arm: 9/12 (`raw/gate.txt`), so the run went on; the full chunk-512 result is 38/52 = 73.1 %, inside the band. No refreeze was needed. But "inside the band" hides that the set is bimodal: `needle` and `needle_dist` are 24/24 at chunk 512 (and at 2048), `needle_hop` is 2/12, `qa_const` 12/16.

## Results

Source: `raw/analysis.md` (from `raw/analyze.py`; per-item replies `raw/replies-ab0.jsonl`, `raw/replies-ab8.jsonl`).

| arm | items | correct | accuracy | Wilson 95 % CI | truncated replies (48 tokens) | call errors | cached tokens | mean TTFT |
|---|---|---|---|---|---|---|---|---|
| chunk 512 (ab0) | 52 | 38 | 0.731 | 0.597 to 0.832 | 1 | 0 | 0 | 52.0 s |
| chunk 2048 (ab8) | 52 | 39 | 0.750 | 0.618 to 0.848 | 0 | 0 | 0 | 40.0 s |

Mean time to first token of these 31K prompts is 23 % lower at 2048 (localeval TTFT, which here is the prefill; spec decoding is on in both arms).

| both right | only 512 right | only 2048 right | both wrong | exact McNemar p (two-sided) | 2048 minus 512 | Newcombe (paired, method 10) 95 % CI |
|---|---|---|---|---|---|---|
| 38 | **0** | **1** | 13 | 1.000 | +0.019 (+1 item) | -0.028 to +0.070 |

- Largest loss the CI allows: **2.8 percentage points (1.5 of 52 items)**. A cruder bound that ignores the one gain: with 0 items lost in 52 pairs, the one-sided 95 % bound on the loss rate is 5.6 points (rule of three, 3/52 = 5.8). A loss of one item in ten is excluded; a loss of two or three items in 52 is not.
- Reply text differs for **4 of 52 items**, all `needle_hop`: `needle_hop-00-d08` (512 ran out of tokens, a 48-token "let me check" ramble; 2048 answered with an alias; both wrong), `-01-d55` (two different wrong tokens), `-06-d80` (wrong in both, different answers), `-09-d71` (512 gave an alias, 2048 gave the right token: the one discordant pair). Six of the nine items both arms got wrong have the same wrong reply in both arms (the decoy token), i.e. the failures are mostly systematic, not rounding noise.
- Truncated replies (48 tokens): 1 at chunk 512 (`needle_hop-00-d08`, wrong in both arms anyway), 0 at 2048.
- Scoring artifact (applies equally to both arms): three `qa_const` items ask for a string constant whose expected text includes the double quotes; the model answered the right value without quotes (`mxfp4_xprep_bf16`, `fp8_xprep_bf16`, `tp-protocol=`) and "Contains expected" marks them wrong. The localeval score above is the primary result. Re-scoring the same replies quote-insensitively: 41/52 (512) and 42/52 (2048), still 0 / 1 discordant, McNemar p 1.0, CI -3.3 to +7.6 points.

By kind (correct at 512 / correct at 2048; only-512 / only-2048 / both wrong):

| kind | items | 512 | 2048 | only 512 | only 2048 | both wrong |
|---|---|---|---|---|---|---|
| needle | 12 | 12 | 12 | 0 | 0 | 0 |
| needle_dist | 12 | 12 | 12 | 0 | 0 | 0 |
| needle_hop | 12 | 2 | 3 | 0 | 1 | 9 |
| qa_const | 16 | 12 | 12 | 0 | 0 | 4 |

By depth of the answer-bearing text:

| depth | items | 512 | 2048 | only 512 | only 2048 | both wrong |
|---|---|---|---|---|---|---|
| early (< 0.33) | 16 | 11 | 11 | 0 | 0 | 5 |
| middle (0.33 to 0.67) | 18 | 14 | 14 | 0 | 0 | 4 |
| late (>= 0.67) | 18 | 13 | 14 | 0 | 1 | 4 |

Chunk 512 by kind and depth (correct/items): `needle` 4/4, 4/4, 4/4 and `needle_dist` 4/4, 4/4, 4/4 at every depth; `needle_hop` 0/4, 1/4, 1/4; `qa_const` 3/4, 5/6, 4/6. No depth pattern, and nothing in the middle or early positions that a chunk change could have hurt.

## What the sample size allows

- **n = 52 pairs, and a tie does not prove equality.** Only 28 of the 52 items can show anything: the 24 single-fact items are at 100 % in both arms, and 9 of the 12 hop items fail at both sizes for reasons unrelated to the chunk. So the paired test rests on roughly the 16 `qa_const` and 3 successful hop items plus the failures that could have flipped; a real loss of two or three items out of 52 is inside the CI's range.
- The CI is a paired interval that assumes the items are independent draws. They are not independent in structure (the same four constants, the same prose pool), so it is, if anything, narrower than the truth.
- This tests retrieval and lookup at 31K. The indexer Jaccard drop (0.76) is about which blocks are selected; a task that needs many blocks at once (summaries, counting, multi-hop reasoning where the model does succeed) is barely represented here, because the model fails the hops at both sizes. A pass here says single-fact retrieval at 31K does not suffer; it says little about reasoning over many facts at once.
- It does not tell which chunk is closer to the truth (no f32 reference). It compares two bf16 layouts through the model's answers only.
- 65K prompts, sampled decoding and rank 1 were not tested.

## Recommendation

**Keep chunk 2048 opt-in; do not flip the default on this run.** The data does not support flipping (no significant difference does not mean no difference, the set is half saturated, and the gate that failed, the indexer overlap, is not answered by retrieval items). It does not argue against 2048 either: 0 of 52 items lost, 48 of 52 byte-identical answers, 23 % less prefill time, and the largest loss the CI allows is 2.8 points. If a default change is wanted, the next step is **more and harder items**, not another run of this set: at least 150 items in the non-saturated regime (items the model passes at 512 in about 60 to 80 % of cases, mixed with long-document QA on prose) at 32K and 65K, and the `HC_FUSED=0` noise arm at chunk 512 on the same items to calibrate what rounding alone moves. More pairs narrow the interval roughly with the square root of n.

## Not done

- **The `HC_FUSED=0` noise-control arm.** At 53 s a prompt at chunk 512, one arm is 46 minutes plus about 8 for stopping and restoring vLLM; the two arms already used 82 minutes of the 90 planned, and a third would have passed the 2 h stop. So there is no measure here of how much the score moves under rounding-order changes alone; the nearest evidence is that six of the nine both-wrong items give the same wrong reply in both arms.
- No f32 reference, no TensorFold text comparison (checks 1, 3 and 4 of the chunkdep doc).
- The set is partly saturated (see above); the 40 to 97 % difficulty gate was met on the total, not per kind.

## Reproduce

```
uv run --no-project --with tokenizers python docs/results/2026-11-03-dsv41-chunk32k-quality/raw/gen_items.py --out .scratch/items     # sha256 as above
fleetq run --host spark1,spark2 --takes-down spark1-vllm,spark2-vllm --quiet gpu,cpu --mem spark1=105,spark2=105 --expect 125 \
   --label "dsv41 chunk32k: 52 long-context items at chunk 512 vs 2048 (localeval, vLLM restored after)" -- .scratch/launch.sh    # raw/launch.sh, raw/run-full-chunk32k.sh
python3 docs/results/2026-11-03-dsv41-chunk32k-quality/raw/analyze.py --arms ab0=512 ab8=2048 --since 20261103-1440 \
   --manifest docs/results/2026-11-03-dsv41-chunk32k-quality/raw/items.manifest.json --out docs/results/2026-11-03-dsv41-chunk32k-quality/raw
```

Files in `raw/`: `gen_items.py`, `items.manifest.json` (kind, depths, token estimates, expected answer, prompt sha256 per item), `items.sha256`, `launch.sh`, `run-full-chunk32k.sh`, `analyze.py`, `analysis.md`, `replies-ab0.jsonl` and `replies-ab8.jsonl` (reply, expected, success, prompt/cached/completion tokens per item), `gate.txt`, `win-steps.log`, `ab0-steps.log`, `ab8-steps.log`, `le-logs/`, `ab*-spec.txt` (the engine's speculation lines), `fleetq-job-8174.log`, `mem.log.gz`. The 5 MB question files themselves are not committed (the generator reproduces them).

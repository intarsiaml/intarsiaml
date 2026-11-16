# Qwen3.8-27B on HIP: what the chunked GDN delta rule (default since PR #378) does to served speed

Lane `gdn27-speed`, 2026-11-15 22:40-23:03 EST, tatooine (2x R9700), build `head=b574f19 code-tree=594cb469dc5a dirty=no` (origin/main 930411f58 plus `scripts/gdn27-speed/` only; no code change). Raw: `2026-11-16-hip-gdn27-speed/raw/{tp1,tp2}/` (localeval logs, steps.log, serve logs, guard.log).

## Method

One stamped release CLI (`--features hip`). Per arm a fresh `intarsia serve --backend hip --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --profile-every 0 --max-batch 1 --mem-budget 29 --max-seq 18432 --no-supervisor` (tp 2 adds `--tp 2 --devices 0,1`), recipe defaults otherwise: activation FP8 (recipe default), prefill chunk 2048, and speculative decoding left at its default, which is **MTP on** in both windows (the server log says `speculative decoding: Mtp`, up to 4 drafts; no `--spec` flag). "off" = `INTARSIA_GDN_CHUNK=0`, "on" = knob unset (the default). Then `localeval speed --sizes 1024,4096,16384 --gen 64 --reps 1 --thinking off --wait-idle 60` (prompts came out at about 990 / 4090 / 16655 tokens). Rounds interleaved off/on, on/off, off/on, 3 rounds per tp; tp 1 and tp 2 were separate fleetq windows (#9588, #9595; each took radiance down and restored it, guard.log). The 27B MXFP4 loads in about 2 min per arm.

## Results (mean of 3 rounds, [min-max]; tok/s)

Prefill:

| tp | prompt | off | on | on vs off |
|---|---|---|---|---|
| 1 | 1k (991, includes request overhead) | 2852 [2636-2962] | 3196 [3180-3209] | +12.1 % (off has a 2636 outlier; others +7 %) |
| 1 | 4k | 2935 [2926-2949] | 3166 [3152-3176] | +7.9 % |
| 1 | 16k | 2609 [2596-2618] | 2793 [2784-2810] | +7.1 % |
| 2 | 1k (includes overhead) | 4432 [4410-4443] | 4857 [4851-4863] | +9.6 % |
| 2 | 4k | 4938 [4926-4946] | 5510 [5505-5513] | +11.6 % |
| 2 | 16k | 4453 [4423-4475] | 4919 [4914-4923] | +10.4 % |

Decode (MTP on, one rep per round, random-text prompts, so acceptance varies run to run):

| tp | prompt | off | on | on vs off |
|---|---|---|---|---|
| 1 | 1k | 68.7 [59.8-75.3] | 56.6 [51.8-61.4] | -17.6 % |
| 1 | 4k | 60.1 [55.7-65.2] | 61.9 [59.2-66.1] | +2.9 % |
| 1 | 16k | 51.9 [51.1-53.5] | 55.0 [51.2-58.7] | +5.8 % |
| 2 | 1k | 96.6 [85.3-114.2] | 88.7 [87.9-90.2] | -8.1 % |
| 2 | 4k | 84.4 [81.6-87.2] | 89.5 [85.1-95.8] | +6.1 % |
| 2 | 16k | 94.0 [81.5-101.4] | 88.5 [80.0-94.6] | -5.8 % |

## Reading

* Prefill: on is faster in every round pair, and the on and off ranges do not overlap at any size (round-to-round spread 0.4-1 % at 4k/16k at tp 2, up to 1 % at tp 1 except the one tp 1 1k off outlier). +7-8 % at tp 1 and +10-12 % at tp 2 for 4k/16k, against the 35B's +13-14 % (tp 1) and +17 % (tp 2, 4k).
* Decode: the delta rule is a prefill kernel (decode and MTP verify rows take the per-token path), and the on/off decode ranges overlap at every size with sign flipping between sizes; the 15-25 % within-arm spread is MTP acceptance variance on one rep. No decode effect is claimed either way.
* Method limits: 3 reps of one request per cell; decode noise is too large to resolve a few percent; the 1k row carries per-request overhead (localeval flags it).

## Cleanup

`~/intarsiaml-gdn27-speed` and volume `intarsia-cargo-target-gdn27-speed` removed on tatooine (fleetq #9599, `docker volume ls` shows none); radiance answered on :8080 after each window and at the end.

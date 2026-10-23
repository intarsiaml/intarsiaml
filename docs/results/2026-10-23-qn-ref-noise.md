# qn: `intarsia qualify` acc-short ERROR on tatooine (35B, HIP)

Date: 2026-10-23 00:25-00:45 EDT. Lane qn. Result: qualify quick for Qwen3.6-35B-A3B-NVFP4 on tatooine now ends **exit 0**, every
stage PASS (acc-format SKIP, by design). The cause was a stale golden file on tatooine; the repo needs no code or record change.

## Root cause

`intarsia qualify` finds the short golden by name in the golden directories (`facts::find_golden`, crates/intarsia-cli/src/cmd/qualify/facts.rs).
It deliberately skips any file whose name contains `meta`. On tatooine the golden directory (`/data/models/intarsia/golden`, mounted as
`/models/golden` in the dev container) held two 35B short goldens:

| file | size | FNV-1a 64 | written |
|---|---:|---|---|
| `qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` | 252396 | `b58845c3cb8a3cb6` | 2026-10-12 23:03 (first vLLM golden, no metadata) |
| `qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors` | 246804 | `cd6ef94b3dcd4c82` | 2026-10-13 02:05 (rebuilt with metadata) |

The rebuilt golden was copied in under a `-meta` suffix instead of over the old name. Qualify skips `-meta`, so it kept picking the
stale 10-12 file. The repo's `golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors.json` (and the cuda/hip/metal baselines in
`golden/baselines/`) are bound to `cd6ef94b3dcd4c82`, size 246804, 1441 positions, so `refnoise::check_binding` refused the stale file:
`ref_noise was measured on golden hash cd6ef94b3dcd4c82, this golden is b58845c3cb8a3cb6 ... measure it again`. qualify turned the
accuracy exit 1 into `acc-short: ERROR` and exit 2. The lane hp doc found the mismatch; its direct
`intarsia accuracy --golden ...-vllm-top20-meta.safetensors` passed because that file is the right golden.

## Fix (option a: bring tatooine's golden in line with the canonical one)

On tatooine, in `/data/models/intarsia/golden` (data only, outside the repo; nothing deleted):

```
mv qwen36-35b-a3b-nvfp4-vllm-top20.safetensors qwen36-35b-a3b-nvfp4-vllm-top20.safetensors.stale-b58845c3
cp -p qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
```

The `-meta` copy stays (qualify ignores it; older lane scripts name it). The `.stale-b58845c3` file has no `.safetensors` extension, so no
golden search matches it. Re-measuring the ref-noise (option b) was rejected: the canonical golden is the one CUDA and Metal already
gate against, the record and all three baselines are bound to it, and re-measuring on the stale file would have orphaned them.

## Golden consistency across backends (FNV-1a 64 of the file bytes, same function as `baseline::fnv1a64`)

| host / backend | short `...-vllm-top20.safetensors` | long `...-vllm-long-top20.safetensors` |
|---|---|---|
| repo records (ref_noise sidecars, cuda/hip/metal baselines) | `cd6ef94b3dcd4c82`, 246804 B | `1c62c8e49c68fe34`, 6061444 B |
| endor (Metal): /opt/code/intarsiaml/golden, wt-gc, wt-gc2, wt-m35-sp-moe | `cd6ef94b3dcd4c82` | `1c62c8e49c68fe34` |
| spark2 (CUDA): ~/intarsia-tp2/golden, ~/intarsiaml-golden-s1 | `cd6ef94b3dcd4c82` | `1c62c8e49c68fe34` |
| tatooine (HIP) before: /data/models/intarsia/golden | **`b58845c3cb8a3cb6`**, 252396 B | `1c62c8e49c68fe34` |
| tatooine (HIP) after; also ~/intarsiaml-on/golden, ~/intarsiaml-fx1/golden | `cd6ef94b3dcd4c82` | `1c62c8e49c68fe34` |

All backends now gate against one reference. Not checked: the 4k golden (`...-vllm-4k-top20.safetensors`, endor `27195b6f4d50d2e6`,
2100484 B; qualify quick does not use it) and the spark1 tree (not used, unstable). tatooine has no fp32 golden for the 35B; the vLLM
top-20 golden is the gate there, as on the other backends.

## Verification

Tree: origin/main a312a794 exported to `~/intarsiaml-qn` on tatooine, `cargo build --release -p intarsia-cli --features hip` in the dev
container (fleetq #4395, cpu slot). qualify quick under fleetq #4396 (`--takes-down radiance`, card 0 via HIP_VISIBLE_DEVICES=0), radiance
stopped for the run and started again before the release:

```
intarsia qualify --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --quick-only --gen 128 --reps 5 --no-unim --out ... --mem-budget 29
```

| stage | verdict |
|---|---|
| doctor | PASS |
| recipe | PASS |
| plan | PASS |
| kernels | PASS (1079 cases) |
| gate | PASS (perplexity 1.21, 5/5 sanity prompts) |
| acc-format | SKIP (no format-reference golden for NVFP4) |
| acc-short | **PASS** (prefill top-1 98.61 %, KL 7.63e-3; decode top-1 98.75 %, KL 1.18e-2) |
| acc-oracle | PASS (prefill top-1 95.86 %, KL 1.07e-2) |
| bench | PASS (decode 160.5 tok/s at 128 tokens, 58 % of the byte floor) |

exit 0 (raw/qualify-quick-35b-hip-fixed.txt, raw/quick-report.md). acc-short took 8.1 s.

A first run with the lane-hp flag `--golden-dir /models/golden` (raw/qualify-quick-35b-hip-golden-dir-flag.txt) also passes acc-short but
ends acc-oracle MISSING-BASELINE and exit 1: giving `--golden-dir` replaces the default search list (`./golden`, `/models/golden`,
`$INTARSIA_GOLDEN_DIR`, repo `golden/`), so the CPU oracle in the checkout's `golden/oracle/` is not searched. Run tatooine's qualify
without `--golden-dir` (the container's `/work/golden` and `/models/golden` are both in the default list), or the HIP gates stay red on
acc-oracle for that reason alone. The lane-hp scripts (`hp-stages.sh`, `stage_qual`) pass the flag; drop it there.

## Radiance downtime (tatooine, both cards held by #4396)

| window | radiance down | run |
|---|---|---|
| 1 | 00:32:43-00:34:12 (89 s) | qualify with `--golden-dir` (acc-oracle MISSING-BASELINE) |
| 2 | 00:34:25-00:35:53 (88 s) | accidental repeat of window 1 (a failed `sed` left the script unchanged) |
| 3 | 00:36:10-00:38:13 (123 s) | qualify without `--golden-dir`: exit 0 |

radiance active after each window. Total 300 s.

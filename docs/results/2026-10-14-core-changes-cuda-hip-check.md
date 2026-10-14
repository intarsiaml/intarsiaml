# Metal-branch core changes on CUDA and HIP (2026-10-14)

**Outcome.** The two core-engine changes on `spike/metal` do not change CUDA or HIP behaviour. On both
GB10 (CUDA, sm_121) and R9700 (HIP, gfx1201), Qwen3-8B BF16 gives byte-identical accuracy output and
identical greedy token ids at BASE `40996bb`, HEAD `369cf88`, and HEAD with `INTARSIA_NO_FUSION=1`.
No divergence, so no bisect was needed and no code was changed. Device tests and kernel admission pass
on HEAD. The engine op table on HEAD shows `matmul_pair` (B2b) on both backends and no
`qk_norm_rope_kv` (B4); with `INTARSIA_NO_FUSION=1` `matmul_pair` disappears and the output does not move.

Changes under test (see [metal phase B](2026-10-14-metal-phase-b.md), "Phase B part 2b", "Phase B part 2,
merged", "Phase B part 4"):

1. B2b (5d24762, merged in f1131f1): `model.rs` tries `Op::MatmulPair` for bf16 gate+up and k+v. CUDA and
   HIP register `matmul_pair_bf16_gemv`, so dense bf16 decode now takes it.
2. B4 (2ed47c2): optional fused op `QkNormRopeKv` plus a refactor of the attention block in `model.rs`.
   CUDA/HIP do not register it and run the unfused QkNorm, Rope, KvAppend sequence. Small test-table edits
   in `crates/intarsia-backend-{cuda,hip}/src/kernels.rs`.

BASE = `40996bb` (before both). HEAD = `369cf88` (current `spike/metal` head when the run started; it did
not move). Both trees were built from one git bundle of `spike/metal` and checked out detached.

## 1. Results

Model Qwen3-8B BF16 (16.38 GB), `--gate-class bf16`, goldens `qwen3-8b-fp32.safetensors` (`--max-seq 4096`)
and `qwen3-8b-long-fp32.safetensors` (`--max-seq 8192`). "identical" means the whole stdout of
`intarsia accuracy` (every per-prompt row) has the same md5 for all three variants; `ids` is the `ids:`
line of `generate` stderr (256 greedy tokens, `--temperature 0 --ignore-eos`, `--max-seq 8192`).

### CUDA (spark1, GB10 sm_121)

| check | BASE | HEAD | HEAD `INTARSIA_NO_FUSION=1` |
|---|---|---|---|
| `cargo test --release -p intarsia-backend-cuda` | not run | 32 passed, 0 failed, 4 ignored | not run |
| `intarsia kernels test --backend cuda` | not run | 41 kernels, 582 cases, all ADMITTED (`matmul_pair_bf16_gemv`: 6 cases, max_abs 0) | not run |
| accuracy short, prefill (434 pos) | top-1 100.00%, KL 2.897e-4, PASS | identical | identical |
| accuracy short, decode (128 pos) | top-1 100.00%, KL 3.238e-4, PASS | identical | identical |
| accuracy long, prefill (1434 pos) | top-1 99.65%, KL 4.004e-4, PASS | identical | identical |
| accuracy long, decode (160 pos) | top-1 100.00%, KL 3.332e-4, PASS | identical | identical |
| greedy ids, short prompt (122 tok) | md5 of `ids:` 11f0303f... | identical | identical |
| greedy ids, 4k prompt (4062 tok) | md5 of `ids:` ba0d15a9... | identical | identical |
| op table, 65 tokens | no `matmul_pair`, `qk_norm_rope_kv` absent | `matmul_pair` 4608 calls (31.3%), `qk_norm_rope_kv` absent | `matmul_pair` absent, `qk_norm_rope_kv` absent |

### HIP (tatooine, one R9700 gfx1201, `HIP_VISIBLE_DEVICES=1`)

| check | BASE | HEAD | HEAD `INTARSIA_NO_FUSION=1` |
|---|---|---|---|
| `cargo test --release -p intarsia-backend-hip` | not run | 37 passed, 0 failed, 6 ignored | not run |
| `intarsia kernels test --backend hip` | not run | 38 kernels, 626 cases, all ADMITTED (`matmul_pair_bf16_gemv`: 6 cases, max_abs 0) | not run |
| accuracy short, prefill (434 pos) | top-1 100.00%, KL 2.335e-4, PASS | identical | identical |
| accuracy short, decode (128 pos) | top-1 100.00%, KL 3.626e-4, PASS | identical | identical |
| accuracy long, prefill (1434 pos) | top-1 99.58%, KL 3.966e-4, PASS | identical | identical |
| accuracy long, decode (160 pos) | top-1 100.00%, KL 3.573e-4, PASS | identical | identical |
| greedy ids, short prompt | md5 of `ids:` 11f0303f... | identical | identical |
| greedy ids, 4k prompt | md5 of `ids:` d07a410c... | identical | identical |
| op table, 65 tokens | no `matmul_pair`, `qk_norm_rope_kv` absent | `matmul_pair` 4608 calls (34.1%), `qk_norm_rope_kv` absent | `matmul_pair` absent, `qk_norm_rope_kv` absent |

Notes:

* The accuracy gates are the usual bf16 floor gates; every one of the 12 accuracy runs (2 backends x
  3 variants x 2 goldens) reports `golden gate: PASS`. HIP long prefill shows greedy-match 37.7% of
  positions before first argmax divergence (informational, same in all three variants, bf16 near ties).
* The CUDA and HIP short-prompt greedy ids have the same md5; the 4k-prompt ids differ between CUDA
  and HIP (different backends, not part of this question).
* `matmul_pair` calls: 4608 = 64 decode steps x 36 layers x 2 (gate+up, k+v); prefill (M>1) takes the
  plain `matmul`.
* The op table needs `--profile-every 1 --op-table`. Absence of an op is read from the table (ops with 0
  calls do not appear); the `calls` column counts launches, `timed` only the sampled ones.
* Speed was not measured. spark1 ran vLLM beside these jobs (CUDA decode here is about 15 tok/s, not
  representative); the HIP decode rates are the same across variants (37.9, 37.9, 37.9 tok/s short; 36.1, 36.1, 36.1 4k)
  but that was not a timed comparison.

## 2. Not run

* BASE and HEAD-no-fusion were not run through `cargo test` or `kernels test`: those tests exercise the
  kernels, which this branch did not change, and the request was HEAD only.
* No bisect (`649cc37`) because nothing diverged.
* The qualify pipeline, 35B model, tensor-parallel and fp8/nvfp4 paths were not run: the changed code
  paths for this question are bf16 dense decode only.

## 3. Method and commands

Slots: spark1 fleetq #1864 (`--host spark1 --mem 30`, non-exclusive, vLLM left up); tatooine fleetq
#1865 (`--host tatooine --mem 24 --takes-down radiance`, queued behind #1863, granted after about
3 minutes). On tatooine radiance was stopped with `systemctl --user stop radiance` after the grant and
started again (`systemctl --user start radiance`, active, fleetq marked it RESTORING then up) before
`fleetq release 1865`. Both slots released.

Code and models:

* One bundle `git bundle create cc.bundle spike/metal`, copied to both hosts, trees cloned to
  `/tmp/intarsia-cc-{base,head}` (spark1) and `/tmp/intarsia-hip-{base,head}` (tatooine), checked out at
  `40996bb` / `369cf88` (rev-parse verified). Own `CARGO_TARGET_DIR` per tree on spark1
  (`/tmp/intarsia-cc-{base,head}-target`); own docker volumes `ccheck-target-{base,head}` on tatooine
  through `deploy/tatooine/dev.sh` with `INTARSIA_REPO` / `INTARSIA_TARGET_VOL` (registry volume shared,
  not touched otherwise). `~/intarsiaml` on tatooine and `/home/nvidia/intarsiaml` on spark1 were not
  touched.
* spark1 model copied from endor: `rsync -a /Users/josef/models/Qwen3-8B/ spark1:/home/nvidia/models/Qwen3-8B/`
  (16 GB). Goldens: `/home/nvidia/intarsiaml-golden/`. tatooine: `/data/models/intarsia/Qwen3-8B` and
  `/data/models/intarsia/golden` (mounted `/models`, `/models/golden`).

Builds (both trees): `cargo build --release -p intarsia-cli --features cuda` (CUDA_PATH=/usr/local/cuda),
`dev.sh cargo build --release -p intarsia-cli --features hip`. Device tests: `cargo test --release -p
intarsia-backend-cuda` and `dev.sh cargo test --release -p intarsia-backend-hip` on HEAD.

Checks (`scripts/cc-checks.sh`, run per variant; `INTARSIA_NO_FUSION=1` for the third variant):

```
intarsia kernels test --backend <b> --model <8B>
intarsia accuracy --model <8B> --backend <b> --golden <qwen3-8b-fp32|qwen3-8b-long-fp32>.safetensors --gate-class bf16 --max-seq <4096|8192>
intarsia generate --model <8B> --backend <b> --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --show-ids --prompt "$(cat prompt-<short|4k>.txt)"
intarsia generate --model <8B> --backend <b> --max-seq 8192 --profile-every 1 --op-table --temperature 0 --max-tokens 65 --ignore-eos --prompt "$(cat prompt-short.txt)"
```

Prompts: `docs/results/raw/2026-10-14-metal-phase-b/prompt-short.txt` and `prompt-4k.txt`.

## 4. Versions

* spark1: NVIDIA GB10 (sm_121), driver 580.173.02, CUDA 13.0.88 (nvcc), Linux 6.17.0-1029-nvidia, rustc/cargo 1.90.0.
* tatooine: R9700 gfx1201, image `intarsia-hip-dev:rocm7.2.4-rust1.90.0` (ROCm 7.2.4, HIP 7.2.53211, rustc 1.90.0), Linux 7.0.0-38-generic.
* BASE `40996bb479c03ed9a32807accf741985de780794`, HEAD `369cf883d367e5e58b02f2c07f7ff486b48bf260`.

## 5. Raw logs and cleanup

Raw logs and scripts: `docs/results/raw/2026-10-14-core-changes-cuda-hip-check/{cuda,hip,scripts}`
(file names `<variant>-<check>.{out,err}`, `cargo-test-*.log`).

Cleanup: `/tmp` trees, targets and bundle removed on spark1 and tatooine; docker volumes
`ccheck-target-*` removed. The Qwen3-8B copy stays on spark1 at `/home/nvidia/models/Qwen3-8B`.

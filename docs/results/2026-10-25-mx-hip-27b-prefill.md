# Lane mx: Qwen3.8-27B MXFP4 prefill on one R9700 (HIP, gfx1201), 2026-10-25

Lane mx, branch `lane-mx/27b-prefill`, from origin/main 2c2d9941. Every GPU number comes from a fleetq slot on tatooine with
radiance down (`--takes-down radiance --quiet gpu,cpu`), card 0 only, restored and confirmed `active` after each window
(`raw/downtime.log`). Raw output: `2026-10-25-mx-hip-27b-prefill/raw/<window>/`. Scripts: `scripts/mx/`.

<!-- HEADLINE -->

## 1. Profile: where a 27B 4k prefill goes on one card (main)

`generate --max-tokens 1` under rocprofv3 `--kernel-trace` (raw/w1, job #5008; raw/w2 #5036), classified by
`scripts/mx/roofline27.py` (FLOPs = 2 per MAC; byte floor = every MXFP4 weight read once per chunk). Model facts: 64 layers
= 48 GatedDeltaNet + 16 gated full attention (24 q / 4 kv heads of 256), hidden 5120, MLP 17408, every projection MXFP4 W4A4
(Quark AWQ, activations QDQ'd to MXFP4 per 32-group), 48.7 GFLOP of GEMM per token.

Engine prefill-only time on main (raw/w1, w2):

| prompt | chunk 256 (default) | chunk 4096 |
|---|---:|---:|
| 1k | 1006 ms (1018 tok/s) | 973 ms (1053) |
| 4k | 3984 ms (1028) | 3669 ms (1116) |
| 16k | 16911 ms (969) | 15241 ms (1075) |

Kernel time by class, 4096-token prompt (kernel time 3952 ms at chunk 256, 3553 ms at chunk 4096; launch gaps 3 %):

| class | chunk 256 ms | % | TFLOP/s | chunk 4096 ms | % | TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| GEMM mlp gate / up (N 17408, K 5120) | 1443 | 36.5 | 64.8 | 1387 | 39.0 | 67.4 |
| GEMM mlp down (N 5120, K 17408) | 818 | 20.7 | 57.2 | 799 | 22.5 | 58.5 |
| GEMM gdn in_proj_qkv (N 10240) | 326 | 8.3 | 63.2 | 313 | 8.8 | 65.8 |
| GEMM gdn out_proj + attn o_proj (K 6144) | 301 | 7.6 | 54.7 | 248 | 7.0 | 66.4 |
| GEMM gdn in_proj_a / b (N 48) | **227** | 5.7 | **0.9** | 16 | 0.5 | 12.0 |
| GEMM gdn in_proj_z (N 6144) | 187 | 4.7 | 66.0 | 185 | 5.2 | 66.7 |
| GDN delta rule (sequential scan) | 157 | 4.0 | | 158 | 4.4 | |
| GEMM attn q + gate (N 12288) | 123 | 3.1 | 67.0 | 124 | 3.5 | 66.3 |
| flash attention (bf16 WMMA) | 82 | 2.1 | 40.4 | 60 | 1.7 | 54.6 |
| GEMM attn k / v (N 1024) | 77 | 2.0 | 17.8 | 22 | 0.6 | 63.6 |
| MXFP4 activation QDQ pre-pass | 68 | 1.7 | | 87 | 2.5 | |
| everything else (norms, conv, silu, rope) | 147 | 3.7 | | 154 | 4.3 | |
| **all MXFP4 GEMMs** | **3502** | **88.6** | **57.0** | **3095** | **87.1** | **64.5** |

* **The GEMMs are the prefill**: 87-89 % of kernel time, at 57-65 TFLOP/s. That is 30-34 % of the measured bf16 WMMA issue
  peak (190.7 TFLOP/s) and 15-17 % of the fp8 peak (366). Floors for the 4k prompt: 529 ms at the fp8 peak, 1046 ms at the
  bf16 peak, 20 ms of weight bytes per chunk (12.9 GB read once at 640 GB/s; far from binding at prefill M).
* **Measured ceilings at the same shapes** (`scripts/mx/ceiling27.hip`, raw/w2/ceiling.txt, M = 4096): hipBLASLt bf16
  143-175 TFLOP/s, hipBLASLt fp8 259-321, our own W8A8 `f8_gemm_kernel` (ik_f8_gemm.h) 188-218. The MXFP4 kernel
  (`dq_gemm_kernel<Mxfp4Policy>`, bf16-dequant WMMA, one 128 x 128 tile for every M > 64) is the outlier.
* At chunk 256 two shapes are grid-starved, not compute-bound: in_proj_a / b (N = 48: 2 blocks per launch) at 0.9 TFLOP/s
  and attn k / v (N = 1024: 16 blocks on 64 CUs) at 17.8. Chunk 4096 removes both.
* Attention and the GDN scan are 6 % together at 4k (flash 82 ms, scan 157 ms); at 16k flash attention grows to 1210 ms
  (7.3 %, 43.6 TFLOP/s causal) and the scan to 626 ms (3.8 %). Neither is the lever at 1k-16k; the GEMM is.
* The 35B on the same build does 4k in 410 ms (9981 tok/s): the 27B is a dense model with 13x the active GEMM FLOPs.

### How much of the gap to radiance is the kernel

Radiance 1.12.0 TP2 on this box: 5210 / 6133 / 5852 tok/s at 1k / 4k / 16k (2026-10-25, user). At 48.7 GFLOP/token,
6133 tok/s is 299 TFLOP/s of GEMM across two cards, at least 150 TFLOP/s per card even if radiance spent no time outside the
GEMM. Single card, our GEMMs run at 64.5 (chunk 4096). So of the 5.5x (6133 / 1116) gap at 4k: about 2x is the second card
(TP2), about 2.3x is the GEMM kernel (150 / 64.5), and the rest (~1.2x) is everything outside the GEMM that TP2 also halves
and radiance's fused attention/scan. A tp1 radiance run would make this direct; it was not run (not tried in this lane's windows), so this
split is arithmetic, not a measurement.

<!-- REST -->

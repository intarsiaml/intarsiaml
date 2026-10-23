N = 16384 tokens; main burst span 1783.8 ms, kernel time 1740.3 ms, idle 43.4 ms (2.4 %)

| kernel class | launches | ms | share | achieved TFLOPS | ceiling TFLOPS | % of ceiling | ceiling source |
|---|---:|---:|---:|---:|---:|---:|---|
| routed experts gate_up (NVFP4 -> bf16 WMMA) | 160 | 355.5 | 20.4 % | 61.9 | 171 | 36 % | hipBLASLt bf16, all pairs as one GEMM |
| routed experts down (NVFP4 -> bf16 WMMA) | 320 | 205.6 | 11.8 % | 53.5 | 123 | 43 % | hipBLASLt bf16, all pairs as one GEMM |
| shared expert gate/up/down (NVFP4 -> bf16 WMMA) | 480 | 84.8 | 4.9 % | 48.6 | 145 | 33 % | hipBLASLt bf16 at the shapes |
| dense fp8 projections (fp8 x fp8 WMMA) | 520 | 229.0 | 13.2 % | 183.1 | 284 | 65 % | hipBLASLt fp8 at the shapes |
| small bf16 GEMMs (router, GDN a/b) | 560 | 29.4 | 1.7 % | 27.8 | 158 | 18 % | hipBLASLt bf16 (k/v shape, N=512) |
| flash attention (bf16 WMMA) | 40 | 278.8 | 16.0 % | 78.9 | 194 | 41 % | bf16 WMMA issue peak |
| GatedDeltaNet scan (f32 VALU, sequential) | 600 | 343.6 | 19.7 % | 5.2 | 11 | 46 % | measured floor of this lane map: LDS-free probe, 0.324 us/token/layer (raw/w4) |
| GDN conv + gated norm | 390 | 60.1 | 3.5 % |  |  |  |  |
| MoE glue (top-k, group, reduce, combine, silu) | 800 | 70.0 | 4.0 % |  |  |  |  |
| fp8 activation quantise | 520 | 27.1 | 1.6 % |  |  |  |  |
| norms, rope, sigmoid, copies, other | 1320 | 56.3 | 3.2 % |  |  |  |  |
| idle (gaps between kernels) | | 43.4 | | | | | |

prefill-equivalent rate over the burst: 9185 tok/s; useful matrix work 101.84 TFLOP = 57.1 TFLOPS over the span

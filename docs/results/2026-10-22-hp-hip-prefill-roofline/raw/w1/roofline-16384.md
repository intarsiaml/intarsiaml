N = 16384 tokens; main burst span 1836.9 ms, kernel time 1792.5 ms, idle 44.4 ms (2.4 %)

| kernel class | launches | ms | share | achieved TFLOPS | ceiling TFLOPS | % of ceiling | ceiling source |
|---|---:|---:|---:|---:|---:|---:|---|
| routed experts gate_up (NVFP4 -> bf16 WMMA) | 160 | 346.0 | 19.3 % | 63.6 | 171 | 37 % | hipBLASLt bf16, all pairs as one GEMM |
| routed experts down (NVFP4 -> bf16 WMMA) | 320 | 202.3 | 11.3 % | 54.4 | 123 | 44 % | hipBLASLt bf16, all pairs as one GEMM |
| shared expert gate/up/down (NVFP4 -> bf16 WMMA) | 480 | 82.3 | 4.6 % | 50.1 | 145 | 35 % | hipBLASLt bf16 at the shapes |
| dense fp8 projections (fp8 x fp8 WMMA) | 520 | 252.5 | 14.1 % | 166.0 | 284 | 59 % | hipBLASLt fp8 at the shapes |
| small bf16 GEMMs (router, GDN a/b) | 560 | 30.7 | 1.7 % | 26.7 | 158 | 17 % | hipBLASLt bf16 (k/v shape, N=512) |
| flash attention (bf16 WMMA) | 40 | 271.1 | 15.1 % | 81.1 | 194 | 42 % | bf16 WMMA issue peak |
| GatedDeltaNet scan (f32 VALU, sequential) | 720 | 397.5 | 22.2 % | 4.5 | 11 | 40 % | measured floor of this lane map: LDS-free probe, 0.324 us/token/layer (raw/w4) |
| GDN conv + gated norm | 390 | 58.2 | 3.2 % |  |  |  |  |
| MoE glue (top-k, group, reduce, combine, silu) | 800 | 70.2 | 3.9 % |  |  |  |  |
| fp8 activation quantise | 520 | 26.3 | 1.5 % |  |  |  |  |
| norms, rope, sigmoid, copies, other | 1320 | 55.4 | 3.1 % |  |  |  |  |
| idle (gaps between kernels) | | 44.4 | | | | | |

prefill-equivalent rate over the burst: 8919 tok/s; useful matrix work 101.84 TFLOP = 55.4 TFLOPS over the span

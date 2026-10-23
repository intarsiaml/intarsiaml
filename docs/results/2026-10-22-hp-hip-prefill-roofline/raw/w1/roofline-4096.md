N = 4096 tokens; main burst span 420.4 ms, kernel time 404.3 ms, idle 16.1 ms (3.8 %)

| kernel class | launches | ms | share | achieved TFLOPS | ceiling TFLOPS | % of ceiling | ceiling source |
|---|---:|---:|---:|---:|---:|---:|---|
| routed experts gate_up (NVFP4 -> bf16 WMMA) | 40 | 88.8 | 22.0 % | 61.9 | 171 | 36 % | hipBLASLt bf16, all pairs as one GEMM |
| routed experts down (NVFP4 -> bf16 WMMA) | 80 | 48.8 | 12.1 % | 56.4 | 123 | 46 % | hipBLASLt bf16, all pairs as one GEMM |
| shared expert gate/up/down (NVFP4 -> bf16 WMMA) | 120 | 20.7 | 5.1 % | 49.9 | 145 | 34 % | hipBLASLt bf16 at the shapes |
| dense fp8 projections (fp8 x fp8 WMMA) | 130 | 63.8 | 15.8 % | 164.1 | 284 | 58 % | hipBLASLt fp8 at the shapes |
| small bf16 GEMMs (router, GDN a/b) | 140 | 7.7 | 1.9 % | 26.5 | 158 | 17 % | hipBLASLt bf16 (k/v shape, N=512) |
| flash attention (bf16 WMMA) | 10 | 17.2 | 4.3 % | 79.9 | 194 | 41 % | bf16 WMMA issue peak |
| GatedDeltaNet scan (f32 VALU, sequential) | 180 | 99.4 | 24.6 % | 4.5 | 11 | 40 % | measured floor of this lane map: LDS-free probe, 0.324 us/token/layer (raw/w4) |
| GDN conv + gated norm | 120 | 14.8 | 3.7 % |  |  |  |  |
| MoE glue (top-k, group, reduce, combine, silu) | 200 | 17.4 | 4.3 % |  |  |  |  |
| fp8 activation quantise | 130 | 6.7 | 1.6 % |  |  |  |  |
| norms, rope, sigmoid, copies, other | 531 | 19.0 | 4.7 % |  |  |  |  |
| idle (gaps between kernels) | | 16.1 | | | | | |

prefill-equivalent rate over the burst: 9743 tok/s; useful matrix work 21.34 TFLOP = 50.8 TFLOPS over the span

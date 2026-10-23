N = 1024 tokens; main burst span 147.3 ms, kernel time 129.4 ms, idle 18.0 ms (12.2 %)

| kernel class | launches | ms | share | achieved TFLOPS | ceiling TFLOPS | % of ceiling | ceiling source |
|---|---:|---:|---:|---:|---:|---:|---|
| routed experts gate_up (NVFP4 -> bf16 WMMA) | 40 | 39.1 | 30.2 % | 35.2 | 171 | 21 % | hipBLASLt bf16, all pairs as one GEMM |
| routed experts down (NVFP4 -> bf16 WMMA) | 40 | 18.1 | 14.0 % | 38.0 | 123 | 31 % | hipBLASLt bf16, all pairs as one GEMM |
| shared expert gate/up/down (NVFP4 -> bf16 WMMA) | 120 | 7.7 | 5.9 % | 33.6 | 145 | 23 % | hipBLASLt bf16 at the shapes |
| dense fp8 projections (fp8 x fp8 WMMA) | 130 | 16.4 | 12.6 % | 160.1 | 284 | 56 % | hipBLASLt fp8 at the shapes |
| small bf16 GEMMs (router, GDN a/b) | 140 | 5.3 | 4.1 % | 9.6 | 158 | 6 % | hipBLASLt bf16 (k/v shape, N=512) |
| flash attention (bf16 WMMA) | 10 | 1.8 | 1.4 % | 46.5 | 194 | 24 % | bf16 WMMA issue peak |
| GatedDeltaNet scan (f32 VALU, sequential) | 30 | 21.3 | 16.5 % | 5.3 | 11 | 47 % | measured floor of this lane map: LDS-free probe, 0.324 us/token/layer (raw/w4) |
| GDN conv + gated norm | 120 | 3.9 | 3.0 % |  |  |  |  |
| MoE glue (top-k, group, reduce, combine, silu) | 200 | 4.4 | 3.4 % |  |  |  |  |
| fp8 activation quantise | 130 | 2.0 | 1.5 % |  |  |  |  |
| norms, rope, sigmoid, copies, other | 531 | 9.4 | 7.3 % |  |  |  |  |
| idle (gaps between kernels) | | 18.0 | | | | | |

prefill-equivalent rate over the burst: 6950 tok/s; useful matrix work 5.08 TFLOP = 34.5 TFLOPS over the span

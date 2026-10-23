N = 1024 tokens; main burst span 149.0 ms, kernel time 132.5 ms, idle 16.5 ms (11.0 %)

| kernel class | launches | ms | share | achieved TFLOPS | ceiling TFLOPS | % of ceiling | ceiling source |
|---|---:|---:|---:|---:|---:|---:|---|
| routed experts gate_up (NVFP4 -> bf16 WMMA) | 40 | 38.1 | 28.7 % | 36.1 | 171 | 21 % | hipBLASLt bf16, all pairs as one GEMM |
| routed experts down (NVFP4 -> bf16 WMMA) | 40 | 18.1 | 13.6 % | 38.1 | 123 | 31 % | hipBLASLt bf16, all pairs as one GEMM |
| shared expert gate/up/down (NVFP4 -> bf16 WMMA) | 120 | 7.7 | 5.8 % | 33.6 | 145 | 23 % | hipBLASLt bf16 at the shapes |
| dense fp8 projections (fp8 x fp8 WMMA) | 130 | 16.3 | 12.3 % | 160.9 | 284 | 57 % | hipBLASLt fp8 at the shapes |
| small bf16 GEMMs (router, GDN a/b) | 140 | 5.3 | 4.0 % | 9.7 | 158 | 6 % | hipBLASLt bf16 (k/v shape, N=512) |
| flash attention (bf16 WMMA) | 10 | 1.7 | 1.3 % | 50.0 | 194 | 26 % | bf16 WMMA issue peak |
| GatedDeltaNet scan (f32 VALU, sequential) | 30 | 25.9 | 19.5 % | 4.4 | 11 | 38 % | measured floor of this lane map: LDS-free probe, 0.324 us/token/layer (raw/w4) |
| GDN conv + gated norm | 120 | 4.1 | 3.1 % |  |  |  |  |
| MoE glue (top-k, group, reduce, combine, silu) | 200 | 4.4 | 3.3 % |  |  |  |  |
| fp8 activation quantise | 130 | 1.8 | 1.3 % |  |  |  |  |
| norms, rope, sigmoid, copies, other | 531 | 9.2 | 7.0 % |  |  |  |  |
| idle (gaps between kernels) | | 16.5 | | | | | |

prefill-equivalent rate over the burst: 6874 tok/s; useful matrix work 5.08 TFLOP = 34.1 TFLOPS over the span

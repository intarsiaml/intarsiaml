N = 4096 tokens; main burst span 409.7 ms, kernel time 394.9 ms, idle 14.8 ms (3.6 %)

| kernel class | launches | ms | share | achieved TFLOPS | ceiling TFLOPS | % of ceiling | ceiling source |
|---|---:|---:|---:|---:|---:|---:|---|
| routed experts gate_up (NVFP4 -> bf16 WMMA) | 40 | 92.9 | 23.5 % | 59.2 | 171 | 35 % | hipBLASLt bf16, all pairs as one GEMM |
| routed experts down (NVFP4 -> bf16 WMMA) | 80 | 50.1 | 12.7 % | 54.9 | 123 | 45 % | hipBLASLt bf16, all pairs as one GEMM |
| shared expert gate/up/down (NVFP4 -> bf16 WMMA) | 120 | 21.1 | 5.3 % | 48.8 | 145 | 34 % | hipBLASLt bf16 at the shapes |
| dense fp8 projections (fp8 x fp8 WMMA) | 130 | 58.3 | 14.8 % | 179.8 | 284 | 63 % | hipBLASLt fp8 at the shapes |
| small bf16 GEMMs (router, GDN a/b) | 140 | 7.3 | 1.9 % | 28.0 | 158 | 18 % | hipBLASLt bf16 (k/v shape, N=512) |
| flash attention (bf16 WMMA) | 10 | 18.0 | 4.6 % | 76.3 | 194 | 39 % | bf16 WMMA issue peak |
| GatedDeltaNet scan (f32 VALU, sequential) | 150 | 87.0 | 22.0 % | 5.2 | 11 | 46 % | measured floor of this lane map: LDS-free probe, 0.324 us/token/layer (raw/w4) |
| GDN conv + gated norm | 120 | 15.8 | 4.0 % |  |  |  |  |
| MoE glue (top-k, group, reduce, combine, silu) | 200 | 17.4 | 4.4 % |  |  |  |  |
| fp8 activation quantise | 130 | 6.9 | 1.7 % |  |  |  |  |
| norms, rope, sigmoid, copies, other | 531 | 20.1 | 5.1 % |  |  |  |  |
| idle (gaps between kernels) | | 14.8 | | | | | |

prefill-equivalent rate over the burst: 9996 tok/s; useful matrix work 21.34 TFLOP = 52.1 TFLOPS over the span

| arm | mode | prompts | positions | mean KL(21-way) | top-1 agreement | top-1 flips | max position KL |
|---|---|---|---|---|---|---|---|
| HIP tp2 fp8 | prefill | 4k (3) | 12559 | 7.458e-04 | 97.55 % | 2.45 % | 1.861e-02 |
| HIP tp2 fp8 | prefill | 16k (3) | 47990 | 6.233e-04 | 98.05 % | 1.95 % | 1.879e-02 |
| HIP tp2 fp8 | decode | 4k (3) | 384 | 4.379e-04 | 99.22 % | 0.78 % | 6.329e-03 |
| HIP tp2 fp8 | decode | 16k (3) | 384 | 4.406e-04 | 99.74 % | 0.26 % | 9.143e-03 |
| HIP tp2 mxfp4 | prefill | 4k (3) | 12559 | 1.235e-02 | 92.39 % | 7.61 % | 1.768e-01 |
| HIP tp2 mxfp4 | prefill | 16k (3) | 47990 | 1.001e-02 | 94.01 % | 5.99 % | 2.203e-01 |
| HIP tp2 mxfp4 | decode | 4k (3) | 384 | 6.311e-03 | 96.09 % | 3.91 % | 1.280e-01 |
| HIP tp2 mxfp4 | decode | 16k (3) | 384 | 7.350e-03 | 97.14 % | 2.86 % | 7.981e-02 |
| HIP tp2 bf16 control (same build) | prefill | 4k (3) | 12559 | 1.694e-12 | 100.00 % | 0.00 % | 9.594e-15 |
| HIP tp2 bf16 control (same build) | prefill | 16k (3) | 47990 | 1.744e-12 | 100.00 % | 0.00 % | 9.594e-15 |
| HIP tp2 bf16 control (same build) | decode | 4k (3) | 384 | 2.610e-12 | 100.00 % | 0.00 % | 5.208e-15 |
| HIP tp2 bf16 control (same build) | decode | 16k (3) | 384 | 8.060e-13 | 100.00 % | 0.00 % | 5.818e-15 |
| HIP tp2 bf16 at --prefill-chunk 256 | decode | 4k (3) | 384 | 2.610e-12 | 100.00 % | 0.00 % | 5.208e-15 |
| HIP tp2 bf16 at --prefill-chunk 256 | decode | 16k (3) | 384 | 8.060e-13 | 100.00 % | 0.00 % | 5.818e-15 |

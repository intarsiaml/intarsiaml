| arm | mode | prompts | positions | mean KL(21-way) | top-1 agreement | top-1 flips | max position KL |
|---|---|---|---|---|---|---|---|
| Metal mxfp4 | prefill | 4k (3) | 12175 | 1.253e-02 | 92.37 % | 7.63 % | 1.766e-01 |
| Metal mxfp4 | prefill | 16k (3) | 47606 | 1.004e-02 | 94.02 % | 5.98 % | 2.209e-01 |
| Metal mxfp4 | decode | 4k (3) | 384 | 1.160e-02 | 92.71 % | 7.29 % | 7.802e-02 |
| Metal mxfp4 | decode | 16k (3) | 384 | 5.294e-03 | 97.14 % | 2.86 % | 1.136e-01 |
| Metal bf16 control | prefill | 4k (3) | 12175 | 1.781e-12 | 100.00 % | 0.00 % | 9.311e-15 |
| Metal bf16 control | prefill | 16k (3) | 47606 | 1.839e-12 | 100.00 % | 0.00 % | 9.806e-15 |
| Metal bf16 control | decode | 4k (3) | 384 | 2.748e-12 | 100.00 % | 0.00 % | 8.816e-15 |
| Metal bf16 control | decode | 16k (3) | 384 | 2.066e-13 | 100.00 % | 0.00 % | 7.503e-15 |

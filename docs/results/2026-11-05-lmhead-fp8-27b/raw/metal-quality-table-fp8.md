| arm | mode | prompts | positions | mean KL(21-way) | top-1 agreement | top-1 flips | max position KL |
|---|---|---|---|---|---|---|---|
| Metal fp8 | prefill | 4k (3) | 12175 | 7.557e-04 | 97.58 % | 2.42 % | 1.880e-02 |
| Metal fp8 | prefill | 16k (3) | 47606 | 6.252e-04 | 98.02 % | 1.98 % | 1.880e-02 |
| Metal fp8 | decode | 4k (3) | 384 | 7.071e-04 | 98.44 % | 1.56 % | 5.880e-03 |
| Metal fp8 | decode | 16k (3) | 384 | 3.920e-04 | 98.96 % | 1.04 % | 4.720e-03 |

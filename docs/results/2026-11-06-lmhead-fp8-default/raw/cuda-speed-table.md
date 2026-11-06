CUDA spark1, 27B Quark MXFP4, 1024-token prompt, 128 generated, 3 reps per visit, median decode tok/s per visit (ABAB; D = default fp8 head, O = INTARSIA_LMHEAD_QUANT=off)

| Mode | default (fp8 head) visits | off (bf16 head) visits | median default | median off | ratio |
|---|---|---|---|---|---|
| MTP k2 | 27.1, 26.9 | 25.1, 24.3 | 27.01 | 24.70 | 1.094 |
| plain | 15.1, 15.1 | 13.8, 13.8 | 15.06 | 13.76 | 1.095 |

Reference baseline 554.2 GB/s (ref-baseline.json). State factor = arm reference / baseline; valid 0.70-1.30.

**Per visit**

| visit | ref GB/s | factor | busy GPU MHz | decode 128 raw / norm | decode 1024 raw / norm | decode 4096 raw / norm | decode 16384 raw / norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| run1/intarsia-1 | 504.5 | 0.910 | 1122 | 35.3 / 38.8 | 33.2 / 36.5 | 31.0 / 34.1 | 28.3 / 31.1 |
| run1/intarsia-2 | 535.5 | 0.966 | 1276 | 36.3 / 37.6 | 36.4 / 37.6 | 33.7 / 34.9 | 30.1 / 31.1 |
| run1/llamacpp | 541.3 | 0.977 | 1064 | 32.8 / 33.6 | 32.1 / 32.9 | 31.4 / 32.2 | 27.9 / 28.6 |
| run1/mlx-lm | 534.2 | 0.964 | 1180 | 28.2 / 29.3 | 29.4 / 30.5 | 29.0 / 30.1 | 26.2 / 27.2 |
| run2/intarsia-1 | 537.0 | 0.969 | 1184 | 33.8 / 34.8 | 34.0 / 35.1 | 32.8 / 33.9 | 29.2 / 30.2 |
| run2/llamacpp-2 | 531.0 | 0.958 | 1014 | 29.3 / 30.6 | 30.6 / 31.9 | 28.5 / 29.7 | 28.0 / 29.3 |
| run2/llamacpp | 537.7 | 0.970 | 1070 | 33.1 / 34.1 | 32.7 / 33.7 | 30.6 / 31.6 | 28.0 / 28.8 |
| run2/mlx-lm-2 | 473.1 | 0.854 | 1177 | 31.6 / 37.0 | 30.3 / 35.5 | 28.3 / 33.2 | 24.6 / 28.8 |
| run2/mlx-lm | 501.8 | 0.905 | 1206 | 31.1 / 34.3 | 31.3 / 34.6 | 30.5 / 33.6 | 25.6 / 28.2 |

**Per visit, prefill tok/s raw (marginal), unnormalised**

| visit | busy GPU MHz | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---:|---:|---:|---:|
| run1/intarsia-1 | 1122 | 2965 (3178) | 2043 (1861) | 1834 (1775) |
| run1/intarsia-2 | 1276 | 3978 (4457) | 3899 (3876) | 2100 (1827) |
| run1/llamacpp | 1064 | 1448 (1533) | 1480 (1489) | 1206 (1137) |
| run1/mlx-lm | 1180 | 1760 (3344) | 1594 (1547) | 1692 (1727) |
| run2/intarsia-1 | 1184 | 3124 (3382) | 2165 (1972) | 1991 (1940) |
| run2/llamacpp-2 | 1014 | 1999 (2222) | 1535 (1429) | 1199 (1118) |
| run2/llamacpp | 1070 | 2158 (2400) | 1555 (1427) | 1280 (1211) |
| run2/mlx-lm-2 | 1177 | 1895 (3440) | 1799 (1770) | 1571 (1509) |
| run2/mlx-lm | 1206 | 1940 (3399) | 2256 (2382) | 1858 (1758) |

**Decode tok/s, raw** - median over visits (min-max), n visits per cell

| engine | n | 128 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|
| intarsia | 3 | 35.3 (33.8-36.3) | 34.0 (33.2-36.4) | 32.8 (31.0-33.7) | 29.2 (28.3-30.1) |
| mlx-lm | 3 | 31.1 (28.2-31.6) | 30.3 (29.4-31.3) | 29.0 (28.3-30.5) | 25.6 (24.6-26.2) |
| llama.cpp | 3 | 32.8 (29.3-33.1) | 32.1 (30.6-32.7) | 30.6 (28.5-31.4) | 28.0 (27.9-28.0) |

Order (~ = bands overlap, a tie): 128: intarsia > llama.cpp ~ mlx-lm; 1024: intarsia > llama.cpp ~ mlx-lm; 4096: intarsia ~ llama.cpp ~ mlx-lm; 16384: intarsia > llama.cpp > mlx-lm

**Decode tok/s, clock-normalised (raw / state factor, every engine)** - median over visits (min-max), n visits per cell

| engine | n | 128 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|
| intarsia | 3 | 37.6 (34.8-38.8) | 36.5 (35.1-37.6) | 34.1 (33.9-34.9) | 31.1 (30.2-31.1) |
| mlx-lm | 3 | 34.3 (29.3-37.0) | 34.6 (30.5-35.5) | 33.2 (30.1-33.6) | 28.2 (27.2-28.8) |
| llama.cpp | 3 | 33.6 (30.6-34.1) | 32.9 (31.9-33.7) | 31.6 (29.7-32.2) | 28.8 (28.6-29.3) |

Order (~ = bands overlap, a tie): 128: intarsia ~ mlx-lm ~ llama.cpp; 1024: intarsia ~ mlx-lm ~ llama.cpp; 4096: intarsia > mlx-lm ~ llama.cpp; 16384: intarsia > llama.cpp ~ mlx-lm

**Prefill tok/s, raw (per request)** - median over visits (min-max), n visits per cell

| engine | n | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|
| intarsia | 3 | 3124 (2965-3978) | 2165 (2043-3899) | 1991 (1834-2100) |
| mlx-lm | 3 | 1895 (1760-1940) | 1799 (1594-2256) | 1692 (1571-1858) |
| llama.cpp | 3 | 1999 (1448-2158) | 1535 (1480-1555) | 1206 (1199-1280) |

Order (~ = bands overlap, a tie): 1024: intarsia > llama.cpp ~ mlx-lm; 4096: intarsia ~ mlx-lm > llama.cpp; 16384: intarsia ~ mlx-lm > llama.cpp

**Marginal prefill tok/s, raw** - median over visits (min-max), n visits per cell

| engine | n | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|
| intarsia | 3 | 3382 (3178-4457) | 1972 (1861-3876) | 1827 (1775-1940) |
| mlx-lm | 3 | 3399 (3344-3440) | 1770 (1547-2382) | 1727 (1509-1758) |
| llama.cpp | 3 | 2222 (1533-2400) | 1429 (1427-1489) | 1137 (1118-1211) |

Order (~ = bands overlap, a tie): 1024: mlx-lm ~ intarsia > llama.cpp; 4096: intarsia ~ mlx-lm > llama.cpp; 16384: intarsia > mlx-lm > llama.cpp

**Sensitivity: clock-matched by MHz** (decode raw median + 0.014 tok/s per MHz x (intarsia median busy MHz - engine median busy MHz); the slope was measured on intarsia only, docs/results/2026-10-21-g3-metal.md section 2, so for the other engines it is an assumption)

| engine | median busy GPU MHz | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---:|---:|---:|---:|
| intarsia | 1184 | 35.3 | 34.0 | 32.8 | 29.2 |
| mlx-lm | 1180 | 31.1 | 30.4 | 29.1 | 25.6 |
| llama.cpp | 1064 | 34.4 | 33.8 | 32.3 | 29.6 |

**Ratios intarsia / engine** (medians; `tie` = the min-max bands of the visit medians overlap)

| engine | basis | 128 | 1024 | 4096 | 16384 |
|---|---|---:|---:|---:|---:|
| mlx-lm | decode raw | 1.14x | 1.12x | 1.13x | 1.14x |
| mlx-lm | decode clock-normalised (reference) | 1.10x tie | 1.05x tie | 1.03x | 1.10x |
| mlx-lm | decode MHz-matched (sensitivity) | 1.13x | 1.12x | 1.13x | 1.14x |
| mlx-lm | prefill raw | - | 1.65x | 1.20x tie | 1.18x tie |
| mlx-lm | marginal prefill raw | - | 0.99x tie | 1.11x tie | 1.06x |
| llama.cpp | decode raw | 1.08x | 1.06x | 1.07x tie | 1.04x |
| llama.cpp | decode clock-normalised (reference) | 1.12x | 1.11x | 1.08x | 1.08x |
| llama.cpp | decode MHz-matched (sensitivity) | 1.02x | 1.01x | 1.02x | 0.99x |
| llama.cpp | prefill raw | - | 1.56x | 1.41x | 1.65x |
| llama.cpp | marginal prefill raw | - | 1.52x | 1.38x | 1.61x |

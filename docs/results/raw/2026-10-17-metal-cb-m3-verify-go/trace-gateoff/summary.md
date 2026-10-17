Arms: off, on; rounds per arm: off=3, on=3

| scenario | metric | off median (rounds) | on median (rounds) | paired on/off per round | median ratio | min ratio |
|---|---|---|---|---|---|---|
| solo-open | decode tok/s | 36.0 (35.9 / 36.0 / 36.0) | 36.9 (36.8 / 36.9 / 36.9) | 1.025 / 1.024 / 1.025 | 1.025 | 1.024 |
| open-n2 | aggregate tok/s | 70.3 (68.6 / 70.3 / 70.3) | 61.9 (60.8 / 62.9 / 61.9) | 0.885 / 0.896 / 0.880 | 0.885 | 0.880 |
| open-n3 | aggregate tok/s | 99.3 (99.2 / 99.3 / 101.6) | 84.2 (83.3 / 84.5 / 84.2) | 0.840 / 0.851 / 0.828 | 0.840 | 0.828 |
| grounded-n2 | aggregate tok/s | 67.1 (67.1 / 66.7 / 68.5) | 126.4 (126.5 / 126.4 / 119.5) | 1.885 / 1.894 / 1.745 | 1.885 | 1.745 |
| grounded-n3 | aggregate tok/s | 95.0 (94.9 / 95.0 / 97.8) | 124.3 (125.6 / 124.3 / 121.1) | 1.325 / 1.308 / 1.238 | 1.308 | 1.238 |

| scenario | on/off per round (median) |
|---|---|
| solo-open | 1.025 / 1.024 / 1.025 (1.025) |
| open-n2 | 0.885 / 0.896 / 0.880 (0.885) |
| open-n3 | 0.840 / 0.851 / 0.828 (0.840) |
| grounded-n2 | 1.885 / 1.894 / 1.745 (1.885) |
| grounded-n3 | 1.325 / 1.308 / 1.238 (1.308) |

| scenario | arm | open-stream decode tok/s (mean over rounds) | grounded wall s / finished (mixed) |
|---|---|---|---|

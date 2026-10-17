Arms: baseon, off, on; rounds per arm: baseon=8, off=8, on=8

| scenario | metric | baseon median (rounds) | off median (rounds) | on median (rounds) | paired on/off per round | median ratio | min ratio |
|---|---|---|---|---|---|---|---|
| grounded-n4 | aggregate tok/s | 123.1 (124.1 / 120.6 / 123.5 / 122.7 / 123.3 / 124.6 / 120.6 / 123.0) | 123.9 (123.5 / 121.1 / 126.0 / 124.8 / 124.4 / 125.3 / 120.4 / 121.5) | 123.5 (121.6 / 122.3 / 123.3 / 124.5 / 124.4 / 123.8 / 123.9 / 120.7) | 0.984 / 1.010 / 0.979 / 0.998 / 1.000 / 0.988 / 1.029 / 0.994 | 0.996 | 0.979 |

| scenario | baseon/off per round (median) | on/off per round (median) |
|---|---|---|
| grounded-n4 | 1.005 / 0.996 / 0.980 / 0.984 / 0.991 / 0.994 / 1.001 / 1.012 (0.995) | 0.984 / 1.010 / 0.979 / 0.998 / 1.000 / 0.988 / 1.029 / 0.994 (0.996) |

| scenario | arm | open-stream decode tok/s (mean over rounds) | grounded wall s / finished (mixed) |
|---|---|---|---|

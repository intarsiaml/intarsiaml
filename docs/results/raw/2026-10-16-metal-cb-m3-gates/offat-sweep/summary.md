Arms: off, offat3, offat5, on; rounds per arm: off=3, offat3=3, offat5=3, on=3

| scenario | metric | off median (rounds) | offat3 median (rounds) | offat5 median (rounds) | on median (rounds) | paired on/off per round | median ratio | min ratio |
|---|---|---|---|---|---|---|---|---|
| mixed-once | aggregate tok/s | 129.9 (126.7 / 129.9 / 130.5) | 126.7 (124.5 / 130.3 / 126.7) | 95.2 (90.1 / 96.5 / 95.2) | 129.1 (125.2 / 129.1 / 130.3) | 0.988 / 0.994 / 0.999 | 0.994 | 0.988 |
| mixed-loop | aggregate tok/s | 123.7 (120.1 / 123.7 / 124.0) | 120.3 (118.4 / 124.9 / 120.3) | 100.1 (96.0 / 100.1 / 101.1) | 122.7 (118.1 / 122.7 / 123.8) | 0.983 / 0.991 / 0.998 | 0.991 | 0.983 |
| grounded-n4 | aggregate tok/s | 124.2 (122.4 / 124.2 / 125.2) | 121.5 (117.9 / 123.8 / 121.5) | 123.1 (118.8 / 128.7 / 123.1) | 122.1 (117.8 / 122.1 / 123.5) | 0.963 / 0.983 / 0.986 | 0.983 | 0.963 |
| open-n4 | aggregate tok/s | 132.1 (128.9 / 132.1 / 132.5) | 129.2 (126.2 / 131.8 / 129.2) | 106.0 (100.7 / 107.1 / 106.0) | 129.2 (126.9 / 129.2 / 132.1) | 0.984 / 0.978 / 0.997 | 0.984 | 0.978 |

| scenario | offat3/off per round (median) | offat5/off per round (median) | on/off per round (median) |
|---|---|---|---|
| mixed-once | 0.983 / 1.003 / 0.971 (0.983) | 0.711 / 0.743 / 0.729 (0.729) | 0.988 / 0.994 / 0.999 (0.994) |
| mixed-loop | 0.986 / 1.009 / 0.970 (0.986) | 0.799 / 0.809 / 0.815 (0.809) | 0.983 / 0.991 / 0.998 (0.991) |
| grounded-n4 | 0.963 / 0.996 / 0.970 (0.970) | 0.970 / 1.036 / 0.983 (0.983) | 0.963 / 0.983 / 0.986 (0.983) |
| open-n4 | 0.978 / 0.997 / 0.975 (0.978) | 0.781 / 0.811 / 0.799 (0.799) | 0.984 / 0.978 / 0.997 (0.984) |

| scenario | arm | open-stream decode tok/s (mean over rounds) | grounded wall s / finished (mixed) |
|---|---|---|---|
| mixed-once | off | 33.0 | 8.1 / 7.9 / 7.8 |
| mixed-once | offat3 | 32.5 | 8.2 / 7.9 / 8.1 |
| mixed-once | offat5 | 23.8 | 11.4 / 10.6 / 10.8 |
| mixed-once | on | 32.7 | 8.2 / 7.9 / 7.9 |
| mixed-loop | off | 32.5 | 2 / 2 / 2 |
| mixed-loop | offat3 | 32.1 | 2 / 2 / 2 |
| mixed-loop | offat5 | 26.2 | 2 / 2 / 2 |
| mixed-loop | on | 32.2 | 2 / 2 / 2 |

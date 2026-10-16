Arms: off, on; rounds per arm: off=3, on=3

| scenario | metric | off median (rounds) | on median (rounds) | paired on/off per round | median ratio | min ratio |
|---|---|---|---|---|---|---|
| solo-grounded | decode tok/s | 35.2 (34.6 / 35.3 / 35.2) | 125.2 (127.7 / 124.7 / 125.2) | 3.695 / 3.531 / 3.551 | 3.551 | 3.531 |
| solo-open | decode tok/s | 35.6 (35.5 / 35.6 / 35.6) | 35.8 (35.8 / 35.4 / 35.9) | 1.009 / 0.992 / 1.010 | 1.009 | 0.992 |
| mixed-once | aggregate tok/s | 128.5 (128.5 / 129.1 / 128.1) | 128.0 (124.9 / 128.0 / 128.0) | 0.972 / 0.991 / 0.999 | 0.991 | 0.972 |
| mixed-loop | aggregate tok/s | 120.5 (120.3 / 120.5 / 124.0) | 120.2 (120.2 / 122.2 / 118.9) | 1.000 / 1.014 / 0.959 | 1.000 | 0.959 |
| grounded-n2 | aggregate tok/s | 67.7 (67.3 / 68.5 / 67.7) | 123.1 (122.5 / 124.4 / 123.1) | 1.820 / 1.817 / 1.818 | 1.818 | 1.817 |
| grounded-n3 | aggregate tok/s | 95.8 (95.1 / 95.8 / 96.5) | 123.1 (119.4 / 123.5 / 123.1) | 1.256 / 1.289 / 1.275 | 1.275 | 1.256 |
| grounded-n4 | aggregate tok/s | 121.0 (119.4 / 121.0 / 124.1) | 119.5 (119.2 / 120.8 / 119.5) | 0.999 / 0.999 / 0.963 | 0.999 | 0.963 |
| open-n2 | aggregate tok/s | 69.7 (70.1 / 69.7 / 69.3) | 62.6 (58.3 / 63.4 / 62.6) | 0.832 / 0.909 / 0.905 | 0.905 | 0.832 |
| open-n3 | aggregate tok/s | 99.8 (99.3 / 99.8 / 102.2) | 84.7 (84.7 / 85.6 / 83.1) | 0.853 / 0.857 / 0.813 | 0.853 | 0.813 |

| scenario | on/off per round (median) |
|---|---|
| solo-grounded | 3.695 / 3.531 / 3.551 (3.551) |
| solo-open | 1.009 / 0.992 / 1.010 (1.009) |
| mixed-once | 0.972 / 0.991 / 0.999 (0.991) |
| mixed-loop | 1.000 / 1.014 / 0.959 (1.000) |
| grounded-n2 | 1.820 / 1.817 / 1.818 (1.818) |
| grounded-n3 | 1.256 / 1.289 / 1.275 (1.275) |
| grounded-n4 | 0.999 / 0.999 / 0.963 (0.999) |
| open-n2 | 0.832 / 0.909 / 0.905 (0.905) |
| open-n3 | 0.853 / 0.857 / 0.813 (0.853) |

| scenario | arm | open-stream decode tok/s (mean over rounds) | grounded wall s / finished (mixed) |
|---|---|---|---|
| mixed-once | off | 32.9 | 8.0 / 7.9 / 8.0 |
| mixed-once | on | 32.4 | 8.2 / 8.0 / 8.0 |
| mixed-loop | off | 32.2 | 2 / 2 / 2 |
| mixed-loop | on | 31.9 | 2 / 2 / 2 |

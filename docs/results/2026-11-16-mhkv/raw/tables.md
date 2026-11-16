| window | bin | arm | n | TTFT ms | decode tok/s | acceptance | server step ms | cached / prompt | text == main cold |
|---|---|---|---|---|---|---|---|---|---|
| 27b-4k | main | cold, store on (empty store) | 2 | 5790 | 51.0 | 60.4 % | 43.50 | 0 / 4623 | True |
| 27b-4k | main | restored from disk after a restart: a new turn after the restore point | 2 | 837 | 46.5 | 50.0 % | 43.48 | 4486 / 4623 | True |
| 27b-4k | main | the same prompt again, restored from the host tier | 2 | 68 | 45.8 | 48.4 % | 43.53 | 4619 / 4623 | True |
| 27b-4k | fix | cold, store on (empty store) | 2 | 5803 | 51.0 | 60.4 % | 43.52 | 0 / 4623 | True |
| 27b-4k | fix | restored from disk after a restart: a new turn after the restore point | 2 | 844 | 51.2 | 60.4 % | 43.30 | 4486 / 4623 | True |
| 27b-4k | fix | the same prompt again, restored from the host tier | 2 | 68 | 51.1 | 60.4 % | 43.37 | 4619 / 4623 | True |
| 35b-4k | main | cold, store on (empty store) | 2 | 1174 | 148.6 | 59.9 % | 14.80 | 0 / 4619 | True |
| 35b-4k | main | restored from disk after a restart: a new turn after the restore point | 2 | 305 | 128.2 | 42.0 % | 14.42 | 4486 / 4619 | True |
| 35b-4k | main | the same prompt again, restored from the host tier | 2 | 41 | 128.6 | 42.0 % | 14.37 | 4615 / 4619 | True |
| 35b-4k | fix | cold, store on (empty store) | 2 | 1161 | 148.7 | 59.9 % | 14.78 | 0 / 4619 | True |
| 35b-4k | fix | restored from disk after a restart: a new turn after the restore point | 2 | 311 | 148.8 | 59.9 % | 14.77 | 4486 / 4619 | True |
| 35b-4k | fix | the same prompt again, restored from the host tier | 2 | 43 | 149.1 | 59.9 % | 14.74 | 4615 / 4619 | True |

| window | bin | case | extra decode s (256 tokens) | TTFT saved s | cost / saved (rule: <= 20 %) | break-even reply length |
|---|---|---|---|---|---|---|
| 27b-4k | main | new turn, from disk | +0.48 | 4.95 | +10 % | 2650 tokens |
| 27b-4k | main | repeat, from host | +0.57 | 5.72 | +10 % | 2560 tokens |
| 27b-4k | fix | new turn, from disk | -0.02 | 4.96 | -0 % | never (restored decodes at least as fast) |
| 27b-4k | fix | repeat, from host | -0.02 | 5.74 | -0 % | never (restored decodes at least as fast) |
| 35b-4k | main | new turn, from disk | +0.27 | 0.87 | +31 % | 811 tokens |
| 35b-4k | main | repeat, from host | +0.27 | 1.13 | +24 % | 1083 tokens |
| 35b-4k | fix | new turn, from disk | -0.00 | 0.85 | -0 % | never (restored decodes at least as fast) |
| 35b-4k | fix | repeat, from host | -0.00 | 1.12 | -0 % | never (restored decodes at least as fast) |

thermal flags (server step time > 10 % off its arm's median): none

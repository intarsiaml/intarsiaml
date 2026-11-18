# ids vs the single-sequence references (equal / compared; cold prompts only)

| config (visit, mode) | set | cached_tokens != 0 | errors | = s_def | = s_spec0 | = s_store | = s_store_spec0 |
|---|---|---|---|---|---|---|---|
| s_def seq | 8192 | 0 | 0 | 16/16 | 16/16 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) |
| s_def seq | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| s_spec0 seq | 8192 | 0 | 0 | 16/16 | 16/16 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) |
| s_spec0 seq | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| s_spec0_nomacro seq | 8192 | 0 | 0 | 16/16 | 16/16 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) |
| s_spec0_nomacro seq | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| s_store seq | 8192 | 0 | 0 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) | 16/16 | 16/16 |
| s_store seq | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| s_store_spec0 seq | 8192 | 0 | 0 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) | 16/16 | 16/16 |
| s_store_spec0 seq | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| b_off burst | 8192 | 0 | 0 | 16/16 | 16/16 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) |
| b_off burst | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| b_store burst | 8192 | 0 | 0 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) | 16/16 | 16/16 |
| b_store burst | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| b_off_spec0 burst | 8192 | 0 | 0 | 16/16 | 16/16 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) |
| b_off_spec0 burst | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| b_store_spec0 burst | 8192 | 0 | 0 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) | 16/16 | 16/16 |
| b_store_spec0 burst | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| b_off_nomacro burst | 8192 | 0 | 0 | 16/16 | 16/16 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) |
| b_off_nomacro burst | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |
| b_off_plain burst | 8192 | 0 | 0 | 16/16 | 16/16 | 2/16 (first diff at token 0..46) | 2/16 (first diff at token 0..46) |
| b_off_plain burst | 8193 | 0 | 0 | 16/16 | 16/16 | 16/16 | 16/16 |

# pairwise: burst vs burst (the 8 simultaneous prompts of a seed, both seeds)

| A | B | set 8192 | set 8193 |
|---|---|---|---|
| b_off | b_store | 2/16 (tok 0..46) | 16/16 |
| b_off | b_off_spec0 | 16/16 | 16/16 |
| b_off | b_store_spec0 | 2/16 (tok 0..46) | 16/16 |
| b_off | b_off_nomacro | 16/16 | 16/16 |
| b_off | b_off_plain | 16/16 | 16/16 |
| b_store | b_off_spec0 | 2/16 (tok 0..46) | 16/16 |
| b_store | b_store_spec0 | 16/16 | 16/16 |
| b_store | b_off_nomacro | 2/16 (tok 0..46) | 16/16 |
| b_store | b_off_plain | 2/16 (tok 0..46) | 16/16 |
| b_off_spec0 | b_store_spec0 | 2/16 (tok 0..46) | 16/16 |
| b_off_spec0 | b_off_nomacro | 16/16 | 16/16 |
| b_off_spec0 | b_off_plain | 16/16 | 16/16 |
| b_store_spec0 | b_off_nomacro | 2/16 (tok 0..46) | 16/16 |
| b_store_spec0 | b_off_plain | 2/16 (tok 0..46) | 16/16 |
| b_off_nomacro | b_off_plain | 16/16 | 16/16 |

# first diverging token index per prompt: store off vs store on (solo, s_spec0 vs s_store_spec0; and burst, b_off vs b_store)

- s_spec0 vs s_store_spec0 (seq), set 8192: 2/16 equal; first-diff token index of the differing prompts: [0, 0, 3, 4, 4, 4, 5, 6, 7, 10, 12, 22, 40, 46]
- s_spec0 vs s_store_spec0 (seq), set 8193: 16/16 equal; first-diff token index of the differing prompts: []
- s_def vs s_store (seq), set 8192: 2/16 equal; first-diff token index of the differing prompts: [0, 0, 3, 4, 4, 4, 5, 6, 7, 10, 12, 22, 40, 46]
- s_def vs s_store (seq), set 8193: 16/16 equal; first-diff token index of the differing prompts: []
- b_off vs b_store (burst), set 8192: 2/16 equal; first-diff token index of the differing prompts: [0, 0, 3, 4, 4, 4, 5, 6, 7, 10, 12, 22, 40, 46]
- b_off vs b_store (burst), set 8193: 16/16 equal; first-diff token index of the differing prompts: []
- b_off_spec0 vs b_store_spec0 (burst), set 8192: 2/16 equal; first-diff token index of the differing prompts: [0, 0, 3, 4, 4, 4, 5, 6, 7, 10, 12, 22, 40, 46]
- b_off_spec0 vs b_store_spec0 (burst), set 8193: 16/16 equal; first-diff token index of the differing prompts: []

# logprob cells (lp mode, 4 prompts per set, 4 tokens, top-5): token 0 = the prefill's last-row output. Per prompt: ids, top-1 logprob, top-1 minus top-2 margin (nats)


set 8192

| visit | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| s_def | [220, 55413, 197, 197] -5.46979 m=0.6181 | [197, 197, 197, 7] -5.32193 m=0.4102 | [220, 55413, 248046, 198] -5.41864 m=0.4125 | [220, 248046, 198, 248045] -5.38715 m=0.5640 |
| s_spec0 | [220, 55413, 197, 197] -5.46979 m=0.6181 | [197, 197, 197, 7] -5.32193 m=0.4102 | [220, 55413, 248046, 198] -5.41864 m=0.4125 | [220, 248046, 198, 248045] -5.38715 m=0.5640 |
| s_store | [220, 55413, 197, 197] -5.36976 m=0.6130 | [197, 197, 197, 7] -5.23978 m=0.5458 | [220, 55413, 220, 55413] -5.37904 m=0.4764 | [220, 248046, 198, 248045] -5.29103 m=0.3979 |
| b_off | [220, 55413, 197, 197] -5.46979 m=0.6181 | [197, 197, 197, 7] -5.32193 m=0.4102 | [220, 55413, 248046, 198] -5.41864 m=0.4125 | [220, 248046, 198, 248045] -5.38715 m=0.5640 |
| b_store | [220, 55413, 197, 197] -5.36976 m=0.6130 | [197, 197, 197, 7] -5.23978 m=0.5458 | [220, 55413, 220, 55413] -5.37904 m=0.4764 | [220, 248046, 198, 248045] -5.29103 m=0.3979 |

set 8193

| visit | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| s_def | [197, 197, 7, 5513] -5.00310 m=0.4010 | [220, 55413, 197, 197] -5.79943 m=0.2898 | [261, 197, 197, 320] -3.33662 m=0.0148 | [220, 55413, 197, 197] -5.30524 m=0.0814 |
| s_spec0 | [197, 197, 7, 5513] -5.00310 m=0.4010 | [220, 55413, 197, 197] -5.79943 m=0.2898 | [261, 197, 197, 320] -3.33662 m=0.0148 | [220, 55413, 197, 197] -5.30524 m=0.0814 |
| s_store | [197, 197, 7, 5513] -5.00310 m=0.4010 | [220, 55413, 197, 197] -5.79943 m=0.2898 | [261, 197, 197, 320] -3.33662 m=0.0148 | [220, 55413, 197, 197] -5.30524 m=0.0814 |
| b_off | [197, 197, 7, 5513] -5.00310 m=0.4010 | [220, 55413, 197, 197] -5.79943 m=0.2898 | [261, 197, 197, 320] -3.33662 m=0.0148 | [220, 55413, 197, 197] -5.30524 m=0.0814 |
| b_store | [197, 197, 7, 5513] -5.00310 m=0.4010 | [220, 55413, 197, 197] -5.79943 m=0.2898 | [261, 197, 197, 320] -3.33662 m=0.0148 | [220, 55413, 197, 197] -5.30524 m=0.0814 |

# max |delta logprob| over the 4 generated positions, per set, between lp visits (nats; 0 = bitwise on these 4 rows)

| A | B | set 8192 | set 8193 |
|---|---|---|---|
| s_def | s_spec0 | 0.00e+00 (ids equal 4/4) | 0.00e+00 (ids equal 4/4) |
| s_def | s_store | 1.15e+00 (ids equal 3/4) | 0.00e+00 (ids equal 4/4) |
| s_def | b_off | 0.00e+00 (ids equal 4/4) | 0.00e+00 (ids equal 4/4) |
| s_def | b_store | 1.15e+00 (ids equal 3/4) | 0.00e+00 (ids equal 4/4) |
| s_spec0 | s_store | 1.15e+00 (ids equal 3/4) | 0.00e+00 (ids equal 4/4) |
| s_spec0 | b_off | 0.00e+00 (ids equal 4/4) | 0.00e+00 (ids equal 4/4) |
| s_spec0 | b_store | 1.15e+00 (ids equal 3/4) | 0.00e+00 (ids equal 4/4) |
| s_store | b_off | 1.15e+00 (ids equal 3/4) | 0.00e+00 (ids equal 4/4) |
| s_store | b_store | 0.00e+00 (ids equal 4/4) | 0.00e+00 (ids equal 4/4) |
| b_off | b_store | 1.15e+00 (ids equal 3/4) | 0.00e+00 (ids equal 4/4) |

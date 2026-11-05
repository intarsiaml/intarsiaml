# check 2: question first, 32K

| item | expected | A reply FINAL | B reply FINAL | same reply |
|---|---|---|---|---|
| hop2-L3-32k-000-qfirst | FINAL: WG-5384-KT | WG-5384-KT ok | WG-5384-KT ok | True |
| hop3-L3-32k-000-qfirst | FINAL: HW-4357-JS | HW-4357-JS ok | HW-4357-JS ok | True |
| agg_sum-L3-32k-000-qfirst | FINAL: 4730 | 4730 ok | 4730 ok | True |
| agg_sum-L4-32k-001-qfirst | FINAL: 6142 | 6142 ok | 6142 ok | True |
| agg_max-L3-32k-000-qfirst | FINAL: Gornov | Gornov ok | Gornov ok | False |
| repo_sum-L4-32k-000-qfirst | FINAL: 4514 | 4514 ok | 4514 ok | False |
| hop2-L3-32k-001-qfirst | FINAL: SF-3400-FY | SF-3400-FY ok | SF-3400-FY ok | True |
| hop3-L3-32k-001-qfirst | FINAL: GM-1410-MY | GM-1410-MY ok | GM-1410-MY ok | True |
| agg_sum-L3-32k-001-qfirst | FINAL: 5261 | 5261 ok | 5261 ok | True |
| agg_sum-L4-32k-002-qfirst | FINAL: 6488 | 6488 ok | 6488 ok | True |
| agg_max-L3-32k-001-qfirst | FINAL: Derorv | Derorv ok | Derorv ok | True |
| repo_sum-L4-32k-001-qfirst | FINAL: 4251 | 4251 ok | 4251 ok | True |

items 12; exact FINAL right: A 12, B 12; identical replies 10; empty replies 0

DIFFERING REPLIES (A, then B; first 400 chars)

--- agg_max-L3-32k-000-qfirst
A: 'The document contains many tracker notes and ledgers for the Peleld line. The failed-test counts recorded for Peleld releases are:\n\n- Pelkel: 39\n- Falwyn: 32\n- Yarvex: 139\n- Hoxcal: 106\n- Quincal: 333\n- Goreld: 226\n- Dunvex: 351\n- Gornov: 394\n- Wynilk: 145\n\nThe largest is Gornov with 394 failed tests.\n\nFINAL: Gornov'
B: 'The document contains several tracker notes and ledgers for the Peleld line:\n- Falwyn of the Peleld line: 32 failed tests\n- Yarvex of the Peleld line: 139 failed tests\n- Hoxcal of the Peleld line: 106 failed tests\n- Quincal of the Peleld line: 333 failed tests\n- Goreld of the Peleld line: 226 failed tests\n- Dunvex of the Peleld line: 351 failed tests\n- Gornov of the Peleld line: 394 failed tests\n-'

--- repo_sum-L4-32k-000-qfirst
A: 'RING = 4096 (coll_trace.rs)\nWARMUP = 5 (icb_check.rs)\nPRE_MIN_T = 64 (gdn.rs)\nEXIT_FAILED = 87 (health.rs)\nPOS_BYTES = 256 (verify_attn_rows.rs)\nIK_MAX_DIMS = 6 (ffi.rs)\n\nSum: 4096 + 5 + 64 + 87 + 256 + 6 = 4514\n\nFINAL: 4514'
B: 'RING = 4096 (coll_trace.rs); WARMUP = 5 (icb_check.rs); PRE_MIN_T = 64 (gdn.rs); EXIT_FAILED = 87 (health.rs); POS_BYTES = 256 (verify_attn_rows.rs); IK_MAX_DIMS = 6 (ffi.rs).\nSum: 4096 + 5 + 64 + 87 + 256 + 6 = 4514.\nFINAL: 4514'

PASS

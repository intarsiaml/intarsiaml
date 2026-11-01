| prompt | pos | intarsia top-5 (id text: logprob) | TF token | TF rank | gap top-1 - TF (nats) | top-1 = recorded intarsia token |
|---|---|---|---|---|---|---|
| story | 10 | 1205 " what": -1.493, 270 " the": -1.540, 8331 " stories": -1.600, 778 " which": -2.490, 1479 " where": -2.819 | 8331 | 3 | 0.107 | NO: 1205 |
| code | 49 | 3465 "def": -1.273, 30 "<": -1.599, 9854 "```": -1.970, 68521 "undefined": -2.582, 361 "   ": -3.050 | 68521 | 4 | 1.309 | NO: 3465 |
| science | 2 | 21251 " Ru": -0.860, 19105 " Rub": -0.997, 396 " that": -2.641, 8368 " responsible": -3.155, 15215 " rub": -3.339 | 19105 | 2 | 0.137 | yes |
| chat_short | 4 | 4521 " clear": -1.078, 3549 " specific": -1.396, 23926 " realistic": -1.799, 73918 " achievable": -1.932, 260 " a": -3.444 | 4521 | 1 | 0.000 | NO: 4521 |
| chat_system | 45 | 38835 "@@": -0.574, 30 "<": -1.116, 94 "|": -3.289, 0 "": -3.358, 32 ">": -5.296 | 38835 | 1 | 0.000 | NO: 38835 |
| chat_multiturn | 50 | 5292 " usually": -1.077, 270 " the": -1.127, 260 " a": -2.307, 8600 " typically": -2.480, 2951 " based": -2.838 | 5292 | 1 | 0.000 | NO: 5292 |
| chat_reasoning | 2 | 807 " go": -0.991, 9487 " solve": -1.008, 4189 " break": -1.766, 13254 " carefully": -3.548, 1116 " work": -4.103 | 9487 | 2 | 0.016 | yes |

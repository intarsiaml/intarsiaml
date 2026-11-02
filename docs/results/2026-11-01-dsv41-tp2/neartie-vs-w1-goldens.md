| prompt | pos | intarsia top-5 (id text: logprob) | TF token | TF rank | gap top-1 - TF (nats) | top-1 = recorded intarsia token |
|---|---|---|---|---|---|---|
| story | 10 | 1205 " what": -1.493, 270 " the": -1.540, 8331 " stories": -1.600, 778 " which": -2.490, 1479 " where": -2.819 | 8331 | 3 | 0.107 | NO: 1205 |
| code | 47 | 1718 "</": -0.674, 1 "": -0.725, 40761 "PY": -6.571, 28986 "```\n": -6.759, 5 "#": -7.403 | 1 | 2 | 0.051 | yes |
| science | 2 | 21251 " Ru": -0.860, 19105 " Rub": -0.997, 396 " that": -2.641, 8368 " responsible": -3.155, 15215 " rub": -3.339 | 19105 | 2 | 0.137 | yes |
| chat_short | 4 | 4521 " clear": -1.078, 3549 " specific": -1.396, 23926 " realistic": -1.799, 73918 " achievable": -1.932, 260 " a": -3.444 | 4521 | 1 | 0.000 | NO: 4521 |
| chat_system | 28 | 1 "": -0.000, 539 " \n": -10.747, 65 "_": -11.331, 624 "ass": -11.678, 1004 " \n\n": -12.540 | 1 | 1 | 0.000 | NO: 1 |
| chat_multiturn | 50 | 5292 " usually": -1.077, 270 " the": -1.127, 260 " a": -2.307, 8600 " typically": -2.480, 2951 " based": -2.838 | 5292 | 1 | 0.000 | NO: 5292 |
| chat_reasoning | 2 | 807 " go": -0.991, 9487 " solve": -1.008, 4189 " break": -1.766, 13254 " carefully": -3.548, 1116 " work": -4.103 | 9487 | 2 | 0.016 | yes |

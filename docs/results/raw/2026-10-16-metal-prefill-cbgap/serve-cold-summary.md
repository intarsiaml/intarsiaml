| gap s | arm | n | request wall ms (curl, 1 token) | load s |
|---|---|---|---|---|
| 0 | base | 6 | 1151 [1146..1157] | 38.7 |
| 0 | newoff | 6 | 1149 [1146..1152] | 38.8 |
| 0 | new | 6 | 1149 [1146..1152] | 38.6 |
| 3 | base | 6 | 1153 [1147..1158] | 38.6 |
| 3 | newoff | 6 | 1151 [1147..1155] | 38.6 |
| 3 | new | 6 | 1154 [1150..1161] | 38.7 |

paired by round (new - arm), request wall ms, median [min..max]

gap 0: new - base: +0.1 [-8.8..+2.4] (n=6)
gap 0: new - newoff: +1.2 [-2.9..+4.3] (n=6)
gap 3: new - base: +2.3 [-5.0..+7.8] (n=6)
gap 3: new - newoff: +4.0 [-4.3..+6.9] (n=6)

request's first command buffer: GPU start minus commit return (ms), arms with host timestamps

gap 0 newoff: 102 [100..105] (n=6)
gap 0 new: 104 [101..107] (n=6)
gap 3 newoff: 104 [101..108] (n=6)
gap 3 new: 107 [104..115] (n=6)

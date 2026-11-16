| arm | scenario | rule | B cached | B prompt | warm ids == cold ids | first diff | acceptance warm | acceptance cold | tok/step warm | tok/step cold |
|---|---|---|---:|---:|---|---|---|---|---|---|
| main | c4096 | default | 0 | 5300 | yes (128 ids) | - | 84/86 = 97.7% | 84/86 = 97.7% | 2.977 | 2.977 |
| main | c4096 | macro0 | 4096 | 5300 | yes (128 ids) | - | 84/86 = 97.7% | 84/86 = 97.7% | 2.977 | 2.977 |
| pr | c4096 | default | 4096 | 5300 | yes (128 ids) | - | 84/86 = 97.7% | 84/86 = 97.7% | 2.977 | 2.977 |
| pr | c4096 | exact | 4096 | 5300 | yes (128 ids) | - | 84/86 = 97.7% | 84/86 = 97.7% | 2.977 | 2.977 |

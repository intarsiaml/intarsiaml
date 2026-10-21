| arm | prefill 1024 | prefill 4096 | prefill 16384 | decode 1024 | decode 4096 | decode 16384 |
|---|---|---|---|---|---|---|
| tp1 (n=3) | 6520 | 9601 | 8098 | 159.2 | 154.5 | 142.7 |
| tp1eu (n=3) | 6323 | 9447 | 7992 | 117.0 | 114.8 | 107.2 |
| tp2 (n=3) | 8180 | 11330 | 10271 | 131.7 | 131.5 | 127.7 |
| tp2 vs tp1 | +25.5 % | +18.0 % | +26.8 % | -17.3 % | -14.9 % | -10.5 % |
| tp2 vs tp1eu | +29.4 % | +19.9 % | +28.5 % | +12.5 % | +14.6 % | +19.1 % |

spread tp1 1024: prefill 6505-6572 (1.0 %), decode 159.2-159.4 (0.2 tok/s)
spread tp1 4096: prefill 9565-9662 (1.0 %), decode 154.4-155.5 (1.1 tok/s)
spread tp1 16384: prefill 8088-8154 (0.8 %), decode 141.8-143.1 (1.3 tok/s)
spread tp1eu 1024: prefill 6299-6351 (0.8 %), decode 116.9-117.2 (0.3 tok/s)
spread tp1eu 4096: prefill 9438-9521 (0.9 %), decode 114.3-114.9 (0.6 tok/s)
spread tp1eu 16384: prefill 7978-8020 (0.5 %), decode 107.2-107.7 (0.5 tok/s)
spread tp2 1024: prefill 8178-8191 (0.2 %), decode 131.5-131.8 (0.3 tok/s)
spread tp2 4096: prefill 11308-11336 (0.3 %), decode 131.5-131.6 (0.1 tok/s)
spread tp2 16384: prefill 10271-10284 (0.1 %), decode 127.5-127.9 (0.3 tok/s)

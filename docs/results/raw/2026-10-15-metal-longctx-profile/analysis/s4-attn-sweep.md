| invocation | variant / policy | pos0 or ctx | us/call (median) | TFLOPS or GB/s | median MHz | % samples < 1450 | W | max C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sweep-40000-128,120 a1 | attn_decode_bf16_splitkv | 40000 | 288 | 568 GB/s | few-samples |  |  |  |
| sweep-40000-128,60 a1 | attn_decode_bf16_splitkv | 40000 | 297 | 552 GB/s | few-samples |  |  |  |
| sweep-40000-128,240 a1 | attn_decode_bf16_splitkv | 40000 | 306 | 535 GB/s | few-samples |  |  |  |
| sweep-40000-128,360 a1 | attn_decode_bf16_splitkv | 40000 | 323 | 507 GB/s | few-samples |  |  |  |
| sweep-40000-128,512 a1 | attn_decode_bf16_splitkv | 40000 | 320 | 513 GB/s | few-samples |  |  |  |
| sweep-40000-256,120 a1 | attn_decode_bf16_splitkv | 40000 | 289 | 567 GB/s | few-samples |  |  |  |
| sweep-40000-128,120 a1 | attn_decode_bf16_splitkv | 40000 | 286 | 574 GB/s | few-samples |  |  |  |
| sweep-16384-128,120 a1 | attn_decode_bf16_splitkv | 16384 | 122 | 549 GB/s | few-samples |  |  |  |
| sweep-16384-128,60 a1 | attn_decode_bf16_splitkv | 16384 | 126 | 532 GB/s | few-samples |  |  |  |
| sweep-16384-128,240 a1 | attn_decode_bf16_splitkv | 16384 | 137 | 490 GB/s | few-samples |  |  |  |
| sweep-16384-128,360 a1 | attn_decode_bf16_splitkv | 16384 | 142 | 473 GB/s | few-samples |  |  |  |
| sweep-16384-128,512 a1 | attn_decode_bf16_splitkv | 16384 | 137 | 491 GB/s | few-samples |  |  |  |
| sweep-16384-256,120 a1 | attn_decode_bf16_splitkv | 16384 | 122 | 551 GB/s | few-samples |  |  |  |
| sweep-16384-128,120 a1 | attn_decode_bf16_splitkv | 16384 | 123 | 547 GB/s | few-samples |  |  |  |

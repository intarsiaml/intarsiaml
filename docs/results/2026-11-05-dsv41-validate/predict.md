corpus 31700 tokens; A = 6000 tokens (chunks 2048 2048 1904, floor 5872)
| B | tokens | default | REPLAY=0 | REPLAY_SKIP=0 |
|---|---|---|---|---|
| B1_A+300 (A completed (max_tokens 1)) | 6300 | 6000 | 6000 | 6000 |
| B1m_A[:-8]+50 (A completed (max_tokens 1)) | 6042 | 5992 | 5992 | 5992 |
| B2_A[:4100]+50 (A completed (max_tokens 1)) | 4150 | 0 | 0 | 0 |
| B3_A[:2040]+50 (A completed (max_tokens 1)) | 2090 | 0 | 0 | 0 |
| B3p_A[:4088]+50 (A completed (max_tokens 1)) | 4138 | 0 | 0 | 0 |
| B3_A[:2040]+50 (A cancelled, state 4096) | 2090 | 0 | 0 | 0 |
| B3p_A[:4088]+50 (A cancelled, state 4096) | 4138 | 0 | 4088 | 4088 |

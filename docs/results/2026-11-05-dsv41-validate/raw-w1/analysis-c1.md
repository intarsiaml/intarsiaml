# check 1

## arm default

- cold B1 on the fresh server vs cold B1 after a separator request: ids identical = True (64 tokens), cached 0 / 0

| B | predicted cached | measured cached | ids equal to cold B | equal prefix (of 64) | garbage |
|---|---|---|---|---|---|
| B1_A+300 | 6000 | 6000 | True | 64 | False |
| B1m_A[:-8]+50 | 5992 | 5992 | False | 11 | False |
| B2_A[:4100]+50 | 0 | 0 | True | 64 | False |

cancel landing: [0.8, 4096]; calibration: [{"delay_s": 0.8, "probe_cached": 4096, "probe_wall_s": 0.39, "metrics_delta": {"intarsia_requests_finished_total{reason=\"cancelled\"}": ["1", null], "intarsia_request_prompt_tokens_bucket{le=\"8192\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"12288\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"16384\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"24576\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"32768\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"65536\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"131072\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"262144\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"+Inf\"}": ["24", "23"], "intarsia_request_prompt_tokens_sum": ["76452", "70452"], "intarsia_request_prompt_tokens_count": ["24", "23"]}}]
- after the cancel (state 4096): B3p_A[:4088]+50: predicted cached 0, measured 0, ids equal to cold True, equal prefix 64, garbage False
- after the cancel (state 4096): B3_A[:2040]+50: predicted cached 0, measured 0, ids equal to cold True, equal prefix 64, garbage False

## arm off

- cold B1 on the fresh server vs cold B1 after a separator request: ids identical = True (64 tokens), cached 0 / 0

| B | predicted cached | measured cached | ids equal to cold B | equal prefix (of 64) | garbage |
|---|---|---|---|---|---|
| B1_A+300 | 6000 | 6000 | True | 64 | False |
| B1m_A[:-8]+50 | 5992 | 5992 | False | 27 | False |
| B2_A[:4100]+50 | 0 | 0 | True | 64 | False |

cancel landing: [0.8, 2048]; calibration: [{"delay_s": 0.8, "probe_cached": 2048, "probe_wall_s": 2.62, "metrics_delta": {"intarsia_requests_finished_total{reason=\"cancelled\"}": ["1", null], "intarsia_request_prompt_tokens_bucket{le=\"8192\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"12288\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"16384\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"24576\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"32768\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"65536\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"131072\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"262144\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"+Inf\"}": ["24", "23"], "intarsia_request_prompt_tokens_sum": ["76452", "70452"], "intarsia_request_prompt_tokens_count": ["24", "23"]}}]
- after the cancel (state 2048): B3p_A[:4088]+50: predicted cached 2048, measured 2048, ids equal to cold True, equal prefix 64, garbage False
- after the cancel (state 2048): B3_A[:2040]+50: predicted cached 2040, measured 2040, ids equal to cold False, equal prefix 45, garbage False

## arm skip0

- cold B1 on the fresh server vs cold B1 after a separator request: ids identical = True (64 tokens), cached 0 / 0

| B | predicted cached | measured cached | ids equal to cold B | equal prefix (of 64) | garbage |
|---|---|---|---|---|---|
| B1_A+300 | 6000 | 6000 | True | 64 | False |
| B1m_A[:-8]+50 | 5992 | 5992 | False | 11 | False |
| B2_A[:4100]+50 | 0 | 0 | True | 64 | False |

cancel landing: [0.8, 2048]; calibration: [{"delay_s": 0.8, "probe_cached": 2048, "probe_wall_s": 1.6, "metrics_delta": {"intarsia_requests_finished_total{reason=\"cancelled\"}": ["1", null], "intarsia_request_prompt_tokens_bucket{le=\"8192\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"12288\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"16384\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"24576\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"32768\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"65536\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"131072\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"262144\"}": ["24", "23"], "intarsia_request_prompt_tokens_bucket{le=\"+Inf\"}": ["24", "23"], "intarsia_request_prompt_tokens_sum": ["76452", "70452"], "intarsia_request_prompt_tokens_count": ["24", "23"]}}]
- after the cancel (state 2048): B3p_A[:4088]+50: predicted cached 2048, measured 2048, ids equal to cold True, equal prefix 64, garbage False
- after the cancel (state 2048): B3_A[:2040]+50: predicted cached 2040, measured 2040, ids equal to cold False, equal prefix 45, garbage False

## default against off, same B

| B | hit ids equal | equal prefix | cold ids equal |
|---|---|---|---|
| B1_A+300 | True | 64 | True |
| B1m_A[:-8]+50 | False | 27 | False |
| B2_A[:4100]+50 | False | 7 | False |

PASS

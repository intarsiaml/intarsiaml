# cbsoak burst

16 simultaneous requests (max_queue 4): status {'ok': 8, 'shed': 8}, http {'200': 8, '429': 8}, Retry-After on the 429s {'1': 8}, `intarsia_requests_rejected_total` delta 8.0; /metrics peaks {'queue_max': 4.0, 'running_max': 4.0, 'waiting_max': 4.0}; admitted TTFT p50 5197.0 p95 10197.2 max 10263.6 ms, admitted ITL p50 29.6 p95 30.2 max 101.9 ms; after: health {'status': 'ok', 'running': 0, 'waiting': 0}, follow-up {'status': 'ok', 'ttft_ms': 67.83670792356133}.
429 body: ['HTTP 429: {"error":{"message":"the request queue is full (4 waiting); retry shortly","type":"rate_limit_error","param":n']

VIOLATIONS: none

# cbsoak burst

24 simultaneous requests (max_queue 4): status {'shed': 16, 'ok': 8}, http {'429': 16, '200': 8}, Retry-After on the 429s {'1': 16}, `intarsia_requests_rejected_total` delta 16.0; /metrics peaks {'queue_max': 6.0, 'running_max': 4.0, 'waiting_max': 6.0}; admitted TTFT p50 4804.3 p95 9180.1 max 9389.4 ms, admitted ITL p50 30.8 p95 34.3 max 111.9 ms; after: health {'status': 'ok', 'running': 0, 'waiting': 0}, follow-up {'status': 'ok', 'ttft_ms': 84.26287490874529}.
429 body: ['HTTP 429: {"error":{"message":"the request queue is full (4 waiting); retry shortly","type":"rate_limit_error","param":n']

VIOLATIONS: none

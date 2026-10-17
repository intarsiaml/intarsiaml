# cbsoak burst

24 simultaneous requests (max_queue 4): status {'ok': 8, 'shed': 16}, http {'200': 8, '429': 16}, Retry-After on the 429s {'1': 16}, `intarsia_requests_rejected_total` delta 16.0; /metrics peaks {'queue_max': 6.0, 'running_max': 4.0, 'waiting_max': 6.0}; admitted TTFT p50 4877.6 p95 9230.9 max 9454.8 ms, admitted ITL p50 30.9 p95 83.8 max 108.1 ms; after: health {'status': 'ok', 'running': 0, 'waiting': 0}, follow-up {'status': 'ok', 'ttft_ms': 85.22124984301627}.
429 body: ['HTTP 429: {"error":{"message":"the request queue is full (4 waiting); retry shortly","type":"rate_limit_error","param":n']

VIOLATIONS: none

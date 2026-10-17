# cbsoak burst

16 simultaneous requests (max_queue 4): status {'shed': 8, 'ok': 8}, http {'429': 8, '200': 8}, Retry-After on the 429s {'1': 8}, `intarsia_requests_rejected_total` delta 8.0; /metrics peaks {'queue_max': 4.0, 'running_max': 4.0, 'waiting_max': 4.0}; admitted TTFT p50 5313.3 p95 10350.2 max 10416.4 ms, admitted ITL p50 30.1 p95 31.6 max 100.7 ms; after: health {'status': 'ok', 'running': 0, 'waiting': 0}, follow-up {'status': 'ok', 'ttft_ms': 68.1289171334356}.
429 body: ['HTTP 429: {"error":{"message":"the request queue is full (4 waiting); retry shortly","type":"rate_limit_error","param":n']

VIOLATIONS: none

#!/usr/bin/env python3
"""Cache-affinity router prototype: send each chat request to the replica most likely to hold its prefix.

Two (or more) single-sequence intarsia replicas, each with its own prefix store (e.g. tatooine's two R9700s at tp=1),
behind one OpenAI-compatible endpoint. The router never sees token ids; it keys on the *messages*: the hash of every
leading run of messages (system, system+user1, system+user1+assistant1, ...) and of the tool schema. A request goes to
the replica that served the longest such run most recently (its store holds that prefix, modulo eviction), unless that
replica is busier than the least-loaded one by more than `--imbalance` requests; then load wins and the chosen replica
learns the prefix too. New conversations go to the least-loaded replica (ties: rendezvous hash of the first message,
so the same system prompt keeps landing on the same replica).

Stdlib only, prototype quality: no streaming backpressure tuning, no auth, no health checks.
  python3 affinity_router.py --listen 127.0.0.1:8100 --replica http://tatooine:8091 --replica http://tatooine:8092
  python3 affinity_router.py --self-test
"""

import argparse
import hashlib
import http.server
import json
import threading
import time
import urllib.request
from collections import OrderedDict


def run_keys(body):
    """Hash of every leading run of messages (with the tools), longest last."""
    tools = json.dumps(body.get("tools"), sort_keys=True)
    h = hashlib.sha256(tools.encode())
    keys = []
    for m in body.get("messages", []):
        h.update(json.dumps(m, sort_keys=True).encode())
        keys.append(h.copy().hexdigest()[:24])
    return keys


class Router:
    def __init__(self, replicas, capacity=4096, imbalance=2):
        self.replicas = list(replicas)
        self.owner = OrderedDict()  # run key -> replica index (LRU)
        self.capacity = capacity
        self.imbalance = imbalance
        self.inflight = [0] * len(self.replicas)
        self.lock = threading.Lock()
        self.stats = {"affinity": 0, "new": 0, "overflow": 0}

    def pick(self, body):
        keys = run_keys(body)
        with self.lock:
            least = min(range(len(self.replicas)), key=lambda i: (self.inflight[i], i))
            # the last message is the new turn: only runs before it can be cached. A system prompt alone is no
            # affinity: every replica pins it at startup (--prefix-store-pin), so it must not pull every new
            # conversation onto one replica.
            msgs = body.get("messages", [])
            lo = 1 if msgs and msgs[0].get("role") == "system" else 0
            for k in reversed(keys[lo:-1]):
                r = self.owner.get(k)
                if r is not None:
                    if self.inflight[r] - self.inflight[least] > self.imbalance:
                        self.stats["overflow"] += 1
                        choice = least
                    else:
                        self.stats["affinity"] += 1
                        choice = r
                    break
            else:
                self.stats["new"] += 1
                first = keys[min(lo, len(keys) - 1)] if keys else ""
                tied = [i for i in range(len(self.replicas)) if self.inflight[i] == self.inflight[least]]
                choice = max(tied, key=lambda i: hashlib.sha256((first + str(i)).encode()).digest())
            for k in keys:
                self.owner[k] = choice
                self.owner.move_to_end(k)
            while len(self.owner) > self.capacity:
                self.owner.popitem(last=False)
            self.inflight[choice] += 1
            return choice

    def done(self, i):
        with self.lock:
            self.inflight[i] -= 1


def serve(router, listen):
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            try:
                body = json.loads(raw)
            except ValueError:
                body = {}
            i = router.pick(body) if self.path.endswith("/chat/completions") else 0
            try:
                req = urllib.request.Request(router.replicas[i] + self.path, data=raw, method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    self.send_response(r.status)
                    for k, v in r.headers.items():
                        if k.lower() not in ("transfer-encoding", "connection"):
                            self.send_header(k, v)
                    self.send_header("X-Intarsia-Replica", str(i))
                    self.end_headers()
                    while True:
                        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            finally:
                router.done(i)

        def do_GET(self):
            if self.path == "/router/stats":
                b = json.dumps({**router.stats, "inflight": router.inflight, "keys": len(router.owner)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b)
                return
            with urllib.request.urlopen(router.replicas[0] + self.path, timeout=30) as r:
                self.send_response(r.status)
                self.end_headers()
                self.wfile.write(r.read())

    host, port = listen.rsplit(":", 1)
    http.server.ThreadingHTTPServer((host, int(port)), H).serve_forever()


def self_test():
    r = Router(["a", "b"])
    sys_msg = {"role": "system", "content": "fixed system prompt"}
    conv = lambda i, turns: [sys_msg] + [{"role": "user", "content": f"c{i} t{t}"} for t in range(turns)]
    # four conversations, three turns each, interleaved; every request completes before the next
    placed = {}
    for t in range(1, 4):
        for c in range(4):
            i = r.pick({"messages": conv(c, t)})
            r.done(i)
            placed.setdefault(c, set()).add(i)
    # a conversation sticks to its replica after its first turn
    assert all(len(v) == 1 for v in placed.values()), placed
    # turns 2 and 3 follow their conversation; the shared system prompt does not pull them all onto one replica
    assert r.stats["affinity"] == 8 and r.stats["new"] == 4, r.stats
    assert len(set().union(*placed.values())) == 2, placed
    # overload: a busy owner hands a request to the idle replica
    r2 = Router(["a", "b"], imbalance=1)
    i0 = r2.pick({"messages": conv(0, 1)})
    r2.done(i0)
    for _ in range(3):
        r2.inflight[i0] += 1
    i1 = r2.pick({"messages": conv(0, 2)})
    assert i1 != i0 and r2.stats["overflow"] == 1, (i0, i1, r2.stats)
    print("affinity router self-test: ok", r.stats, r2.stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:8100")
    ap.add_argument("--replica", action="append", default=[])
    ap.add_argument("--imbalance", type=int, default=2)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        if len(a.replica) < 1:
            ap.error("give --replica at least once")
        serve(Router(a.replica, imbalance=a.imbalance), a.listen)

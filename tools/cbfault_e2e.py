#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""End-to-end fatal-failure test of Metal continuous batching through `intarsia serve` (M2 item D3, docs/research/continuous-batching-design.md 2.5).

Needs a binary built with `--features metal,fault-inject` and the GPU; run it as the command of an exclusive fleetq slot
(tools/cbfault_slot.sh does). For each case it starts `intarsia serve --max-batch 8` with INTARSIA_FAULT_INJECT set,
builds the mixed load, lets the injected fault fire, and checks the designed behaviour:

  load     7 decoding streams (4 streamed, 3 not) + one 4k prompt mid-prefill (streamed) + 3 requests queued behind the
           full house (2 streamed, 1 not). Fault spec `<mode>:mixed=7,waiting=2,pos=512`: it fires on the first decode
           step with 7 decoding rows, at least 2 queued and the 4k prompt at least 512 tokens into its prefill.
  checks   every request ends with a clean error (streamed: an error event then [DONE]; not streamed: HTTP 500 JSON body),
           none hangs or is cut off, later requests get 503 JSON, /health is 503 "failed", /metrics counters match what the
           clients saw, the process stays up refusing (--no-supervisor) or exits 87 and the supervisor restarts it
           (default), and a fresh server serves a normal request.

    tools/cbfault_e2e.py --bin PATH --model ~/models/Qwen3-8B --out DIR [--modes err,panic] [--cases nosup,sup] [--port 8117]

Standard library only. Exit 0 when every check passed, 1 otherwise, 2 for harness problems. Only processes this script
started are ever signalled (its own process group).
"""

import argparse
import http.client
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "prompts" / "corpus_docs.txt"
CLIENT_DEADLINE_S = 120.0  # whole-request budget of every client
READ_TIMEOUT_S = 40.0  # one socket read (a healthy stream sends a chunk every ~40 ms; a stall this long is a hang)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------------------------------------------------

def http_json(port, method, path, body=None, timeout=15.0):
    """(status, parsed json or raw text). Never raises on HTTP status; raises on transport errors."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=None if body is None else json.dumps(body).encode(),
                     headers={"Content-Type": "application/json", "Connection": "close"})
        r = conn.getresponse()
        raw = r.read().decode("utf-8", "replace")
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw
    finally:
        conn.close()


def http_text(port, path, timeout=15.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path, headers={"Connection": "close"})
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", "replace")
    finally:
        conn.close()


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.rpartition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def try_metrics(port):
    try:
        st, txt = http_text(port, "/metrics")
        return parse_metrics(txt) if st == 200 else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------------------------------
# one client = one request, recorded in full
# ---------------------------------------------------------------------------------------------------------------------

class Client(threading.Thread):
    """Sends one /v1/completions request and records how it ended. outcome:
    done            finished normally (finish_reason set)
    error_event     streamed: HTTP 200, an `error` SSE event, then [DONE]
    error_event_nodone  streamed: an error event but no [DONE] before EOF
    http_error      not streamed (or refused before streaming): non-200 status with a JSON error body
    truncated       streamed: EOF with neither [DONE] nor an error event, or a non-streamed body that did not parse
    hang            no terminal event within the deadline / a read timed out
    transport       connection error (refused, reset)"""

    def __init__(self, name, port, prompt, max_tokens, stream, model, t0, path="/v1/completions"):
        super().__init__(name=name, daemon=True)
        self.cname, self.port, self.stream, self.t0 = name, port, stream, t0
        self.path = path
        if path == "/v1/completions":
            self.body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": stream,
                         "ignore_eos": True}
        else:
            self.body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0.0, "stream": stream, "chat_template_kwargs": {"enable_thinking": False}}
        self.tokens = 0
        self.status = None
        self.outcome = None
        self.error_message = None
        self.error_body = None
        self.finish_reason = None
        self.prompt_tokens = None
        self.t_start = self.t_first = self.t_end = None
        self.saw_done_marker = False
        self.detail = ""

    def rel(self):
        return time.time() - self.t0

    def run(self):
        self.t_start = self.rel()
        deadline = time.time() + CLIENT_DEADLINE_S
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=READ_TIMEOUT_S)
            conn.request("POST", self.path, body=json.dumps(self.body).encode(),
                         headers={"Content-Type": "application/json", "Connection": "close"})
            r = conn.getresponse()
            self.status = r.status
            if r.status != 200:
                self._finish_nonstream(r.read())
            elif not self.stream:
                self._finish_nonstream(r.read())
            else:
                self._read_stream(r, deadline)
            conn.close()
        except TimeoutError:
            self.outcome, self.detail = "hang", f"read timed out after {READ_TIMEOUT_S:.0f}s"
        except (ConnectionError, http.client.HTTPException, OSError) as e:
            self.outcome, self.detail = ("hang", str(e)) if "timed out" in str(e) else ("transport", f"{type(e).__name__}: {e}")
        self.t_end = self.rel()

    def _finish_nonstream(self, raw):
        try:
            obj = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            self.outcome, self.detail = "truncated", f"body is not JSON ({len(raw)} bytes): {raw[:200]!r}"
            return
        if self.status == 200:
            ch = (obj.get("choices") or [{}])[0]
            self.finish_reason = ch.get("finish_reason")
            u = obj.get("usage") or {}
            self.tokens, self.prompt_tokens = u.get("completion_tokens", 0), u.get("prompt_tokens")
            self.outcome = "done"
        else:
            self.error_body = obj
            self.error_message = ((obj.get("error") or {}).get("message")) if isinstance(obj, dict) else None
            self.outcome = "http_error"

    def _read_stream(self, r, deadline):
        while True:
            if time.time() > deadline:
                self.outcome, self.detail = "hang", "client deadline"
                return
            line = r.readline()
            if not line:
                self.outcome = "error_event_nodone" if self.error_message else "truncated"
                self.detail = "EOF" + (" after error event" if self.error_message else " without [DONE] or error event")
                return
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                self.saw_done_marker = True
                self.outcome = "error_event" if self.error_message else "done"
                return
            try:
                ev = json.loads(data)
            except ValueError:
                self.detail += f" bad-json:{data[:80]!r}"
                continue
            if "error" in ev:
                self.error_body = ev
                self.error_message = (ev["error"] or {}).get("message") or str(ev["error"])
                continue
            ch = (ev.get("choices") or [{}])[0] if ev.get("choices") else {}
            if ch.get("text") or (ch.get("delta") or {}).get("content"):
                self.tokens += 1
                if self.t_first is None:
                    self.t_first = self.rel()
            if ch.get("finish_reason"):
                self.finish_reason = ch["finish_reason"]
            if ev.get("usage"):
                self.prompt_tokens = ev["usage"].get("prompt_tokens", self.prompt_tokens)

    def summary(self):
        return {"name": self.cname, "stream": self.stream, "path": self.path, "http_status": self.status,
                "outcome": self.outcome, "tokens": self.tokens, "finish_reason": self.finish_reason,
                "prompt_tokens": self.prompt_tokens, "error_message": self.error_message, "error_body": self.error_body,
                "done_marker": self.saw_done_marker, "t_start": self.t_start, "t_first": self.t_first, "t_end": self.t_end,
                "detail": self.detail}


# ---------------------------------------------------------------------------------------------------------------------
# server process
# ---------------------------------------------------------------------------------------------------------------------

class Server:
    def __init__(self, args, out, port, fault, supervised, label):
        self.args, self.out, self.port, self.fault, self.supervised, self.label = args, out, port, fault, supervised, label
        self.proc = None
        self.log_path = out / f"server-{label}.log"

    def start(self):
        if subprocess.run(["lsof", "-nP", f"-iTCP:{self.port}", "-sTCP:LISTEN", "-t"], capture_output=True).stdout.strip():
            raise SystemExit(f"port {self.port} already in use")
        cmd = [self.args.bin, "serve", "--model", self.args.model, "--backend", "metal", "--max-seq", "8192",
               "--profile-every", "0", "--max-batch", "8", "--no-ui", "--host", "127.0.0.1", "--port", str(self.port)]
        if not self.supervised:
            cmd.append("--no-supervisor")
        env = dict(os.environ, DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer")
        env.pop("INTARSIA_FAULT_INJECT", None)
        if self.fault:
            env["INTARSIA_FAULT_INJECT"] = self.fault
        self.logf = open(self.log_path, "wb")
        self.t_start = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        log(f"server [{self.label}] pid {self.proc.pid} fault={self.fault!r} supervised={self.supervised}")

    def wait_ready(self, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise SystemExit(f"server [{self.label}] exited early with {self.proc.returncode}; see {self.log_path}")
            try:
                st, body = http_json(self.port, "GET", "/health", timeout=3)
                if st == 200:
                    return time.time() - t0
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
        raise SystemExit(f"server [{self.label}] not healthy after {timeout}s")

    def log_text(self):
        self.logf.flush()
        return self.log_path.read_text(errors="replace")

    def group_pids(self):
        return subprocess.run(["pgrep", "-g", str(self.proc.pid)], capture_output=True, text=True).stdout.split()

    def stop(self):
        """SIGTERM the whole process group this script started, SIGKILL after a grace; returns True if nothing is left."""
        if self.proc is None:
            return True
        for sig, wait in ((signal.SIGTERM, 40), (signal.SIGKILL, 10)):
            if not self.group_pids():
                break
            try:
                os.killpg(self.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            t0 = time.time()
            while time.time() - t0 < wait and self.group_pids():
                time.sleep(0.25)
        try:
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        left = self.group_pids()
        self.logf.close()
        return not left


# ---------------------------------------------------------------------------------------------------------------------
# the scenario
# ---------------------------------------------------------------------------------------------------------------------

class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        log(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))

    def all_ok(self):
        return all(r["ok"] for r in self.rows)


class Prompts:
    def __init__(self, port, model):
        self.text = CORPUS.read_text()
        self.cpt = 4.0
        self.port, self.model, self.n = port, model, 0

    def calibrate(self):
        chars = 6000
        st, obj = http_json(self.port, "POST", "/v1/completions",
                            {"model": self.model, "prompt": "[cal] " + self.text[:chars], "max_tokens": 1, "temperature": 0.0},
                            timeout=120)
        if st == 200 and obj.get("usage", {}).get("prompt_tokens"):
            self.cpt = chars / obj["usage"]["prompt_tokens"]
        log(f"calibrated {self.cpt:.2f} chars/token")

    def nonce(self, tag):
        self.n += 1
        return f"[{tag}-{self.n}-{int(time.time())}] "

    def tokens(self, n, tag):
        return self.nonce(tag) + self.text[: int(n * self.cpt)]

    def story(self, tag):
        return self.nonce(tag) + "Write a very long and detailed story about a lighthouse keeper who finds a map. Begin:"


def snapshot_delta(before, after):
    keys = sorted(k for k in after if k.startswith(("intarsia_batch_requests", "intarsia_batch_decode", "intarsia_engine",
                                                    "intarsia_requests", "intarsia_batch_iterations",
                                                    "intarsia_batch_prefill_chunks", "intarsia_batch_queue",
                                                    "intarsia_batch_running")))
    return {k: {"before": (before or {}).get(k), "after": after[k]} for k in keys}


def run_case(args, out, mode, supervised, port):
    label = f"{mode}-{'sup' if supervised else 'nosup'}"
    cdir = out / label
    cdir.mkdir(parents=True, exist_ok=True)
    chk = Checks()
    fault = f"{mode}:mixed=7,waiting=2,pos=512"
    srv = Server(args, cdir, port, fault, supervised, label)
    result = {"case": label, "fault": fault, "supervised": supervised}
    srv.start()
    try:
        result["ready_s"] = round(srv.wait_ready(), 1)
        st, models = http_json(port, "GET", "/v1/models")
        model = models["data"][0]["id"] if st == 200 else "model"
        prompts = Prompts(port, model)
        prompts.calibrate()
        # a normal request first: proves the armed-but-unfired plan is inert for ordinary (1-row) work
        st, obj = http_json(port, "POST", "/v1/completions",
                            {"model": model, "prompt": prompts.nonce("warm") + "Hello", "max_tokens": 8, "temperature": 0.0,
                             "ignore_eos": True}, timeout=60)
        chk.add("armed plan is inert for a normal request", st == 200 and obj["usage"]["completion_tokens"] == 8, f"HTTP {st}")
        m_before = try_metrics(port)
        t0 = time.time()
        clients = []

        def spawn(name, prompt, max_tokens, stream, path="/v1/completions"):
            c = Client(name, port, prompt, max_tokens, stream, model, t0, path)
            clients.append(c)
            c.start()
            return c

        # 7 decoding streams: 4 streamed, 3 not
        decoders = []
        for i in range(7):
            decoders.append(spawn(f"dec{i}-{'stream' if i < 4 else 'plain'}", prompts.story(f"d{i}"), 3000, i < 4))
        t_wait = time.time()
        while time.time() - t_wait < 90:
            st, h = http_json(port, "GET", "/health")
            if st == 200 and h.get("running") == 7 and all(c.tokens >= 3 for c in decoders[:4]):
                break
            time.sleep(0.2)
        else:
            chk.add("7 decoders reached steady decode", False, f"health={h}")
            raise RuntimeError("load did not build")
        chk.add("7 decoders reached steady decode", True, f"running={h.get('running')}")
        t_load = time.time() - t0
        # the 4k prompt (streamed) and 3 queued behind the full house (2 streamed, 1 not)
        # the 4k prompt goes first and must hold the 8th slot before the queued ones are sent (a short prompt sent at the same
        # time could win the slot, prefill in one chunk, and leave nothing mid-prefill for the fault to hit)
        big = spawn("big4k-stream", prompts.tokens(4096, "big"), 64, True)
        t_adm = time.time()
        while time.time() - t_adm < 10:
            st, h = http_json(port, "GET", "/health")
            if st == 200 and h.get("running") == 8:
                break
            time.sleep(0.03)
        else:
            chk.add("4k prompt took the 8th slot", False, f"health={h}")
            raise RuntimeError("4k prompt not admitted")
        q = [spawn("queued0-stream", prompts.story("q0"), 32, True),
             spawn("queued1-stream", prompts.story("q1"), 32, True),
             spawn("queued2-plain", prompts.story("q2"), 32, False)]
        # wait for every client to end (each has its own hard deadline)
        for c in clients:
            c.join(timeout=CLIENT_DEADLINE_S + 20)
        t_clients_done = time.time() - t0
        m_after = try_metrics(port)
        # who ended how
        log("client outcomes:")
        for c in clients:
            log(f"  {c.cname:16s} http={c.status} outcome={c.outcome} tokens={c.tokens} err={c.error_message!r}")
        result["clients"] = [c.summary() for c in clients]
        result["load_ready_s"] = round(t_load, 1)
        result["all_clients_done_s"] = round(t_clients_done, 1)
        still = [c.cname for c in clients if c.is_alive()]
        chk.add("no client hangs (all returned within the deadline)", not still and not any(c.outcome == "hang" for c in clients),
                f"alive={still} hang={[c.cname for c in clients if c.outcome == 'hang']}")
        ok_end = lambda c: (c.outcome == "error_event" and c.status == 200 and c.saw_done_marker) or (  # noqa: E731
            c.outcome == "http_error" and c.status == 500 and not c.stream)
        bad = [c.cname for c in clients if not ok_end(c)]
        chk.add("every in-flight and queued request ends with a clean error (stream: error event + [DONE]; plain: 500 JSON)",
                not bad, f"not clean: {bad}")
        msgs = [c.error_message or "" for c in clients]
        want = "injected fault"
        chk.add("error messages name the failure (engine failed + injected fault)",
                all("engine failed" in m and want in m for m in msgs), f"{set(msgs)}")
        chk.add("plain requests: HTTP 500 with JSON {error:{message,type}} body",
                all(c.status == 500 and isinstance(c.error_body, dict) and "message" in (c.error_body.get("error") or {})
                    for c in clients if not c.stream), "")
        chk.add("streamed requests: HTTP 200 + error event (headers were already sent) then [DONE]",
                all(c.status == 200 and c.saw_done_marker for c in clients if c.stream), "")
        # decoders and the 4k prompt really were mid-flight when it fired (tokens were streamed before the error)
        chk.add("streamed decoders had produced tokens before the error", all(c.tokens >= 3 for c in decoders[:4]),
                str([c.tokens for c in decoders[:4]]))
        chk.add("4k prompt was mid-prefill (no token streamed)", big.tokens == 0, f"tokens={big.tokens}")
        chk.add("queued requests never got a token", all(c.tokens == 0 for c in q), str([c.tokens for c in q]))

        # later requests are refused cleanly
        refusals = []
        for name, path, stream in (("late-plain", "/v1/completions", False), ("late-stream", "/v1/completions", True),
                                   ("late-chat-plain", "/v1/chat/completions", False)):
            c = Client(name, port, "Hello there", 8, stream, model, t0, path)
            c.run()
            refusals.append(c)
        result["refused"] = [c.summary() for c in refusals]
        # supervised: the watchdog exits the child ~2.5 s after the failure; a probe that already lost the race gets a refused
        # connection (clean, not a hang) and is recorded as such
        def refused_ok(c):
            if c.status == 503 and c.outcome == "http_error" and "engine failed" in (c.error_message or ""):
                return True
            return supervised and c.outcome == "transport" and "Connection refused" in c.detail
        chk.add("later submits refused with 503 + JSON error body (supervised: or a refused connection once the child exited)",
                all(refused_ok(c) for c in refusals), str([(c.cname, c.status, c.outcome, c.detail) for c in refusals]))
        # /health
        deadline = time.time() + 10
        health = None
        while time.time() < deadline:
            try:
                health = http_json(port, "GET", "/health")
            except Exception as e:  # noqa: BLE001
                health = (None, str(e))
            if health[0] == 503 or (supervised and health[0] is None):
                break
            time.sleep(0.25)
        result["health_after"] = health
        gone = supervised and health[0] is None  # supervised: the child already exited (connection refused)
        chk.add("/health is 503 with status=failed" + (" (supervised: or the child already exited)" if supervised else ""),
                gone or (health[0] == 503 and isinstance(health[1], dict) and health[1].get("status") == "failed"),
                json.dumps(health[1])[:300])
        if not gone:
            chk.add("/health reports running=0 waiting=0 (no leaked requests)",
                    isinstance(health[1], dict) and health[1].get("running") == 0 and health[1].get("waiting") == 0,
                    f"running={health[1].get('running') if isinstance(health[1], dict) else None} waiting={health[1].get('waiting') if isinstance(health[1], dict) else None}")
        # /metrics: counters vs what the clients saw (best effort in supervised mode, the child exits ~2.5 s after the failure)
        m_final = try_metrics(port)
        result["metrics_before_load"] = m_before
        result["metrics_after_clients"] = m_after
        result["metrics_after_health"] = m_final
        if m_final is None and m_after is not None:
            m_final = m_after
        if m_final is None:
            chk.add("/metrics readable after the failure", False, "metrics unreachable")
        else:
            d = lambda k: m_final.get(k, 0) - (m_before or {}).get(k, 0)  # noqa: E731
            n_err = sum(1 for c in clients if c.error_message)
            n_admitted = 8  # 7 decoders + the 4k prompt got a slot; 3 stayed queued
            failed_d = d("intarsia_batch_requests_failed_total")
            chk.add("metrics: failed_total delta == requests that saw the error (admitted + queued)", failed_d == n_err,
                    f"delta={failed_d} clients_with_error={n_err}")
            chk.add("metrics: admitted_total delta == 8 (7 decoders + 4k prompt), finished delta == 0",
                    d("intarsia_batch_requests_admitted_total") == n_admitted and d("intarsia_batch_requests_finished_total") == 0,
                    f"admitted={d('intarsia_batch_requests_admitted_total')} finished={d('intarsia_batch_requests_finished_total')}")
            chk.add("metrics: engine_failed gauge is 1", m_final.get("intarsia_engine_failed") == 1.0,
                    str(m_final.get("intarsia_engine_failed")))
            chk.add("metrics: scheduler running 0 and queue empty after the failure (no leaked slot)",
                    m_final.get("intarsia_batch_running", -1) == 0 and m_final.get("intarsia_batch_queue_depth", -1) == 0,
                    f"running={m_final.get('intarsia_batch_running')} queue={m_final.get('intarsia_batch_queue_depth')}")
            chk.add("metrics: request gauges (intarsia_requests_running / _waiting) back to 0",
                    m_final.get("intarsia_requests_running", -1) == 0 and m_final.get("intarsia_requests_waiting", -1) == 0,
                    f"running={m_final.get('intarsia_requests_running')} waiting={m_final.get('intarsia_requests_waiting')}")
            result["metrics_delta"] = snapshot_delta(m_before, m_final)

        # process behaviour
        t_fail = time.time()
        if not supervised:
            time.sleep(8)
            alive = srv.proc.poll() is None
            chk.add("process stays up refusing (--no-supervisor: watchdog does not exit)", alive, f"returncode={srv.proc.poll()}")
            st2, h2 = http_json(port, "GET", "/health")
            chk.add("still refusing 8 s later: /health 503", st2 == 503, f"HTTP {st2}")
        else:
            t_exit = time.time()
            while time.time() - t_exit < 30:
                txt = srv.log_text()
                if "restart #1" in txt:
                    break
                time.sleep(0.5)
            txt = srv.log_text()
            m_died = re.search(r"\[supervisor\] server died after [\d.]+s: (exit \d+[^;]*); restart #1", txt)
            result["supervisor_line"] = m_died.group(0) if m_died else None
            chk.add("child exits 87 (engine worker failed) and the supervisor restarts it",
                    bool(m_died) and "exit 87" in m_died.group(1), m_died.group(0) if m_died else "no restart line")
            chk.add("watchdog logged the failure before exiting", "[watchdog] engine failed" in txt, "")
            pids = re.findall(r"\[supervisor\] started server pid (\d+)", txt)
            result["child_pids"] = pids
            t_up = time.time()
            srv.wait_ready(timeout=300)
            result["restart_ready_s"] = round(time.time() - t_up, 1)
            st3, h3 = http_json(port, "GET", "/health")
            chk.add("after the restart /health is 200", st3 == 200, json.dumps(h3)[:200])
            normal_after_restart(chk, port, model, prompts, t0, result)

        # the fault line the engine printed
        txt = srv.log_text()
        fl = [l for l in txt.splitlines() if "fault-inject" in l or "scheduler panicked" in l or "panicked at" in l]
        result["server_fault_lines"] = fl[:12]
        chk.add("server logged the injection point", any("fault-inject: arming" in l for l in fl), fl[0] if fl else "")
        if mode == "panic":
            chk.add("panic mode: the panic was caught in run_scheduler (message names 'scheduler panicked')",
                    "scheduler panicked" in " ".join(msgs) or "scheduler panicked" in str(result.get("health_after")),
                    str(result.get("health_after"))[:200])
        else:
            chk.add("err mode: the backend error message reaches the clients", all("command buffer failed" in m for m in msgs), "")
    except Exception as e:  # noqa: BLE001
        chk.add("scenario ran to the end", False, f"{type(e).__name__}: {e}")
    finally:
        clean = srv.stop()
        chk.add("server stopped, no process left in its group", clean, "")
    if not supervised:
        # a fresh restart (no fault in the environment) serves a normal request
        srv2 = Server(args, cdir, port, None, False, label + "-fresh")
        srv2.start()
        try:
            srv2.wait_ready()
            st, models = http_json(port, "GET", "/v1/models")
            model = models["data"][0]["id"] if st == 200 else "model"
            normal_after_restart(chk, port, model, Prompts(port, model), time.time(), result, fresh=True)
        except Exception as e:  # noqa: BLE001
            chk.add("fresh restart serves a normal request", False, f"{type(e).__name__}: {e}")
        finally:
            chk.add("fresh server stopped, no process left", srv2.stop(), "")
    result["checks"] = chk.rows
    result["pass"] = chk.all_ok()
    (cdir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def normal_after_restart(chk, port, model, prompts, t0, result, fresh=False):
    tag = "fresh" if fresh else "restart"
    a = Client(f"{tag}-plain", port, prompts.nonce(tag) + "Count from one to ten.", 24, False, model, t0)
    a.run()
    b = Client(f"{tag}-stream", port, prompts.nonce(tag) + "Count from one to ten.", 24, True, model, t0)
    b.run()
    result.setdefault("after_restart", []).extend([a.summary(), b.summary()])
    chk.add(f"{tag}: a normal request succeeds (plain 200, 24 tokens, finish_reason length)",
            a.status == 200 and a.outcome == "done" and a.tokens == 24 and a.finish_reason == "length", str(a.summary()))
    chk.add(f"{tag}: a normal streamed request succeeds ([DONE], 24 tokens)",
            b.status == 200 and b.outcome == "done" and b.tokens > 0 and b.saw_done_marker, f"tokens={b.tokens} {b.outcome}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="intarsia binary built with --features metal,fault-inject")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-8B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="err,panic")
    ap.add_argument("--cases", default="nosup,sup", help="nosup: --no-supervisor (stays up refusing); sup: default supervisor (exit 87, restart)")
    ap.add_argument("--port", type=int, default=8117)
    args = ap.parse_args()
    if not os.access(args.bin, os.X_OK):
        print(f"no binary {args.bin}", file=sys.stderr)
        return 2
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in args.modes.split(","):
        for case in args.cases.split(","):
            log(f"=== case {mode}/{case} ===")
            results.append(run_case(args, out, mode, case == "sup", args.port))
    (out / "summary.json").write_text(json.dumps([{"case": r["case"], "pass": r["pass"],
                                                   "failed_checks": [c["check"] for c in r["checks"] if not c["ok"]]} for r in results], indent=1))
    log("=== summary ===")
    for r in results:
        log(f"{r['case']}: {'PASS' if r['pass'] else 'FAIL'} ({sum(c['ok'] for c in r['checks'])}/{len(r['checks'])} checks)")
        for c in r["checks"]:
            if not c["ok"]:
                log(f"    FAIL {c['check']}  [{c['detail']}]")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

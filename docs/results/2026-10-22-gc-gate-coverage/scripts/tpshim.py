#!/usr/bin/env python3
"""gc lane: `qualify --bin` shim that runs the 35B tp=2 stage across the two Sparks.

qualify runs `intarsia <sub> ... --tp 2` on one box and cannot pass `--tp-remote`. For `accuracy` / `serve` / `generate` with
`--tp 2` this shim (on spark1, rank 0) starts a matching `intarsia tp-worker` on spark2 (same model flags, which the
handshake checks; `--serving` for serve), adds `--tp-remote`, runs rank 0, then stops the worker. GC_WORKER_ENV
(e.g. the scale:1.01 test-bug variable) is set on the worker (rank 1) only. Everything else is passed through unchanged."""
import os, re, subprocess, sys, shlex, time

BIN = os.path.expanduser("~/intarsia-tp2/bin/gc-intarsia")
WIP, WPORT = "10.100.152.3", "29555"
NCCL = ("INTARSIA_NCCL_LIB=$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 "
        "NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN")
args = sys.argv[1:]
sub = args[0] if args else ""
tp2 = any(a == "--tp" and i + 1 < len(args) and args[i + 1] == "2" for i, a in enumerate(args)) or "--tp=2" in args
env = dict(os.environ, INTARSIA_NCCL_LIB=os.path.expanduser("~/intarsia-nccl/libnccl.so.2"), NCCL_SOCKET_IFNAME="enp1s0f1np1",
           NCCL_IB_HCA="rocep1s0f1,roceP2p1s0f1", NCCL_IB_GID_INDEX="3", NCCL_DEBUG="WARN")
if not (tp2 and sub in ("accuracy", "serve", "generate")):
    os.execve(BIN, [BIN] + args, env)

help_txt = subprocess.run([BIN, "tp-worker", "--help"], capture_output=True, text=True).stdout
opts = {}  # long option -> takes a value
for m in re.finditer(r"^\s+(?:-\w, )?(--[\w-]+)( <[^>]+>)?", help_txt, re.M):
    opts[m.group(1)] = bool(m.group(2))
keep, i = [], 1
while i < len(args):
    a = args[i]
    name = a.split("=", 1)[0]
    if name in opts and name not in ("--listen", "--tp-remote", "--help"):
        keep.append(a)
        if opts[name] and "=" not in a and i + 1 < len(args):
            i += 1
            keep.append(args[i])
    elif name.startswith("--") and name not in opts:
        # unknown to the worker: a rank-0-only option; skip its value when it has one (next token not an option)
        if "=" not in a and i + 1 < len(args) and not args[i + 1].startswith("--"):
            i += 1
    i += 1
wenv = os.environ.get("GC_WORKER_ENV", "")
serving = "--serving" if sub == "serve" else ""
wcmd = f"{wenv} {NCCL} {BIN} tp-worker --listen {WIP}:{WPORT} {serving} " + " ".join(shlex.quote(k) for k in keep)
sys.stderr.write(f"[tpshim] worker: ssh spark2 {wcmd}\n")
wlog = os.path.expanduser(f"~/gc/worker-{int(time.time())}.log")
job = os.environ.get("FLEETQ_JOB", "")
w = subprocess.Popen(["ssh", "-n", "-o", "BatchMode=yes", "spark2", f"export FLEETQ_JOB={job}; {wcmd} 2>&1"],
                     stdout=open(wlog, "w"), stderr=subprocess.STDOUT)
r = subprocess.run([BIN] + args + ["--tp-remote", f"{WIP}:{WPORT}"], env=env)
try:
    w.wait(timeout=60)
except subprocess.TimeoutExpired:
    subprocess.run(["ssh", "-n", "spark2", "pkill -TERM -f '[t]p-worker --listen'"])
    time.sleep(3)
    w.kill()
sys.stderr.write(f"[tpshim] rank 0 exit {r.returncode}; worker exit {w.returncode}; worker log {wlog}\n")
sys.exit(r.returncode)

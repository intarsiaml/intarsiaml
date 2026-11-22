"""Label a COMMIT_EVERY=1 cb trace (one dispatch per command buffer, pipelined) with the op sequence of a decode step from a
kernel-mode trace of the same prompt. usage: prof_cb1.py KERNEL_TRACE CB1_TRACE STEPS
Aligns from the end: the last STEPS * L records of the cb trace are STEPS decode steps of L dispatches each."""
import sys, json, collections
k = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
c = sorted((json.loads(l) for l in open(sys.argv[2]) if l.strip()), key=lambda r: r["s"])
steps = int(sys.argv[3])
starts = [i for i, r in enumerate(k) if r["op"].startswith("embedding")]
step = k[starts[-1]:]  # the last decode step's labelled sequence
names = []
for r in step:
    op = r["op"]
    if op.startswith("logits"):
        op += " (unembed)" if r["n"] > 100000 else " (router)"
    names += [op.split("#")[0]] * r["nd"]
L = len(names)
assert all(r["nd"] == 1 for r in c[-steps * L:]), "cb trace must have one dispatch per command buffer"
win = c[-steps * L:]
agg = collections.defaultdict(lambda: [0.0, 0])
for i, r in enumerate(win):
    a = agg[names[i % L]]; a[0] += (r["e"] - r["s"]) * 1e6; a[1] += 1
busy = sum(a[0] for a in agg.values()) / steps
span = (win[-1]["e"] - win[0]["s"]) * 1e6 / steps
print(f"{L} dispatches/step, {steps} steps: GPU busy {busy:.0f} us/token, span {span:.0f} us/token (idle {100*(1-busy/span):.0f} %)")
for op, a in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print(f"| {op} | {a[0]/steps:.0f} | {100*a[0]/steps/busy:.1f} | {a[1]/steps:.0f} | {a[0]/a[1]:.1f} |")

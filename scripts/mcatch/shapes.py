#!/usr/bin/env python3
"""Lane mcatch: distribution of the MTP head's catch-up shapes per 2-stream iteration, from the server logs of lane mhead's ABBA and gate runs.

usage: shapes.py LOG...   (serve logs, .log or .log.gz, run with INTARSIA_CB_TRACE=1)

A drafting step runs, per sequence, a catch-up pass of the head over the n rows its verify pass left queued: n = emits of that
sequence's previous verify pass (accepted + 1; `state.rs`: `pending_rows = keep - pos` after the rollback). The Emit line of an
iteration lists `(sequence, emitted)`, so the catch-up shape of iteration i + 1 is the emitted counts of iteration i. Counted: iterations that
drafted for 2 sequences (`verify_seqs: 2`) right after an iteration that emitted for the same 2 sequences. Everything else (a request's first
draft after its prefill, iterations with 1 sequence, a sequence joining or leaving) is not counted, and is reported as `skipped`.
Prints the shape table and the ceiling of batching the all-n = 1 case (the only bitwise-batchable catch-up shape)."""
import collections, gzip, re, sys

E = re.compile(r"Emit \{ iter: (\d+), tokens: \[(.*?)\], prepare_us: (\d+), verify_us: (\d+), verify_seqs: (\d+)")
tot = collections.Counter(); prep = collections.defaultdict(list); skipped = 0; iters = 0
for f in sys.argv[1:]:
    L = (gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")).read().splitlines()
    em = []
    for l in L:
        m = E.search(l)
        if m:
            em.append((int(m[1]), dict((int(a), int(b)) for a, b in re.findall(r"\((\d+), (\d+)\)", m[2])), int(m[3]), int(m[5])))
    c = collections.Counter(); n2 = 0
    for (i0, t0, _, _), (i1, t1, p1, vs1) in zip(em, em[1:]):
        if vs1 != 2:
            continue
        n2 += 1
        if len(t1) == 2 and len(t0) == 2 and all(s in t0 for s in t1):
            sh = tuple(sorted(t0[s] for s in t1)); c[sh] += 1; prep[sh].append(p1)
        else:
            skipped += 1
    iters += n2; tot += c
    n = sum(c.values())
    print(f"{f}: {n2} two-sequence iterations, {n} counted; all n=1: {c[(1, 1)]} ({100 * c[(1, 1)] / max(n, 1):.2f}%)")
n = sum(tot.values())
print(f"\nall logs: {iters} two-sequence iterations, {n} counted, {skipped} skipped")
print("shape (n of the two catch-ups) | iterations | share | mean prepare_us (draft time of the iteration; confounded, see doc)")
for k, v in sorted(tot.items()):
    print(f"  {k}  {v:6d}  {100 * v / n:6.2f}%  {sum(prep[k]) / len(prep[k]):.0f}")
p11 = tot[(1, 1)] / n; anyone = sum(v for k, v in tot.items() if 1 in k) / n; multi = sum(v for k, v in tot.items() if 1 not in k) / n
print(f"\nall n=1: {100 * p11:.2f}%   any n=1: {100 * anyone:.2f}%   no n=1 (2-3 rows each, picks moe_experts_bf16): {100 * multi:.2f}%")
# ceiling: share of iterations x ms saved on one all-n=1 iteration
for name, ms in [("mhead's measured saving of one batched 2-row pass (0.39 ms)", 0.39),
                 ("a whole one-row head pass vanishing (mtp2s nsys 1.13 ms)", 1.13),
                 ("the whole draft vanishing (mean prepare, absurd bound)", sum(sum(v) for v in prep.values()) / sum(len(v) for v in prep.values()) / 1000)]:
    print(f"ceiling, saving = {name}: {p11 * ms:.3f} ms per iteration")

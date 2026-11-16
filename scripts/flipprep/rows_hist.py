#!/usr/bin/env python3
"""Rows per batched verify pass from a server log with INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1: between two `DecodePhase` lines, rows = sum(drafted + 1) of the
`spec: token ... drafted N` lines (one per verified sequence; sequences that did not draft take a plain row, not counted). Prints the histogram of rows
(>= 2 drafted sequences => the batched pass) and how many passes had 1 or 2 verified sequences."""
import re, sys, collections
rows, seqs, hist, nseq = 0, 0, collections.Counter(), collections.Counter()
def flush():
    global rows, seqs
    if seqs:
        hist[rows] += 1; nseq[seqs] += 1
    rows = seqs = 0
for line in open(sys.argv[1], errors="replace"):
    if "DecodePhase" in line:
        flush()
    m = re.search(r"spec: token \d+ pos \d+ drafted (\d+) accepted", line)
    if m:
        rows += int(m.group(1)) + 1; seqs += 1
flush()
print("verify-pass rows histogram:", dict(sorted(hist.items())))
print("verified sequences per pass:", dict(sorted(nseq.items())))

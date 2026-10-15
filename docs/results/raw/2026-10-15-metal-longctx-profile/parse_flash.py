#!/usr/bin/env python3
"""Parse the gated `micro` outputs (flash_bench / attn_bw) into one markdown row per invocation, with the invocation's median GPU clock.
usage: parse_flash.py <file> [more files]"""
import re, sys
print("| invocation | variant / policy | pos0 or ctx | us/call (median) | TFLOPS or GB/s | median MHz | % samples < 1450 | W | max C |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
for f in sys.argv[1:]:
    txt = open(f).read()
    for blk in re.split(r"(?m)^(?=## )", txt):
        m = re.match(r"## (\S+) attempt (\d+) clock\(med p10 %<1450 W C n\)=\[(.*?)\] clockok=(\d)", blk)
        if not m: continue
        lab, att, ck = m.group(1), m.group(2), m.group(3).split()
        for l in blk.splitlines():
            r = re.match(r"\s*(\d+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+\((\d+)%)?", l)
            if r and (r.group(2).startswith("attn_") or r.group(2) in ("nax", "flash", "old")):
                if r.group(2) in ("attn_decode_bf16",): continue
                ck2 = ck + ["", "", "", "", "", ""]
                print(f"| {lab} a{att} | {r.group(2)} | {r.group(1)} | {float(r.group(3)):.0f} | {r.group(6)}{' GB/s' if 'decode' in r.group(2) else ''} | {ck2[0]} | {ck2[2]} | {ck2[3]} | {ck2[4]} |")

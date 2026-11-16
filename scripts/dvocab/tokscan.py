#!/usr/bin/env python3
"""dvocab: where the scripts live in a BPE vocabulary's id space (the draft head is the first N ids, so this says what it can draft).
usage: tokscan.py tokenizer.json   (byte-level BPE: token strings are GPT-2 byte-to-unicode encoded)"""
import collections, json, statistics, sys

v = json.load(open(sys.argv[1]))["model"]["vocab"]
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
inv = {}; part = collections.Counter()
for s, i in v.items():
    try: inv[bytes(u2b[c] for c in s).decode("utf-8")] = i
    except Exception: part[i // 16384] += 1
B = 16384
rows = collections.defaultdict(collections.Counter)
for d, i in inv.items():
    k = "cjk" if any("一" <= c <= "鿿" for c in d) else "ascii" if all(ord(c) < 128 for c in d) else "other"
    rows[i // B][k] += 1
print(f"vocab {len(v)} tokens; draft head = ids [0, N)")
print(f"{'ids':>15} {'ascii':>6} {'cjk':>6} {'other':>6} {'partial-utf8':>12}")
for b in sorted(rows):
    r = rows[b]; print(f"{b*B:>7}-{(b+1)*B-1:>7} {r['ascii']:>6} {r['cjk']:>6} {r['other']:>6} {part[b]:>12}")
cjk = sorted(i for d, i in inv.items() if any("一" <= c <= "鿿" for c in d))
print(f"first cjk id {cjk[0]}; last pure-ascii id below 98304: {max(i for d, i in inv.items() if i < 98304 and all(ord(c) < 128 for c in d))}")
print("cjk tokens below N:", {N: sum(1 for i in cjk if i < N) for N in (65536, 81920, 98304, 114688, 131072, 147456, 248320)}, "of", len(cjk))
common = "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工话儿者向情部正名定女问力机给等几很业最间新什写使"
ids = [inv[c] for c in common if c in inv]
print(f"top ~{len(ids)} common Chinese characters: ids {min(ids)}..{max(ids)}, median {statistics.median(ids):.0f}")
words = "我们 可以 这个 因为 所以 如果 什么 没有 已经 他们 但是 问题 使用 一个 需要 进行 函数 解释 例子 首先 其次 然后 最后 因此 例如 其中 通过 表示 返回 哈希".split()
print("common words:", " ".join(f"{w}={inv[w]}" for w in words if w in inv))

#!/usr/bin/env python3
"""Lane dfh: compare two tap dumps of tests/dflash2_real.rs (u32 taps, rows, hidden, then f32 [taps][rows][hidden]):
    cmp_taps.py a.bin b.bin  -> rel L2 per tap; exit 1 if any tap's rel L2 exceeds 0.05 (a tap reading the wrong buffer is O(1))."""
import array, math, struct, sys

def load(p):
    b = open(p, "rb").read()
    n, rows, h = struct.unpack("<III", b[:12])
    a = array.array("f"); a.frombytes(b[12:])
    return n, rows, h, a

na, ra, ha, a = load(sys.argv[1])
nb, rb, hb, b = load(sys.argv[2])
assert (na, ra, ha) == (nb, rb, hb), "shapes differ"
bad = False
for t in range(na):
    s = slice(t * ra * ha, (t + 1) * ra * ha)
    x, y = a[s], b[s]
    num = math.fsum((u - v) ** 2 for u, v in zip(x, y)); den = math.fsum(v * v for v in y)
    rel = math.sqrt(num / den) if den > 0 else float("inf")
    print(f"tap {t}: rel l2 {rel:.3e} over {ra} rows")
    bad |= not rel <= 0.05
sys.exit(1 if bad else 0)

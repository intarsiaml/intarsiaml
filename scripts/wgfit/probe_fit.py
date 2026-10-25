#!/usr/bin/env python3
"""Lane wg: fit a scalar model to the raw dumps of scripts/wg/probe_wmma.cpp (one gfx12 fp8 WMMA instruction, 16 products + C).

    probe_fit.py selftest                      # the search recovers a known synthetic model (run this first)
    probe_fit.py fit  <probe-dir> [n_fit=100000]   # crafted cases first, then random; prints every parameter set that is bitwise exact
    probe_fit.py verify <probe-dir> <params>   # whole random + chain dump with one parameter set (the >= 1e6 case gate)

Model family (all arithmetic exact in Python ints):
  AL(width, drop, rnd, c_in): the 16 products and (c_in) C are aligned to the largest leading-bit exponent E, keeping `width` bits
      (positions E .. E-width+1); bits below are dropped with `drop` = 'tz' (toward zero on the magnitude) or 'floor' (two's complement);
      the integers are summed exactly; the result is rounded to f32 with `rnd` = 'rne' | 'tz'. With c_in = False, C is added after the
      16-term sum with an ordinary f32 add (rne).
  GR(g, width, drop, rnd): products in consecutive groups of g aligned-and-summed with `width`, the group sums and C chained by f32 RNE adds.
Subnormal f32 results keep IEEE gradual underflow (the first thing to check if only those cases fail).
"""
import struct, sys, itertools, random, math

def dec_e4m3(b):
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    nan = (b & 0x7f) == 0x7f
    if e == 0:
        mant, ex = m, -6 - 3
    else:
        mant, ex = 8 + m, e - 7 - 3
    return nan, (-1 if s else 1), mant, ex          # value = sign * mant * 2^ex

def dec_f32(u):
    s, e, m = u >> 31, (u >> 23) & 255, u & 0x7fffff
    if e == 255:
        return ('nan' if m else 'inf'), (-1 if s else 1), 0, 0
    if e == 0:
        return 'fin', (-1 if s else 1), m, -149
    return 'fin', (-1 if s else 1), (1 << 23) | m, e - 127 - 23

def to_f32_bits(sign, M, e2, rnd):
    """sign * M * 2^e2 (M >= 0 integer) -> f32 bit pattern with rounding rnd ('rne' | 'tz')."""
    if M == 0:
        return 0
    bl = M.bit_length()
    lsb = max(e2 + bl - 24, -149)
    sh = lsb - e2
    if sh > 0:
        q, r = divmod(M, 1 << sh)
        if rnd == 'rne':
            half = 1 << (sh - 1)
            if r > half or (r == half and (q & 1)):
                q += 1
    else:
        q = M << (-sh)
    if q == 0:
        return (1 << 31) if sign < 0 else 0
    # q * 2^lsb ; q may be 2^24 after rounding
    if q >= (1 << 24):
        q >>= 1; lsb += 1
    if lsb > 104:
        return ((1 << 31) if sign < 0 else 0) | 0x7f800000
    if q < (1 << 23):                                # subnormal (lsb == -149)
        bits = q
    else:
        bits = ((lsb + 23 + 127) << 23) | (q & 0x7fffff)
    return ((1 << 31) if sign < 0 else 0) | bits

def f32_val(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]

def terms_of(a, b, c):
    ts, nan = [], False
    for k in range(16):
        n1, s1, m1, e1 = dec_e4m3(a[k]); n2, s2, m2, e2 = dec_e4m3(b[k])
        if n1 or n2:
            nan = True
        if m1 and m2:
            ts.append((s1 * s2, m1 * m2, e1 + e2))
    kind, sc, mc, ec = dec_f32(c)
    if kind != 'fin':
        return ts, None, kind
    return ts, ((sc, mc, ec) if mc else None), ('nan' if nan else 'ok')

def align_sum(ts, width, drop):
    """ts: [(sign, mant, exp)]; returns (sign, M, e2) of the aligned integer sum (exact value = sign*M*2^e2)."""
    ts = [t for t in ts if t[1]]
    if not ts:
        return 1, 0, 0
    E = max(t[2] + t[1].bit_length() - 1 for t in ts)
    L = E - width + 1
    S = 0
    for s, m, e in ts:
        sh = e - L
        v = (m << sh) if sh >= 0 else (m >> -sh)
        if sh < 0 and drop == 'floor' and s < 0 and (m & ((1 << -sh) - 1)):
            v += 1                                   # two's complement floor of a negative value moves away from zero
        S += s * v
    return (-1 if S < 0 else 1), abs(S), L

def model(case, params):
    a, b, c = case
    kind = params[0]
    ts, ct, status = terms_of(a, b, c)
    if status in ('nan', 'inf'):
        return None                                  # specials are compared by class, not bits, in this fit
    if kind == 'AL':
        _, width, drop, rnd, c_in = params
        if c_in:
            s, M, e2 = align_sum(ts + ([ct] if ct else []), width, drop)
            return to_f32_bits(s, M, e2, rnd)
        s, M, e2 = align_sum(ts, width, drop)
        p = to_f32_bits(s, M, e2, rnd)
        return f32_add_rne(p, c)
    if kind == 'GR':
        _, g, width, drop, rnd = params
        acc = c
        for i in range(0, 16, g):
            sub = [t for k, t in enumerate(ts_pos(a, b)) if i <= k < i + g and t]
            s, M, e2 = align_sum(sub, width, drop)
            acc = f32_add_rne(acc, to_f32_bits(s, M, e2, rnd))
        return acc
    raise ValueError(kind)

def ts_pos(a, b):
    out = []
    for k in range(16):
        n1, s1, m1, e1 = dec_e4m3(a[k]); n2, s2, m2, e2 = dec_e4m3(b[k])
        out.append((s1 * s2, m1 * m2, e1 + e2) if (m1 and m2) else None)
    return out

def f32_add_rne(u, v):
    k1, s1, m1, e1 = dec_f32(u); k2, s2, m2, e2 = dec_f32(v)
    if k1 != 'fin' or k2 != 'fin':
        return None
    if m1 == 0 and m2 == 0:
        return u if u == v else 0
    e = min(e1 if m1 else 10**9, e2 if m2 else 10**9)
    S = s1 * (m1 << (e1 - e) if m1 else 0) + s2 * (m2 << (e2 - e) if m2 else 0)
    return to_f32_bits(-1 if S < 0 else 1, abs(S), e, 'rne')

# ---------------------------------------------------------------- IO
def load(path):
    d = open(path, 'rb').read()
    assert d[:8] in (b'WMMAPRB1',), d[:8]
    n = struct.unpack_from('<Q', d, 8)[0]
    out, off = [], 16
    for _ in range(n):
        a = d[off:off + 16]; b = d[off + 16:off + 32]
        c, o = struct.unpack_from('<II', d, off + 32)
        out.append(((a, b, c), o)); off += 40
    return out

def is_special(case, hw):
    ts, ct, status = terms_of(*case)
    return status in ('nan', 'inf')

def score(data, params, limit=None, maxbad=None):
    ok = bad = 0; first = None
    for case, hw in data[:limit]:
        if maxbad is not None and bad >= maxbad:
            break
        if is_special(case, hw):
            continue
        m = model(case, params)
        if m is None:
            continue
        if m == hw:
            ok += 1
        else:
            bad += 1
            if first is None:
                first = (case, hw, m)
    return ok, bad, first

def candidates():
    for width in range(18, 49):
        for drop in ('tz', 'floor'):
            for rnd in ('rne', 'tz'):
                yield ('AL', width, drop, rnd, True)
                yield ('AL', width, drop, rnd, False)
    for g in (2, 4, 8):
        for width in range(18, 49, 1):
            for drop in ('tz', 'floor'):
                for rnd in ('rne', 'tz'):
                    yield ('GR', g, width, drop, rnd)

def synthetic(params, n, seed=1):
    rnd = random.Random(seed); data = []
    for _ in range(n):
        a = bytes(rnd.choice([x for x in range(256) if (x & 0x7f) != 0x7f]) for _ in range(16))
        b = bytes(rnd.choice([x for x in range(256) if (x & 0x7f) != 0x7f]) for _ in range(16))
        c = rnd.getrandbits(32) & ~0x40000000 if rnd.random() < 0.5 else struct.unpack('<I', struct.pack('<f', rnd.uniform(-8, 8)))[0]
        if (c >> 23) & 255 == 255:
            c &= ~0x7f800000
        case = (a, b, c); m = model(case, params)
        if m is not None:
            data.append((case, m))
    return data

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        true = ('AL', 27, 'floor', 'tz', True)
        data = synthetic(true, 3000)
        found = [p for p in candidates() if score(data, p, maxbad=1)[1] == 0]
        print('selftest: true', true, 'recovered', found)
        assert true in found and len(found) <= 3, found
        print('selftest ok')
        return
    if cmd == 'fit':
        d = sys.argv[2]; nfit = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
        crafted = load(f'{d}/probe-crafted.bin'); rnd = load(f'{d}/probe-random.bin')
        print('crafted', len(crafted), 'random', len(rnd))
        best = []
        for p in candidates():
            ok, bad, first = score(crafted, p, maxbad=400)
            best.append((bad, p))
        best.sort(key=lambda x: x[0])
        print('best on crafted cases (bad count, params):')
        for bad, p in best[:8]:
            print(' ', bad, p)
        for bad, p in best[:8]:
            ok, bad2, first = score(rnd, p, nfit)
            print('random', nfit, p, 'ok', ok, 'bad', bad2)
            if first and bad2:
                case, hw, m = first
                print('   first miss: a', case[0].hex(), 'b', case[1].hex(), 'c', hex(case[2]), 'hw', hex(hw), f32_val(hw), 'model', hex(m) if m is not None else None)
        return
    if cmd == 'verify':
        d = sys.argv[2]; params = eval(sys.argv[3])
        rnd = load(f'{d}/probe-random.bin')
        ok, bad, first = score(rnd, params)
        spec = sum(1 for c, h in rnd if is_special(c, h))
        print('random', len(rnd), 'specials skipped', spec, 'ok', ok, 'bad', bad)
        sub_in = sum(1 for c, h in rnd if (c[2] & 0x7f800000) == 0 and (c[2] & 0x7fffff))
        sub_out = sum(1 for c, h in rnd if (h & 0x7f800000) == 0 and (h & 0x7fffff))
        print('cases with subnormal C', sub_in, 'subnormal hw outputs', sub_out)
        if first:
            case, hw, m = first
            print('first miss: a', case[0].hex(), 'b', case[1].hex(), 'c', hex(case[2]), 'hw', hex(hw), 'model', hex(m) if m is not None else None)
        print('BITWISE EXACT' if bad == 0 and ok >= 10**6 else 'NOT EXACT (or fewer than 1e6 cases)')

if __name__ == '__main__':
    main()

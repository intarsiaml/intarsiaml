import re, collections, sys, os
D = sys.argv[1]
d = collections.defaultdict(dict)
for f in ['bench3-qo.txt', 'bench3-down.txt', 'pair3.txt']:
    for l in open(os.path.join(D, f)):
        m = re.match(r'RESULT shape=(\S+) M=(\d+) N=(\d+) K=(\d+) cand=(\S+) swz=\d+ median_tflops=([\d.]+)', l)
        if m:
            d[(m[1], int(m[2]))][m[5].replace('gemm_nax_t', '')] = float(m[6])
        m = re.match(r'PAIR shape=(\S+) M=(\d+) N=(\d+) K=(\d+) tile=(\S+) mode=pair median_tflops=([\d.]+)', l)
        if m:
            d[({'gate_up': 'gu', 'kv': 'kv'}[m[1]], int(m[2]))][m[5]] = float(m[6])
shapes = {'qo': (4096, 4096, 2), 'kv': (2048, 4096, 1), 'gu': (24576, 4096, 1), 'down': (4096, 12288, 1)}


def shipped(m, n):
    if 128 <= m <= 512:
        if m * n < 700_000: return '32x32w1'
        if 0 < m % 64 <= 32 and n < 8192: return '32x128w2'
        return '64x64w2'
    return cur(m, n)


def ruleA(m, n):
    if m < 128: return '64x64w4'
    if m * n < 700_000: return '32x32w1'
    if 0 < m % 64 <= 32: return '32x128w2'
    if m >= 1024 and n >= 8192: return '256x64w8'
    return '64x64w2'


def ruleB(m, n):
    if m < 128: return '64x64w4'
    if m * n < 700_000: return '32x32w1'
    if 0 < m % 64 <= 32: return '32x128w2'
    return '64x64w2'


def cur(m, n):
    return '128x64w4' if m >= 128 and -(-m // 128) * (-(-n // 64)) >= 192 else '64x64w4'


def tot(policy, Ms):
    t = 0
    for M in Ms:
        for s, (n, k, c) in shapes.items():
            v = d.get((s, M))
            if not v:
                return None
            name = policy(M, n) if policy else max(v, key=v.get)
            tf = v.get(name)
            if tf is None:
                tf = 0.001
            t += c * 2 * M * n * k / (tf * 1e12)
    return t * 1e3


for label, Ms in [('c256 4k', [256] * 15 + [222]), ('c256 1k', [256] * 3 + [205]), ('c2048 4k', [2048, 2014]), ('c2048 1k', [973]), ('c512 4k', [512] * 7 + [478])]:
    r = [tot(p, Ms) for p in (cur, shipped, ruleA, None)]
    if all(x is not None for x in r):
        print(label, 'P1 rule %.1f ms  shipped %.1f  (variant: tails at any N, 256x64 wide) %.1f  oracle %.1f  (x36 layers)' % tuple(x * 36 for x in r))
    else:
        print(label, 'missing data')

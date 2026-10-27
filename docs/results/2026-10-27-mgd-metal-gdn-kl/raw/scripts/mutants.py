import sys
m=sys.argv[1]; p='kernels/metal/gdn.metal'; s=open(p).read()
R={
 'm1':("const float qr = (1.0f / precise::sqrt(qq + p.eps)) * scale;\n    const float kr = 1.0f / precise::sqrt(kk + p.eps);\n    device float* dst","const float qr = (1.0f / precise::sqrt(qq + p.eps));\n    const float kr = 1.0f / precise::sqrt(kk + p.eps);\n    device float* dst"),
 'm2':("const float cdecay = sgt[tt][1], cbeta = sgt[tt][0];","const float cdecay = sgt[tt][0], cbeta = sgt[tt][1];"),
 'm3':("        const uint n = min(TILE, p.t - t0);\n        for (uint tt = 0; tt < n; ++tt) {\n            const float cdecay","        const uint n = TILE;\n        for (uint tt = 0; tt < n; ++tt) {\n            const float cdecay"),
 'm4':("vp[((T0) + tt) * C + (idx % GDN_PRE_ROWS)]","vp[((T0) + tt) * C + ((idx % GDN_PRE_ROWS) ^ 1)]"),
 'm5':("    const uint kh = h / (p.hv / p.hk);\n    const ulong C = 2ul * p.hk * DK + (ulong)p.hv * p.dv;\n    device const float* qk = scratch","    const uint kh = h % p.hk;\n    const ulong C = 2ul * p.hk * DK + (ulong)p.hv * p.dv;\n    device const float* qk = scratch"),
 'm6':("        for (uint i = 0; i < PL; ++i) srow[lane + 32 * i] = s[r][i];\n    }\n}","        if (r + 1 < RPS) for (uint i = 0; i < PL; ++i) srow[lane + 32 * i] = s[r][i];\n    }\n}"),
 'm7':("rg = tid < 2 * TILE && tid / 2 < fn ? gates[((T0) + tid / 2) * gs + (tid & 1)] : 0.0f;","rg = tid < 2 * TILE && tid / 2 < fn ? gates[((T0) + tid / 2) * gs + (tid & 1)] : 0.0f; if (tid == 5 && (T0) > 0) rg *= 1.0001f;"),
}
a,b=R[m]
assert s.count(a)==1,(m,s.count(a))
open(p,'w').write(s.replace(a,b))

// Inkling-2-Mini sublayer-boundary ops (lane ink-cuda-mixmoe): SconvRes (69), AttnResMix (67), ConvMix (74), LoopBlend (68).
// Contract: crates/intarsia-core/src/inkling2/ops.rs; reference: intarsia-backend-cpu ink_ref.rs. bf16 activations.
//
// The row kernel (one block per row, the same code for every row count, so rows-exact and chunk-invariant): the static conv
// with its residual epilogue, then the depth mix of the next sublayer (sources, queries, softmax over the sources), then the
// norm, the f32 mix, or the controller pair of the blend. Thread `tid` owns elements tid + j * blockDim (j < NE); every
// per-row sum is a per-thread partial in element order, a butterfly over the warp, then the warp partials in warp order, so
// the bits depend on D only. Block size and NE depend on D only.
//
// The ring: the conv reads ring rows of positions < pos0 and the call writes the rows of its last Rr positions. When
// T + K - 1 <= Rr (every decode and verify call: Rr = K - 1 + 16) the rows read and written are disjoint and the row kernel
// writes its own row; otherwise a second launch writes them after the reads.
//
// The blend (spec a12): the row kernel writes the pair rA([d w(d), m w(m)]) and the dest d (f32), then
//   h      = rA(pair . down^T)       the tensor-core GEMV of ink_proj.cu, the streams as its segments
//   gate   y = (1 - tanh(fg)) d + tanh(gn) m with gn / fg = gain / forget . h (one thread per channel, 16-row chunks), and the
//          slice partials of sum(y^2)
//   norm   rA(y inv(y) norm) with the partials in slice order
// in chunks of kBlendRows rows (the scratch holds one chunk).
#include "ik_launch.h"
#include "ik_cuda_abi_ink.h"
#include "ink_common.cuh"

namespace {

constexpr float kSrcEps = 1e-6f;
constexpr int kMaxThreads = 768;
constexpr int kNsMax = 8;    // stored sources + last
constexpr int kQMax = 4;     // queries
constexpr int kSMax = 3;     // blend streams
constexpr int kGMax = 128;   // controller rank
constexpr int kDMax = 6 * kMaxThreads;
constexpr int kNvMax = kNsMax + kQMax * kNsMax;  // values of the widest reduction
constexpr int kBlendRows = 256;                  // rows per blend chunk
constexpr int kGateCh = 128;                     // channels per gate block (one thread each)
constexpr int kGateRows = 16;                    // rows per gate block

IK_DEVI float bsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// red: [32][kNvMax] warp partials; fin: [kNvMax]. Every thread must call with the same nv. Adds value i of the warp into slot i.
IK_DEVI void warp_put(float* red, int i, float v) {
    v = bsum(v);
    if ((threadIdx.x & 31) == 0) red[(threadIdx.x >> 5) * kNvMax + i] = v;
}
IK_DEVI void block_fin(const float* red, float* fin, int nv) {
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    if ((int)threadIdx.x < nv) {
        float s = 0.f;
        for (int w = 0; w < nw; ++w) s += red[w * kNvMax + threadIdx.x];
        fin[threadIdx.x] = s;
    }
    __syncthreads();
}

struct RowArgs {
    // conv (x null: no conv)
    const ik_bf16* x;
    const ik_bf16* taps;
    const ik_bf16* res;
    const ik_bf16* add;
    ik_bf16* ring;
    ik_bf16* y;
    ik_bf16* sum;
    int K, Rr;
    int ring_now;
    // AttnResMix: last [R,D] and its addend (used when x is null)
    const ik_bf16* last;
    const ik_bf16* last_add;
    // mix
    const ik_bf16* src;  // [N,T,D]
    int N;
    const float* q;  // [Q,D]
    int Q;
    const ik_bf16* norm;  // [Q,D] (no blend) or [S,D] (unused by the row kernel with a blend)
    ik_bf16* out_b;       // [Q,T,D] with norm
    float* out_f;         // [Q,T,D] without
    // blend: dest from the mix (dest null) or the LoopBlend input, msg f32 [S,T,D]
    const float* dest;  // [Sd,T,D]
    int dest_bcast;     // Sd == 1
    const float* msg;
    int S;
    ik_bf16* pair;  // scratch [S,Tc,2D]
    float* dbuf;    // scratch [S,Tc,D]: the dest of each stream (f32)
    // geometry
    int T, D, t0;  // rows of the whole call, width, first row of this launch (blend chunks)
    int64_t pos0;
    const int* pos;
    float eps;
};

template <int NE>
__global__ void __launch_bounds__(kMaxThreads) ink_row_kernel(RowArgs a) {
    __shared__ float red[32 * kNvMax];
    __shared__ float fin[kNvMax];
    const int t = a.t0 + blockIdx.x;
    const int tl = blockIdx.x;  // row in the blend chunk
    const int D = a.D;
    const int64_t T = a.T;
    const int nt = blockDim.x;
    int ix[NE];
    bool ok[NE];
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        ix[j] = threadIdx.x + j * nt;
        ok[j] = ix[j] < D;
        if (!ok[j]) ix[j] = 0;
    }
    float last[NE];
    if (a.x) {
        const int64_t pos0 = a.pos ? (int64_t)*a.pos : a.pos0;
        const int64_t p = pos0 + t;
        const float sc = rsqrtf((float)(a.K + 1));
        float yv[NE];
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const int c = ix[j];
            float acc = 0.f;
            for (int l = 0; l < a.K; ++l) {
                const int64_t qp = p - l;
                float xv = 0.f;
                if (qp >= pos0) xv = ik_b2f(a.x[(qp - pos0) * D + c]);
                else if (qp >= 0) xv = ik_b2f(a.ring[(qp % a.Rr) * D + c]);
                acc = fmaf(ik_b2f(a.taps[(int64_t)l * D + c]), xv, acc);
            }
            yv[j] = ink_rbf((acc + ik_b2f(a.x[(int64_t)t * D + c])) * sc);
        }
        // y, the ring row of this position (no row of the launch reads it), the sum
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            if (!ok[j]) continue;
            const int c = ix[j];
            const int64_t o = (int64_t)t * D + c;
            a.y[o] = ik_f2b(yv[j]);
            if (a.ring_now && p >= pos0 + T - a.Rr) a.ring[(p % a.Rr) * D + c] = a.x[o];
            if (a.sum) {
                const float ad = a.add ? ik_b2f(a.add[o]) : 0.f;
                const float s = ink_rbf((ik_b2f(a.res[o]) + ad) + yv[j]);
                a.sum[o] = ik_f2b(s);
                last[j] = s;
            } else if (a.res) {
                last[j] = ik_b2f(a.res[o]) + yv[j];
            } else {
                last[j] = yv[j];
            }
        }
    } else if (a.last) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const int64_t o = (int64_t)t * D + ix[j];
            last[j] = ok[j] ? ik_b2f(a.last[o]) + (a.last_add ? ik_b2f(a.last_add[o]) : 0.f) : 0.f;
        }
    }
    float mix[kQMax][NE];
    const int Q = a.Q;
    if (Q > 0 && !a.dest) {
        // ---- depth mix: sources src[0..N) and last ----
        const int NS = a.N + 1;
        float xs[kNsMax][NE];
#pragma unroll
        for (int n = 0; n < kNsMax; ++n) {
#pragma unroll
            for (int j = 0; j < NE; ++j) {
                float v = 0.f;
                if (n < a.N) v = ok[j] ? ik_b2f(a.src[((int64_t)n * T + t) * D + ix[j]]) : 0.f;
                else if (n == a.N) v = ok[j] ? last[j] : 0.f;
                xs[n][j] = v;
            }
        }
#pragma unroll
        for (int n = 0; n < kNsMax; ++n) {
            if (n >= NS) break;
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < NE; ++j) s = fmaf(xs[n][j], xs[n][j], s);
            warp_put(red, n, s);
        }
#pragma unroll
        for (int qi = 0; qi < kQMax; ++qi) {
            if (qi >= Q) break;
            float qv[NE];
#pragma unroll
            for (int j = 0; j < NE; ++j) qv[j] = ok[j] ? a.q[(int64_t)qi * D + ix[j]] : 0.f;
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                if (n >= NS) break;
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < NE; ++j) s = fmaf(qv[j], xs[n][j], s);
                warp_put(red, kNsMax + qi * kNsMax + n, s);
            }
        }
        block_fin(red, fin, kNvMax);
        float wn[kNsMax];
#pragma unroll
        for (int n = 0; n < kNsMax; ++n) wn[n] = n < NS ? 1.f / sqrtf(fin[n] / (float)D + kSrcEps) : 0.f;
#pragma unroll
        for (int qi = 0; qi < kQMax; ++qi) {
            if (qi >= Q) break;
            float l[kNsMax], mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                l[n] = n < NS ? fin[kNsMax + qi * kNsMax + n] * wn[n] : -INFINITY;
                mx = fmaxf(mx, l[n]);
            }
            float e[kNsMax], se = 0.f;
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                e[n] = n < NS ? expf(l[n] - mx) : 0.f;
                se += e[n];
            }
#pragma unroll
            for (int j = 0; j < NE; ++j) mix[qi][j] = 0.f;
#pragma unroll
            for (int n = 0; n < kNsMax; ++n) {
                if (n >= NS) break;
                const float cf = e[n] / se * wn[n];
#pragma unroll
                for (int j = 0; j < NE; ++j) mix[qi][j] = fmaf(cf, xs[n][j], mix[qi][j]);
            }
        }
    }
    if (Q > 0 && !a.msg) {
        if (a.norm) {
#pragma unroll
            for (int qi = 0; qi < kQMax; ++qi) {
                if (qi >= Q) break;
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < NE; ++j) s = fmaf(mix[qi][j], mix[qi][j], s);
                warp_put(red, qi, s);
            }
            block_fin(red, fin, Q);
#pragma unroll
            for (int qi = 0; qi < kQMax; ++qi) {
                if (qi >= Q) break;
                const float r = 1.f / sqrtf(fin[qi] / (float)D + a.eps);
#pragma unroll
                for (int j = 0; j < NE; ++j)
                    if (ok[j]) a.out_b[((int64_t)qi * T + t) * D + ix[j]] = ik_f2b(mix[qi][j] * r * ik_b2f(a.norm[(int64_t)qi * D + ix[j]]));
            }
        } else {
#pragma unroll
            for (int qi = 0; qi < kQMax; ++qi) {
                if (qi >= Q) break;
#pragma unroll
                for (int j = 0; j < NE; ++j)
                    if (ok[j]) a.out_f[((int64_t)qi * T + t) * D + ix[j]] = mix[qi][j];
            }
        }
        return;
    }
    if (!a.msg) return;
    // ---- blend: the pair of every stream ----
    const int S = a.S;
    float dv[kSMax][NE], mv[kSMax][NE];
#pragma unroll
    for (int s = 0; s < kSMax; ++s) {
        if (s >= S) break;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            float d;
            if (a.dest) d = ok[j] ? a.dest[((int64_t)(a.dest_bcast ? 0 : s) * T + t) * D + ix[j]] : 0.f;
            else d = Q == 1 ? mix[0][j] : mix[s < kQMax ? s : 0][j];
            dv[s][j] = ok[j] ? d : 0.f;
            mv[s][j] = ok[j] ? a.msg[((int64_t)s * T + t) * D + ix[j]] : 0.f;
        }
        float sd = 0.f, sm = 0.f;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            sd = fmaf(dv[s][j], dv[s][j], sd);
            sm = fmaf(mv[s][j], mv[s][j], sm);
        }
        warp_put(red, 2 * s, sd);
        warp_put(red, 2 * s + 1, sm);
    }
    block_fin(red, fin, 2 * S);
#pragma unroll
    for (int s = 0; s < kSMax; ++s) {
        if (s >= S) break;
        const float wd = 1.f / sqrtf(fin[2 * s] / (float)D + kSrcEps);
        const float wm = 1.f / sqrtf(fin[2 * s + 1] / (float)D + kSrcEps);
        ik_bf16* pr = a.pair + ((int64_t)s * gridDim.x + tl) * 2 * D;
        float* db = a.dbuf + ((int64_t)s * gridDim.x + tl) * D;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            if (!ok[j]) continue;
            pr[ix[j]] = ik_f2b(dv[s][j] * wd);
            pr[D + ix[j]] = ik_f2b(mv[s][j] * wm);
            db[ix[j]] = dv[s][j];
        }
    }
}

// the deferred ring write: positions max(pos0, pos0 + T - Rr) .. pos0 + T
__global__ void ink_ring_kernel(const ik_bf16* __restrict__ x, ik_bf16* __restrict__ ring, int T, int D, int Rr, int64_t pos0h,
                                const int* pos) {
    const int64_t pos0 = pos ? (int64_t)*pos : pos0h;
    const int64_t q0 = (T > Rr) ? pos0 + T - Rr : pos0;
    const int64_t q = q0 + blockIdx.y;
    if (q >= pos0 + T) return;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < D; c += gridDim.x * blockDim.x)
        ring[(q % Rr) * D + c] = x[(q - pos0) * D + c];
}

// gate: y = (1 - tanh(fg)) d + tanh(gn) m for one stream, 128 channels x 16 rows of a chunk; writes y and the block's slice
// partial of sum(y^2) per row
__global__ void __launch_bounds__(kGateCh) ink_blend_gate_kernel(const ik_bf16* __restrict__ h, const ik_bf16* __restrict__ gain,
                                                                 const ik_bf16* __restrict__ forget, const float* __restrict__ dbuf,
                                                                 const float* __restrict__ msg, float* __restrict__ ybuf,
                                                                 float* __restrict__ part, int Tc, int T, int t0, int D, int G) {
    __shared__ float hs[kGateRows][kGMax];
    __shared__ float red[kGateRows][kGateCh / 32];
    const int s = blockIdx.y;
    const int r0 = blockIdx.z * kGateRows;
    const int rows = min(kGateRows, Tc - r0);
    for (int i = threadIdx.x; i < rows * G; i += blockDim.x) hs[i / G][i % G] = ik_b2f(h[((int64_t)s * Tc + r0 + i / G) * G + i % G]);
    __syncthreads();
    const int c = blockIdx.x * kGateCh + threadIdx.x;
    const bool on = c < D;
    const int cc = on ? c : 0;
    float gn[kGateRows], fg[kGateRows];
#pragma unroll
    for (int r = 0; r < kGateRows; ++r) gn[r] = fg[r] = 0.f;
    const ik_bf16* gr = gain + ((int64_t)s * D + cc) * G;
    const ik_bf16* fr = forget + ((int64_t)s * D + cc) * G;
    for (int g0 = 0; g0 < G; g0 += 8) {
        const uint4 gu = __ldg(reinterpret_cast<const uint4*>(gr + g0));
        const uint4 fu = __ldg(reinterpret_cast<const uint4*>(fr + g0));
        float gw[8], fw[8];
        ik_unpack8(gu, gw);
        ik_unpack8(fu, fw);
#pragma unroll
        for (int r = 0; r < kGateRows; ++r) {
            if (r >= rows) break;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                gn[r] = fmaf(gw[e], hs[r][g0 + e], gn[r]);
                fg[r] = fmaf(fw[e], hs[r][g0 + e], fg[r]);
            }
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < kGateRows; ++r) {
        if (r >= rows) break;
        const int64_t o = ((int64_t)s * Tc + r0 + r) * D + cc;
        float yv = 0.f;
        if (on) {
            const float d = dbuf[o];
            const float m = msg[((int64_t)s * T + t0 + r0 + r) * D + cc];
            yv = (1.f - tanhf(fg[r])) * d + tanhf(gn[r]) * m;
            ybuf[o] = yv;
        }
        const float q = bsum(yv * yv);
        if (lane == 0) red[r][warp] = q;
    }
    __syncthreads();
    if ((int)threadIdx.x < rows) {
        float q = 0.f;
        for (int w = 0; w < kGateCh / 32; ++w) q += red[threadIdx.x][w];
        part[((int64_t)s * Tc + r0 + threadIdx.x) * gridDim.x + blockIdx.x] = q;
    }
}

// norm: out[s, t] = rA(y inv(y) norm[s]); the slice partials summed in slice order
__global__ void ink_blend_norm_kernel(const float* __restrict__ ybuf, const float* __restrict__ part, const ik_bf16* __restrict__ norm,
                                      ik_bf16* __restrict__ out, int Tc, int T, int t0, int D, int nsl, float eps) {
    __shared__ float r;
    const int tl = blockIdx.x, s = blockIdx.y;
    if (threadIdx.x == 0) {
        float q = 0.f;
        for (int i = 0; i < nsl; ++i) q += part[((int64_t)s * Tc + tl) * nsl + i];
        r = 1.f / sqrtf(q / (float)D + eps);
    }
    __syncthreads();
    const float* yr = ybuf + ((int64_t)s * Tc + tl) * D;
    ik_bf16* orow = out + ((int64_t)s * T + t0 + tl) * D;
    for (int c = threadIdx.x; c < D; c += blockDim.x) orow[c] = ik_f2b(yr[c] * r * ik_b2f(norm[(int64_t)s * D + c]));
}

int64_t rnd(int64_t v) { return (v + 255) & ~(int64_t)255; }

struct BlendScratch {
    ik_bf16* pair;
    float* dbuf;
    ik_bf16* h;
    float* ybuf;
    float* part;
};

int64_t blend_bytes(int64_t s, int64_t r, int64_t d, int64_t g) {
    const int64_t rc = r < kBlendRows ? r : kBlendRows;
    const int64_t nsl = (d + kGateCh - 1) / kGateCh;
    return rnd(s * rc * 2 * d * 2) + rnd(s * rc * d * 4) + rnd(s * rc * g * 2) + rnd(s * rc * d * 4) + rnd(s * rc * nsl * 4);
}

BlendScratch blend_carve(void* base, int64_t s, int64_t rc, int64_t d, int64_t g) {
    char* p = (char*)base;
    const int64_t nsl = (d + kGateCh - 1) / kGateCh;
    BlendScratch b;
    b.pair = (ik_bf16*)p;
    p += rnd(s * rc * 2 * d * 2);
    b.dbuf = (float*)p;
    p += rnd(s * rc * d * 4);
    b.h = (ik_bf16*)p;
    p += rnd(s * rc * g * 2);
    b.ybuf = (float*)p;
    p += rnd(s * rc * d * 4);
    b.part = (float*)p;
    return b;
}

int row_threads(int D, int* ne) {
    int e = (D + kMaxThreads - 1) / kMaxThreads;
    if (e < 1) e = 1;
    *ne = e;
    const int th = (D + e - 1) / e;
    return (th + 31) / 32 * 32;
}

int launch_rows(const RowArgs& a, int rows, cudaStream_t st) {
    if (rows <= 0) return IK_OK;
    int ne = 0;
    const int th = row_threads(a.D, &ne);
    switch (ne) {
        case 1: ink_row_kernel<1><<<rows, th, 0, st>>>(a); break;
        case 2: ink_row_kernel<2><<<rows, th, 0, st>>>(a); break;
        case 3: ink_row_kernel<3><<<rows, th, 0, st>>>(a); break;
        case 4: ink_row_kernel<4><<<rows, th, 0, st>>>(a); break;
        case 5: ink_row_kernel<5><<<rows, th, 0, st>>>(a); break;
        case 6: ink_row_kernel<6><<<rows, th, 0, st>>>(a); break;
        default: return IK_ERR_SHAPE;
    }
    return ik_launch_status();
}

const ik_bf16* bp(const ik_tensor* t) { return ink_absent(t) ? nullptr : (const ik_bf16*)t->data; }

int chk_opt(const ik_tensor* t, int dtype, int ndim) { return ink_absent(t) ? IK_OK : ik_check(t, dtype, ndim); }

// the blend over the chunks of `a` (row kernel with msg set), then gemv / gate / norm into out [S,T,D]
int run_blend(RowArgs a, const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, const ik_tensor* norm, ik_tensor* out,
              const ik_scratch* sc, cudaStream_t st) {
    const int S = a.S, T = a.T, D = a.D;
    const int G = (int)down->shape[1];
    if (S < 1 || S > kSMax || G < 1 || G > kGMax || G % 8 != 0 || (2 * D) % 16 != 0) return IK_ERR_SHAPE;
    if (!ik_aligned16(down->data) || !ik_aligned16(gain->data) || !ik_aligned16(forget->data)) return IK_ERR_INVALID_ARG;
    if (!sc || !sc->data || sc->bytes < blend_bytes(S, T, D, G)) return IK_ERR_INVALID_ARG;
    const int rcmax = T < kBlendRows ? T : kBlendRows;
    const BlendScratch b = blend_carve(sc->data, S, rcmax, D, G);
    const int nsl = (D + kGateCh - 1) / kGateCh;
    for (int t0 = 0; t0 < T; t0 += kBlendRows) {
        const int rc = T - t0 < kBlendRows ? T - t0 : kBlendRows;
        RowArgs r = a;
        r.t0 = t0;
        r.pair = b.pair;
        r.dbuf = b.dbuf;
        IK_TRY(launch_rows(r, rc, st));
        InkSegs sg{};
        sg.nseg = S;
        for (int s = 0; s < S; ++s)
            sg.s[s] = InkSeg{b.pair + (int64_t)s * rc * 2 * D, (const ik_bf16*)down->data + (int64_t)s * G * 2 * D, b.h + (int64_t)s * rc * G, G};
        IK_TRY(ink_mma_gemv(sg, rc, 2 * D, st));
        const dim3 gg((unsigned)nsl, (unsigned)S, (unsigned)((rc + kGateRows - 1) / kGateRows));
        ink_blend_gate_kernel<<<gg, kGateCh, 0, st>>>(b.h, (const ik_bf16*)gain->data, (const ik_bf16*)forget->data, b.dbuf, a.msg, b.ybuf,
                                                     b.part, rc, T, t0, D, G);
        IK_TRY(ik_launch_status());
        ink_blend_norm_kernel<<<dim3((unsigned)rc, (unsigned)S), 256, 0, st>>>(b.ybuf, b.part, (const ik_bf16*)norm->data,
                                                                                (ik_bf16*)out->data, rc, T, t0, D, nsl, a.eps);
        IK_TRY(ik_launch_status());
    }
    return IK_OK;
}

int check_blend_w(const ik_tensor* msg, const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, const ik_tensor* norm,
                  int64_t T, int64_t D) {
    IK_TRY(ik_check(msg, IK_DTYPE_F32, 3));
    IK_TRY(ik_check(down, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(gain, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(forget, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(norm, IK_DTYPE_BF16, 2));
    const int64_t S = msg->shape[0], G = down->shape[1];
    if (msg->shape[1] != T || msg->shape[2] != D || down->shape[0] != S || down->shape[2] != 2 * D || gain->shape[0] != S ||
        gain->shape[1] != D || gain->shape[2] != G || forget->shape[0] != S || forget->shape[1] != D || forget->shape[2] != G ||
        norm->shape[0] != S || norm->shape[1] != D)
        return IK_ERR_SHAPE;
    return IK_OK;
}

// the conv part's checks and args (SconvRes / ConvMix)
int conv_args(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend, int64_t pos0, const ik_tensor* pos,
              ik_tensor* y, ik_tensor* ring, ik_tensor* sum, RowArgs* a) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(taps, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(y, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(ring, IK_DTYPE_BF16, 2));
    IK_TRY(chk_opt(res, IK_DTYPE_BF16, 2));
    IK_TRY(chk_opt(addend, IK_DTYPE_BF16, 2));
    IK_TRY(chk_opt(sum, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check_pos(pos));
    const int64_t T = x->shape[0], D = x->shape[1], K = taps->shape[0], Rr = ring->shape[0];
    if (taps->shape[1] != D || y->shape[0] != T || y->shape[1] != D || ring->shape[1] != D || Rr < K || K < 1) return IK_ERR_SHAPE;
    if (D < 1 || D > kDMax || T > 0x7fffffff) return IK_ERR_SHAPE;
    *a = RowArgs{};
    a->x = (const ik_bf16*)x->data;
    a->taps = (const ik_bf16*)taps->data;
    a->res = bp(res);
    a->add = bp(addend);
    a->ring = (ik_bf16*)ring->data;
    a->y = (ik_bf16*)y->data;
    a->sum = ink_absent(sum) ? nullptr : (ik_bf16*)sum->data;
    a->K = (int)K;
    a->Rr = (int)Rr;
    a->ring_now = T + K - 1 <= Rr;
    a->T = (int)T;
    a->D = (int)D;
    a->pos0 = pos0;
    a->pos = pos ? (const int*)pos->data : nullptr;
    return IK_OK;
}

int ring_after(const RowArgs& a, cudaStream_t st) {
    if (a.ring_now || a.T == 0) return IK_OK;
    const int rows = a.T < a.Rr ? a.T : a.Rr;
    ink_ring_kernel<<<dim3((unsigned)((a.D + 255) / 256), (unsigned)rows), 256, 0, st>>>(a.x, a.ring, a.T, a.D, a.Rr, a.pos0, a.pos);
    return ik_launch_status();
}

}  // namespace

extern "C" {

int64_t ik_ink_blend_scratch_bytes(int64_t s, int64_t r, int64_t d, int64_t g) { return blend_bytes(s, r, d, g); }

int ik_sconv_res_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend, int64_t pos0,
                      const ik_tensor* pos, ik_tensor* y, ik_tensor* ring, ik_tensor* sum, void* stream) {
    RowArgs a;
    IK_TRY(conv_args(x, taps, res, addend, pos0, pos, y, ring, sum, &a));
    if (a.T == 0) return IK_OK;
    cudaStream_t st = IK_STREAM(stream);
    IK_TRY(launch_rows(a, a.T, st));
    return ring_after(a, st);
}

int ik_attn_res_mix_bf16(const ik_tensor* src, const ik_tensor* last, const ik_tensor* addend, const ik_tensor* queries,
                         const ik_tensor* norm, float eps, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(last, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(queries, IK_DTYPE_F32, 2));
    IK_TRY(chk_opt(addend, IK_DTYPE_BF16, 2));
    IK_TRY(chk_opt(norm, IK_DTYPE_BF16, 2));
    const int64_t R = last->shape[0], D = last->shape[1], N = src->ndim == 3 ? src->shape[0] : -1, Q = queries->shape[0];
    if (N < 0 || N + 1 > kNsMax || Q < 1 || Q > kQMax || D < 1 || D > kDMax || queries->shape[1] != D) return IK_ERR_SHAPE;
    if (N > 0) {
        IK_TRY(ik_check(src, IK_DTYPE_BF16, 3));
        if (src->shape[1] != R || src->shape[2] != D) return IK_ERR_SHAPE;
    }
    IK_TRY(ik_check(out, ink_absent(norm) ? IK_DTYPE_F32 : IK_DTYPE_BF16, 3));
    if (R == 0) return IK_OK;
    RowArgs a{};
    a.last = (const ik_bf16*)last->data;
    a.last_add = bp(addend);
    a.src = N > 0 ? (const ik_bf16*)src->data : nullptr;
    a.N = (int)N;
    a.q = (const float*)queries->data;
    a.Q = (int)Q;
    a.norm = bp(norm);
    if (a.norm) a.out_b = (ik_bf16*)out->data;
    else a.out_f = (float*)out->data;
    a.T = (int)R;
    a.D = (int)D;
    a.eps = eps;
    return launch_rows(a, (int)R, IK_STREAM(stream));
}

int ik_conv_mix_bf16(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend,
                     const ik_tensor* src, const ik_tensor* queries, const ik_tensor* norm, const ik_tensor* msg,
                     const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, int64_t pos0, const ik_tensor* pos,
                     float eps, ik_tensor* y, ik_tensor* ring, ik_tensor* sum, ik_tensor* mix, const ik_scratch* scratch,
                     void* stream) {
    RowArgs a;
    IK_TRY(conv_args(x, taps, res, addend, pos0, pos, y, ring, sum, &a));
    a.eps = eps;
    const int64_t T = a.T, D = a.D;
    const int64_t Q = queries->ndim == 2 ? queries->shape[0] : -1;
    if (Q < 0 || Q > kQMax) return IK_ERR_SHAPE;
    cudaStream_t st = IK_STREAM(stream);
    if (Q > 0) {
        IK_TRY(ik_check(queries, IK_DTYPE_F32, 2));
        const int64_t N = src->ndim == 3 ? src->shape[0] : -1;
        if (N < 0 || N + 1 > kNsMax || queries->shape[1] != D) return IK_ERR_SHAPE;
        if (N > 0) {
            IK_TRY(ik_check(src, IK_DTYPE_BF16, 3));
            if (src->shape[1] != T || src->shape[2] != D) return IK_ERR_SHAPE;
        }
        a.src = N > 0 ? (const ik_bf16*)src->data : nullptr;
        a.N = (int)N;
        a.q = (const float*)queries->data;
        a.Q = (int)Q;
    }
    if (T == 0) return IK_OK;
    if (Q > 0 && !ink_absent(msg)) {
        IK_TRY(check_blend_w(msg, down, gain, forget, norm, T, D));
        IK_TRY(ik_check(mix, IK_DTYPE_BF16, 3));
        const int64_t S = msg->shape[0];
        if (!(Q == 1 || Q == S) || mix->shape[0] != S) return IK_ERR_SHAPE;
        a.msg = (const float*)msg->data;
        a.S = (int)S;
        // every chunk's row kernel reads the ring before the deferred write (in-kernel writes only when T + K - 1 <= Rr: one chunk)
        IK_TRY(run_blend(a, down, gain, forget, norm, mix, scratch, st));
        return ring_after(a, st);
    }
    if (Q > 0) {
        IK_TRY(chk_opt(norm, IK_DTYPE_BF16, 2));
        IK_TRY(ik_check(mix, ink_absent(norm) ? IK_DTYPE_F32 : IK_DTYPE_BF16, 3));
        a.norm = bp(norm);
        if (a.norm) a.out_b = (ik_bf16*)mix->data;
        else a.out_f = (float*)mix->data;
    }
    IK_TRY(launch_rows(a, (int)T, st));
    return ring_after(a, st);
}

int ik_loop_blend_bf16(const ik_tensor* dest, const ik_tensor* msg, const ik_tensor* down, const ik_tensor* gain,
                       const ik_tensor* forget, const ik_tensor* norm, float eps, ik_tensor* out, const ik_scratch* scratch,
                       void* stream) {
    IK_TRY(ik_check(dest, IK_DTYPE_F32, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t S = out->shape[0], R = out->shape[1], D = out->shape[2];
    if (D < 1 || D > kDMax || R > 0x7fffffff) return IK_ERR_SHAPE;
    IK_TRY(check_blend_w(msg, down, gain, forget, norm, R, D));
    if (msg->shape[0] != S || !(dest->shape[0] == 1 || dest->shape[0] == S) || dest->shape[1] != R || dest->shape[2] != D) return IK_ERR_SHAPE;
    if (R == 0) return IK_OK;
    RowArgs a{};
    a.dest = (const float*)dest->data;
    a.dest_bcast = dest->shape[0] == 1;
    a.msg = (const float*)msg->data;
    a.S = (int)S;
    a.Q = 1;  // (the row kernel's blend path; no mix is computed with a dest)
    a.T = (int)R;
    a.D = (int)D;
    a.eps = eps;
    return run_blend(a, down, gain, forget, norm, out, scratch, IK_STREAM(stream));
}

}  // extern "C"

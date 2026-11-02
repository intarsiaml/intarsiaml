// EXL3 routed experts (Op::MoeExperts in its EXL3 encoding, crates/intarsia-core/src/quant/exl3_moe.rs): the launcher
// ik_moe_experts_exl3 and its small kernels (grouping, input rotation, gate / up epilogue, down combine). The grouped
// GEMVs are in exl3_moe_cp.cu (decode windows) and exl3_moe_rows.cu (prompt chunks); provenance, license and the changes
// made to TensorFold's kernels: exl3_moe.cuh.
//
// A call runs in slices of rows (all T rows when T < kDecodeRows, else kPromptRows at a time), six launches a slice:
//   group     one block: the distinct routed experts of the slice's (row, slot) pairs in id order ("places"), each
//             place's members (row * 32 + slot, rows in order) as one list with starts and counts, and the prompt
//             kernel's work list of (place, 32-member group); ids outside 0..E belong to no place
//   rot_in    xg[p] = fp16(H (x[row] * suh_g[e]) / sqrt(128)) and xu[p] likewise, a warp per (pair, 128 block, matrix)
//   gate/up   grouped GEMV (decode: grouped_cp_kernel, SK K splits) or the prompt kernel (K splits folded) -> Z
//   epilogue  per pair and 128 columns: the splits summed in order, H, * svh / sqrt(128), SwiGLU with the clamp
//             (silu(min(g, L)) * clamp(u, -L, L), fp32), * suh_d, H, / sqrt(128) -> xd (fp16)
//   down      the same grouped kernels over xd -> Zd (one K split; Zd reuses xg / xu)
//   combine   per row and 128 columns: each slot's H(Zd) * svh_d / sqrt(128), then out = sum over slots in order of
//             fmaf(weight, y, acc) from 0, in the output dtype; a slot outside 0..E adds weight * 0
// Every row's arithmetic is fixed by (H, I, its experts' widths): K ranges by shape, warps and splits added in order, no
// atomics, so a row's bits depend on neither the batch nor the path (decode or prompt), and runs repeat bit for bit.
#include "exl3_moe.cuh"

namespace ik_exl3x {
namespace {

constexpr int kDecodeRows = 64;   // rows below which the decode kernel serves (TensorFold's EXACT_ROWS)
constexpr int kPromptRows = 2048;  // rows of a prompt slice (the scratch bound; DeepSeek-V4.1's served chunk is the same)
constexpr int kMaxE = 1024;       // experts a call (the grouping block's threads)
constexpr int kChunk = 4096;      // ids the grouping block stages at a time

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ float from_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ half from_f<half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

__global__ void __launch_bounds__(1024) group_kernel(const int* __restrict__ ids, int n, int K, int E,
                                                     int* __restrict__ uids, int* __restrict__ ucount,
                                                     int* __restrict__ mstart, int* __restrict__ mcount,
                                                     int* __restrict__ members, int* __restrict__ work, int wlb, int grp_rows) {
    __shared__ int chunk[kChunk];
    __shared__ int sa[kMaxE], sb[kMaxE], sc[kMaxE];
    const int e = threadIdx.x;
    int cnt = 0;
    for (int c0 = 0; c0 < n; c0 += kChunk) {
        const int len = min(kChunk, n - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += blockDim.x) chunk[i] = ids[c0 + i];
        __syncthreads();
        if (e < E)
            for (int i = 0; i < len; ++i) cnt += chunk[i] == e;
    }
    const int pres = e < E && cnt > 0, grp = (cnt + grp_rows - 1) / grp_rows;
    sa[e] = pres;
    sb[e] = cnt;
    sc[e] = grp;
    __syncthreads();
    for (int off = 1; off < kMaxE; off <<= 1) {  // inclusive scans over expert ids
        int a = 0, b = 0, c = 0;
        if (e >= off) {
            a = sa[e - off];
            b = sb[e - off];
            c = sc[e - off];
        }
        __syncthreads();
        if (e >= off) {
            sa[e] += a;
            sb[e] += b;
            sc[e] += c;
        }
        __syncthreads();
    }
    const int place = sa[e] - pres, start = sb[e] - cnt, wl0 = sc[e] - grp;
    if (pres) {
        uids[place] = e;
        mstart[place] = start;
        mcount[place] = cnt;
        for (int g = 0; g < grp; ++g) {
            work[2 * (wl0 + g)] = place;
            work[2 * (wl0 + g) + 1] = g;
        }
    }
    if (e == 0) ucount[0] = sa[kMaxE - 1];
    for (int i = sc[kMaxE - 1] + e; i < wlb; i += blockDim.x) {
        work[2 * i] = 0x7fffffff;  // past the list: no place
        work[2 * i + 1] = 0;
    }
    int pos = 0;
    for (int c0 = 0; c0 < n; c0 += kChunk) {
        const int len = min(kChunk, n - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += blockDim.x) chunk[i] = ids[c0 + i];
        __syncthreads();
        if (pres)
            for (int i = 0; i < len; ++i)
                if (chunk[i] == e) members[start + pos++] = ((c0 + i) / K) * 32 + (c0 + i) % K;
    }
}

// warp per (pair p, 128 block, matrix)
template <typename TIN>
__global__ void __launch_bounds__(256) rot_in_kernel(const TIN* __restrict__ x, const int* __restrict__ ids, Experts ex,
                                                     half* __restrict__ xg, half* __restrict__ xu, int P, int K) {
    const int H = ex.H, nblk = H / 128;
    const long long item = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= (long long)P * nblk * 2) return;
    const int p = (int)(item / (nblk * 2)), mat = (int)((item / nblk) % 2), blk = (int)(item % nblk);
    const int e = ids[p];
    if (e < 0 || e >= ex.E) return;
    const int lane = threadIdx.x & 31;
    const half* suh = (mat ? ex.suh_u(e) : ex.suh_g(e)) + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)(p / K) * H + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? xu : xg) + (size_t)p * H + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// grid (P, I / 128), a warp: gate / up partials Z [2, SKz, P, I] -> xd [P, I]
__global__ void __launch_bounds__(32) epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ ids,
                                                      Experts ex, const float* __restrict__ limit,
                                                      half* __restrict__ xd, int P, int SKz) {
    const int p = blockIdx.x, I = ex.I, lane = threadIdx.x;
    const int e = ids[p];
    if (e < 0 || e >= ex.E) return;
    const int n = blockIdx.y * 128 + 4 * lane;
    float g[4], u[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SKz; ++s) {
            sg += Z[((size_t)(0 * SKz + s) * P + p) * I + n + j];
            su += Z[((size_t)(1 * SKz + s) * P + p) * I + n + j];
        }
        g[j] = sg;
        u[j] = su;
    }
    fwht128(g, lane);
    fwht128(u, lane);
    const float L = *limit;
    const half *svg = ex.svh_g(e) + n, *svu = ex.svh_u(e) + n, *sud = ex.suh_d(e) + n;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gg = fminf(g[j] * HAD_SCALE * __half2float(svg[j]), L);
        const float uu = fminf(fmaxf(u[j] * HAD_SCALE * __half2float(svu[j]), -L), L);
        v[j] = gg / (1.f + expf(-gg)) * uu * __half2float(sud[j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * I + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// grid (rows, H / 128), K warps: down partials Zd [P, H] -> out [rows, H]
template <typename TOUT>
__global__ void __launch_bounds__(1024) combine_kernel(const float* __restrict__ Zd, const int* __restrict__ ids,
                                                       const float* __restrict__ wts, Experts ex,
                                                       TOUT* __restrict__ out, int K) {
    __shared__ float4 part[32][32];  // [slot][lane]
    const int H = ex.H, r = blockIdx.x;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.y * 128 + 4 * lane;
    const int p = r * K + k;
    const int e = ids[p];
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    if (e >= 0 && e < ex.E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = Zd[(size_t)p * H + n + j];
        fwht128(v, lane);
        const half* sv = ex.svh_d(e) + n;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(sv[j]);
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < K; ++q) {
        const float w = wts[r * K + q];
        const float4 y = part[q][lane];
        acc[0] = fmaf(w, y.x, acc[0]);
        acc[1] = fmaf(w, y.y, acc[1]);
        acc[2] = fmaf(w, y.z, acc[2]);
        acc[3] = fmaf(w, y.w, acc[3]);
    }
    TOUT* orow = out + (size_t)r * H + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) orow[j] = from_f<TOUT>(acc[j]);
}

// K splits of the gate / up projection: the most of 4, 2, 1 that divide its k tiles among 4 warps
int gateup_splits(int H) {
    for (int sk : {4, 2, 1})
        if ((H / 16) % (4 * sk) == 0) return sk;
    return 0;
}

size_t r256(size_t b) { return (b + 255) & ~(size_t)255; }

struct Layout {
    size_t ints, xg, xu, z, xd, total;
    int R, P, places, wlb, skz;
};

// The scratch of a call (T rows): crates/intarsia-backend-cuda/src/exl3.rs moe_scratch_bytes mirrors this.
Layout layout(long long T, int H, int I, int E, int K) {
    Layout l{};
    const bool decode = T < kDecodeRows;
    l.R = (int)(decode ? T : (T < kPromptRows ? T : kPromptRows));
    l.P = l.R * K;
    l.places = E < l.P ? E : l.P;
    l.wlb = l.P / 32 + l.places + 1;
    l.skz = decode ? gateup_splits(H) : 1;
    l.ints = r256((size_t)4 * (3 * (size_t)E + 1 + l.P + 2 * (size_t)l.wlb));
    l.xg = r256((size_t)l.P * H * 2);
    l.xu = l.xg;
    l.z = r256((size_t)2 * l.skz * l.P * I * 4);
    l.xd = r256((size_t)l.P * I * 2);
    l.total = l.ints + l.xg + l.xu + l.z + l.xd;
    return l;
}

bool is_act(int dt) { return dt == IK_DTYPE_BF16 || dt == IK_DTYPE_F16 || dt == IK_DTYPE_F32; }

template <typename F>
int by_dtype(int dt, F&& f) {
    switch (dt) {
        case IK_DTYPE_BF16: f((__nv_bfloat16*)nullptr); return IK_OK;
        case IK_DTYPE_F16: f((half*)nullptr); return IK_OK;
        case IK_DTYPE_F32: f((float*)nullptr); return IK_OK;
        default: return IK_ERR_DTYPE;
    }
}

}  // namespace
}  // namespace ik_exl3x

using namespace ik_exl3x;

extern "C" long long ik_moe_experts_exl3_scratch_bytes(long long T, long long H, long long I, long long E, long long K) {
    if (T <= 0) return 0;
    return (long long)layout(T, (int)H, (int)I, (int)E, (int)K).total;
}

extern "C" int ik_moe_experts_exl3(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* arena,
                                   const ik_tensor* scales, const ik_tensor* meta, const ik_tensor* cb,
                                   const ik_tensor* limit, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    if (!x || !out) return IK_ERR_INVALID_ARG;
    if (!is_act(x->dtype) || !is_act(out->dtype)) return IK_ERR_DTYPE;
    IK_TRY(ik_check_a(x, x->dtype, 2, 8));
    IK_TRY(ik_check_a(out, out->dtype, 2, 8));
    IK_TRY(ik_check_a(ids, IK_DTYPE_I32, 2, 4));
    IK_TRY(ik_check_a(wts, IK_DTYPE_F32, 2, 4));
    IK_TRY(ik_check_a(arena, IK_DTYPE_I16, 1, 16));
    IK_TRY(ik_check_a(scales, IK_DTYPE_F16, 2, 4));
    IK_TRY(ik_check_a(meta, IK_DTYPE_I32, 2, 4));
    IK_TRY(ik_check_a(cb, IK_DTYPE_I32, 1, 4));
    IK_TRY(ik_check_a(limit, IK_DTYPE_F32, 1, 4));
    const long long T = x->shape[0], H = x->shape[1], K = ids->shape[1], E = meta->shape[0], C = scales->shape[1];
    if (ids->shape[0] != T || wts->shape[0] != T || wts->shape[1] != K || out->shape[0] != T || out->shape[1] != H)
        return IK_ERR_SHAPE;
    if (meta->shape[1] != META_COLS || scales->shape[0] != E || C <= 3 * H || (C - 3 * H) % 3 != 0) return IK_ERR_SHAPE;
    const long long I = (C - 3 * H) / 3;
    if (H % 128 != 0 || I % 128 != 0 || E < 1 || E > kMaxE || K < 1 || K > 32 || cb->shape[0] != 1 || limit->shape[0] != 1)
        return IK_ERR_SHAPE;
    if (H > (1 << 20) || I > (1 << 20)) return IK_ERR_SHAPE;
    if (T == 0) return IK_OK;
    const int sk = gateup_splits((int)H);
    if (sk == 0) return IK_ERR_SHAPE;
    const Layout l = layout(T, (int)H, (int)I, (int)E, (int)K);
    if (!scratch || !scratch->data || scratch->bytes < (int64_t)l.total || !ik_aligned16(scratch->data))
        return IK_ERR_INVALID_ARG;
    cudaStream_t s = IK_STREAM(stream);
    char* base = (char*)scratch->data;
    int* ints = (int*)base;
    int *uids = ints, *ucount = uids + E, *mstart = ucount + 1, *mcount = mstart + E, *members = mcount + E;
    int* work = members + l.P;
    half* xg = (half*)(base + l.ints);
    half* xu = (half*)(base + l.ints + l.xg);
    float* z = (float*)(base + l.ints + l.xg + l.xu);
    half* xd = (half*)(base + l.ints + l.xg + l.xu + l.z);
    float* zd = (float*)xg;  // [P, H] f32 over xg and xu (dead after gate / up)

    Experts ex;
    ex.arena = (const uint32_t*)arena->data;
    ex.words = arena->shape[0] / 2;
    ex.sc = (const half*)scales->data;
    ex.C = (int)C;
    ex.meta = (const int*)meta->data;
    ex.cbp = (const int*)cb->data;
    ex.E = (int)E;
    ex.H = (int)H;
    ex.I = (int)I;
    const bool decode = T < kDecodeRows;
    const int xsize = x->dtype == IK_DTYPE_F32 ? 4 : 2, osize = out->dtype == IK_DTYPE_F32 ? 4 : 2;

    for (long long r0 = 0; r0 < T; r0 += l.R) {
        const int R = (int)(T - r0 < l.R ? T - r0 : l.R);
        const int P = R * (int)K;
        const int places = (int)(E < P ? E : P);
        const int wlb = P / 32 + places + 1;
        const int* idp = (const int*)ids->data + r0 * K;
        const float* wp = (const float*)wts->data + r0 * K;
        group_kernel<<<1, kMaxE, 0, s>>>(idp, P, (int)K, (int)E, uids, ucount, mstart, mcount, members, work, wlb,
                                         decode ? 32 : launch_rows_group_rows());
        IK_TRY(ik_launch_status());
        const char* xp = (const char*)x->data + r0 * H * xsize;
        const long long items = (long long)P * (H / 128) * 2;
        IK_TRY(by_dtype(x->dtype, [&](auto tag) {
            using TIN = std::remove_pointer_t<decltype(tag)>;
            rot_in_kernel<TIN><<<(unsigned)((items + 7) / 8), 256, 0, s>>>((const TIN*)xp, idp, ex, xg, xu, P, (int)K);
        }));
        IK_TRY(ik_launch_status());

        GroupedArgs a{};
        a.ex = ex;
        a.uids = uids;
        a.ucount = ucount;
        a.mstart = mstart;
        a.mcount = mcount;
        a.members = members;
        a.P = P;
        a.slots = (int)K;
        a.maxm = P;  // any place has at most every pair of the slice
        // gate / up
        a.x0 = xg;
        a.x1 = xu;
        a.j0 = 0;
        a.z = z;
        a.K = (int)H;
        a.N = (int)I;
        a.SK = sk;
        a.mats = 2;
        if (decode) {
            a.nexp = places;
            a.work = nullptr;
            IK_TRY(launch_cp(a, s));
        } else {
            a.nexp = wlb;
            a.work = work;
            IK_TRY(launch_rows(a, true, s));
        }
        epilogue_kernel<<<dim3((unsigned)P, (unsigned)(I / 128)), 32, 0, s>>>(z, idp, ex, (const float*)limit->data, xd, P,
                                                                             decode ? sk : 1);
        IK_TRY(ik_launch_status());
        // down
        a.x0 = xd;
        a.x1 = xd;
        a.j0 = 2;
        a.z = zd;
        a.K = (int)I;
        a.N = (int)H;
        a.SK = 1;
        a.mats = 1;
        IK_TRY(decode ? launch_cp(a, s) : launch_rows(a, false, s));
        char* op = (char*)out->data + r0 * H * osize;
        IK_TRY(by_dtype(out->dtype, [&](auto tag) {
            using TOUT = std::remove_pointer_t<decltype(tag)>;
            combine_kernel<TOUT><<<dim3((unsigned)R, (unsigned)(H / 128)), 32 * (unsigned)K, 0, s>>>(zd, idp, wp, ex,
                                                                                                    (TOUT*)op, (int)K);
        }));
        IK_TRY(ik_launch_status());
    }
    return IK_OK;
}

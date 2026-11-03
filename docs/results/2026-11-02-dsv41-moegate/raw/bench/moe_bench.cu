// standalone microbench of the prompt-chunk gate/up kernel: the repo's grouped_rows_kernel (baseline) against variants (bitwise check)
// build: nvcc -O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=sm_121 -I kernels moe_bench.cu -o moe_bench   (from the repo root)
#include "cuda/exl3_moe.cu"
#include "cuda/exl3_moe_rows.cu"
namespace ik_exl3x {
int launch_cp(const GroupedArgs&, cudaStream_t) { return IK_ERR_SHAPE; }
int launch_dense_cp(const half*, const uint32_t*, float*, const int*, int, int, int, int, int, cudaStream_t) { return IK_ERR_SHAPE; }
}  // namespace ik_exl3x
#include <vector>
#include <algorithm>
#include <random>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <functional>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

namespace ik_exl3x {
namespace v2 {

constexpr int W = 4;  // K quarters (fixed by the numerics)

// work list with `gsz` members per group (group_kernel's, generalized)
__global__ void __launch_bounds__(1024) group_kernel2(const int* __restrict__ ids, int n, int K, int E, int* __restrict__ uids,
                                                      int* __restrict__ ucount, int* __restrict__ mstart, int* __restrict__ mcount,
                                                      int* __restrict__ members, int* __restrict__ work, int wlb, int gsz) {
    __shared__ int chunk[4096];
    __shared__ int sa[1024], sb[1024], sc[1024];
    const int e = threadIdx.x;
    int cnt = 0;
    for (int c0 = 0; c0 < n; c0 += 4096) {
        const int len = min(4096, n - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += blockDim.x) chunk[i] = ids[c0 + i];
        __syncthreads();
        if (e < E) for (int i = 0; i < len; ++i) cnt += chunk[i] == e;
    }
    const int pres = e < E && cnt > 0, grp = (cnt + gsz - 1) / gsz;
    sa[e] = pres; sb[e] = cnt; sc[e] = grp;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        int a = 0, b = 0, c = 0;
        if (e >= off) { a = sa[e - off]; b = sb[e - off]; c = sc[e - off]; }
        __syncthreads();
        if (e >= off) { sa[e] += a; sb[e] += b; sc[e] += c; }
        __syncthreads();
    }
    const int place = sa[e] - pres, start = sb[e] - cnt, wl0 = sc[e] - grp;
    if (pres) {
        uids[place] = e; mstart[place] = start; mcount[place] = cnt;
        for (int g = 0; g < grp; ++g) { work[2 * (wl0 + g)] = place; work[2 * (wl0 + g) + 1] = g; }
    }
    if (e == 0) ucount[0] = sa[1023];
    for (int i = sc[1023] + e; i < wlb; i += blockDim.x) { work[2 * i] = 0x7fffffff; work[2 * i + 1] = 0; }
    int pos = 0;
    for (int c0 = 0; c0 < n; c0 += 4096) {
        const int len = min(4096, n - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += blockDim.x) chunk[i] = ids[c0 + i];
        __syncthreads();
        if (pres) for (int i = 0; i < len; ++i) if (chunk[i] == e) members[start + pos++] = ((c0 + i) / K) * 32 + (c0 + i) % K;
    }
}

// the widths of the synthetic experts only (compile time): mul1 codebook, k2 4 / 6 / 8 / 10
template <class F>
__device__ __forceinline__ bool with_width_sub(int cb, int k2, F&& f) {
    if (cb != 2) return false;
    switch (k2) {
        case 4: f(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{}); return true;
        case 6: f(std::integral_constant<int, 2>{}, std::integral_constant<int, 6>{}); return true;
        case 8: f(std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{}); return true;
        case 10: f(std::integral_constant<int, 2>{}, std::integral_constant<int, 10>{}); return true;
        default: return false;
    }
}

template <int CB, int K2, int G, int NTW, int PF, int KO>
__device__ __forceinline__ void warp_tiles2(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0, const half* __restrict__ X,
                                            const long long (&off)[G][2], int live, int lane, float (&acc)[G][NTW][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;
    uint32_t pf[PF][NTW][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NTW; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);
    const half* xr[G][2];
    bool ok[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < 2; ++j) { ok[g][j] = off[g][j] >= 0; xr[g][j] = X + (ok[g][j] ? off[g][j] : 0) + kt0 * 16; }
    uint32_t an[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g)
        if (g < live) {
            if (KO & 1) { an[g][0] = an[g][1] = an[g][2] = an[g][3] = 0x3c003c00u; }
            else {
                an[g][0] = load_pair(xr[g][0], ok[g][0]); an[g][1] = load_pair(xr[g][1], ok[g][1]);
                an[g][2] = load_pair(xr[g][0] + 8, ok[g][0]); an[g][3] = load_pair(xr[g][1] + 8, ok[g][1]);
            }
        }
    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NTW][LW];
#pragma unroll
                for (int i = 0; i < NTW; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NTW; ++i) load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                uint32_t a[G][4];
#pragma unroll
                for (int g = 0; g < G; ++g)
#pragma unroll
                    for (int c = 0; c < 4; ++c) a[g][c] = an[g][c];
                if (it + 1 < nkt && !(KO & 1)) {
                    const int k = (it + 1) * 16;
#pragma unroll
                    for (int g = 0; g < G; ++g)
                        if (g < live) {
                            an[g][0] = load_pair(xr[g][0] + k, ok[g][0]); an[g][1] = load_pair(xr[g][1] + k, ok[g][1]);
                            an[g][2] = load_pair(xr[g][0] + k + 8, ok[g][0]); an[g][3] = load_pair(xr[g][1] + k + 8, ok[g][1]);
                        }
                }
#pragma unroll
                for (int i = 0; i < NTW; ++i) {
                    uint32_t b0[2], b1[2];
                    if (KO & 2) { b0[0] = w[i][0]; b0[1] = w[i][0] ^ 0x1111u; b1[0] = w[i][0] ^ 0x2222u; b1[1] = w[i][0] ^ 0x3333u; }
                    else decode_tile<CB, K2>(w[i], map, lane, b0, b1);
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        if (g < live) {
                            if (KO & 4) { acc[g][i][0][0] += __uint_as_float(b0[0] ^ a[g][0]); acc[g][i][1][1] += __uint_as_float(b1[1] ^ a[g][3]); }
                            else { mma16816(acc[g][i][0], a[g], b0); mma16816(acc[g][i][1], a[g], b1); }
                        }
                    }
                }
            }
        }
    }
}

// G member tiles (16 rows each), NTW n tiles a warp, CS warps share a K quarter (a block: W * CS warps, NTW * CS n tiles = 16 * NTW * CS columns)
template <bool FOLD, int G, int NTW, int CS, int PF, int KO, int MINB>
__global__ void __launch_bounds__(W * CS * 32, MINB) grouped_rows2_kernel(GroupedArgs a) {
    constexpr int NTB = NTW * CS, GR = 16 * G, NTHR = W * CS * 32;
    constexpr int TPT = 16 * NTB * 16 / NTHR;
    const int K = a.K, N = a.N, P = a.P, SK = a.SK, slots = a.slots, nexp = a.nexp;
    const int zz = (int)(blockIdx.z % nexp);
    const int u = a.work[2 * zz];
    if (u >= a.ucount[0]) return;
    const int mg = a.work[2 * zz + 1];
    const int zi = (int)(blockIdx.z / nexp);
    const int splits = FOLD ? 1 : SK;
    const int split0 = FOLD ? 0 : zi % SK;
    const int mat = zi / splits;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.uids[u];
    const int cb = *a.ex.cbp;
    int k2 = 0;
    const uint32_t* T = mat_words(a.ex, e, a.j0 + mat, K, N, cb, k2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wq = warp % W, cs = warp / W;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    __shared__ int rows_sh[GR];
    for (int i = threadIdx.x; i < GR; i += NTHR) {
        const int m = mg * GR + i;
        const int code = m < a.mcount[u] ? a.members[a.mstart[u] + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;
    int live = 0;
#pragma unroll
    for (int g = 0; g < G; ++g) live += rows_sh[g * 16] >= 0;
    long long off[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int r0 = rows_sh[g * 16 + g8], r1 = rows_sh[g * 16 + g8 + 8];
        off[g][0] = r0 < 0 ? -1 : (long long)r0 * K + 2 * t;
        off[g][1] = r1 < 0 ? -1 : (long long)r1 * K + 2 * t;
    }
    const int per_split = KT / SK, per_warp = per_split / W;
    const int nt0 = blockIdx.y * NTB + cs * NTW;
    __shared__ float red[W][16][NTB * 16];
    for (int split = split0; split < (FOLD ? SK : split0 + 1); ++split) {
        const int kt0 = split * per_split + wq * per_warp;
        float acc[G][NTW][2][4];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < NTW; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[g][i][h][c] = 0.f;
        const bool ran = T != nullptr && with_width_sub(cb, k2, [&](auto cbc, auto k2c) {
                             warp_tiles2<decltype(cbc)::value, decltype(k2c)::value, G, NTW, PF, KO>(T, NTILES, kt0, per_warp, nt0, X, off, live, lane, acc);
                         });
        if (!ran) {
#pragma unroll
            for (int g = 0; g < G; ++g)
#pragma unroll
                for (int i = 0; i < NTW; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) acc[g][i][h][c] = __int_as_float(0x7fc00000);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < live) {
#pragma unroll
                for (int i = 0; i < NTW; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (cs * NTW + i) * 16 + h * 8 + 2 * t;
                        red[wq][g8][col] = acc[g][i][h][0];
                        red[wq][g8][col + 1] = acc[g][i][h][1];
                        red[wq][g8 + 8][col] = acc[g][i][h][2];
                        red[wq][g8 + 8][col + 1] = acc[g][i][h][3];
                    }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < TPT; ++j) {
                    const int idx = threadIdx.x + j * NTHR;
                    const int row = idx / (NTB * 16), col = idx % (NTB * 16);
                    float s = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) s += red[w][row][col];
                    const int r = rows_sh[g * 16 + row];
                    if (r >= 0) {
                        if constexpr (FOLD) {
                            float* zp = a.z + ((size_t)mat * P + r) * N + blockIdx.y * NTB * 16 + col;
                            *zp = (split == 0 ? 0.f : *zp) + s;
                        } else {
                            a.z[(((size_t)mat * SK + split) * P + r) * N + blockIdx.y * NTB * 16 + col] = s;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

template <bool FOLD, int G, int NTW, int CS, int PF, int KO, int MINB = 1>
int launch2(const GroupedArgs& a, cudaStream_t s) {
    constexpr int NTB = NTW * CS;
    if (a.N % (16 * NTB) != 0 || (a.K / 16) % (W * a.SK) != 0 || a.nexp <= 0 || a.work == nullptr) return IK_ERR_SHAPE;
    const long long gz = (long long)a.mats * (FOLD ? 1 : a.SK) * a.nexp;
    if (gz > 65535) return IK_ERR_SHAPE;
    const dim3 grid(1, (unsigned)(a.N / (16 * NTB)), (unsigned)gz);
    grouped_rows2_kernel<FOLD, G, NTW, CS, PF, KO, MINB><<<grid, W * CS * 32, 0, s>>>(a);
    return ik_launch_status();
}

// ---- v3: A (the activation rows) staged through shared memory with cp.async, fragments by ldmatrix (TensorFold's path) ----
constexpr int KB = 4;            // k tiles a stage holds
constexpr int ASTR = KB * 16 + 8;  // halves per smem row (padded: 144 B, conflict-free ldmatrix)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool ok) {
    const int sz = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

template <bool FOLD, int G, int NTW, int CS, int PF, int ST, int MINB>
__global__ void __launch_bounds__(W * CS * 32, MINB) grouped_rows3_kernel(GroupedArgs a) {
    constexpr int NTB = NTW * CS, GR = 16 * G, NTHR = W * CS * 32;
    constexpr int TPT = 16 * NTB * 16 / NTHR;
    const int K = a.K, N = a.N, P = a.P, SK = a.SK, slots = a.slots, nexp = a.nexp;
    const int zz = (int)(blockIdx.z % nexp);
    const int u = a.work[2 * zz];
    if (u >= a.ucount[0]) return;
    const int mg = a.work[2 * zz + 1];
    const int zi = (int)(blockIdx.z / nexp);
    const int splits = FOLD ? 1 : SK;
    const int split0 = FOLD ? 0 : zi % SK;
    const int mat = zi / splits;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.uids[u];
    const int cb = *a.ex.cbp;
    int k2 = 0;
    const uint32_t* T = mat_words(a.ex, e, a.j0 + mat, K, N, cb, k2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wq = warp % W, cs = warp / W;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    __shared__ int rows_sh[GR];
    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* Asm = reinterpret_cast<half*>(smem_raw);  // [ST][W][GR][ASTR]
    float* red = reinterpret_cast<float*>(smem_raw);  // reused after the k loop: [W][16][NTB*16]
    for (int i = threadIdx.x; i < GR; i += NTHR) {
        const int m = mg * GR + i;
        const int code = m < a.mcount[u] ? a.members[a.mstart[u] + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;
    int live = 0;
#pragma unroll
    for (int g = 0; g < G; ++g) live += rows_sh[g * 16] >= 0;
    const int per_split = KT / SK, per_warp = per_split / W;
    const int nchunk = per_warp / KB;
    const int nt0 = blockIdx.y * NTB + cs * NTW;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;
    for (int split = split0; split < (FOLD ? SK : split0 + 1); ++split) {
        float acc[G][NTW][2][4];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < NTW; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[g][i][h][c] = 0.f;
        const int kbase = split * per_split;  // k tile of warp quarter wq: kbase + wq * per_warp
        auto issue = [&](int chunk) {
            const int stage = chunk % ST;
            constexpr int SEG = GR * (KB * 2);  // 16-byte segments a quarter's stage holds
            constexpr int PER = W * SEG / NTHR;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int idx = threadIdx.x + j * NTHR;
                const int q = idx / SEG, rem = idx % SEG;
                const int row = rem / (KB * 2), seg = rem % (KB * 2);
                const int r = rows_sh[row];
                const half* src = X + (r >= 0 ? (size_t)r * K : 0) + (size_t)(kbase + q * per_warp + chunk * KB) * 16 + seg * 8;
                cp_async16(smem_u32(Asm + (((size_t)stage * W + q) * GR + row) * ASTR + seg * 8), src, r >= 0);
            }
        };
        const bool okT = T != nullptr;
        // prefetch ST-1 stages
#pragma unroll
        for (int s0 = 0; s0 < ST - 1; ++s0) { if (s0 < nchunk) issue(s0); cp_commit(); }
        const bool ran = okT && with_width_sub(cb, k2, [&](auto cbc, auto k2c) {
            constexpr int CBc = decltype(cbc)::value, K2c = decltype(k2c)::value;
            constexpr int TW = Fmt<K2c>::TW, LW = Fmt<K2c>::LW;
            const LaneMap<K2c> map(lane);
            const size_t kstride = (size_t)NTILES * TW;
            const uint32_t* tp = T + ((size_t)(kbase + wq * per_warp) * NTILES + nt0) * TW + lane;
            const int nkt = per_warp;
            uint32_t pf[PF][NTW][LW];
#pragma unroll
            for (int d = 0; d < PF; ++d)
                if (d < nkt)
#pragma unroll
                    for (int i = 0; i < NTW; ++i) load_words<K2c>(pf[d][i], tp + d * kstride + i * TW, lane);
            for (int chunk = 0; chunk < nchunk; ++chunk) {
                cp_wait<ST - 2>();
                __syncthreads();
                if (chunk + ST - 1 < nchunk) issue(chunk + ST - 1);
                cp_commit();
                const half* As = Asm + (((size_t)(chunk % ST) * W + wq) * GR) * ASTR;
#pragma unroll
                for (int kk = 0; kk < KB; ++kk) {
                    // PF slots rotate over the unrolled kk (KB % PF == 0)
                    constexpr int dummy = 0; (void)dummy;
                    const int it = chunk * KB + kk;
                    uint32_t w[NTW][LW];
#pragma unroll
                    for (int i = 0; i < NTW; ++i)
#pragma unroll
                        for (int l = 0; l < LW; ++l) w[i][l] = pf[kk % PF][i][l];
                    if (it + PF < nkt)
#pragma unroll
                        for (int i = 0; i < NTW; ++i) load_words<K2c>(pf[kk % PF][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                    uint32_t af[G][4];
#pragma unroll
                    for (int g = 0; g < G; ++g) ldsm4(af[g], smem_u32(As + (size_t)(g * 16 + lrow) * ASTR + kk * 16 + lcol));
#pragma unroll
                    for (int i = 0; i < NTW; ++i) {
                        uint32_t b0[2], b1[2];
                        decode_tile<CBc, K2c>(w[i], map, lane, b0, b1);
#pragma unroll
                        for (int g = 0; g < G; ++g)
                            if (g < live) { mma16816(acc[g][i][0], af[g], b0); mma16816(acc[g][i][1], af[g], b1); }
                    }
                }
            }
        });
        cp_wait<0>();
        __syncthreads();
        if (!ran) {
#pragma unroll
            for (int g = 0; g < G; ++g)
#pragma unroll
                for (int i = 0; i < NTW; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) acc[g][i][h][c] = __int_as_float(0x7fc00000);
        }
        float (*redp)[16][NTB * 16] = reinterpret_cast<float (*)[16][NTB * 16]>(red);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < live) {
#pragma unroll
                for (int i = 0; i < NTW; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (cs * NTW + i) * 16 + h * 8 + 2 * t;
                        redp[wq][g8][col] = acc[g][i][h][0];
                        redp[wq][g8][col + 1] = acc[g][i][h][1];
                        redp[wq][g8 + 8][col] = acc[g][i][h][2];
                        redp[wq][g8 + 8][col + 1] = acc[g][i][h][3];
                    }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < TPT; ++j) {
                    const int idx = threadIdx.x + j * NTHR;
                    const int row = idx / (NTB * 16), col = idx % (NTB * 16);
                    float sm = redp[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) sm += redp[w][row][col];
                    const int r = rows_sh[g * 16 + row];
                    if (r >= 0) {
                        float* zp = a.z + ((size_t)mat * P + r) * N + blockIdx.y * NTB * 16 + col;
                        *zp = (split == 0 ? 0.f : *zp) + sm;
                    }
                }
                __syncthreads();
            }
        }
    }
}

template <int G, int NTW, int CS, int PF, int ST, int MINB = 1>
int launch3(const GroupedArgs& a, cudaStream_t s) {
    constexpr int NTB = NTW * CS, GR = 16 * G;
    static_assert(KB % PF == 0, "PF must divide KB");
    if (a.N % (16 * NTB) != 0 || (a.K / 16) % (W * a.SK) != 0 || (((a.K / 16) / a.SK) / W) % KB != 0 || a.nexp <= 0 || a.work == nullptr) return IK_ERR_SHAPE;
    const long long gz = (long long)a.mats * a.nexp;
    if (gz > 65535) return IK_ERR_SHAPE;
    constexpr size_t sm_a = (size_t)ST * W * GR * ASTR * sizeof(half), sm_r = (size_t)W * 16 * NTB * 16 * sizeof(float);
    constexpr size_t sm = sm_a > sm_r ? sm_a : sm_r;
    static bool set = false;
    if (!set) { cudaFuncSetAttribute(grouped_rows3_kernel<true, G, NTW, CS, PF, ST, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); set = true; }
    const dim3 grid(1, (unsigned)(a.N / (16 * NTB)), (unsigned)gz);
    grouped_rows3_kernel<true, G, NTW, CS, PF, ST, MINB><<<grid, W * CS * 32, sm, s>>>(a);
    return ik_launch_status();
}
}  // namespace v2
}  // namespace ik_exl3x

using namespace ik_exl3x;

static uint64_t g_s = 0x9E3779B97F4A7C15ull;
static uint64_t rnd64() { g_s ^= g_s << 13; g_s ^= g_s >> 7; g_s ^= g_s << 17; return g_s; }
static double rndu() { return (rnd64() >> 11) * (1.0 / 9007199254740992.0); }
static double rndn() { double u = rndu() + 1e-12, v = rndu(); return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v); }

__global__ void fill_random_words(uint32_t* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u + seed;
        h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
        p[i] = h;
    }
}
__global__ void fill_random_half(half* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u + seed;
        h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
        p[i] = __float2half(((h & 0xffff) / 65535.f - 0.5f) * 2.f);
    }
}

__global__ void stream_read(const uint4* p, size_t n, uint32_t* sink) {
    uint32_t acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = __ldcs(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) sink[0] = acc;
}

struct Variant { std::string name; std::function<int()> run; int gsz; bool is_base; };

int main(int argc, char** argv) {
    setenv("INTARSIA_DSV41_MOE_SMEMA", "0", 1);  // base = the per-lane global A kernel
    int R = argc > 1 ? atoi(argv[1]) : 2048;
    double sigma = argc > 2 ? atof(argv[2]) : 0.5;
    int reps = argc > 3 ? atoi(argv[3]) : 7;
    const char* only = (argc > 4 && argv[4][0] != 0x2d) ? argv[4] : "";
    const int H = 5120, I = 1152, E = 385, KS = 7, SK = 4, CB = 2;
    const int P = R * KS;
    // routing
    std::vector<double> lw(384);
    for (auto& v : lw) v = sigma * rndn();
    std::vector<int> ids((size_t)R * KS);
    for (int r = 0; r < R; ++r) {
        std::vector<std::pair<double, int>> key(384);
        for (int e = 0; e < 384; ++e) key[e] = {lw[e] - std::log(-std::log(rndu() + 1e-300) + 1e-300), e};
        std::partial_sort(key.begin(), key.begin() + 6, key.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (int j = 0; j < 6; ++j) ids[(size_t)r * KS + j] = key[j].second;
        ids[(size_t)r * KS + 6] = 384;
    }
    std::vector<int> cnt(E, 0);
    for (int v : ids) cnt[v]++;
    {
        double g32 = 0, g64 = 0; int mx = 0;
        for (int e = 0; e < E; ++e) { g32 += (cnt[e] + 31) / 32; g64 += (cnt[e] + 63) / 64; mx = std::max(mx, cnt[e] == 2048 ? 0 : cnt[e]); }
        printf("R=%d sigma=%.2f pairs=%d; routed members/expert mean %.1f max %d; groups of 32: %.0f, of 64: %.0f (shared expert %d members)\n", R, sigma, P, (R * 6.0) / 384, mx, g32, g64, cnt[384]);
    }
    // experts: widths and arena
    std::vector<int> k2of(E * 3);
    const int k2r = argc > 5 ? atoi(argv[5]) : 6, k2s = argc > 6 ? atoi(argv[6]) : 8;  // Mia: routed 3-bit (k2 6) in 35 layers, 2-bit (4) in 5; shared 4/5-bit
    for (int e = 0; e < E; ++e) {
        int base = e == 384 ? k2s : k2r;
        for (int j = 0; j < 3; ++j) k2of[e * 3 + j] = base;
    }
    std::vector<int> meta(E * 8, 0);
    size_t cursor = 0;  // words
    double gu_bytes = 0, tot_bytes = 0;
    for (int e = 0; e < E; ++e)
        for (int j = 0; j < 3; ++j) {
            int k2 = k2of[e * 3 + j];
            size_t words = (size_t)(5120 >> 4) * (1152 >> 4) * 4 * k2;
            meta[e * 8 + j] = (int)(cursor / 4);
            meta[e * 8 + 3 + j] = k2;
            cursor += words;
            tot_bytes += words * 4;
            if (j < 2) gu_bytes += words * 4;
        }
    printf("arena %.2f GB; gate+up bytes of all 385 experts %.3f GB (%.2f bpw); at 254 GB/s the DRAM floor of gate/up is %.2f ms, of all three %.2f ms\n",
           cursor * 4 / 1e9, gu_bytes / 1e9, gu_bytes * 8 / (385.0 * 2 * 5120 * 1152), gu_bytes / 254e9 * 1e3, tot_bytes / 254e9 * 1e3);
    uint32_t* d_arena; CK(cudaMalloc(&d_arena, cursor * 4));
    fill_random_words<<<2048, 256>>>(d_arena, cursor, 12345u);
    CK(cudaDeviceSynchronize());
    {   // empirical DRAM read bandwidth: stream the whole arena (2 GB, far over the L2)
        uint4* a4 = (uint4*)d_arena; size_t n4 = cursor / 4; uint32_t* d_sink; CK(cudaMalloc(&d_sink, 4096 * 4));
        cudaEvent_t b0, b1; cudaEventCreate(&b0); cudaEventCreate(&b1); float best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            cudaEventRecord(b0); stream_read<<<48 * 8, 256>>>(a4, n4, d_sink); cudaEventRecord(b1); cudaEventSynchronize(b1);
            float ms; cudaEventElapsedTime(&ms, b0, b1); best = std::min(best, ms);
        }
        printf("stream read of the arena: %.2f GB in %.2f ms = %.0f GB/s (the practical DRAM floor rate)\n", cursor * 4 / 1e9, best, cursor * 4 / (best * 1e-3) / 1e9);
    }
    int *d_meta, *d_cb, *d_ids; CK(cudaMalloc(&d_meta, meta.size() * 4)); CK(cudaMalloc(&d_cb, 4)); CK(cudaMalloc(&d_ids, ids.size() * 4));
    CK(cudaMemcpy(d_meta, meta.data(), meta.size() * 4, cudaMemcpyHostToDevice));
    int cbv = CB; CK(cudaMemcpy(d_cb, &cbv, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    half *d_xg, *d_xu; CK(cudaMalloc(&d_xg, (size_t)P * H * 2)); CK(cudaMalloc(&d_xu, (size_t)P * H * 2));
    fill_random_half<<<2048, 256>>>(d_xg, (size_t)P * H, 1u); fill_random_half<<<2048, 256>>>(d_xu, (size_t)P * H, 2u);
    float *d_z, *d_zref; size_t zn = (size_t)2 * P * I;
    CK(cudaMalloc(&d_z, zn * 4)); CK(cudaMalloc(&d_zref, zn * 4));
    int *d_uids, *d_ucount, *d_mstart, *d_mcount, *d_members, *d_work;
    CK(cudaMalloc(&d_uids, E * 4)); CK(cudaMalloc(&d_ucount, 4)); CK(cudaMalloc(&d_mstart, E * 4)); CK(cudaMalloc(&d_mcount, E * 4)); CK(cudaMalloc(&d_members, P * 4));
    int wlb_max = P / 16 + E + 1;
    CK(cudaMalloc(&d_work, (size_t)2 * wlb_max * 4));
    GroupedArgs a{};
    a.x0 = d_xg; a.x1 = d_xu;
    a.ex.arena = d_arena; a.ex.words = (long long)cursor; a.ex.sc = nullptr; a.ex.C = 0; a.ex.meta = d_meta; a.ex.cbp = d_cb; a.ex.E = E; a.ex.H = H; a.ex.I = I;
    a.j0 = 0; a.uids = d_uids; a.ucount = d_ucount; a.mstart = d_mstart; a.mcount = d_mcount; a.members = d_members; a.z = d_z;
    a.K = H; a.N = I; a.P = P; a.SK = SK; a.maxm = P; a.slots = KS; a.mats = 2;
    // work lists per group size
    std::vector<int*> works;
    auto make_work = [&](int gsz) {
        int places = std::min(E, P), wlb = P / gsz + places + 1;
        int* w; CK(cudaMalloc(&w, (size_t)2 * wlb * 4));
        v2::group_kernel2<<<1, 1024>>>(d_ids, P, KS, E, d_uids, d_ucount, d_mstart, d_mcount, d_members, w, wlb, gsz);
        CK(cudaDeviceSynchronize());
        return std::make_pair(w, wlb);
    };
    auto [w32, wlb32] = make_work(32);
    auto [w48, wlb48] = make_work(48);
    auto [w64, wlb64] = make_work(64);
    auto [w16, wlb16] = make_work(16);
    cudaStream_t s = 0;
    std::vector<Variant> vs;
    auto V = [&](const std::string& n, int gsz, std::function<int(GroupedArgs&)> f, bool base = false) {
        int* w = gsz == 32 ? w32 : gsz == 48 ? w48 : gsz == 64 ? w64 : w16; int wlb = gsz == 32 ? wlb32 : gsz == 48 ? wlb48 : gsz == 64 ? wlb64 : wlb16;
        vs.push_back({n, [=, &a]() { GroupedArgs b = a; b.work = w; b.nexp = wlb; return f(b); }, gsz, base});
    };
    V("base (repo launch_rows, G2 NT8 W4, 128 thr)", 32, [&](GroupedArgs& b) { return launch_rows(b, true, s); }, true);
    V("base shape via v2 (G2 NTW8 CS1), minB 1", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 8, 1, 2, 0, 1>(b, s); });
    V("base shape via v2, minB 3", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 8, 1, 2, 0, 3>(b, s); });
    V("G2 NTW4 CS2 minB 2 (<=128 regs)", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 4, 2, 2, 0, 2>(b, s); });
    V("G4 NTW2 CS2 minB 2 (<=128 regs)", 64, [&](GroupedArgs& b) { return v2::launch2<true, 4, 2, 2, 2, 0, 2>(b, s); });
    V("G4 NTW1 CS4 minB 1 (512 thr)", 64, [&](GroupedArgs& b) { return v2::launch2<true, 4, 1, 4, 2, 0, 1>(b, s); });
    V("REPO smem kernel (exl3_moe_rows.cu)", 32, [&](GroupedArgs& b) {
        static bool set = false; if (!set) { cudaFuncSetAttribute(grouped_rows_smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmemBytes); set = true; }
        const dim3 grid(1, (unsigned)(b.N / 128), (unsigned)((long long)b.mats * b.nexp));
        grouped_rows_smem_kernel<<<grid, SNTHR, kSmemBytes, s>>>(b); return ik_launch_status(); });
    V("SMEM-A G4 NTW2 CS2 ST2 PF2 (64x64, 74 KB)", 64, [&](GroupedArgs& b) { return v2::launch3<4, 2, 2, 2, 2>(b, s); });
    V("SMEM-A G4 NTW4 CS2 ST2 PF2 (64x128, 74 KB)", 64, [&](GroupedArgs& b) { return v2::launch3<4, 4, 2, 2, 2>(b, s); });
    V("SMEM-A G3 NTW4 CS2 ST3 PF2 (48x128, 83 KB)", 48, [&](GroupedArgs& b) { return v2::launch3<3, 4, 2, 2, 3>(b, s); });
    V("SMEM-A G2 NTW4 CS2 ST3 PF2", 32, [&](GroupedArgs& b) { return v2::launch3<2, 4, 2, 2, 3>(b, s); });
    V("SMEM-A G2 NTW4 CS2 ST2 PF2", 32, [&](GroupedArgs& b) { return v2::launch3<2, 4, 2, 2, 2>(b, s); });
    V("SMEM-A G2 NTW8 CS1 ST3 PF2 (128 thr)", 32, [&](GroupedArgs& b) { return v2::launch3<2, 8, 1, 2, 3>(b, s); });
    V("SMEM-A G4 NTW2 CS2 ST3 PF2 (64x64)", 64, [&](GroupedArgs& b) { return v2::launch3<4, 2, 2, 2, 3>(b, s); });
    V("SMEM-A G4 NTW4 CS2 ST3 PF2 (64x128)", 64, [&](GroupedArgs& b) { return v2::launch3<4, 4, 2, 2, 3>(b, s); });
    V("SMEM-A G2 NTW4 CS2 ST3 PF1", 32, [&](GroupedArgs& b) { return v2::launch3<2, 4, 2, 1, 3>(b, s); });
    V("G2 NTW4 CS2 (256 thr, 32 rows x 128 cols)", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 4, 2, 2, 0>(b, s); });
    V("G2 NTW2 CS2 (256 thr, 32 rows x 64 cols)", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 2, 2, 2, 0>(b, s); });
    V("G4 NTW2 CS2 (256 thr, 64 rows x 64 cols)", 64, [&](GroupedArgs& b) { return v2::launch2<true, 4, 2, 2, 2, 0>(b, s); });
    V("G4 NTW2 CS2 PF1", 64, [&](GroupedArgs& b) { return v2::launch2<true, 4, 2, 2, 1, 0>(b, s); });
    V("G3 NTW2 CS2 (256 thr, 48 rows x 64 cols)", 48, [&](GroupedArgs& b) { return v2::launch2<true, 3, 2, 2, 2, 0>(b, s); });
    V("G4 NTW1 CS4 (512 thr, 64 rows x 64 cols)", 64, [&](GroupedArgs& b) { return v2::launch2<true, 4, 1, 4, 2, 0>(b, s); });
    V("G2 NTW1 CS4 (512 thr, 32 rows x 64 cols)", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 1, 4, 2, 0>(b, s); });
    V("G1 NTW2 CS2 (256 thr, 16 rows x 64 cols)", 16, [&](GroupedArgs& b) { return v2::launch2<true, 1, 2, 2, 2, 0>(b, s); });
    // knockouts on the base shape (timing only)
    V("KO base-shape no A loads", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 8, 1, 2, 1>(b, s); });
    V("KO base-shape no decode", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 8, 1, 2, 2>(b, s); });
    V("KO base-shape no mma", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 8, 1, 2, 4>(b, s); });
    V("KO base-shape no A, no decode", 32, [&](GroupedArgs& b) { return v2::launch2<true, 2, 8, 1, 2, 3>(b, s); });
    V("KO G4 NTW2 CS2 no decode", 64, [&](GroupedArgs& b) { return v2::launch2<true, 4, 2, 2, 2, 2>(b, s); });
    V("KO G4 NTW2 CS2 no A", 64, [&](GroupedArgs& b) { return v2::launch2<true, 4, 2, 2, 2, 1>(b, s); });
    V("KO G4 NTW2 CS2 no mma", 64, [&](GroupedArgs& b) { return v2::launch2<true, 4, 2, 2, 2, 4>(b, s); });
    // correctness + timing
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> zref(zn), zgot(zn);
    std::vector<std::vector<float>> times(vs.size());
    for (size_t vi = 0; vi < vs.size(); ++vi) {
        if (only[0] && vs[vi].name.find(only) == std::string::npos && !vs[vi].is_base) continue;
        CK(cudaMemset(d_z, 0xff, zn * 4));
        int st = vs[vi].run(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (st != IK_OK) { printf("%-50s launch status %d\n", vs[vi].name.c_str(), st); continue; }
        if (vs[vi].is_base) { CK(cudaMemcpy(zref.data(), d_z, zn * 4, cudaMemcpyDeviceToHost)); }
        else if (vs[vi].name.find("KO") == std::string::npos) {
            CK(cudaMemcpy(zgot.data(), d_z, zn * 4, cudaMemcpyDeviceToHost));
            size_t bad = 0; for (size_t i = 0; i < zn; ++i) if (memcmp(&zgot[i], &zref[i], 4) != 0) ++bad;
            printf("%-50s bitwise vs base: %s (%zu differ)\n", vs[vi].name.c_str(), bad ? "NO" : "yes", bad);
        }
    }
    for (int rep = 0; rep < reps; ++rep)
        for (size_t vi = 0; vi < vs.size(); ++vi) {
            if (only[0] && vs[vi].name.find(only) == std::string::npos && !vs[vi].is_base) continue;
            cudaEventRecord(e0, s); vs[vi].run(); cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); times[vi].push_back(ms);
        }
    double base_med = 0;
    for (size_t vi = 0; vi < vs.size(); ++vi) {
        if (times[vi].empty()) continue;
        auto t = times[vi]; std::sort(t.begin(), t.end());
        if (vs[vi].is_base) base_med = t[t.size() / 2];
        printf("%-50s min %7.3f  median %7.3f ms  (x%.2f of base)\n", vs[vi].name.c_str(), t[0], t[t.size() / 2], t[t.size() / 2] / base_med);
    }
    return 0;
}

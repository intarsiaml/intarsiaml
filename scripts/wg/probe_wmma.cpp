// Lane wg: hardware probe of ONE gfx12 fp8 WMMA instruction, v_wmma_f32_16x16x16_fp8_fp8 (the builtin ik_f8_wmma uses).
// Question: can a scalar emulation reproduce its fp32 output bitwise (alignment width, truncation vs rounding, how C joins the add,
// dependence on the k position)?  This program only gathers raw evidence: crafted cases + a large random set, dumped with the hardware's
// outputs so the model is fitted on CPU afterwards.  Needs the GPU for well under a minute.  Not a kernel of the engine.
//   probe_wmma <outdir> [n_random=2000000]
// Case d of a call is the diagonal element D[d][d] = c + sum_k A[d][k] * B[k][d]; 16 independent cases per instruction. Layout (gfx12, wave32):
//   A frag lane l: row l%16, k = 8*(l/16) .. +7 ; B frag lane l: col l%16, k = 8*(l/16) .. +7 ; C/D v8f lane l: col l%16, row 8*(l/16)+i.
// The layout is verified first with a full 16x16 integer matrix product; a failed check prints LAYOUT CHECK FAILED and stops.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

struct Case { uint8_t a[16], b[16]; float c; };

static float f32_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t u32_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// ---------------------------------------------------------------- device
// General 16x16 instruction: A[16][16] (row-major fp8 bytes), B[16][16], C/D [16][16] f32, one wave.
__global__ void wmma_full(const uint8_t* A, const uint8_t* B, const float* C, float* D) {
    const int l = threadIdx.x, r = l & 15, h = l >> 4;
    uint32_t a[2], b[2];
    for (int w = 0; w < 2; ++w) {
        uint32_t av = 0, bv = 0;
        for (int j = 0; j < 4; ++j) {
            av |= (uint32_t)A[r * 16 + 8 * h + 4 * w + j] << (8 * j);
            bv |= (uint32_t)B[(8 * h + 4 * w + j) * 16 + r] << (8 * j);
        }
        a[w] = av; b[w] = bv;
    }
    v8f c;
    for (int i = 0; i < 8; ++i) c[i] = C[(8 * h + i) * 16 + r];
    const v2i av = {(int)a[0], (int)a[1]}, bv = {(int)b[0], (int)b[1]};
    v8f d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av, bv, c);
    for (int i = 0; i < 8; ++i) D[(8 * h + i) * 16 + r] = d[i];
}

// 16 diagonal cases per instruction. Case n*16+d: a[16], b[16], c in cases[]; out[n*16+d] = D[d][d]. One block = one wave handles calls
// n = blockIdx.x, blockIdx.x + gridDim.x, ...
// perm (optional, 16 bytes): the k order applied to BOTH a and b of every case (k' = perm[k]) - the model must not depend on it, or must be fitted with it.
// The WMMA must run with every lane active: the result is stored UNCONDITIONALLY (all 8 accumulator values of all 32 lanes, dall[(n*32+l)*8+i]) and
// the host picks the diagonal. (A first version stored only d[r-8h] inside `if ((r>>3)==h)`; the compiler sank the instruction into the branch and the
// partial-EXEC result was garbage: window E dump 2026-10-25, discarded.)
__global__ void wmma_diag(const Case* cases, float* dall, int ncalls, const uint8_t* perm) {
    const int l = threadIdx.x, r = l & 15, h = l >> 4;
    for (int n = blockIdx.x; n < ncalls; n += gridDim.x) {
        const Case& cs = cases[(size_t)n * 16 + r];
        uint32_t a[2] = {0, 0}, b[2] = {0, 0};
        for (int w = 0; w < 2; ++w)
            for (int j = 0; j < 4; ++j) {
                const int k = 8 * h + 4 * w + j;
                const int kk = perm ? perm[k] : k;
                a[w] |= (uint32_t)cs.a[kk] << (8 * j);
                b[w] |= (uint32_t)cs.b[kk] << (8 * j);
            }
        v8f c;
        for (int i = 0; i < 8; ++i) c[i] = (8 * h + i == r) ? cs.c : 0.f;   // C[row][col=r]: only the diagonal element carries the case's c
        const v2i av = {(int)a[0], (int)a[1]}, bv = {(int)b[0], (int)b[1]};
        v8f d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av, bv, c);
        for (int i = 0; i < 8; ++i) dall[((size_t)n * 32 + l) * 8 + i] = d[i];
    }
}

// two chained instructions (k 0..15 then 16..31): out0 = wmma(a0,b0,c), out1 = wmma(a1,b1,out0), diagonal cases
struct Chain { uint8_t a0[16], b0[16], a1[16], b1[16]; float c; };
__global__ void wmma_chain(const Chain* cs, float* d0all, float* d1all, int ncalls) {   // unconditional stores, see wmma_diag
    const int l = threadIdx.x, r = l & 15, h = l >> 4;
    for (int n = blockIdx.x; n < ncalls; n += gridDim.x) {
        const Chain& ch = cs[(size_t)n * 16 + r];
        uint32_t a0[2] = {0, 0}, b0[2] = {0, 0}, a1[2] = {0, 0}, b1[2] = {0, 0};
        for (int w = 0; w < 2; ++w)
            for (int j = 0; j < 4; ++j) {
                const int k = 8 * h + 4 * w + j;
                a0[w] |= (uint32_t)ch.a0[k] << (8 * j); b0[w] |= (uint32_t)ch.b0[k] << (8 * j);
                a1[w] |= (uint32_t)ch.a1[k] << (8 * j); b1[w] |= (uint32_t)ch.b1[k] << (8 * j);
            }
        v8f c;
        for (int i = 0; i < 8; ++i) c[i] = (8 * h + i == r) ? ch.c : 0.f;
        v8f d0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(v2i{(int)a0[0], (int)a0[1]}, v2i{(int)b0[0], (int)b0[1]}, c);
        v8f d1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(v2i{(int)a1[0], (int)a1[1]}, v2i{(int)b1[0], (int)b1[1]}, d0);
        for (int i = 0; i < 8; ++i) { d0all[((size_t)n * 32 + l) * 8 + i] = d0[i]; d1all[((size_t)n * 32 + l) * 8 + i] = d1[i]; }
    }
}

// ---------------------------------------------------------------- host helpers
static double dec_e4m3(uint8_t b, bool* nan) {   // OCP e4m3 (bias 7, no inf, 0x7f / 0xff = NaN)
    *nan = (b & 0x7f) == 0x7f;
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    double v = e == 0 ? std::ldexp((double)m / 8.0, -6) : std::ldexp(1.0 + (double)m / 8.0, e - 7);
    return s ? -v : v;
}
static uint8_t enc_pow2(int x, int sign) {      // 2^x, x in [-9, 8]
    uint8_t v;
    if (x >= -6) v = (uint8_t)(((x + 7) << 3));
    else v = (uint8_t)(1 << (x + 9));           // subnormal: m/8 * 2^-6 = 2^x -> m = 2^(x+9)
    return v | (sign ? 0x80 : 0);
}
// product 2^e (e in [-18, 16]) as a (a, b) pair, a carrying the sign
static void prod_pow2(int e, int sign, uint8_t* a, uint8_t* b) {
    if (e >= -9 && e <= 8) { *a = enc_pow2(e, sign); *b = enc_pow2(0, 0); }
    else if (e > 8) { *a = enc_pow2(8, sign); *b = enc_pow2(e - 8, 0); }
    else { *a = enc_pow2(-9, sign); *b = enc_pow2(e + 9, 0); }
}
static Case zero_case() { Case c; memset(&c, 0, sizeof c); c.c = 0.f; return c; }
static void set_term(Case& c, int k, int e, int sign) { prod_pow2(e, sign, &c.a[k], &c.b[k]); }

struct Exact { bool nan; double sum; };
static Exact exact_sum(const Case& c) {          // products are exact in double; the sum of 17 terms is not always (huge range), so also use long double
    long double s = c.c; bool nan = std::isnan(c.c);
    for (int k = 0; k < 16; ++k) {
        bool n1, n2; const double x = dec_e4m3(c.a[k], &n1), y = dec_e4m3(c.b[k], &n2);
        if (n1 || n2) nan = true;
        s += (long double)x * (long double)y;
    }
    return {nan, (double)s};
}
// candidate models (all on the exact sum as long double, which holds 64 mantissa bits: enough for products 2^-18..2^16 plus a f32 c)
static float model_exact_rne(const Case& c) {     // correctly rounded exact sum
    long double s = c.c; for (int k = 0; k < 16; ++k) { bool n; s += (long double)dec_e4m3(c.a[k], &n) * (long double)dec_e4m3(c.b[k], &n); }
    return (float)s;
}
static float model_seq_rne(const Case& c) {       // ((c + p0) + p1) + ... each add rounded to f32 (RNE), products exact
    float s = c.c; for (int k = 0; k < 16; ++k) { bool n; s = (float)((double)s + dec_e4m3(c.a[k], &n) * dec_e4m3(c.b[k], &n)); } return s;
}
static float model_exact_trunc(const Case& c) {  // exact sum truncated toward zero to f32
    long double s = c.c; for (int k = 0; k < 16; ++k) { bool n; s += (long double)dec_e4m3(c.a[k], &n) * (long double)dec_e4m3(c.b[k], &n); }
    float f = (float)s; if ((long double)f > s && s > 0) f = std::nextafterf(f, 0.f); if ((long double)f < s && s < 0) f = std::nextafterf(f, 0.f); return f;
}

// diagonal element D[r][r] of call n: lane r + 16*(r>>3), accumulator register r&7
static float diag_of(const std::vector<float>& all, size_t n, int r) { return all[((n * 32) + (size_t)(r + 16 * (r >> 3))) * 8 + (r & 7)]; }

static std::vector<float> run_cases(const std::vector<Case>& cs, const uint8_t* perm_h = nullptr) {
    size_t n = cs.size(); const size_t calls = (n + 15) / 16;
    std::vector<Case> padded(cs); padded.resize(calls * 16, zero_case());
    Case* dc; float* dout; uint8_t* dperm = nullptr;
    CK(hipMalloc(&dc, padded.size() * sizeof(Case))); CK(hipMalloc(&dout, padded.size() * 32 * 8 / 16 * 4));
    CK(hipMemcpy(dc, padded.data(), padded.size() * sizeof(Case), hipMemcpyHostToDevice));
    if (perm_h) { CK(hipMalloc(&dperm, 16)); CK(hipMemcpy(dperm, perm_h, 16, hipMemcpyHostToDevice)); }
    wmma_diag<<<(unsigned)std::min<size_t>(calls, 1024), 32>>>(dc, dout, (int)calls, dperm);
    CK(hipGetLastError()); CK(hipDeviceSynchronize());
    std::vector<float> all(calls * 32 * 8), out(padded.size());
    CK(hipMemcpy(all.data(), dout, all.size() * 4, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < padded.size(); ++i) out[i] = diag_of(all, i / 16, (int)(i % 16));
    CK(hipFree(dc)); CK(hipFree(dout)); if (dperm) CK(hipFree(dperm));
    out.resize(n);
    return out;
}

static bool same_bits(float a, float b) { return (std::isnan(a) && std::isnan(b)) || u32_of(a) == u32_of(b); }

struct Stats { size_t n = 0, nan = 0, eq_exact = 0, eq_seq = 0, eq_trunc = 0, eq_any = 0, within1ulp = 0; };
static void tally(Stats& s, const Case& c, float hw) {
    ++s.n;
    const Exact ex = exact_sum(c);
    if (ex.nan || std::isnan(hw)) { ++s.nan; return; }
    const float m0 = model_exact_rne(c), m1 = model_seq_rne(c), m2 = model_exact_trunc(c);
    const bool e0 = same_bits(hw, m0), e1 = same_bits(hw, m1), e2 = same_bits(hw, m2);
    s.eq_exact += e0; s.eq_seq += e1; s.eq_trunc += e2; s.eq_any += (e0 || e1 || e2);
    if (std::fabs((double)hw - (double)m0) <= std::fabs((double)std::nextafterf(m0, INFINITY) - (double)m0) * 1.0001) ++s.within1ulp;
}
static void print_stats(const char* name, const Stats& s) {
    printf("%-34s n=%-8zu nan/inf=%-6zu hw==exactRNE %6.2f%%  hw==seqRNE %6.2f%%  hw==exactTRUNC %6.2f%%  hw==any %6.2f%%  within1ulp(exact) %6.2f%%\n", name, s.n, s.nan,
           100.0 * s.eq_exact / std::max<size_t>(1, s.n - s.nan), 100.0 * s.eq_seq / std::max<size_t>(1, s.n - s.nan), 100.0 * s.eq_trunc / std::max<size_t>(1, s.n - s.nan),
           100.0 * s.eq_any / std::max<size_t>(1, s.n - s.nan), 100.0 * s.within1ulp / std::max<size_t>(1, s.n - s.nan));
}

static void dump_cases(const std::string& path, const std::vector<Case>& cs, const std::vector<float>& out) {
    FILE* f = fopen(path.c_str(), "wb"); if (!f) { perror(path.c_str()); exit(2); }
    uint64_t n = cs.size(); fwrite("WMMAPRB1", 1, 8, f); fwrite(&n, 8, 1, f);
    for (size_t i = 0; i < cs.size(); ++i) { fwrite(cs[i].a, 1, 16, f); fwrite(cs[i].b, 1, 16, f); fwrite(&cs[i].c, 4, 1, f); fwrite(&out[i], 4, 1, f); }
    fclose(f);
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    const size_t NR = argc > 2 ? (size_t)atoll(argv[2]) : 2000000;
    int dev = 0; if (getenv("DEV")) dev = atoi(getenv("DEV"));
    CK(hipSetDevice(dev));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, dev));
    printf("probe device %d: %s %s\n", dev, prop.name, prop.gcnArchName);
    std::mt19937_64 rng(20261025);

    // ---- 0. layout check: full 16x16 product on small integers (exact in any sane accumulator)
    {
        std::vector<uint8_t> A(256), B(256); std::vector<float> C(256), D(256);
        const uint8_t small[] = {0x00, 0x38, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x44, 0x34};   // 0, 1, -1, 2, -2, .5, -.5, 3, .25 ... (all exact)
        for (auto& x : A) x = small[rng() % 9];
        for (auto& x : B) x = small[rng() % 9];
        for (auto& x : C) x = (float)((int)(rng() % 9) - 4);
        uint8_t *dA, *dB; float *dC, *dD;
        CK(hipMalloc(&dA, 256)); CK(hipMalloc(&dB, 256)); CK(hipMalloc(&dC, 1024)); CK(hipMalloc(&dD, 1024));
        CK(hipMemcpy(dA, A.data(), 256, hipMemcpyHostToDevice)); CK(hipMemcpy(dB, B.data(), 256, hipMemcpyHostToDevice)); CK(hipMemcpy(dC, C.data(), 1024, hipMemcpyHostToDevice));
        wmma_full<<<1, 32>>>(dA, dB, dC, dD); CK(hipGetLastError()); CK(hipDeviceSynchronize());
        CK(hipMemcpy(D.data(), dD, 1024, hipMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
            double s = C[r * 16 + c]; bool n;
            for (int k = 0; k < 16; ++k) s += dec_e4m3(A[r * 16 + k], &n) * dec_e4m3(B[k * 16 + c], &n);
            if (s != (double)D[r * 16 + c]) ++bad;
        }
        printf("layout check: %d of 256 elements differ from the exact integer product\n", bad);
        if (bad) { printf("LAYOUT CHECK FAILED\n"); return 3; }
        CK(hipFree(dA)); CK(hipFree(dB)); CK(hipFree(dC)); CK(hipFree(dD));
    }

    // ---- 1. crafted cases
    std::vector<Case> S; std::vector<std::string> tag;
    auto add = [&](const Case& c, const std::string& t) { S.push_back(c); tag.push_back(t); };
    char buf[96];
    // F1: one big term 2^e at position i, the other 15 terms 1.0 (c = 0): position / alignment / truncation of the small terms
    for (int i = 0; i < 16; ++i) for (int e = 0; e <= 16; ++e) {
        Case c = zero_case(); for (int k = 0; k < 16; ++k) set_term(c, k, 0, 0); set_term(c, i, e, 0);
        snprintf(buf, sizeof buf, "F1 pos=%d e=%d", i, e); add(c, buf);
    }
    // F1b: one big term 2^e, the others tiny 2^-t: how many bits below the max exponent survive
    for (int i : {0, 7, 8, 15}) for (int e = 0; e <= 16; ++e) for (int t : {1, 4, 8, 12, 16, 18}) {
        Case c = zero_case(); for (int k = 0; k < 16; ++k) set_term(c, k, -t, 0); set_term(c, i, e, 0);
        snprintf(buf, sizeof buf, "F1b pos=%d e=%d tiny=-%d", i, e, t); add(c, buf);
    }
    // F2: C = 2^s, sixteen products of 1.0 / of 2^-t: does C join the same aligned add
    for (int s = -20; s <= 30; ++s) for (int t : {0, 6, 12, 18}) {
        Case c = zero_case(); for (int k = 0; k < 16; ++k) set_term(c, k, -t, 0); c.c = std::ldexp(1.0f, s);
        snprintf(buf, sizeof buf, "F2 c=2^%d prod=2^-%d x16", s, t); add(c, buf);
    }
    // F3: +2^16 and -2^16 at positions i, j, the rest tiny: a pairwise tree cancels them before the tiny terms meet them; a single aligned add does not care
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j) if (i != j) for (int t : {0, 6, 12, 18}) {
        Case c = zero_case(); for (int k = 0; k < 16; ++k) set_term(c, k, -t, 0); set_term(c, i, 16, 0); set_term(c, j, 16, 1);
        snprintf(buf, sizeof buf, "F3 +@%d -@%d tiny=-%d", i, j, t); add(c, buf);
    }
    // F4: rounding / ties around c = 2^6 (ulp 2^-17): n copies of 2^-18 (n/2 ulp), c even and odd mantissa
    for (int odd = 0; odd < 2; ++odd) for (int n = 0; n <= 16; ++n) for (int sgn = 0; sgn < 2; ++sgn) {
        Case c = zero_case(); for (int k = 0; k < n; ++k) set_term(c, k, -18, sgn);
        c.c = std::ldexp(1.0f, 6) + (odd ? std::ldexp(1.0f, -17) : 0.f);
        snprintf(buf, sizeof buf, "F4 c=2^6+%s n=%d sign=%d", odd ? "ulp" : "0", n, sgn); add(c, buf);
    }
    // F5: subnormal / extreme C with zero or tiny products
    {
        const float cs[] = {f32_of(0x00000001), f32_of(0x80000001), f32_of(0x00400000), f32_of(0x007fffff), f32_of(0x00800000), f32_of(0x80800000), f32_of(0x00000000), f32_of(0x80000000),
                            std::ldexp(1.0f, -127), 3.0f * std::ldexp(1.0f, -130), std::ldexp(1.5f, -126), std::ldexp(1.0f, -120)};
        for (float cv : cs) for (int mode = 0; mode < 3; ++mode) {
            Case c = zero_case(); c.c = cv;
            if (mode == 1) set_term(c, 3, -18, 0);
            if (mode == 2) { set_term(c, 3, -18, 0); set_term(c, 9, -18, 1); }
            snprintf(buf, sizeof buf, "F5 c=%08x mode=%d", u32_of(cv), mode); add(c, buf);
        }
    }
    // F6: fp8 subnormal inputs, fp8 NaN, f32 inf / nan in C
    {
        Case c = zero_case(); for (int k = 0; k < 16; ++k) { c.a[k] = (uint8_t)(1 + (k & 7)); c.b[k] = (uint8_t)(1 + ((k * 3) & 7)); } add(c, "F6 fp8 subnormals only");
        c = zero_case(); set_term(c, 0, 0, 0); c.a[5] = 0x7f; c.b[5] = 0x38; add(c, "F6 fp8 NaN in a");
        c = zero_case(); set_term(c, 0, 0, 0); c.c = INFINITY; add(c, "F6 c=+inf");
        c = zero_case(); set_term(c, 0, 0, 0); c.c = -INFINITY; add(c, "F6 c=-inf");
        c = zero_case(); set_term(c, 0, 16, 0); c.c = f32_of(0x7f7fffff); add(c, "F6 c=FLT_MAX + 2^16");
        c = zero_case(); set_term(c, 0, 16, 0); set_term(c, 1, 16, 0); c.c = f32_of(0x7f7fffff); add(c, "F6 c=FLT_MAX + 2 * 2^16");
        c = zero_case(); c.c = f32_of(0x7fc00000); add(c, "F6 c=qNaN");
    }
    // F7: the same random case at every k permutation tested below (see 'perm')
    std::vector<float> Sout = run_cases(S);
    {   // sanity: the diag kernel must agree with the exact value where the answer is not in doubt (16 x 1.0*1.0 = 16, one 2^e term, c = 0)
        std::vector<Case> T; for (int e = 0; e <= 10; ++e) { Case c = zero_case(); for (int k = 0; k < 16; ++k) set_term(c, k, 0, 0); set_term(c, 3, e, 0); T.push_back(c); }
        std::vector<float> o = run_cases(T); size_t bad = 0;
        for (size_t i = 0; i < T.size(); ++i) bad += !same_bits(o[i], (float)(15.0 + std::ldexp(1.0, (int)i)));
        printf("diag sanity: %zu of %zu easy cases differ from the exact sum%s\n", bad, T.size(), bad ? "  <-- PROBE BROKEN, results below are not hardware semantics" : "");
    }
    {   // determinism
        std::vector<float> again = run_cases(S); size_t diff = 0; for (size_t i = 0; i < S.size(); ++i) diff += !same_bits(Sout[i], again[i]);
        printf("crafted cases: %zu, repeat-run differences: %zu\n", S.size(), diff);
    }
    {   // structured summary lines (F1 and F2 are the informative ones)
        Stats st; for (size_t i = 0; i < S.size(); ++i) tally(st, S[i], Sout[i]);
        print_stats("crafted (all)", st);
        for (size_t i = 0; i < S.size(); ++i) {
            const std::string& t = tag[i];
            const bool show = t.rfind("F1 pos=0 ", 0) == 0 || t.rfind("F1 pos=15 ", 0) == 0 || t.rfind("F2 c=2^", 0) == 0 || t.rfind("F4", 0) == 0 || t.rfind("F5", 0) == 0 || t.rfind("F6", 0) == 0;
            if (!show) continue;
            const float m0 = model_exact_rne(S[i]);
            printf("  %-34s hw=%-14.9g (%08x)  exactRNE=%-14.9g  %s\n", t.c_str(), Sout[i], u32_of(Sout[i]), m0, same_bits(Sout[i], m0) ? "same" : "DIFF");
        }
        // F3 summary: how many cancel cases disagree with the exact sum, by tiny exponent
        for (int t : {0, 6, 12, 18}) {
            size_t n = 0, d = 0;
            for (size_t i = 0; i < S.size(); ++i) if (tag[i].rfind("F3", 0) == 0 && tag[i].find("tiny=-" + std::to_string(t)) == tag[i].size() - ("tiny=-" + std::to_string(t)).size()) { ++n; d += !same_bits(Sout[i], model_exact_rne(S[i])); }
            printf("  F3 tiny=-%-2d: %zu of %zu cancel cases differ from the exact sum\n", t, d, n);
        }
    }
    dump_cases(dir + "/probe-crafted.bin", S, Sout);
    { FILE* f = fopen((dir + "/probe-crafted.tags").c_str(), "w"); for (auto& t : tag) fprintf(f, "%s\n", t.c_str()); fclose(f); }

    // ---- 2. random differential set
    auto rand_e4m3 = [&](bool allow_nan) { uint8_t b; do { b = (uint8_t)rng(); } while (!allow_nan && (b & 0x7f) == 0x7f); return b; };
    auto rand_c = [&](int mode) -> float {
        switch (mode) {
            case 0: { uint32_t u = (uint32_t)rng(); if ((u & 0x7f800000) == 0x7f800000) u &= ~0x40000000u; return f32_of(u); }       // any finite float
            case 1: return std::ldexp((float)((double)(int32_t)rng() / 2147483648.0), (int)(rng() % 40) - 25);                          // moderate
            case 2: return f32_of((uint32_t)(rng() & 0x80ffffffu) & ~0x7f000000u);                                                      // subnormal / small exponents
            default: return 0.f;
        }
    };
    std::vector<Case> R(NR); std::vector<int> kind(NR);
    for (size_t i = 0; i < NR; ++i) {
        Case& c = R[i]; const int kd = (int)(i % 8); kind[i] = kd;
        for (int k = 0; k < 16; ++k) { c.a[k] = rand_e4m3(false); c.b[k] = rand_e4m3(false); }
        c.c = rand_c(kd % 4 == 3 ? 3 : (int)(rng() % 3));
        if (kd == 1) {   // clustered binades: a, b exponents within +-1 of a base; c of the product magnitude
            const int base = 4 + (int)(rng() % 8);
            for (int k = 0; k < 16; ++k) { c.a[k] = (uint8_t)((rng() & 0x80) | ((base + (int)(rng() % 3) - 1) << 3) | (rng() & 7)); c.b[k] = (uint8_t)((rng() & 0x80) | ((base + (int)(rng() % 3) - 1) << 3) | (rng() & 7)); }
            c.c = std::ldexp((float)((double)(int32_t)rng() / 2147483648.0), 2 * (base - 7) + 2);
        } else if (kd == 2) {   // cancellation: later products undo the earlier ones, c cancels the rest
            for (int k = 8; k < 16; ++k) { c.a[k] = c.a[k - 8] ^ 0x80; c.b[k] = c.b[k - 8]; }
            if (rng() & 1) c.c = 0.f;
        } else if (kd == 4) {   // sparse: most products zero
            for (int k = 0; k < 16; ++k) if (rng() % 4) c.a[k] = (uint8_t)(rng() & 0x80);
        } else if (kd == 5) {   // fp8 subnormal inputs
            for (int k = 0; k < 16; ++k) { c.a[k] = (uint8_t)((rng() & 0x80) | (rng() % 8)); if (rng() & 1) c.b[k] = (uint8_t)((rng() & 0x80) | (rng() % 8)); }
        } else if (kd == 6) {   // wide dynamic range: a,b exponents drawn from the extremes
            for (int k = 0; k < 16; ++k) { const bool hi = rng() & 1; c.a[k] = (uint8_t)((rng() & 0x80) | ((hi ? 11 + rng() % 4 : rng() % 3) << 3) | (rng() & 7)); c.b[k] = (uint8_t)((rng() & 0x80) | ((hi ? 11 + rng() % 4 : rng() % 3) << 3) | (rng() & 7)); }
        }
    }
    std::vector<float> Rout = run_cases(R);
    {
        std::vector<float> again = run_cases(R); size_t diff = 0; for (size_t i = 0; i < NR; ++i) diff += !same_bits(Rout[i], again[i]);
        printf("random cases: %zu, repeat-run differences: %zu\n", NR, diff);
        const char* names[8] = {"R0 any bytes, c any", "R1 clustered binades", "R2 cancellation", "R3 c = 0", "R4 sparse", "R5 fp8 subnormals", "R6 wide range", "R7 any bytes"};
        Stats all; Stats per[8];
        for (size_t i = 0; i < NR; ++i) { tally(per[kind[i]], R[i], Rout[i]); tally(all, R[i], Rout[i]); }
        for (int k = 0; k < 8; ++k) print_stats(names[k], per[k]);
        print_stats("random (all)", all);
        size_t sub_in = 0, sub_out = 0, zero_out = 0;
        for (size_t i = 0; i < NR; ++i) { if (std::fpclassify(R[i].c) == FP_SUBNORMAL) ++sub_in; if (std::fpclassify(Rout[i]) == FP_SUBNORMAL) ++sub_out; if (Rout[i] == 0.f) ++zero_out; }
        printf("random set: c subnormal in %zu cases, hw output subnormal in %zu, hw output zero in %zu\n", sub_in, sub_out, zero_out);
    }
    dump_cases(dir + "/probe-random.bin", R, Rout);

    // ---- 3. position / permutation dependence: 4000 random cases at k permutations (the same permutation applied to a and b) must give the same bits if the add is order-free
    {
        const size_t NP = 4000; std::vector<Case> P(R.begin(), R.begin() + NP);
        std::vector<float> base = run_cases(P);
        size_t total_diff = 0; std::vector<size_t> pd;
        for (int trial = 0; trial < 12; ++trial) {
            uint8_t perm[16]; for (int k = 0; k < 16; ++k) perm[k] = (uint8_t)k;
            if (trial == 0) for (int k = 0; k < 16; ++k) perm[k] = (uint8_t)(15 - k);
            else if (trial == 1) for (int k = 0; k < 16; ++k) perm[k] = (uint8_t)(k ^ 1);
            else if (trial == 2) for (int k = 0; k < 16; ++k) perm[k] = (uint8_t)(k ^ 8);
            else std::shuffle(perm, perm + 16, rng);
            std::vector<float> o = run_cases(P, perm); size_t d = 0; for (size_t i = 0; i < NP; ++i) d += !same_bits(o[i], base[i]);
            printf("  k-permutation trial %2d (%s): %zu of %zu cases change bits\n", trial, trial == 0 ? "reverse" : trial == 1 ? "swap pairs" : trial == 2 ? "swap halves" : "random", d, NP);
            total_diff += d;
        }
        printf("permutation dependence: %zu changed outputs in total (0 = the 16-term add is order-free)\n", total_diff);
    }

    // ---- 4. chained instructions (the engine feeds D back as C): out1 = wmma(a1, b1, wmma(a0, b0, c))
    {
        const size_t NC = 200000; std::vector<Chain> ch(NC);
        for (auto& c : ch) { for (int k = 0; k < 16; ++k) { c.a0[k] = rand_e4m3(false); c.b0[k] = rand_e4m3(false); c.a1[k] = rand_e4m3(false); c.b1[k] = rand_e4m3(false); } c.c = rand_c((int)(rng() % 4)); }
        const size_t calls = (NC + 15) / 16; ch.resize(calls * 16, ch[0]);
        Chain* dch; float *d0, *d1;
        CK(hipMalloc(&dch, ch.size() * sizeof(Chain))); CK(hipMalloc(&d0, calls * 32 * 8 * 4)); CK(hipMalloc(&d1, calls * 32 * 8 * 4));
        CK(hipMemcpy(dch, ch.data(), ch.size() * sizeof(Chain), hipMemcpyHostToDevice));
        wmma_chain<<<(unsigned)std::min<size_t>(calls, 1024), 32>>>(dch, d0, d1, (int)calls); CK(hipGetLastError()); CK(hipDeviceSynchronize());
        std::vector<float> a0(calls * 256), a1(calls * 256), o0(ch.size()), o1(ch.size());
        CK(hipMemcpy(a0.data(), d0, a0.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(a1.data(), d1, a1.size() * 4, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < ch.size(); ++i) { o0[i] = diag_of(a0, i / 16, (int)(i % 16)); o1[i] = diag_of(a1, i / 16, (int)(i % 16)); }
        // consistency: the single-instruction results must match the diag kernel's own first step, i.e. step 2 with c = o0 must reproduce o1
        std::vector<Case> second(NC); for (size_t i = 0; i < NC; ++i) { memcpy(second[i].a, ch[i].a1, 16); memcpy(second[i].b, ch[i].b1, 16); second[i].c = o0[i]; }
        std::vector<float> o1b = run_cases(second); size_t diff = 0; for (size_t i = 0; i < NC; ++i) diff += !same_bits(o1[i], o1b[i]);
        printf("chained: %zu cases, chain vs two separate instructions differ in %zu\n", NC, diff);
        FILE* f = fopen((dir + "/probe-chain.bin").c_str(), "wb"); uint64_t n = NC; fwrite("WMMACHN1", 1, 8, f); fwrite(&n, 8, 1, f);
        for (size_t i = 0; i < NC; ++i) { fwrite(ch[i].a0, 1, 16, f); fwrite(ch[i].b0, 1, 16, f); fwrite(ch[i].a1, 1, 16, f); fwrite(ch[i].b1, 1, 16, f); fwrite(&ch[i].c, 4, 1, f); fwrite(&o0[i], 4, 1, f); fwrite(&o1[i], 4, 1, f); }
        fclose(f);
        CK(hipFree(dch)); CK(hipFree(d0)); CK(hipFree(d1));
    }
    printf("PROBE DONE\n");
    return 0;
}

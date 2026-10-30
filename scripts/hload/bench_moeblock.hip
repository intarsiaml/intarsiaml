// Lane hload: standalone microbench + output dump of the fused HIP MoE block (kernels/hip/moe_block.hip, ik_moe_block_bf16 /
// ik_moe_block_bf16_rows: the decode MoE at 1 and 2 rows) at the shapes of Qwen3.6-35B-A3B NVFP4 (E 256, top-8 normalised,
// H 2048, I 512, shared IS 512; TP=2 shard I = IS = 256). Includes moe_block.hip itself, so the numbers are the tree's kernel.
//   BENCH_TS=1,2 BENCH_I=512,256 BENCH_REPS=5 BENCH_LAYERS=12 BENCH_DUMP=<dir> ./bench_moeblock
// T = 1 runs ik_moe_block_bf16 (the solo step), T = 2 ik_moe_block_bf16_rows (the 2-row batched step). Router logits are random
// per layer copy (BENCH_LAYERS copies of all expert and shared weights rotate: about 5.6 GB at I 512, so the routed experts come
// from DRAM). us per launch: best of BENCH_REPS reps of the mean over the layer copies. Dump: the output rows of every layer copy.
#include "moe_block.hip"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int ik_rt_stamp(void*, const char*, int) { return 0; }

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(2); } } while (0)

static ik_tensor T_(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t{};
    t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t st = 1;
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = st; st *= sh[i]; }
    return t;
}
static std::vector<int> csv(const char* s, const char* def) {
    std::string v = s ? s : def;
    std::vector<int> r;
    size_t a = 0;
    while (a < v.size()) { size_t b = v.find(',', a); if (b == std::string::npos) b = v.size(); r.push_back(atoi(v.substr(a, b - a).c_str())); a = b + 1; }
    return r;
}
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

int main() {
    const char* e;
    const auto Ts = csv(getenv("BENCH_TS"), "1,2");
    const auto Is = csv(getenv("BENCH_I"), "512");
    const int reps = (e = getenv("BENCH_REPS")) ? atoi(e) : 5;
    const int L = (e = getenv("BENCH_LAYERS")) ? atoi(e) : 12;
    const char* dump = getenv("BENCH_DUMP");
    if (dump) { std::string c = std::string("mkdir -p ") + dump; if (system(c.c_str()) != 0) return 2; }
    const int E = 256, K = 8, H = 2048;
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    std::mt19937 rng(777);
    std::vector<uint8_t> hb;
    auto dev = [&](size_t n, int kind) {  // kind 0: fp4 bytes, 1: e4m3 block scales, 2: f32 global scales (n floats)
        void* d; CK(hipMalloc(&d, kind == 2 ? n * 4 : n));
        if (kind == 2) {
            std::vector<float> f(n);
            for (auto& v : f) v = 0.5f + (rng() % 1000) * 1e-3f;
            CK(hipMemcpy(d, f.data(), n * 4, hipMemcpyHostToDevice));
        } else {
            hb.resize(n);
            for (auto& b : hb) b = kind ? (uint8_t)(0x30 + rng() % 16) : (uint8_t)rng();
            CK(hipMemcpy(d, hb.data(), n, hipMemcpyHostToDevice));
        }
        return d;
    };
    for (int I : Is) {
        const int IS = I;
        struct Layer { void *guw, *gus, *dnw, *dns, *gsc, *sgw, *sgs, *sgg, *suw, *sus, *sug, *sdw, *sds, *sdg, *logits, *sl; };
        std::vector<Layer> ly(L);
        for (auto& l : ly) {
            l.guw = dev((size_t)E * 2 * I * H / 2, 0); l.gus = dev((size_t)E * 2 * I * H / 16, 1);
            l.dnw = dev((size_t)E * H * I / 2, 0); l.dns = dev((size_t)E * H * I / 16, 1); l.gsc = dev((size_t)E * 3, 2);
            l.sgw = dev((size_t)IS * H / 2, 0); l.sgs = dev((size_t)IS * H / 16, 1); l.sgg = dev(1, 2);
            l.suw = dev((size_t)IS * H / 2, 0); l.sus = dev((size_t)IS * H / 16, 1); l.sug = dev(1, 2);
            l.sdw = dev((size_t)H * IS / 2, 0); l.sds = dev((size_t)H * IS / 16, 1); l.sdg = dev(1, 2);
            std::vector<uint16_t> lg(2 * E), s1(2);
            for (auto& v : lg) v = f2bf(((int)(rng() % 20001) - 10000) * 4e-4f);
            for (auto& v : s1) v = f2bf(((int)(rng() % 2001) - 1000) * 1e-3f);
            CK(hipMalloc(&l.logits, lg.size() * 2)); CK(hipMemcpy(l.logits, lg.data(), lg.size() * 2, hipMemcpyHostToDevice));
            CK(hipMalloc(&l.sl, 4)); CK(hipMemcpy(l.sl, s1.data(), 4, hipMemcpyHostToDevice));
        }
        std::vector<uint16_t> hx(2 * H);
        for (auto& v : hx) v = f2bf(((int)(rng() % 20001) - 10000) * 1e-4f);
        void* dx; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        void* dout; CK(hipMalloc(&dout, (size_t)L * 2 * H * 2));
        const int64_t sb = ik_moe_block_scratch_bytes(2, K, I, IS);
        void* dscr; CK(hipMalloc(&dscr, sb));
        for (int T : Ts) {
            auto run = [&](int l) {
                const Layer& y = ly[l];
                ik_tensor in[17] = {
                    T_(dx, IK_DTYPE_BF16, {T, H}), T_(y.logits, IK_DTYPE_BF16, {T, E}), T_(y.sl, IK_DTYPE_BF16, {T, 1}),
                    T_(y.guw, IK_DTYPE_U8, {E, 2 * I, H / 2}), T_(y.gus, IK_DTYPE_FP8_E4M3, {E, 2 * I, H / 16}),
                    T_(y.dnw, IK_DTYPE_U8, {E, H, I / 2}), T_(y.dns, IK_DTYPE_FP8_E4M3, {E, H, I / 16}), T_(y.gsc, IK_DTYPE_F32, {E, 3}),
                    T_(y.sgw, IK_DTYPE_U8, {IS, H / 2}), T_(y.sgs, IK_DTYPE_FP8_E4M3, {IS, H / 16}), T_(y.sgg, IK_DTYPE_F32, {1}),
                    T_(y.suw, IK_DTYPE_U8, {IS, H / 2}), T_(y.sus, IK_DTYPE_FP8_E4M3, {IS, H / 16}), T_(y.sug, IK_DTYPE_F32, {1}),
                    T_(y.sdw, IK_DTYPE_U8, {H, IS / 2}), T_(y.sds, IK_DTYPE_FP8_E4M3, {H, IS / 16}), T_(y.sdg, IK_DTYPE_F32, {1})};
                ik_tensor out = T_((char*)dout + (size_t)l * 2 * H * 2, IK_DTYPE_BF16, {T, H});
                ik_scratch scr{dscr, sb};
                const int rc = T == 1 ? ik_moe_block_bf16(in, K, 1, &out, &scr, st) : ik_moe_block_bf16_rows(in, K, 1, &out, &scr, st);
                if (rc) { fprintf(stderr, "ik_moe_block rc %d (T %d I %d)\n", rc, T, I); exit(2); }
            };
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                for (int l = 0; l < L; ++l) run(l);
                CK(hipEventRecord(e0, st));
                for (int l = 0; l < L; ++l) run(l);
                CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                best = std::min(best, ms * 1e3 / L);
            }
            if (dump) {
                std::vector<uint16_t> o((size_t)L * 2 * H);
                CK(hipMemcpy(o.data(), dout, o.size() * 2, hipMemcpyDeviceToHost));
                char fn[512];
                snprintf(fn, sizeof fn, "%s/block-T%d-I%d.bin", dump, T, I);
                FILE* f = fopen(fn, "wb");
                if (!f) { fprintf(stderr, "cannot write %s\n", fn); return 2; }
                for (int l = 0; l < L; ++l) fwrite(o.data() + (size_t)l * 2 * H, 2, (size_t)T * H, f);
                fclose(f);
            }
            printf("moe_block I=%d T=%d %8.2f us\n", I, T, best);
            fflush(stdout);
        }
        for (auto& l : ly)
            for (void* p : {l.guw, l.gus, l.dnw, l.dns, l.gsc, l.sgw, l.sgs, l.sgg, l.suw, l.sus, l.sug, l.sdw, l.sds, l.sdg, l.logits, l.sl})
                CK(hipFree(p));
        CK(hipFree(dx)); CK(hipFree(dout)); CK(hipFree(dscr));
    }
    return 0;
}

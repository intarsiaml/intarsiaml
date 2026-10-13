// Native replica of one Qwen3.6-35B-A3B decode step (the exact launcher sequence of Model::run_chunk for T = 1,
// random weights, per-layer dense weights so L2 does not flatter the timings; the routed experts of all 40 layers
// share one set, which only moves which 8 of 256 experts a layer reads). It exists to measure, without the Rust
// engine, what the step costs eagerly (one launch per op, the host position) versus replayed from a CUDA graph
// (device position), and to look at the per-kernel GPU times under nsys.
//
//   nvcc -O2 --std=c++17 -arch=sm_121 --expt-relaxed-constexpr -I kernels kernels/cuda_checks/decode_replica.cu \
//        obj/*.o -lcublas -o decode_replica          (obj/*.o: the kernels/cuda/*.cu objects, see graph_pos_check.cu)
//   ./decode_replica [--ctx 1024] [--steps 200] [--mode eager|graph|both]
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include "abi/ik_abi.h"
#include "cuda/ik_runtime.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define IKOK(x) do { int r_ = (x); if (r_ != 0) { printf("launcher %s returned %d at line %d\n", #x, r_, __LINE__); exit(2); } } while (0)

namespace {
const int H = 2048, NL = 40, VOC = 248320;
const int NH = 16, NKV = 2, HD = 256, ROT = 64;               // gated attention
const int HK = 16, HV = 32, DK = 128, DV = 128, CONVK = 4;     // GatedDeltaNet
const int CONV_C = 2 * HK * DK + HV * DV, VDIM = HV * DV;      // 8192, 4096
const int NE = 256, TOPK = 8, EI = 512, SI = 512;

uint64_t rs = 0x9E3779B97F4A7C15ull;
uint32_t rnd32() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
float rndf() { return (rnd32() & 0xFFFFFF) / 16777216.f - 0.5f; }

void* dalloc(size_t n) { void* p; CK(cudaMalloc(&p, n + 256)); CK(cudaMemset(p, 0, n + 256)); return p; }
void* bf16_rand(size_t n, float s) {
    std::vector<__nv_bfloat16> h(n);
    for (auto& x : h) x = __float2bfloat16(rndf() * s);
    void* p = dalloc(n * 2); CK(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice)); return p;
}
void* bytes_rand(size_t n, uint8_t mask) {
    std::vector<uint8_t> h(n);
    for (auto& x : h) x = (uint8_t)rnd32() & mask;
    void* p = dalloc(n); CK(cudaMemcpy(p, h.data(), n, cudaMemcpyHostToDevice)); return p;
}
void* f32_const(std::vector<float> v) { void* p = dalloc(v.size() * 4); CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return p; }

ik_tensor T(void* p, int dt, std::vector<int64_t> sh) {
    ik_tensor t; memset(&t, 0, sizeof t);
    t.data = p; t.dtype = dt; t.ndim = (int)sh.size();
    int64_t acc = 1;
    for (int i = 0; i < IK_MAX_DIMS; ++i) { t.shape[i] = 1; t.stride[i] = 0; }
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[i]; t.stride[i] = acc; acc *= sh[i]; }
    return t;
}
const int BF = IK_DTYPE_BF16, F8 = IK_DTYPE_FP8_E4M3, F32 = IK_DTYPE_F32, I32 = IK_DTYPE_I32, U8 = IK_DTYPE_U8;

struct Fp8W { void* w; void* sc; int n, k; };
Fp8W fp8w(int n, int k) { return {bytes_rand((size_t)n * k, 0x77), f32_const({0.01f, 0.02f}), n, k}; }
struct Bf16W { void* w; int n, k; };
Bf16W bfw(int n, int k) { return {bf16_rand((size_t)n * k, 0.05f), n, k}; }
struct Nv4W { void* w; void* bs; void* gs; int n, k; };
Nv4W nv4(int n, int k) { return {bytes_rand((size_t)n * k / 2, 0xff), bytes_rand((size_t)n * k / 16, 0x77), f32_const({0.003f}), n, k}; }

struct Layer {
    bool gdn;
    void *pre_attn, *pre_moe;  // norm weights
    // gdn
    Fp8W qkv, z, out; Bf16W b, a; void *conv, *a_log, *dt_bias, *gnorm, *state;
    // attn
    Fp8W q, k, v, o; void *qn, *kn, *kc, *vc;
    // moe (dense parts)
    Bf16W router, sgate; Nv4W sg, su, sd;
};

struct Model {
    std::vector<Layer> L;
    void *embed, *final_norm, *ids, *pos, *tok, *logits;
    Nv4W head;
    // activations
    void *h[2], *n, *blk, *qg, *gate, *q[2], *k[2], *v, *qkv, *zz, *aa, *bb, *conv, *core, *normed, *moe_logits, *topk_w, *topk_id,
        *routed, *sh_g, *sh_u, *sh_a, *sh_out, *sh_logit;
    // shared routed experts
    void *gu_w, *gu_s, *dn_w, *dn_s, *gscales;
    ik_scratch moe_scratch, attn_scratch;
    int cap;
};

Model build(int cap) {
    Model m; m.cap = cap;
    for (int i = 0; i < NL; ++i) {
        Layer l; memset(&l, 0, sizeof l);
        l.gdn = (i % 4) != 3;
        l.pre_attn = bf16_rand(H, 1.f); l.pre_moe = bf16_rand(H, 1.f);
        if (l.gdn) {
            l.qkv = fp8w(CONV_C, H); l.z = fp8w(VDIM, H); l.out = fp8w(H, VDIM);
            l.b = bfw(HV, H); l.a = bfw(HV, H);
            l.conv = bf16_rand((size_t)CONV_C * CONVK, 0.3f); l.a_log = bf16_rand(HV, 0.5f); l.dt_bias = bf16_rand(HV, 0.5f);
            l.gnorm = bf16_rand(DV, 1.f);
            l.state = dalloc((size_t)CONV_C * (CONVK - 1) * 4 + 256 + (size_t)HV * DV * DK * 4);
        } else {
            l.q = fp8w(NH * HD * 2, H); l.k = fp8w(NKV * HD, H); l.v = fp8w(NKV * HD, H); l.o = fp8w(H, NH * HD);
            l.qn = bf16_rand(HD, 1.f); l.kn = bf16_rand(HD, 1.f);
            l.kc = bf16_rand((size_t)NKV * cap * HD, 1.f); l.vc = bf16_rand((size_t)NKV * cap * HD, 1.f);
        }
        l.router = bfw(NE, H); l.sgate = bfw(1, H);
        l.sg = nv4(SI, H); l.su = nv4(SI, H); l.sd = nv4(H, SI);
        m.L.push_back(l);
    }
    m.embed = bf16_rand((size_t)1024 * H, 1.f);  // gather: only the row read matters
    m.final_norm = bf16_rand(H, 1.f);
    m.head = nv4(VOC, H);
    m.ids = dalloc(16); m.pos = dalloc(16); m.tok = dalloc(16); m.logits = dalloc((size_t)VOC * 4);
    for (int i = 0; i < 2; ++i) { m.h[i] = bf16_rand(H, 1.f); m.q[i] = dalloc(NH * HD * 2); m.k[i] = dalloc(NKV * HD * 2); }
    m.n = dalloc(H * 2); m.blk = dalloc(H * 2); m.qg = dalloc(NH * HD * 2 * 2); m.gate = dalloc(NH * HD * 2); m.v = dalloc(NKV * HD * 2);
    m.qkv = dalloc(CONV_C * 2); m.zz = dalloc(VDIM * 2); m.aa = dalloc(HV * 2); m.bb = dalloc(HV * 2); m.conv = dalloc(CONV_C * 2);
    m.core = dalloc(VDIM * 2); m.normed = dalloc(VDIM * 2);
    m.moe_logits = dalloc(NE * 2); m.topk_w = dalloc(TOPK * 4); m.topk_id = dalloc(TOPK * 4);
    m.routed = dalloc(H * 2); m.sh_g = dalloc(SI * 2); m.sh_u = dalloc(SI * 2); m.sh_a = dalloc(SI * 2); m.sh_out = dalloc(H * 2); m.sh_logit = dalloc(16);
    m.gu_w = bytes_rand((size_t)NE * 2 * EI * H / 2, 0xff); m.gu_s = bytes_rand((size_t)NE * 2 * EI * H / 16, 0x77);
    m.dn_w = bytes_rand((size_t)NE * H * EI / 2, 0xff); m.dn_s = bytes_rand((size_t)NE * H * EI / 16, 0x77);
    {
        std::vector<float> g(NE * 3, 0.002f);
        m.gscales = f32_const(g);
    }
    {  // routing: make the router pick varied experts: ids are written by topk from the random logits
        m.moe_scratch.bytes = 64 << 20; m.moe_scratch.data = dalloc(m.moe_scratch.bytes);
        int nsplit = 64; (void)nsplit;
        m.attn_scratch.bytes = (int64_t)1 * NH * 64 * (HD + 2) * 4; m.attn_scratch.data = dalloc(m.attn_scratch.bytes);
    }
    return m;
}

// ---- the step ----------------------------------------------------------------------------------------------
struct Ctx { Model* m; cudaStream_t s; int pos0; bool dev_pos; cudaStream_t s2; bool branches; cudaEvent_t ev[8]; int evn; };
// fork: stream s2 waits for everything queued on s so far; join: s waits for everything queued on s2
void fork_(Ctx& c) { cudaEvent_t e = c.ev[c.evn++ % 8]; CK(cudaEventRecord(e, c.s)); CK(cudaStreamWaitEvent(c.s2, e, 0)); }
void join_(Ctx& c) { cudaEvent_t e = c.ev[c.evn++ % 8]; CK(cudaEventRecord(e, c.s2)); CK(cudaStreamWaitEvent(c.s, e, 0)); }

#define LAUNCH(call) IKOK(call)

void gemv_fp8(Ctx& c, const Fp8W& w, void* x, void* out) {
    ik_tensor tx = T(x, BF, {1, w.k}), tw = T(w.w, F8, {w.n, w.k}), ts = T(w.sc, F32, {2}), to = T(out, BF, {1, w.n});
    LAUNCH(ik_matmul_fp8_gemv(&tx, &tw, &ts, &to, c.s));
}
void gemv_bf(Ctx& c, const Bf16W& w, void* x, void* out) {
    ik_tensor tx = T(x, BF, {1, w.k}), tw = T(w.w, BF, {w.n, w.k}), to = T(out, BF, {1, w.n});
    LAUNCH(ik_matmul_bf16_gemv(&tx, &tw, &to, c.s));
}
void gemv_nv4(Ctx& c, const Nv4W& w, void* x, void* out, int odt) {
    ik_tensor tx = T(x, BF, {1, w.k}), tw = T(w.w, U8, {w.n, w.k / 2}), tb = T(w.bs, F8, {w.n, w.k / 16}), tg = T(w.gs, F32, {1}),
              to = T(out, odt, {1, w.n});
    LAUNCH(ik_matmul_nvfp4_gemv(&tx, &tw, &tb, &tg, &to, c.s));
}
void rmsnorm_add(Ctx& c, void* blk, void* res, void* w, void* out, void* sum) {
    ik_tensor a = T(blk, BF, {1, H}), r = T(res, BF, {1, H}), tw = T(w, BF, {H}), o = T(out, BF, {1, H}), s = T(sum, BF, {1, H});
    LAUNCH(ik_rmsnorm_add_bf16(&a, &r, &tw, 1e-6f, 1.0f, &o, &s, c.s));
}
void silu(Ctx& c, void* g, void* u, void* out, int rows, int cols) {
    ik_tensor a = T(g, BF, {rows, cols}), b = T(u, BF, {rows, cols}), o = T(out, BF, {rows, cols});
    LAUNCH(ik_silu_glu_bf16(&a, &b, &o, c.s));
}

void moe_block(Ctx& c, Layer& l, void* n, void* out) {
    Model& m = *c.m;
    if (c.branches) {
        // shared expert on a second stream, in parallel with router -> topk -> routed experts
        fork_(c);
        Ctx b = c; b.s = c.s2;
        gemv_nv4(b, l.sg, n, m.sh_g, BF);
        gemv_nv4(b, l.su, n, m.sh_u, BF);
        silu(b, m.sh_g, m.sh_u, m.sh_a, 1, SI);
        gemv_nv4(b, l.sd, m.sh_a, m.sh_out, BF);
        gemv_bf(b, l.sgate, n, m.sh_logit);
        c.evn = b.evn;
    }
    gemv_bf(c, l.router, n, m.moe_logits);
    {
        ik_tensor tl = T(m.moe_logits, BF, {1, NE}), tw = T(m.topk_w, F32, {1, TOPK}), ti = T(m.topk_id, I32, {1, TOPK});
        LAUNCH(ik_moe_topk_bf16(&tl, TOPK, 1, &tw, &ti, c.s));
        ik_tensor tx = T(n, BF, {1, H}), tgu = T(m.gu_w, U8, {NE, 2 * EI, H / 2}), tgs = T(m.gu_s, F8, {NE, 2 * EI, H / 16}),
                  tdn = T(m.dn_w, U8, {NE, H, EI / 2}), tds = T(m.dn_s, F8, {NE, H, EI / 16}), tgsc = T(m.gscales, F32, {NE, 3}),
                  to = T(m.routed, BF, {1, H});
        LAUNCH(ik_moe_experts_bf16(&tx, &ti, &tw, &tgu, &tgs, &tdn, &tds, &tgsc, &to, &m.moe_scratch, c.s));
    }
    if (c.branches) join_(c);
    else {
        gemv_nv4(c, l.sg, n, m.sh_g, BF);
        gemv_nv4(c, l.su, n, m.sh_u, BF);
        silu(c, m.sh_g, m.sh_u, m.sh_a, 1, SI);
        gemv_nv4(c, l.sd, m.sh_a, m.sh_out, BF);
        gemv_bf(c, l.sgate, n, m.sh_logit);
    }
    {
        ik_tensor r = T(m.routed, BF, {1, H}), s = T(m.sh_out, BF, {1, H}), g = T(m.sh_logit, BF, {1, 1}), o = T(out, BF, {1, H});
        LAUNCH(ik_moe_combine_bf16(&r, &s, &g, &o, c.s));
    }
}

void gdn_block(Ctx& c, Layer& l, void* n, void* out) {
    Model& m = *c.m;
    if (c.branches) {
        fork_(c);
        Ctx b = c; b.s = c.s2;
        gemv_fp8(b, l.z, n, m.zz);
        gemv_bf(b, l.b, n, m.bb);
        gemv_bf(b, l.a, n, m.aa);
        gemv_fp8(c, l.qkv, n, m.qkv);
    } else {
        gemv_fp8(c, l.qkv, n, m.qkv);
        gemv_fp8(c, l.z, n, m.zz);
        gemv_bf(c, l.b, n, m.bb);
        gemv_bf(c, l.a, n, m.aa);
    }
    void* conv_state = l.state;
    void* s_state = (char*)l.state + ((size_t)CONV_C * (CONVK - 1) * 4 + 255) / 256 * 256;
    {
        ik_tensor x = T(m.qkv, BF, {1, CONV_C}), w = T(l.conv, BF, {CONV_C, CONVK}), y = T(m.conv, BF, {1, CONV_C}), st = T(conv_state, F32, {CONV_C, CONVK - 1});
        LAUNCH(ik_gdn_conv_bf16(&x, &w, &y, &st, c.s));
        if (c.branches) join_(c);
        ik_tensor qkv = T(m.conv, BF, {1, CONV_C}), a = T(m.aa, BF, {1, HV}), b = T(m.bb, BF, {1, HV}), al = T(l.a_log, BF, {HV}),
                  dt = T(l.dt_bias, BF, {HV}), o = T(m.core, BF, {1, HV, DV}), ss = T(s_state, F32, {HV, DV, DK});
        LAUNCH(ik_gdn_delta_rule_bf16(&qkv, &a, &b, &al, &dt, 1e-6f, &o, &ss, c.s));
        ik_tensor xr = T(m.core, BF, {HV, DV}), gw = T(l.gnorm, BF, {DV}), nr = T(m.normed, BF, {HV, DV});
        LAUNCH(ik_rmsnorm_bf16(&xr, &gw, 1e-6f, 0.0f, &nr, c.s));
    }
    silu(c, m.zz, m.normed, m.core, HV, DV);
    gemv_fp8(c, l.out, m.core, out);
}

void attn_block(Ctx& c, Layer& l, void* n, void* out) {
    Model& m = *c.m;
    gemv_fp8(c, l.q, n, m.qg);
    {
        ik_tensor x = T(m.qg, BF, {1, NH, 2 * HD}), q = T(m.q[0], BF, {1, NH, HD}), g = T(m.gate, BF, {1, NH, HD});
        LAUNCH(ik_gate_split_bf16(&x, &q, &g, c.s));
    }
    ik_tensor tpos = T(m.pos, I32, {1});
    const ik_tensor* pos = c.dev_pos ? &tpos : nullptr;
    int64_t p0 = c.dev_pos ? 0 : c.pos0;
    ik_tensor q0 = T(m.q[0], BF, {1, NH, HD}), q1 = T(m.q[1], BF, {1, NH, HD}), qw = T(l.qn, BF, {HD});
    LAUNCH(ik_qk_norm_bf16(&q0, &qw, 1e-6f, 1.0f, &q1, c.s));
    LAUNCH(ik_rope_bf16(&q1, p0, 1e7f, ROT, &q0, pos, c.s));
    gemv_fp8(c, l.k, n, m.k[0]);
    ik_tensor k0 = T(m.k[0], BF, {1, NKV, HD}), k1 = T(m.k[1], BF, {1, NKV, HD}), kw = T(l.kn, BF, {HD});
    LAUNCH(ik_qk_norm_bf16(&k0, &kw, 1e-6f, 1.0f, &k1, c.s));
    LAUNCH(ik_rope_bf16(&k1, p0, 1e7f, ROT, &k0, pos, c.s));
    gemv_fp8(c, l.v, n, m.v);
    ik_tensor kc = T(l.kc, BF, {NKV, c.m->cap, HD}), vc = T(l.vc, BF, {NKV, c.m->cap, HD}), vn = T(m.v, BF, {1, NKV, HD});
    LAUNCH(ik_kv_append_bf16(&k0, p0, &kc, pos, c.s));
    LAUNCH(ik_kv_append_bf16(&vn, p0, &vc, pos, c.s));
    ik_tensor ao = T(m.q[1], BF, {1, NH, HD});
    LAUNCH(ik_attn_decode_bf16_splitkv(&q0, &kc, &vc, p0, 1.f / 16.f, &ao, &m.attn_scratch, pos, c.s));
    {
        ik_tensor a = T(m.q[1], BF, {NH * HD}), g = T(m.gate, BF, {NH * HD}), o = T(m.q[0], BF, {NH * HD});
        a.ndim = 2; a.shape[0] = 1; a.shape[1] = NH * HD; a.stride[1] = 1; a.stride[0] = NH * HD;
        g.ndim = 2; g.shape[0] = 1; g.shape[1] = NH * HD; g.stride[1] = 1; g.stride[0] = NH * HD;
        o.ndim = 2; o.shape[0] = 1; o.shape[1] = NH * HD; o.stride[1] = 1; o.stride[0] = NH * HD;
        LAUNCH(ik_sigmoid_mul_bf16(&a, &g, &o, c.s));
    }
    gemv_fp8(c, l.o, m.q[0], out);
}

int g_launch_count = 0;
void step(Ctx& c) {
    Model& m = *c.m;
    {
        ik_tensor ids = T(m.ids, I32, {1}), tab = T(m.embed, BF, {1024, H}), o = T(m.h[0], BF, {1, H});
        LAUNCH(ik_embedding_bf16(&ids, &tab, &o, c.s));
    }
    int cur = 0; bool pending = false;
    for (int i = 0; i < NL; ++i) {
        Layer& l = m.L[i];
        // block norm
        auto norm_in = [&](void* w) {
            if (pending) {
                int nx = 1 - cur;
                rmsnorm_add(c, m.blk, m.h[cur], w, m.n, m.h[nx]);
                cur = nx;
            } else {
                ik_tensor x = T(m.h[cur], BF, {1, H}), tw = T(w, BF, {H}), o = T(m.n, BF, {1, H});
                LAUNCH(ik_rmsnorm_bf16(&x, &tw, 1e-6f, 1.0f, &o, c.s));
            }
        };
        norm_in(l.pre_attn);
        if (l.gdn) gdn_block(c, l, m.n, m.blk); else attn_block(c, l, m.n, m.blk);
        pending = true;
        norm_in(l.pre_moe);
        moe_block(c, l, m.n, m.blk);
    }
    {
        int nx = 1 - cur;
        rmsnorm_add(c, m.blk, m.h[cur], m.final_norm, m.n, m.h[nx]);
        cur = nx;
    }
    gemv_nv4(c, m.head, m.n, m.logits, F32);
    ik_tensor lg = T(m.logits, F32, {1, VOC}), tk = T(m.tok, I32, {1});
    LAUNCH(ik_argmax_f32(&lg, &tk, c.s));
}

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

int main(int argc, char** argv) {
    int ctx = 1024, steps = 200; std::string mode = "both";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
    }
    CK(cudaSetDevice(0));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Model m = build(ctx + steps + 64);
    printf("built %d layers (%d GDN, %d attention), ctx %d\n", NL, 30, 10, ctx);
    // routing values: random logits come from random router weights x random hidden: fine
    cudaStream_t s2; CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    bool branches = false;
    auto run = [&](bool graph, const char* name) {
        Ctx c{&m, s, ctx, graph, s2, branches, {}, 0};
        for (auto& e : c.ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        void* exec = nullptr;
        double t_cap = 0;
        if (graph) {
            int32_t p = ctx; CK(cudaMemcpy(m.pos, &p, 4, cudaMemcpyHostToDevice));
            double t0 = now_ms();
            IKOK(ik_rt_graph_begin(s)); step(c); IKOK(ik_rt_graph_end(s, &exec));
            t_cap = now_ms() - t0;
        }
        // warmup + timed
        std::vector<double> ms;
        for (int i = -5; i < steps; ++i) {
            int32_t id = 100 + (i & 7), p = ctx + (i < 0 ? 0 : i);
            c.pos0 = p;
            double t0 = now_ms();
            CK(cudaMemcpyAsync(m.ids, &id, 4, cudaMemcpyHostToDevice, s));
            if (graph) { CK(cudaMemcpyAsync(m.pos, &p, 4, cudaMemcpyHostToDevice, s)); IKOK(ik_rt_graph_launch(exec, s)); }
            else step(c);
            int32_t tok; CK(cudaMemcpyAsync(&tok, m.tok, 4, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
            if (i >= 0) ms.push_back(now_ms() - t0);
        }
        double sum = 0, mn = 1e9; for (double x : ms) { sum += x; mn = x < mn ? x : mn; }
        std::sort(ms.begin(), ms.end());
        printf("%-6s: mean %.3f ms  median %.3f ms  min %.3f ms  => %.1f tok/s (median)%s\n", name, sum / ms.size(), ms[ms.size() / 2], mn,
               1000.0 / ms[ms.size() / 2], graph ? "" : "");
        if (graph) printf("        capture took %.1f ms\n", t_cap);
        if (exec) ik_rt_graph_destroy(exec);
    };
    if (mode == "eager" || mode == "both") run(false, "eager");
    if (mode == "graph" || mode == "both") run(true, "graph");
    if (mode == "branches" || mode == "both") { branches = true; run(true, "graph+branches"); }
    return 0;
}

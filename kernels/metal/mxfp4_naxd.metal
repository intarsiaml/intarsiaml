// matmul_mxfp4_nax with the weights decoded inside the GEMM (lane mprefill, Mac roadmap idea #6; opt-in
// `INTARSIA_METAL_MXFP4_NAXD=1`, launcher mxfp4.rs `launch_mxfp4_nax`). Compiled -std=metal4.0 into intarsia_metal4.metallib
// (build.rs METAL4_SOURCES).
//
// The shipped form decodes the whole weight to bf16 scratch per call (`mxfp4_w_to_bf16`, about 0.5 ms per 17408 x 5120
// call, flat in M) and then runs `gemm_nax_fp8_t` on device operands. Here a threadgroup owns a BM x BN output tile and walks
// K in steps of BK; per step its threads decode the BN x BK weights of the step (16 code bytes + 1 e8m0 byte per 32, the
// exact bf16 values `mxfp4_w_to_bf16` writes) into one of two threadgroup buffers, and the NSG simdgroups run
// mpp::tensor_ops::matmul2d (multiply-accumulate) with A read from device memory (the QDQ'd / copied activations of
// `mxfp4_xprep_bf16`, as before) and B from that buffer. Two buffers: one barrier per step, the next step's weight words are
// fetched before the accelerator runs. Epilogue as gemm_nax_fp8.metal (float destination, bf16 round-to-nearest-even or f32;
// the scratch path's scale pair is [1, 0], i.e. x 1.0).
//
// Numerics: same products (bf16 x bf16, exact) as the scratch path; whether the accelerator's K order makes the result
// bitwise that path is a test (tests/mxfp4_nax.rs `naxd_*`), not an assumption.
//
// bind: 0 A [M,K] bf16 (16-byte aligned), 1 W u8 [N,K/2] (4-byte aligned), 2 scales u8 [N,K/32], 3 C [M,N] bf16 / f32,
// 4 NaxdP. K % BK == 0. Grid: (ceil(N/BN) << swz_log, ceil(ceil(M/BM) / 2^swz_log)) threadgroups of NSG*32 threads.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct NaxdP {
  uint m, n, k, swz_log;
  uint s_c, s_s, pad0, pad1;
};

constant float kNdFp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// e8m0 -> float as `mx_e8m0` (mxfp4.metal): 2^(e - 127), 0 -> 2^-127 (subnormal), 255 -> NaN
inline float nd_e8m0(uint e) {
  return e == 0u ? as_type<float>(0x00400000u) : (e == 255u ? as_type<float>(0x7FC00000u) : as_type<float>(e << 23));
}

// one 32-weight group (16 code bytes) -> 32 bf16 at dst (16-byte aligned), the bits of `mxfp4_w_to_bf16`
inline void nd_dec32(uint4 wv, float s, threadgroup uint4* dst) {
  uint o[16];
  for (int q = 0; q < 4; ++q) {
    const uint v = wv[q];
    for (int b = 0; b < 4; ++b) {
      const float lo = kNdFp4[(v >> (8 * b)) & 0xFu] * s, hi = kNdFp4[(v >> (8 * b + 4)) & 0xFu] * s;
      o[q * 4 + b] = (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xFFFF0000u);
    }
  }
  for (int i = 0; i < 4; ++i) dst[i] = uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}

template <typename OutT, int BM, int BN, int BK, int NSG, typename SA, typename SC>
METAL_FUNC void naxd_tile(SA tA, SC sC, device const uchar* W, device const uchar* S, threadgroup bfloat* Bs, int n0,
                          int m0, int N, int K, uint tid) {
  constexpr int LD = BK + 8;          // bf16 per staged weight row (16-byte multiple)
  constexpr int NT = NSG * 32;
  constexpr int GPS = BN * BK / 32;   // 32-weight groups per step
  constexpr int GPT = (GPS + NT - 1) / NT;
  constexpr int GPR = BK / 32;        // groups per staged row and step
  // this thread's groups of each step: (row, group-in-row) fixed, K offset moves
  int gr[GPT], gq[GPT];
  bool gok[GPT];
  for (int j = 0; j < GPT; ++j) {
    const int g = int(tid) + NT * j;
    gr[j] = g / GPR;
    gq[j] = g % GPR;
    gok[j] = g < GPS && n0 + gr[j] < N;
  }
  uint4 wv[GPT];
  uint sv[GPT];
  auto fetch = [&](int k0) {
    for (int j = 0; j < GPT; ++j) {
      if (gok[j]) {
        const ulong row = ulong(n0 + gr[j]);
        const int g = k0 / 32 + gq[j];
        wv[j] = uint4(*(device const packed_uint4*)(W + row * ulong(K / 2) + ulong(g) * 16u));
        sv[j] = S[row * ulong(K / 32) + ulong(g)];
      } else {
        wv[j] = uint4(0u);
        sv[j] = 127u;
      }
    }
  };
  using ext_b = extents<int, BK, BN>;
  using ten_b = tensor<threadgroup bfloat, ext_b, tensor_inline>;
  constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  using TA = decltype(tA.template slice<BK, dynamic_extent>(0, 0));
  auto ct = op.template get_destination_cooperative_tensor<TA, ten_b, float>();
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < ct.get_capacity(); ++i) ct[i] = 0.f;
  fetch(0);
  array<int, 2> st = {1, LD};
  ext_b eb;
  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup bfloat* B = Bs + buf * BN * LD;
    for (int j = 0; j < GPT; ++j) {
      if (int(tid) + NT * j < GPS) nd_dec32(wv[j], nd_e8m0(sv[j]), (threadgroup uint4*)(B + gr[j] * LD + gq[j] * 32));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + BK < K) fetch(k0 + BK);
    ten_b tB(B, eb, st);
    auto sA = tA.template slice<BK, dynamic_extent>(k0, m0);
    op.run(sA, tB, ct);
    buf ^= 1;
  }
  if constexpr (is_same_v<OutT, float>) {
    ct.store(sC);
  } else {
    auto cb = op.template get_destination_cooperative_tensor<TA, ten_b, bfloat>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
      if (ct.is_valid_element(i)) cb[i] = static_cast<bfloat>(ct[i]);
    }
    cb.store(sC);
  }
}

template <typename OutT, int BM, int BN, int BK, int NSG>
kernel void gemm_naxd_mxfp4_t(device bfloat* A [[buffer(0)]], device const uchar* W [[buffer(1)]],
                              device const uchar* S [[buffer(2)]], device OutT* C [[buffer(3)]],
                              constant NaxdP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat Bs[2 * BN * (BK + 8)];
  const int M = int(p.m), N = int(p.n), K = int(p.k);
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int ty = int((tg.y << p.swz_log) + (tg.x & ((1u << p.swz_log) - 1)));
  const int tx = int(tg.x >> p.swz_log);
  if (tx >= tiles_n || ty >= tiles_m) return;
  const int m0 = ty * BM, n0 = tx * BN;
  tensor<device bfloat, dextents<int, 2>, tensor_inline> tA(A, dextents<int, 2>(K, M));
  tensor<device OutT, dextents<int, 2>, tensor_inline> tC(C + p.s_c, dextents<int, 2>(N, M));
  if (m0 + BM <= M && n0 + BN <= N) {
    naxd_tile<OutT, BM, BN, BK, NSG>(tA, tC.template slice<BN, BM>(n0, m0), W, S + p.s_s, Bs, n0, m0, N, K, tid);
  } else {
    naxd_tile<OutT, BM, BN, BK, NSG>(tA, tC.slice(n0, m0), W, S + p.s_s, Bs, n0, m0, N, K, tid);
  }
}

#define NAXD_T(NAME, OUT, BM, BN, BK, NSG) \
  template [[host_name(NAME)]] kernel void gemm_naxd_mxfp4_t<OUT, BM, BN, BK, NSG>( \
      device bfloat*, device const uchar*, device const uchar*, device OUT*, constant NaxdP&, uint2, uint);

// Keep in step with NAXD_FORMS in src/mxfp4.rs.
NAXD_T("gemm_naxd_mxfp4_bf16_t64x64k64w4", bfloat, 64, 64, 64, 4)
NAXD_T("gemm_naxd_mxfp4_f32_t64x64k64w4", float, 64, 64, 64, 4)
NAXD_T("gemm_naxd_mxfp4_bf16_t128x64k64w4", bfloat, 128, 64, 64, 4)
NAXD_T("gemm_naxd_mxfp4_f32_t128x64k64w4", float, 128, 64, 64, 4)
NAXD_T("gemm_naxd_mxfp4_bf16_t128x128k32w8", bfloat, 128, 128, 32, 8)
NAXD_T("gemm_naxd_mxfp4_f32_t128x128k32w8", float, 128, 128, 32, 8)
NAXD_T("gemm_naxd_mxfp4_bf16_t64x64k64w2", bfloat, 64, 64, 64, 2)
NAXD_T("gemm_naxd_mxfp4_f32_t64x64k64w2", float, 64, 64, 64, 2)

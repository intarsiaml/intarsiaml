//! The kernel table: which Metal kernel implements which op, for which shapes, and how closely it must match the
//! reference. Mirrors `intarsia-backend-hip/src/kernels.rs`. The shader names are the `host_name`s in
//! kernels/metal/*.metal.

use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

/// bf16-rounded outputs: a one-ulp flip is allowed (same budget as the cpu and hip backends).
const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);

/// f32 outputs of bf16 inputs: only accumulation-order noise (same as the hip backend's logits).
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Budget of `matmul_pair_bf16_gemv`, the same as the hip kernel's (`TOL` in intarsia-backend-hip/src/quantmoe.rs).
const PAIR_TOL: Tol = Tol::new(1.5e-2, 1e-3);

/// Budget of the MoE kernels, the hip ones' (`TOL` in intarsia-backend-hip/src/quantmoe.rs).
const MOE_TOL: Tol = Tol::new(1.5e-2, 1e-3);

/// `KernelImpl::name` of the batch-1 bf16 GEMV (the HIP backend calls the same thing `matmul_bf16_gemv`).
pub const MATMUL_BF16_GEMV: &str = "matmul_bf16_gemv";

/// A vectorised GEMV shader variant (an instantiation of `gemv_bf16_t<ROWS, UNROLL, SG>`, or of the row-per-threadgroup
/// `gemv_bf16_rt<UNROLL, SG>` when `row_tg`).
#[derive(Clone, Copy, Debug)]
pub struct GemvVariant {
    pub name: &'static str,
    /// Output rows per simdgroup.
    pub rows: u32,
    /// 16-byte weight chunks per row each lane loads per step.
    pub unroll: u32,
    /// Simdgroups (32 lanes) per threadgroup.
    pub sg: u32,
    /// Row-per-threadgroup form: all `sg` simdgroups of a threadgroup work on the same row (`rows` is 1).
    pub row_tg: bool,
}

impl GemvVariant {
    /// Output rows per threadgroup.
    pub const fn rows_per_tg(&self) -> u32 {
        if self.row_tg {
            self.rows
        } else {
            self.rows * self.sg
        }
    }
    pub const fn threads_per_tg(&self) -> u32 {
        self.sg * 32
    }
}

macro_rules! variant {
    ($r:literal, $u:literal, $s:literal) => {
        GemvVariant { name: concat!("gemv_bf16_r", $r, "u", $u, "s", $s), rows: $r, unroll: $u, sg: $s, row_tg: false }
    };
}

macro_rules! variant_rt {
    ($u:literal, $s:literal) => {
        GemvVariant { name: concat!("gemv_bf16_rt_u", $u, "s", $s), rows: 1, unroll: $u, sg: $s, row_tg: true }
    };
}
macro_rules! variant_rt_a8 {
    ($u:literal, $s:literal) => {
        GemvVariant { name: concat!("gemv_bf16_rt_a8u", $u, "s", $s), rows: 1, unroll: $u, sg: $s, row_tg: true }
    };
}

/// Every vectorised variant compiled into the metallib (needs K % 8 == 0 and 16-byte aligned W, x, y).
pub const GEMV_VARIANTS: &[GemvVariant] = &[
    variant!(1, 1, 8),
    variant!(1, 2, 8),
    variant!(1, 4, 4),
    variant!(1, 4, 8),
    variant!(1, 4, 16),
    variant!(1, 8, 8),
    variant!(2, 1, 4),
    variant!(2, 2, 4),
    variant!(2, 4, 4),
    variant!(4, 1, 4),
    variant!(4, 2, 4),
    variant!(8, 1, 2),
    variant!(8, 2, 2),
    variant_rt!(2, 8),
    variant_rt!(4, 4),
    variant_rt!(2, 4),
    variant_rt!(1, 16),
];

/// The variant `matmul_bf16_gemv` launches when the operands allow vector loads. Phase B part 3: one row per threadgroup
/// of 8 simdgroups (256 lanes, two 16-byte loads each, so a K=4096 row is one load round trip and a threadgroup, the unit
/// whose tail ends a dispatch, is short); docs/results/raw/2026-10-14-metal-phase-b/b3/. The spike's choice was
/// `variant!(1, 4, 4)` (one row per simdgroup, four loads per lane), still compiled in as `GEMV_VARIANTS[2]`.
pub const DEFAULT_GEMV: GemvVariant = variant_rt!(2, 8);

/// 8-byte-load variants (`gemv_bf16_a8_t<UNROLL, SG>`, one row per simdgroup) for K % 4 == 0 and operands that are
/// 8-byte but not 16-byte aligned, e.g. every weight of a safetensors shard whose header length is 8 mod 16.
pub const GEMV_A8_VARIANTS: &[GemvVariant] = &[
    GemvVariant { name: "gemv_bf16_a8u4s4", rows: 1, unroll: 4, sg: 4, row_tg: false },
    GemvVariant { name: "gemv_bf16_a8u8s4", rows: 1, unroll: 8, sg: 4, row_tg: false },
    GemvVariant { name: "gemv_bf16_a8u8s8", rows: 1, unroll: 8, sg: 8, row_tg: false },
    variant_rt_a8!(4, 8),
    variant_rt_a8!(8, 4),
];
/// The 8-byte-load default, the row-per-threadgroup form (`gemv_bf16_rt_a8`): the same eight simdgroups, each lane four
/// 8-byte loads, i.e. the same bytes in flight per row as `DEFAULT_GEMV`.
pub const DEFAULT_GEMV_A8: GemvVariant = variant_rt_a8!(4, 8);

/// Fallback for any K and any 2-byte alignment: one simdgroup per row, 8 rows per threadgroup, 2-byte loads.
pub const GEMV_SCALAR: GemvVariant = GemvVariant { name: "gemv_bf16_scalar", rows: 1, unroll: 1, sg: 8, row_tg: false };

/// `MatmulPair` (the optional fused decode op): two bf16 GEMVs of one `x` in one dispatch. The shader
/// `gemv_pair_bf16_rt_u2s8` is `gemv_bf16_rt_u2s8` (`DEFAULT_GEMV`) over two weight matrices, so each output is bitwise
/// the `Matmul` GEMV's.
pub const MATMUL_PAIR_BF16_GEMV: &str = "matmul_pair_bf16_gemv";
pub const GEMV_PAIR_BF16: &str = "gemv_pair_bf16_rt_u2s8";
/// The 8-byte-load form (`DEFAULT_GEMV_A8` rows: K % 4 == 0, operands 8-byte aligned).
pub const GEMV_PAIR_BF16_A8: &str = "gemv_pair_bf16_rt_a8u4s8";
/// Rows per threadgroup of the pair shaders (one row per threadgroup, like `DEFAULT_GEMV`) and their threads.
pub const GEMV_PAIR_ROWS_PER_TG: u32 = DEFAULT_GEMV.rows_per_tg();
pub const GEMV_PAIR_THREADS: u32 = DEFAULT_GEMV.threads_per_tg();
/// The spike's one-row-per-simdgroup pair shaders (phase B part 2), kept compiled in for comparison: bitwise
/// `gemv_bf16_r1u4s4` / `gemv_bf16_a8u8s4`, 4 rows and 128 threads per threadgroup.
pub const GEMV_PAIR_BF16_R1: &str = "gemv_pair_bf16";
pub const GEMV_PAIR_BF16_R1_A8: &str = "gemv_pair_bf16_a8";

/// bf16 GEMM, any M; the shader is `gemm_bf16_t<OutT>` with a bf16 (`Matmul`) or f32 (`Logits`) output.
pub const MATMUL_BF16: &str = "matmul_bf16";
pub const LOGITS_BF16: &str = "logits_bf16";
/// `Logits` with M == 1: the f32-output GEMV.
pub const LOGITS_BF16_GEMV: &str = "logits_bf16_gemv";
pub const GEMM_O_BF16: &str = "gemm_bf16_o_bf16";
pub const GEMM_O_F32: &str = "gemm_bf16_o_f32";
pub const GEMV_F32O_V: &str = "gemv_f32o_v";
pub const GEMV_F32O_SCALAR: &str = "gemv_f32o_scalar";

/// Dense elementwise / row ops. The kernel names are the hip launcher names (without `ik_`) and the shaders'
/// `host_name`s, one shader per kernel except `qk_norm_bf16`, which runs `rmsnorm_bf16`.
pub const EMBEDDING_BF16: &str = "embedding_bf16";
pub const RMSNORM_BF16: &str = "rmsnorm_bf16";
pub const RMSNORM_ADD_BF16: &str = "rmsnorm_add_bf16";
pub const ADD_BF16: &str = "add_bf16";
pub const QK_NORM_BF16: &str = "qk_norm_bf16";
pub const ROPE_BF16: &str = "rope_bf16";
/// `RopeScaled` (llama3 / yarn): `rope_bf16` with a frequency table input and a cos / sin multiplier.
pub const ROPE_SCALED_BF16: &str = "rope_scaled_bf16";
/// `AddBias`: `x + bias[col]`, one rounding.
pub const ADD_BIAS_BF16: &str = "add_bias_bf16";
pub const SILU_GLU_BF16: &str = "silu_glu_bf16";
pub const KV_APPEND_BF16: &str = "kv_append_bf16";
/// `QkNormRopeKv` (qk_rope_kv.metal): qk-norm, rope and the k / v cache appends of a layer in one dispatch, bitwise the
/// unfused `qk_norm_bf16` (`rmsnorm_bf16_ex1` at head_dim <= 256) + `rope_bf16` + `kv_append_bf16` sequence.
pub const QK_NORM_ROPE_KV_BF16: &str = "qk_norm_rope_kv_bf16";
/// The prefill form of the same registry entry (T > 1, head_dim = rotary_dim = 128), chosen by the launcher: bitwise the
/// decode form, with the 64 (cos, sin) pairs of a token computed once per threadgroup instead of per head and thread.
pub const QK_NORM_ROPE_KV_BF16_PF: &str = "qk_norm_rope_kv_bf16_pf";
/// Gated-attention / GatedDeltaNet helper ops (gdn.metal, launchers gdn.rs); the conv and delta-rule shaders are per
/// tap count / dk (`gdn::CONV_SHADERS`, `gdn::DELTA_SHADERS`), picked by the launcher.
pub const GATE_SPLIT_BF16: &str = "gate_split_bf16";
pub const SIGMOID_MUL_BF16: &str = "sigmoid_mul_bf16";
pub const GDN_CONV_BF16: &str = "gdn_conv_bf16";
pub const GDN_DELTA_RULE_BF16: &str = "gdn_delta_rule_bf16";
/// `GdnDeltaRule` for T >= `gdn::PRE_MIN_T`: the per-token norms and gates once per head instead of once per value row, bitwise
/// `gdn_delta_rule_bf16` (kernels/metal/gdn.metal, tests/gdn.rs).
pub const GDN_DELTA_RULE_BF16_PRE: &str = "gdn_delta_rule_bf16_pre";
pub const ARGMAX_F32: &str = "argmax_f32";
pub const ARGMAX_BF16: &str = "argmax_bf16";
/// Low-latency forms (phase B2b), chosen by the launcher of the op when the operands allow: row norms that issue all
/// loads before the reduction (`rmsnorm_bf16_ex<NIT>`, `rmsnorm_add_bf16_ex<NIT>`, bitwise the same results as the
/// kernels above; `NIT` = elements per thread) and `argmax_f32_v4`. Not separate registry entries. The last three of
/// each norm list are the `_f` forms for H == NIT * threads (no bounds selects, phase B6).
pub const RMSNORM_EX: [&str; 6] = [
    "rmsnorm_bf16_ex1",
    "rmsnorm_bf16_ex16",
    "rmsnorm_bf16_ex32",
    "rmsnorm_bf16_ex1_f",
    "rmsnorm_bf16_ex16_f",
    "rmsnorm_bf16_ex32_f",
];
pub const RMSNORM_ADD_EX: [&str; 6] = [
    "rmsnorm_add_bf16_ex1",
    "rmsnorm_add_bf16_ex16",
    "rmsnorm_add_bf16_ex32",
    "rmsnorm_add_bf16_ex1_f",
    "rmsnorm_add_bf16_ex16_f",
    "rmsnorm_add_bf16_ex32_f",
];
pub const ARGMAX_F32_V4: &str = "argmax_f32_v4";

pub const ATTN_PREFILL_BF16: &str = "attn_prefill_bf16";
/// `AttnPrefill`, flash-style on `simdgroup_matrix` with GQA-shared K / V tiles (attention_flash.metal, launcher flash.rs);
/// ranks above `attn_prefill_bf16`, which it falls back to for misaligned operands.
pub const ATTN_PREFILL_BF16_FLASH: &str = "attn_prefill_bf16_flash";
pub const ATTN_DECODE_BF16: &str = "attn_decode_bf16";
/// `AttnDecode` with 16-byte loads (head_dim a power of two in 8..=256, 16-byte aligned K / V caches); the launcher of
/// `attn_decode_bf16` picks it, it is not a separate registry entry.
pub const ATTN_DECODE_BF16_V8: &str = "attn_decode_bf16_v8";
/// `AttnDecode`, GQA-grouped split-KV (attention_splitkv.metal, launcher attn.rs): stage 1 `attn_decode_bf16_splitkv_g{GP}`
/// (GP = query heads per kv head rounded up to a power of two) and the combine pass. Needs scratch.
pub const ATTN_DECODE_BF16_SPLITKV: &str = "attn_decode_bf16_splitkv";
pub const ATTN_DECODE_SPLITKV_SHADERS: [&str; 5] = [
    "attn_decode_bf16_splitkv_g1",
    "attn_decode_bf16_splitkv_g2",
    "attn_decode_bf16_splitkv_g4",
    "attn_decode_bf16_splitkv_g8",
    "attn_decode_bf16_combine",
];
/// `AttnPrefill` with 2..=8 query rows (speculative-decoding verification), bitwise the 1-row `attn_decode_bf16_splitkv`
/// call per row (attention_splitkv.metal, launcher `attn.rs`, G0b). Registered below the prefill kernels (priority -1).
pub const ATTN_PREFILL_BF16_SPLITKV_ROWS: &str = "attn_prefill_bf16_splitkv_rows";
pub const ATTN_ROWS_SHADERS: [&str; 5] =
    ["attn_rows_splitkv_g1", "attn_rows_splitkv_g2", "attn_rows_splitkv_g4", "attn_rows_splitkv_g8", "attn_rows_combine"];
pub const COPY_U4: &str = "copy_u4";
pub const READ_XOR_U4: &str = "read_xor_u4";
/// `Backend::zero` / `Backend::copy` kernels (copy.metal): 16-byte vector forms plus byte forms for the unaligned rest.
pub const ZERO_U4: &str = "zero_u4";
pub const ZERO_BYTES: &str = "zero_bytes";
pub const COPY_BYTES: &str = "copy_bytes";
/// 16-byte vectors each copy/read thread handles (`PER_THREAD` in copy.metal).
pub const COPY_PER_THREAD: usize = 4;
/// Same for `read_xor_u4` (`READ_PER_THREAD`).
pub const READ_PER_THREAD: usize = 16;

/// Every shader entry point the backend builds a pipeline for.
/// Phase B part 3 experiment shaders (row-per-threadgroup GEMV, templated pair), benchmarked by `examples/gemv_b3.rs`.
pub const EXPERIMENT_SHADERS: &[&str] = &[
    "gemv_bf16_r1u4s1",
    "gemv_bf16_r1u4s2",
    "gemv_bf16_r1u2s4",
    "gemv_bf16_r1u8s4",
    "gemv_bf16_r1u8s2",
    "gemv_bf16_r1u2s2",
    "gemv_bf16_rt_u1s4",
    "gemv_bf16_rt_u2s4",
    "gemv_bf16_rt_u4s4",
    "gemv_bf16_rt_u1s8",
    "gemv_bf16_rt_u2s8",
    "gemv_bf16_rt_u1s2",
    "gemv_bf16_rt_u2s2",
    "gemv_bf16_rt_u4s2",
    "gemv_bf16_rt_u8s2",
    "gemv_bf16_rt_u1s16",
    "gemv_bf16_rt_u2s16",
    "gemv_bf16_rt_u3s16",
    "gemv_bf16_rt_u3s8",
    "gemv_bf16_rt_u6s8",
    "gemv_bf16_rt_u3s4",
    "gemv_bf16_rt_u4s8",
    "gemv_bf16_rt_a8u4s8",
    "gemv_bf16_rt_a8u8s4",
    "gemv_bf16_rt_a8u2s8",
    "gemv_bf16_rt_a8u4s4",
    "gemv_bf16_rt_a8u8s8",
    "gemv_pair_bf16_rt_a8u4s8",
    "gemv_pair_bf16_rt_a8u8s4",
    "gemv_pair_bf16_rt_a8u2s8",
    "gemv_pair_bf16_rt_a8u4s4",
    "gemv_pair_bf16_rt_a8u8s8",
    "gemv_pair_bf16_u4s4",
    "gemv_pair_bf16_u4s2",
    "gemv_pair_bf16_u4s1",
    "gemv_pair_bf16_u2s4",
    "gemv_pair_bf16_u8s4",
    "gemv_pair_bf16_rt_u1s4",
    "gemv_pair_bf16_rt_u2s4",
    "gemv_pair_bf16_rt_u4s4",
    "gemv_pair_bf16_rt_u1s8",
    "gemv_pair_bf16_rt_u2s8",
    "gemv_pair_bf16_rt_u2s2",
    "gemv_pair_bf16_rt_u4s2",
    "gemv_pair_bf16_rt_u1s16",
    "gemv_pair_bf16_rt_u2s16",
    "gemv_pair_bf16_rt_u4s8",
];

/// Entry points of `intarsia_metal4.metallib` (`METAL4_SOURCES` in build.rs, compiled `-std=metal4.0`): one per
/// `gemm_nax::NAX_VARIANTS` entry. Not part of [`shader_names`]: a failure to load them only drops the kernels in
/// [`requires_metal4`], never `MetalBackend::new`.
pub fn metal4_shaders() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = crate::gemm_nax::NAX_VARIANTS.iter().flat_map(|v| [v.shader, v.pair_shader]).collect();
    v.extend(crate::flash_nax::NAX_FLASH_VARIANTS.iter().map(|f| f.name));
    v.extend(crate::fp8::NAX_FP8_SHADERS.iter().flatten());
    v.extend(crate::moe::NAX_MM_GU.iter().chain(&crate::moe::NAX_MM_DN));
    v
}

/// `Op::Matmul`, M >= 16, bf16 on the M5 neural accelerators (`mpp::tensor_ops::matmul2d`, gemm_nax.rs), strict
/// precision. The tile is picked per call (`gemm_nax::pick_variant`); every tile gives the same bits.
pub const MATMUL_BF16_NAX: &str = "matmul_bf16_nax";
/// `Op::MatmulPair` (gate + up, k + v), M >= 16: bitwise two `matmul_bf16_nax` launches in one dispatch.
pub const MATMUL_PAIR_BF16_NAX: &str = "matmul_pair_bf16_nax";
/// Registered kernels that run on a `metal4_shaders()` pipeline; `Backend::kernels` hides them when the metal4.0 library
/// could not be loaded.
/// `AttnPrefill` on the neural accelerators (attention_flash_nax.metal, launcher flash_nax.rs): head_dim 128; ranks above
/// the P2 flash kernel and falls back to it for operands it cannot take.
pub const ATTN_PREFILL_BF16_FLASH_NAX: &str = "attn_prefill_bf16_flash_nax";
pub const NAX_KERNELS: &[&str] =
    &[MATMUL_BF16_NAX, MATMUL_PAIR_BF16_NAX, ATTN_PREFILL_BF16_FLASH_NAX, crate::fp8::MATMUL_FP8_NAX];

pub fn requires_metal4(kernel: &str) -> bool {
    NAX_KERNELS.contains(&kernel)
}

pub fn shader_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = GEMV_VARIANTS.iter().map(|g| g.name).collect();
    v.extend(crate::mxfp4::SHADERS);
    v.extend(crate::gdn::shader_names());
    v.extend(crate::moe::SHADERS);
    v.extend(crate::fp8::SHADERS);
    v.extend(crate::nvfp4::SHADERS);
    v.extend(GEMV_A8_VARIANTS.iter().map(|g| g.name));
    v.extend([GEMV_SCALAR.name, COPY_U4, READ_XOR_U4, ZERO_U4, ZERO_BYTES, COPY_BYTES]);
    v.extend([GEMM_O_BF16, GEMM_O_F32, GEMV_F32O_V, GEMV_F32O_SCALAR]);
    v.extend([EMBEDDING_BF16, RMSNORM_BF16, RMSNORM_ADD_BF16, ADD_BF16, ROPE_BF16, SILU_GLU_BF16, KV_APPEND_BF16]);
    v.extend([ROPE_SCALED_BF16, ADD_BIAS_BF16]);
    v.extend([ARGMAX_F32, ARGMAX_BF16]);
    v.extend([QK_NORM_ROPE_KV_BF16, QK_NORM_ROPE_KV_BF16_PF]);
    v.extend([ATTN_PREFILL_BF16, ATTN_DECODE_BF16, ATTN_DECODE_BF16_V8]);
    v.extend(ATTN_DECODE_SPLITKV_SHADERS);
    v.extend(ATTN_ROWS_SHADERS);
    v.extend(crate::flash::FLASH_VARIANTS.iter().map(|f| f.name));
    v.extend([GEMV_PAIR_BF16, GEMV_PAIR_BF16_A8, GEMV_PAIR_BF16_R1, GEMV_PAIR_BF16_R1_A8]);
    v.extend(RMSNORM_EX);
    v.extend(RMSNORM_ADD_EX);
    v.push(ARGMAX_F32_V4);
    v.extend(EXPERIMENT_SHADERS);
    v.extend(crate::multirow::shader_names());
    v.extend(crate::rowtab::shader_names());
    let mut seen = std::collections::HashSet::new();
    v.retain(|n| seen.insert(*n)); // the experiment list repeats the promoted variants
    v
}

/// (kernel name, op) for every implementation, in registration order. Ops without an entry (most quantised and the
/// GatedDeltaNet ops) report `NoKernel`.
pub const TABLE: &[(&str, Op)] = &[
    (EMBEDDING_BF16, Op::Embedding),
    (RMSNORM_BF16, Op::RmsNorm),
    (RMSNORM_ADD_BF16, Op::RmsNormAdd),
    (ADD_BF16, Op::Add),
    (QK_NORM_BF16, Op::QkNorm),
    (ROPE_BF16, Op::Rope),
    (ROPE_SCALED_BF16, Op::RopeScaled),
    (ADD_BIAS_BF16, Op::AddBias),
    (MATMUL_BF16, Op::Matmul),
    (MATMUL_BF16_GEMV, Op::Matmul),
    (LOGITS_BF16, Op::Logits),
    (LOGITS_BF16_GEMV, Op::Logits),
    (KV_APPEND_BF16, Op::KvAppend),
    (SILU_GLU_BF16, Op::SiluGlu),
    (ARGMAX_F32, Op::Argmax),
    (ARGMAX_BF16, Op::Argmax),
    (ATTN_PREFILL_BF16, Op::AttnPrefill),
    (ATTN_PREFILL_BF16_FLASH, Op::AttnPrefill),
    (ATTN_PREFILL_BF16_FLASH_NAX, Op::AttnPrefill),
    (ATTN_DECODE_BF16, Op::AttnDecode),
    (ATTN_DECODE_BF16_SPLITKV, Op::AttnDecode),
    (MATMUL_PAIR_BF16_GEMV, Op::MatmulPair),
    (QK_NORM_ROPE_KV_BF16, Op::QkNormRopeKv),
    (MATMUL_BF16_NAX, Op::Matmul),
    (MATMUL_PAIR_BF16_NAX, Op::MatmulPair),
    (crate::multirow::MATMUL_BF16_GEMV_ROWS, Op::Matmul),
    (crate::multirow::MATMUL_PAIR_BF16_GEMV_ROWS, Op::MatmulPair),
    (crate::multirow::LOGITS_BF16_GEMV_ROWS, Op::Logits),
    (ATTN_PREFILL_BF16_SPLITKV_ROWS, Op::AttnPrefill),
    (crate::mxfp4::MATMUL_MXFP4_GEMV, Op::MatmulMxfp4),
    (GATE_SPLIT_BF16, Op::GateSplit),
    (SIGMOID_MUL_BF16, Op::SigmoidMul),
    (GDN_CONV_BF16, Op::GdnConv),
    (GDN_DELTA_RULE_BF16, Op::GdnDeltaRule),
    (GDN_DELTA_RULE_BF16_PRE, Op::GdnDeltaRule),
    (crate::moe::MOE_TOPK_BF16, Op::MoeTopk),
    (crate::moe::MOE_EXPERTS_BF16, Op::MoeExperts),
    (crate::moe::MOE_COMBINE_BF16, Op::MoeCombine),
    (crate::fp8::MATMUL_FP8_GEMV, Op::MatmulFp8),
    (crate::fp8::MATMUL_FP8_MMA, Op::MatmulFp8),
    (crate::fp8::MATMUL_FP8_NAX, Op::MatmulFp8),
    (crate::nvfp4::MATMUL_NVFP4_GEMV, Op::MatmulNvfp4),
    (crate::nvfp4::MATMUL_NVFP4_MMA, Op::MatmulNvfp4),
    (crate::fp8::MATMUL_FP8_GEMV_ROWS, Op::MatmulFp8),
    (crate::nvfp4::MATMUL_NVFP4_GEMV_ROWS, Op::MatmulNvfp4),
    (crate::moe::MOE_EXPERTS_BF16_ROWS, Op::MoeExperts),
    (crate::rowtab::GDN_CONV_BF16_ROWS, Op::GdnConv),
    (crate::rowtab::GDN_DELTA_RULE_BF16_ROWS, Op::GdnDeltaRule),
    (crate::rowtab::KV_APPEND_BF16_ROWS, Op::KvAppend),
    (crate::rowtab::QK_NORM_ROPE_KV_BF16_ROWS, Op::QkNormRopeKv),
    (crate::rowtab::ATTN_DECODE_BF16_SPLITKV_ROWS, Op::AttnDecode),
];

/// Every kernel implementation this backend registers. Tolerances and exactness match the hip entries of the same ops.
pub fn all() -> Vec<KernelImpl> {
    let bf = [DType::Bf16];
    TABLE
        .iter()
        .map(|&(name, op)| {
            let k = KernelImpl::new(op, name, "metal");
            match name {
                EMBEDDING_BF16 => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT),
                RMSNORM_BF16 | RMSNORM_ADD_BF16 | ADD_BF16 | QK_NORM_BF16 | SILU_GLU_BF16 => {
                    k.acts(bf).exact(Exactness::Deterministic).tol(BF16_TOL)
                }
                ROPE_BF16 | ROPE_SCALED_BF16 => {
                    k.acts(bf).when(|s| s.head_dim % 2 == 0).exact(Exactness::Deterministic).tol(BF16_TOL)
                }
                ADD_BIAS_BF16 => k.acts(bf).exact(Exactness::Deterministic).tol(BF16_TOL),
                // tiled GEMM, f32 accumulate in a fixed K order, run-to-run identical; any M
                MATMUL_BF16 => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.out.is_none_or(|d| d == DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(0),
                // decode GEMV: fixed reduction order, run-to-run identical
                MATMUL_BF16_GEMV => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.m == 1 && s.out.is_none_or(|d| d == DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(10),
                LOGITS_BF16 => k.acts(bf).weights(bf).exact(Exactness::Deterministic).tol(F32_TOL).priority(0),
                LOGITS_BF16_GEMV => {
                    k.acts(bf).weights(bf).when(|s| s.m == 1).exact(Exactness::Deterministic).tol(F32_TOL).priority(10)
                }
                // the cache dtype is the output's; only bf16 caches are ported (fp8 / f32 caches fall to the cpu)
                KV_APPEND_BF16 => {
                    k.acts(bf).when(|s| s.out == Some(DType::Bf16)).exact(Exactness::Exact).tol(Tol::EXACT)
                }
                ARGMAX_F32 => k.acts([DType::F32]).exact(Exactness::Exact).tol(Tol::EXACT),
                ARGMAX_BF16 => k.acts(bf).exact(Exactness::Exact).tol(Tol::EXACT),
                // causal GQA, f32 online softmax; `weights` is the K cache dtype (bf16 only, as KvAppend). Same
                // predicate and tolerance as the hip `attn_prefill_bf16` / `attn_decode_bf16`.
                ATTN_PREFILL_BF16 | ATTN_DECODE_BF16 => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.head_dim > 0 && s.head_dim <= 256 && s.head_dim % 2 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL),
                // flash-style prefill on simdgroup matrices (half operands, float accumulate and softmax); the cache dtype
                // (`weights`) is bf16 only, so fp8 / f32 caches stay on `attn_prefill_bf16`
                ATTN_PREFILL_BF16_FLASH => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::flash::flash_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(10),
                // NAX flash-style prefill (metal4.0 library, hidden without it): P2's contract on the neural accelerators
                ATTN_PREFILL_BF16_FLASH_NAX => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::flash_nax::nax_flash_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(12),
                // GQA-grouped split-KV decode (each K / V row read once per step, the sequence spread over
                // threadgroups): needs scratch for the partials; ranks above the single-pass kernel.
                ATTN_DECODE_BF16_SPLITKV => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::attn::splitkv_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(10)
                    .scratch(crate::attn::splitkv_scratch),
                // fused decode pair (gate + up, k + v): bitwise two `matmul_bf16_gemv`s; the hip entry's predicate and tolerance
                MATMUL_PAIR_BF16_GEMV => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.m == 1 && s.k % 8 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .priority(10),
                // fused attention prologue; `weights` is the cache dtype (bf16 only, as KvAppend), head_dim as the norm kernel's
                // one-element-per-thread threadgroup allows. Any T (prefill rows included). Bitwise the unfused kernels', graded
                // against the cpu composition with the budget of the ops it fuses.
                QK_NORM_ROPE_KV_BF16 => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.head_dim > 0 && s.head_dim <= 256 && s.head_dim % 2 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(10),
                // NAX GEMM (metal4.0 library): M >= NAX_MIN_M (2: M = 1 is the decode GEMV's). A prompt's short tail chunk (2..15 rows)
                // runs here too, not on the tiled GEMM (181 ms against 21 + 20 ms per 4k prompt tail of 11 rows, and a row then
                // gets the same bits whatever chunk it lands in: docs/results/2026-10-15-metal-p3-noise.md); any K
                // (the primitive zero-fills a K tail, tests/gemm_nax.rs `nax_k_tails`). Priority 5: above the tiled GEMM
                // (0) and below the GEMV-class kernels (10, which only take M == 1). Hidden when the metal4.0 library did
                // not load, then the tiled GEMM runs.
                MATMUL_BF16_NAX => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.m >= crate::gemm_nax::NAX_MIN_M && s.k >= 16 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(5),
                // fused gate + up / k + v for M >= NAX_MIN_M (the GEMV pair above serves M == 1)
                MATMUL_PAIR_BF16_NAX => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| s.m >= crate::gemm_nax::NAX_MIN_M && s.k >= 16 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(5),
                // G0 multi-row GEMVs (bitwise the 1-row serving GEMV per row, tests/multirow.rs). `rows_exact`: only
                // speculative verify calls (`OpCall::exact_rows`) select them, ordinary calls never do (priority -1 is
                // kept but no longer matters); `kernels test` grades them against the cpu reference like the 1-row
                // kernels (the harness cannot grade against the 1-row Metal launch, see the G0 report).
                crate::multirow::MATMUL_BF16_GEMV_ROWS => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| {
                        (2..=crate::multirow::MR_MAX_REGISTERED).contains(&s.m)
                            && s.out.is_none_or(|d| d == DType::Bf16)
                    })
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(-1)
                    .rows_exact(),
                // the only pair kernel accepting M = 2..8: being `rows_exact`, ordinary prefill chunks never fuse with it
                crate::multirow::MATMUL_PAIR_BF16_GEMV_ROWS => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| {
                        (2..=crate::multirow::MR_MAX_REGISTERED).contains(&s.m)
                            && s.k % 8 == 0
                            && s.out == Some(DType::Bf16)
                    })
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .priority(-1)
                    .rows_exact(),
                crate::multirow::LOGITS_BF16_GEMV_ROWS => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| (2..=crate::multirow::MR_MAX_REGISTERED).contains(&s.m))
                    .exact(Exactness::Deterministic)
                    .tol(F32_TOL)
                    .priority(-1)
                    .rows_exact(),
                // G0b: 2..=8 verification rows, bitwise the 1-row split-KV decode (host position) per row. `rows_exact`:
                // only verify calls select it; `kernels test` grades it against the cpu reference.
                ATTN_PREFILL_BF16_SPLITKV_ROWS => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::attn::rows_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(-1)
                    .rows_exact()
                    .scratch(crate::attn::rows_scratch),
                // OCP MXFP4 (kernels/metal/mxfp4.metal, mxfp4.rs): the activations are QDQ'd once into scratch, then a
                // simdgroup-per-row GEMV runs; any M, K % 32 == 0. The hip `matmul_mxfp4_gemv` budgets and scratch bound.
                crate::mxfp4::MATMUL_MXFP4_GEMV => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| !s.mxfp4_a8 && s.m >= 1 && s.k > 0 && s.k % 32 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .tol_f32(F32_TOL)
                    .scratch(crate::mxfp4::scratch),
                // pure bit copy / f32 elementwise, one rounding: the hip entries' tolerance
                GATE_SPLIT_BF16 | SIGMOID_MUL_BF16 => k.acts(bf).exact(Exactness::Deterministic).tol(BF16_TOL),
                // depthwise causal conv + SiLU; `weights` is the tap dtype, `k` the tap count
                GDN_CONV_BF16 => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::gdn::conv_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL),
                // sequential recurrence, one simdgroup per state row: fixed reduction order; `weights` is A_log's dtype
                GDN_DELTA_RULE_BF16 => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::gdn::delta_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL),
                // the same recurrence for T >= 64 with the per-token norms and gates done once per head in a prep pass: bitwise
                // `gdn_delta_rule_bf16` (tests/gdn.rs). `INTARSIA_METAL_GDN_PRE=0` hides it.
                GDN_DELTA_RULE_BF16_PRE => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::gdn::pre_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .scratch(crate::gdn::pre_scratch)
                    .priority(5),
                // MoE (kernels/metal/moe.metal, moe.rs), the hip entries' predicates and budget. top-k: bf16 or f32 logits.
                crate::moe::MOE_TOPK_BF16 => k
                    .acts([DType::Bf16, DType::F32])
                    .when(|s| s.experts <= 512 && (1..=32).contains(&s.top_k) && s.top_k <= s.experts)
                    .exact(Exactness::Deterministic)
                    .tol(MOE_TOL),
                // NVFP4 experts, bf16 x (h / y scratch are bf16); decode GEMVs at T*K <= 16, else grouped by expert
                crate::moe::MOE_EXPERTS_BF16 => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.experts <= crate::moe::MAX_EXPERTS && s.k % 16 == 0 && s.n % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(MOE_TOL)
                    .scratch(intarsia_core::quant::ops::moe_experts_scratch),
                crate::moe::MOE_COMBINE_BF16 => k.acts(bf).exact(Exactness::Deterministic).tol(MOE_TOL),
                // FP8 e4m3 W8A8 / W8A16 (kernels/metal/fp8.metal, fp8.rs), the hip budgets. The GEMV (M <= 8, one dispatch, q(x)
                // staged in threadgroup memory) takes decode; the GEMM (priority 0) every other shape, QDQ'ing x into scratch.
                crate::fp8::MATMUL_FP8_GEMV => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(crate::fp8::gemv_ok)
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .tol_f32(F32_TOL)
                    .priority(10),
                crate::fp8::MATMUL_FP8_MMA => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .tol_f32(F32_TOL)
                    .scratch(crate::fp8::scratch)
                    .priority(0),
                // fp8 GEMM on the neural accelerators (gemm_nax_fp8.metal, fp8.rs): M above the decode GEMV's 8 rows (the
                // GEMV-class kernels and the M <= 8 `mma` calls are untouched), any K % 16 == 0. The weights are decoded to
                // bf16 into the scratch per call and the product runs on matmul2d, so the K accumulation order differs from
                // `matmul_fp8_mma` (KL-gated, docs/results/2026-10-27-nxp-metal-nax-prefill.md). `INTARSIA_METAL_FP8_NAX=0`
                // hides it. Hidden when the metal4.0 library did not load, then `mma` runs.
                crate::fp8::MATMUL_FP8_NAX => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(crate::fp8::nax_ok)
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .tol_f32(F32_TOL)
                    .scratch(crate::fp8::nax_scratch)
                    .priority(5),
                // NVFP4 (kernels/metal/nvfp4.metal, nvfp4.rs): GEMV for M <= 8 (no scratch), GEMM for every other shape.
                crate::nvfp4::MATMUL_NVFP4_GEMV => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(crate::nvfp4::gemv_ok)
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .tol_f32(F32_TOL)
                    .priority(10),
                crate::nvfp4::MATMUL_NVFP4_MMA => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| s.k % 16 == 0)
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .tol_f32(F32_TOL)
                    .priority(0),
                // Row-exact multi-row forms for batched decode of the hybrid 35B (rows_exact: only `OpCall::exact_rows` calls
                // select them): row r bitwise the M = 1 serving kernel on row r alone, M = 2..=8 (tests/rows_exact.rs).
                crate::fp8::MATMUL_FP8_GEMV_ROWS => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(crate::fp8::rows_ok)
                    .scratch(crate::fp8::rows_scratch)
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .tol_f32(F32_TOL)
                    .priority(-1)
                    .rows_exact(),
                crate::nvfp4::MATMUL_NVFP4_GEMV_ROWS => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(crate::nvfp4::rows_ok)
                    .exact(Exactness::Deterministic)
                    .tol(PAIR_TOL)
                    .tol_f32(F32_TOL)
                    .priority(-1)
                    .rows_exact(),
                crate::moe::MOE_EXPERTS_BF16_ROWS => k
                    .acts(bf)
                    .weights([DType::U8])
                    .when(|s| {
                        (2..=crate::moe::ROWS_MAX).contains(&s.m)
                            && s.experts <= crate::moe::MAX_EXPERTS
                            && s.k % 16 == 0
                            && s.n % 16 == 0
                    })
                    .exact(Exactness::Deterministic)
                    .tol(MOE_TOL)
                    .priority(-1)
                    .rows_exact()
                    .scratch(intarsia_core::quant::ops::moe_experts_scratch),
                // Row-table twins (lane mr3, `OpCall::rows`): the solo one-row decode kernel of the same name without `_rows`
                // on 1..=16 rows at once, each row on its own state / cache from a table of device addresses. Bitwise the
                // solo launch per row (tests/row_tables.rs); selected only for calls with a row table, and those select only
                // these. They take the admission of the kernel they twin (`Registry::effective_admission`).
                crate::rowtab::GDN_CONV_BF16_ROWS => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::rowtab::conv_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .row_table(),
                crate::rowtab::GDN_DELTA_RULE_BF16_ROWS => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::rowtab::delta_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .row_table(),
                crate::rowtab::KV_APPEND_BF16_ROWS => k
                    .acts(bf)
                    .when(|s| s.out == Some(DType::Bf16) && crate::rowtab::rows_in_range(s))
                    .exact(Exactness::Exact)
                    .tol(Tol::EXACT)
                    .row_table(),
                crate::rowtab::QK_NORM_ROPE_KV_BF16_ROWS => k
                    .acts(bf)
                    .weights(bf)
                    .when(|s| {
                        s.head_dim > 0
                            && s.head_dim <= 256
                            && s.head_dim % 2 == 0
                            && s.out == Some(DType::Bf16)
                            && crate::rowtab::rows_in_range(s)
                    })
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .row_table(),
                crate::rowtab::ATTN_DECODE_BF16_SPLITKV_ROWS => k
                    .acts(bf)
                    .weights(bf)
                    .when(crate::rowtab::attn_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .row_table()
                    .scratch(crate::rowtab::attn_scratch),
                other => unreachable!("kernel {other} has no registration"),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn names_unique_and_named_after_their_op() {
        let names: HashSet<_> = TABLE.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), TABLE.len());
        for (name, op) in TABLE {
            assert!(name.starts_with(op.as_str()), "{name} is not named after op {op}");
        }
        assert_eq!(all().len(), TABLE.len());
    }

    #[test]
    fn metal4_shaders_are_the_nax_variants_and_not_metal3_shaders() {
        let m4 = metal4_shaders();
        assert_eq!(
            m4.len(),
            2 * crate::gemm_nax::NAX_VARIANTS.len()
                + crate::flash_nax::NAX_FLASH_VARIANTS.len()
                + 2 * crate::gemm_nax::NAX_VARIANTS.len()
                + 8
        );
        assert!(!shader_names().iter().any(|n| m4.contains(n)));
        let set: HashSet<_> = m4.iter().collect();
        assert_eq!(set.len(), m4.len());
    }

    #[test]
    fn attention_is_registered_for_both_ops() {
        // decode has the single-pass kernel (any even head_dim), the split-KV one (ATTN_DECODE_BF16_SPLITKV) and its row-table twin
        for (op, n) in [(Op::AttnPrefill, 4), (Op::AttnDecode, 3)] {
            assert_eq!(TABLE.iter().filter(|(_, o)| *o == op).count(), n, "{op}");
        }
        assert!(shader_names().contains(&ATTN_PREFILL_BF16) && shader_names().contains(&ATTN_DECODE_BF16));
        for s in ATTN_DECODE_SPLITKV_SHADERS {
            assert!(shader_names().contains(&s), "{s}");
        }
    }

    // Metal has no fp8 / nvfp4 / MoE kernels: the 35B NVFP4 checkpoint does not load here (the '-' rows), before or after
    const PHASE1_PINS: &[&str] = &[
        "35b attn input norm: rmsnorm_add_bf16",
        "35b k/v pair: -",
        "35b k proj: matmul_fp8_gemv",
        "35b output gate: sigmoid_mul_bf16",
        "35b o_proj: matmul_fp8_gemv",
        "35b shared down: matmul_nvfp4_gemv",
        "35b combine: moe_combine_bf16",
        "35b fold P5: -",
        "35b fold P4b: -",
        "35b fold P2a: -",
        "8b attn input norm: rmsnorm_add_bf16",
        "8b k/v pair: matmul_pair_bf16_gemv",
    ];

    /// Core-op fusion phase 1 (docs/results/2026-10-19-core-op-fusion-p1.md): this backend registers none of the three new
    /// ops, so `Model::try_fused` returns false for each and the model runs the unfused chain it ran before (the selection of
    /// those ops at the Qwen3.6-35B-A3B and Qwen3-8B decode shapes is pinned below: the names origin/main's table resolves).
    #[test]
    fn phase1_folds_are_absent_and_the_unfused_selection_is_pinned() {
        use intarsia_core::ShapeInfo;
        let ks = all();
        for op in [Op::RmsNormAddMatmulFp8Pair, Op::SigmoidMulMatmulFp8, Op::SharedDownCombine] {
            assert!(!ks.iter().any(|k| k.op == op), "{op} is registered: the model would take the fused path here");
        }
        let pick = |op: Op, s: &ShapeInfo| -> Option<String> {
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == op && k.accepts(s)) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.clone())
        };
        let (bf, f8, u8_) = (Some(DType::Bf16), Some(DType::Fp8E4m3), Some(DType::U8));
        let sh = |m: usize, n: usize, k: usize, w: Option<DType>| ShapeInfo { m, n, k, act: bf, out: bf, weight: w, ..Default::default() };
        // (what, op, shape): the ops each fold replaces, at the 35B decode shapes (hidden 2048, k / v 2 x 256, o_proj 2048 x
        // 4096, shared down 2048 x 512), and the 8B's input norm (hidden 4096, bf16 weights: no fold applies to the 8B)
        let cases: Vec<(&str, Op, ShapeInfo)> = vec![
            ("35b attn input norm", Op::RmsNormAdd, sh(1, 0, 2048, None)),
            ("35b k/v pair", Op::MatmulFp8Pair, sh(1, 512, 2048, f8)),
            ("35b k proj", Op::MatmulFp8, sh(1, 512, 2048, f8)),
            ("35b output gate", Op::SigmoidMul, sh(1, 0, 4096, None)),
            ("35b o_proj", Op::MatmulFp8, sh(1, 2048, 4096, f8)),
            ("35b shared down", Op::MatmulNvfp4, sh(1, 2048, 512, u8_)),
            ("35b combine", Op::MoeCombine, sh(1, 0, 2048, None)),
            ("35b fold P5", Op::RmsNormAddMatmulFp8Pair, sh(1, 512, 2048, f8)),
            ("35b fold P4b", Op::SigmoidMulMatmulFp8, sh(1, 2048, 4096, f8)),
            ("35b fold P2a", Op::SharedDownCombine, sh(1, 2048, 512, u8_)),
            ("8b attn input norm", Op::RmsNormAdd, sh(1, 0, 4096, None)),
            ("8b k/v pair", Op::MatmulPair, sh(1, 1024, 4096, bf)),
        ];
        let got: Vec<String> = cases.iter().map(|(w, op, s)| format!("{w}: {}", pick(*op, s).unwrap_or_else(|| "-".into()))).collect();
        eprintln!("PINS\n{}", got.join("\n"));
        assert_eq!(got, PHASE1_PINS, "the selection of the unfused ops moved");
    }

    #[rustfmt::skip]
    const MAIN_KERNEL_LIST: &[&str] = &[
        "add_bf16:add",
        "add_bias_bf16:add_bias",
        "argmax_bf16:argmax",
        "argmax_f32:argmax",
        "attn_decode_bf16:attn_decode",
        "attn_decode_bf16_splitkv:attn_decode",
        "attn_decode_bf16_splitkv_rows:attn_decode",
        "attn_prefill_bf16:attn_prefill",
        "attn_prefill_bf16_flash:attn_prefill",
        "attn_prefill_bf16_flash_nax:attn_prefill",
        "attn_prefill_bf16_splitkv_rows:attn_prefill",
        "embedding_bf16:embedding",
        "gate_split_bf16:gate_split",
        "gdn_conv_bf16:gdn_conv",
        "gdn_conv_bf16_rows:gdn_conv",
        "gdn_delta_rule_bf16:gdn_delta_rule",
        "gdn_delta_rule_bf16_pre:gdn_delta_rule",
        "gdn_delta_rule_bf16_rows:gdn_delta_rule",
        "kv_append_bf16:kv_append",
        "kv_append_bf16_rows:kv_append",
        "logits_bf16:logits",
        "logits_bf16_gemv:logits",
        "logits_bf16_gemv_rows:logits",
        "matmul_bf16:matmul",
        "matmul_bf16_gemv:matmul",
        "matmul_bf16_gemv_rows:matmul",
        "matmul_bf16_nax:matmul",
        "matmul_fp8_gemv:matmul_fp8",
        "matmul_fp8_gemv_rows:matmul_fp8",
        "matmul_fp8_mma:matmul_fp8",
        "matmul_fp8_nax:matmul_fp8",
        "matmul_mxfp4_gemv:matmul_mxfp4",
        "matmul_nvfp4_gemv:matmul_nvfp4",
        "matmul_nvfp4_gemv_rows:matmul_nvfp4",
        "matmul_nvfp4_mma:matmul_nvfp4",
        "matmul_pair_bf16_gemv:matmul_pair",
        "matmul_pair_bf16_gemv_rows:matmul_pair",
        "matmul_pair_bf16_nax:matmul_pair",
        "moe_combine_bf16:moe_combine",
        "moe_experts_bf16:moe_experts",
        "moe_experts_bf16_rows:moe_experts",
        "moe_topk_bf16:moe_topk",
        "qk_norm_bf16:qk_norm",
        "qk_norm_rope_kv_bf16:qk_norm_rope_kv",
        "qk_norm_rope_kv_bf16_rows:qk_norm_rope_kv",
        "rmsnorm_add_bf16:rmsnorm_add",
        "rmsnorm_bf16:rmsnorm",
        "rope_bf16:rope",
        "rope_scaled_bf16:rope_scaled",
        "sigmoid_mul_bf16:sigmoid_mul",
        "silu_glu_bf16:silu_glu",
    ];

    /// Core-op fusion phase 1 changes no kernel table here: the full registered list (name and op, sorted) is pinned to
    /// origin/main's (`4962d8fa`) plus the two ops added by rope scaling / projection bias (`add_bias`, `rope_scaled`: no existing
    /// op gains or loses a kernel), so the plan has the same candidates as before.
    #[test]
    fn kernel_list_is_unchanged_by_core_op_fusion_p1() {
        let mut got: Vec<String> = all().iter().map(|k| format!("{}:{}", k.name, k.op)).collect();
        got.sort();
        if MAIN_KERNEL_LIST.is_empty() {
            panic!("LIST {}", got.join(" "));
        }
        assert_eq!(got, MAIN_KERNEL_LIST);
    }

    // The gated-attention split and the GatedDeltaNet conv / delta rule are registered (lane M35 gdn); the fused ops are not
    // (the '-' rows); the fused prologue serves the dense models
    const PHASE2_PINS: &[&str] = &[
        "35b m=1 gate split: gate_split_bf16",
        "35b m=1 q norm: qk_norm_bf16",
        "35b m=1 k norm: qk_norm_bf16",
        "35b m=1 rope: rope_bf16",
        "35b m=1 kv append: kv_append_bf16",
        "35b m=1 prologue: qk_norm_rope_kv_bf16",
        "35b m=1 fold P4a: -",
        "35b m=1 gdn conv: gdn_conv_bf16",
        "35b m=1 gdn delta: gdn_delta_rule_bf16",
        "35b m=1 fold P3: -",
        "35b m=32 gate split: gate_split_bf16",
        "35b m=32 q norm: qk_norm_bf16",
        "35b m=32 k norm: qk_norm_bf16",
        "35b m=32 rope: rope_bf16",
        "35b m=32 kv append: kv_append_bf16",
        "35b m=32 prologue: qk_norm_rope_kv_bf16",
        "35b m=32 fold P4a: -",
        "35b m=32 gdn conv: gdn_conv_bf16",
        "35b m=32 gdn delta: gdn_delta_rule_bf16",
        "35b m=32 fold P3: -",
    ];

    /// Core-op fusion phase 2 (docs/results/2026-10-19-core-op-fusion-p2.md): this backend registers neither new op
    /// (`GateSplitQkNormRopeKv`, `GdnConvDelta`), so `Model::fused_ok` is false for both and the model runs the chains it
    /// ran before. The selection of the ops those chains consist of, at the Qwen3.6-35B-A3B decode shapes (gated attention:
    /// 16 query / 2 kv heads of 256, bf16 cache; GatedDeltaNet: 16 key / 32 value heads of 128, conv width 8192, 4 taps),
    /// and at a 32-token chunk (the longest the HIP fold takes), is pinned below: the names origin/main's table resolves.
    #[test]
    fn phase2_folds_are_absent_and_the_unfused_selection_is_pinned() {
        use intarsia_core::ShapeInfo;
        let ks = all();
        for op in [Op::GateSplitQkNormRopeKv, Op::GdnConvDelta] {
            assert!(!ks.iter().any(|k| k.op == op), "{op} is registered: the model would take the fused path here");
        }
        let pick = |op: Op, s: &ShapeInfo| -> Option<String> {
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == op && k.accepts(s)) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.clone())
        };
        let bf = Some(DType::Bf16);
        let heads = |m: usize, n: usize, nkv: usize, d: usize| ShapeInfo {
            m,
            n_heads: n,
            n_kv_heads: nkv,
            head_dim: d,
            k: d,
            act: bf,
            out: bf,
            weight: bf,
            ..Default::default()
        };
        let gdn = |m: usize, k: usize| ShapeInfo {
            m,
            n: 128,
            k,
            n_heads: 32,
            n_kv_heads: 16,
            head_dim: 128,
            act: bf,
            out: bf,
            weight: bf,
            ..Default::default()
        };
        let mut cases: Vec<(String, Op, ShapeInfo)> = Vec::new();
        for m in [1usize, 32] {
            cases.extend([
                (format!("35b m={m} gate split"), Op::GateSplit, heads(m, 16, 0, 256)),
                (format!("35b m={m} q norm"), Op::QkNorm, heads(m, 16, 0, 256)),
                (format!("35b m={m} k norm"), Op::QkNorm, heads(m, 2, 0, 256)),
                (format!("35b m={m} rope"), Op::Rope, heads(m, 16, 0, 256)),
                (format!("35b m={m} kv append"), Op::KvAppend, heads(m, 2, 0, 256)),
                (format!("35b m={m} prologue"), Op::QkNormRopeKv, heads(m, 16, 2, 256)),
                (format!("35b m={m} fold P4a"), Op::GateSplitQkNormRopeKv, heads(m, 16, 2, 256)),
                (format!("35b m={m} gdn conv"), Op::GdnConv, ShapeInfo { m, n: 8192, k: 4, act: bf, out: bf, weight: bf, ..Default::default() }),
                (format!("35b m={m} gdn delta"), Op::GdnDeltaRule, gdn(m, 128)),
                (format!("35b m={m} fold P3"), Op::GdnConvDelta, gdn(m, 4)),
            ]);
        }
        let got: Vec<String> = cases.iter().map(|(w, op, s)| format!("{w}: {}", pick(*op, s).unwrap_or_else(|| "-".into()))).collect();
        eprintln!("PINS\n{}", got.join("\n"));
        assert_eq!(got, PHASE2_PINS, "the selection of the unfused ops moved");
    }

    const MOE_BLOCK_PINS: &[&str] = &[
        "35b m=1 topk: moe_topk_bf16",
        "35b m=1 experts: moe_experts_bf16",
        "35b m=1 shared glu: -",
        "35b m=1 shared gate: matmul_nvfp4_gemv",
        "35b m=1 silu glu: silu_glu_bf16",
        "35b m=1 shared down: matmul_nvfp4_gemv",
        "35b m=1 shared down combine: -",
        "35b m=1 combine: moe_combine_bf16",
        "35b m=1 fold P2': -",
        "35b m=2 topk: moe_topk_bf16",
        "35b m=2 experts: moe_experts_bf16",
        "35b m=2 shared glu: -",
        "35b m=2 shared gate: matmul_nvfp4_gemv",
        "35b m=2 silu glu: silu_glu_bf16",
        "35b m=2 shared down: matmul_nvfp4_gemv",
        "35b m=2 shared down combine: -",
        "35b m=2 combine: moe_combine_bf16",
        "35b m=2 fold P2': -",
    ];

    /// Core-op fusion P2' (docs/results/2026-10-19-p2prime-moeblock.md): this backend does not register `MoeBlock`, so
    /// `Model::fused_ok` is false for it and `MoeWeights::forward_folded` runs the chain it ran before. The selection of the
    /// ops that chain consists of, at the Qwen3.6-35B-A3B decode shapes (256 experts, top-8, hidden 2048, routed and shared
    /// intermediate 512; one row and the two-row batched step), is pinned below: the names origin/main's table resolves.
    #[test]
    fn moe_block_is_absent_and_the_unfused_selection_is_pinned() {
        use intarsia_core::ShapeInfo;
        let ks = all();
        assert!(!ks.iter().any(|k| k.op == Op::MoeBlock), "MoeBlock is registered: the model would take the fused path here");
        let pick = |op: Op, s: &ShapeInfo| -> Option<String> {
            let mut best: Option<&KernelImpl> = None;
            for k in ks.iter().filter(|k| k.op == op && k.accepts(s)) {
                if best.is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name.clone())
        };
        let (bf, u8_) = (Some(DType::Bf16), Some(DType::U8));
        let sh = |m: usize, n: usize, k: usize, w: Option<DType>| ShapeInfo { m, n, k, act: bf, out: bf, weight: w, ..Default::default() };
        let moe = |m: usize, n: usize, k: usize, w: Option<DType>| ShapeInfo { experts: 256, top_k: 8, ..sh(m, n, k, w) };
        let mut cases: Vec<(String, Op, ShapeInfo)> = Vec::new();
        for m in [1usize, 2] {
            cases.extend([
                (format!("35b m={m} topk"), Op::MoeTopk, ShapeInfo { k: 256, ..moe(m, 256, 256, None) }),
                (format!("35b m={m} experts"), Op::MoeExperts, moe(m, 512, 2048, u8_)),
                (format!("35b m={m} shared glu"), Op::MatmulNvfp4Glu, sh(m, 512, 2048, u8_)),
                (format!("35b m={m} shared gate"), Op::MatmulNvfp4, sh(m, 512, 2048, u8_)),
                (format!("35b m={m} silu glu"), Op::SiluGlu, sh(m, 0, 512, None)),
                (format!("35b m={m} shared down"), Op::MatmulNvfp4, sh(m, 2048, 512, u8_)),
                (format!("35b m={m} shared down combine"), Op::SharedDownCombine, sh(m, 2048, 512, u8_)),
                (format!("35b m={m} combine"), Op::MoeCombine, sh(m, 0, 2048, None)),
                (format!("35b m={m} fold P2'"), Op::MoeBlock, moe(m, 512, 2048, u8_)),
            ]);
        }
        let got: Vec<String> = cases.iter().map(|(w, op, s)| format!("{w}: {}", pick(*op, s).unwrap_or_else(|| "-".into()))).collect();
        eprintln!("PINS\n{}", got.join("\n"));
        assert_eq!(got, MOE_BLOCK_PINS, "the selection of the unfused ops moved");
    }
}

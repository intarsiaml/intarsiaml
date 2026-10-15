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
pub const SILU_GLU_BF16: &str = "silu_glu_bf16";
pub const KV_APPEND_BF16: &str = "kv_append_bf16";
/// `QkNormRopeKv` (qk_rope_kv.metal): qk-norm, rope and the k / v cache appends of a layer in one dispatch, bitwise the
/// unfused `qk_norm_bf16` (`rmsnorm_bf16_ex1` at head_dim <= 256) + `rope_bf16` + `kv_append_bf16` sequence.
pub const QK_NORM_ROPE_KV_BF16: &str = "qk_norm_rope_kv_bf16";
/// The prefill form of the same registry entry (T > 1, head_dim = rotary_dim = 128), chosen by the launcher: bitwise the
/// decode form, with the 64 (cos, sin) pairs of a token computed once per threadgroup instead of per head and thread.
pub const QK_NORM_ROPE_KV_BF16_PF: &str = "qk_norm_rope_kv_bf16_pf";
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

// ---- fp8 e4m3 KV cache (`--kv-dtype fp8_e4m3`) -------------------------------------------------------------------------
// Opt-in: every entry below takes a Fp8E4m3 cache (the `weights` dtype of attention and QkNormRopeKv, the output dtype of
// KvAppend) and nothing else, so the bf16 registrations and the default path are untouched. Same ops as their bf16
// twins, same priorities.
/// `KvAppend` into an fp8 cache (dense_ops.metal): round to nearest even, saturating (the oracle's `f32_to_fp8_e4m3`).
pub const KV_APPEND_F8: &str = "kv_append_f8";
/// `QkNormRopeKv` with fp8 k / v caches (qk_rope_kv.metal): the bf16 kernel with the cache stores encoded; `_pf` the prefill
/// form chosen by the launcher as for bf16.
pub const QK_NORM_ROPE_KV_F8: &str = "qk_norm_rope_kv_f8";
pub const QK_NORM_ROPE_KV_F8_PF: &str = "qk_norm_rope_kv_f8_pf";
/// `AttnDecode` split-KV over an fp8 cache (attention_splitkv.metal): stage 1 `attn_decode_f8_splitkv_g{GP}`; the combine
/// pass is the bf16 one (it never reads the cache).
pub const ATTN_DECODE_F8_SPLITKV: &str = "attn_decode_f8_splitkv";
pub const ATTN_DECODE_F8_SPLITKV_SHADERS: [&str; 12] = [
    "attn_decode_f8_splitkv_g1",
    "attn_decode_f8_splitkv_g2",
    "attn_decode_f8_splitkv_g4",
    "attn_decode_f8_splitkv_g8",
    "attn_decode_f8u2_splitkv_g1",
    "attn_decode_f8u2_splitkv_g2",
    "attn_decode_f8u2_splitkv_g4",
    "attn_decode_f8u2_splitkv_g8",
    "attn_decode_f8u8_splitkv_g1",
    "attn_decode_f8u8_splitkv_g2",
    "attn_decode_f8u8_splitkv_g4",
    "attn_decode_f8u8_splitkv_g8",
];
/// `AttnPrefill` 2..=8 rows over an fp8 cache: bitwise the 1-row `attn_decode_f8_splitkv` per row (`rows_exact`).
pub const ATTN_PREFILL_F8_SPLITKV_ROWS: &str = "attn_prefill_f8_splitkv_rows";
pub const ATTN_ROWS_F8_SHADERS: [&str; 12] = [
    "attn_rows_f8_splitkv_g1",
    "attn_rows_f8_splitkv_g2",
    "attn_rows_f8_splitkv_g4",
    "attn_rows_f8_splitkv_g8",
    "attn_rows_f8u2_splitkv_g1",
    "attn_rows_f8u2_splitkv_g2",
    "attn_rows_f8u2_splitkv_g4",
    "attn_rows_f8u2_splitkv_g8",
    "attn_rows_f8u8_splitkv_g1",
    "attn_rows_f8u8_splitkv_g2",
    "attn_rows_f8u8_splitkv_g4",
    "attn_rows_f8u8_splitkv_g8",
];
/// `AttnPrefill` over an fp8 cache: the cache is decoded to a bf16 staging buffer (`kv_dequant_f8`, exact: every e4m3 value
/// is a bf16 value) and the bf16 flash kernel (the NAX one, or the P2 one without metal4.0) runs on it. See flash_nax.rs.
pub const ATTN_PREFILL_F8_FLASH_NAX: &str = "attn_prefill_f8_flash_nax";
pub const ATTN_PREFILL_F8_FLASH: &str = "attn_prefill_f8_flash";
pub const KV_DEQUANT_F8: &str = "kv_dequant_f8";
/// Test-only shaders (fp8_conv.metal): the exhaustive conversion checks of tests/kv_fp8.rs.
pub const FP8_TEST_SHADERS: [&str; 2] = ["fp8_test_decode", "fp8_test_encode"];

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
    &[MATMUL_BF16_NAX, MATMUL_PAIR_BF16_NAX, ATTN_PREFILL_BF16_FLASH_NAX, ATTN_PREFILL_F8_FLASH_NAX];

pub fn requires_metal4(kernel: &str) -> bool {
    NAX_KERNELS.contains(&kernel)
}

pub fn shader_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = GEMV_VARIANTS.iter().map(|g| g.name).collect();
    v.extend(GEMV_A8_VARIANTS.iter().map(|g| g.name));
    v.extend([GEMV_SCALAR.name, COPY_U4, READ_XOR_U4, ZERO_U4, ZERO_BYTES, COPY_BYTES]);
    v.extend([GEMM_O_BF16, GEMM_O_F32, GEMV_F32O_V, GEMV_F32O_SCALAR]);
    v.extend([EMBEDDING_BF16, RMSNORM_BF16, RMSNORM_ADD_BF16, ADD_BF16, ROPE_BF16, SILU_GLU_BF16, KV_APPEND_BF16]);
    v.extend([ARGMAX_F32, ARGMAX_BF16]);
    v.extend([QK_NORM_ROPE_KV_BF16, QK_NORM_ROPE_KV_BF16_PF]);
    v.extend([KV_APPEND_F8, QK_NORM_ROPE_KV_F8, QK_NORM_ROPE_KV_F8_PF, KV_DEQUANT_F8]);
    v.extend(["qk_norm_rope_kv_fq", "qk_norm_rope_kv_fq_pf"]);
    v.extend(ATTN_DECODE_F8_SPLITKV_SHADERS);
    v.extend(ATTN_ROWS_F8_SHADERS);
    v.extend(FP8_TEST_SHADERS);
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
    let mut seen = std::collections::HashSet::new();
    v.retain(|n| seen.insert(*n)); // the experiment list repeats the promoted variants
    v
}

/// (kernel name, op) for every implementation, in registration order. The quantised / MoE / GatedDeltaNet ops are not
/// ported yet, so the registry reports `NoKernel` for them.
pub const TABLE: &[(&str, Op)] = &[
    (EMBEDDING_BF16, Op::Embedding),
    (RMSNORM_BF16, Op::RmsNorm),
    (RMSNORM_ADD_BF16, Op::RmsNormAdd),
    (ADD_BF16, Op::Add),
    (QK_NORM_BF16, Op::QkNorm),
    (ROPE_BF16, Op::Rope),
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
    (KV_APPEND_F8, Op::KvAppend),
    (QK_NORM_ROPE_KV_F8, Op::QkNormRopeKv),
    (ATTN_DECODE_F8_SPLITKV, Op::AttnDecode),
    (ATTN_PREFILL_F8_FLASH, Op::AttnPrefill),
    (ATTN_PREFILL_F8_FLASH_NAX, Op::AttnPrefill),
    (ATTN_PREFILL_F8_SPLITKV_ROWS, Op::AttnPrefill),
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
                ROPE_BF16 => k.acts(bf).when(|s| s.head_dim % 2 == 0).exact(Exactness::Deterministic).tol(BF16_TOL),
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
                // ---- fp8 e4m3 KV cache: the bf16 twins' predicates with the cache dtype swapped. Tolerances: the cache holds
                // the e4m3 rounding of the bf16 values in both the oracle and here, so the cache bytes agree except where
                // a one-ulp difference of the f32 stages of the fused prologue lands on a rounding boundary (one e4m3
                // step); attention reads that same cache, so its tolerance is the bf16 one.
                KV_APPEND_F8 => {
                    k.acts(bf).when(|s| s.out == Some(DType::Fp8E4m3)).exact(Exactness::Exact).tol(Tol::EXACT)
                }
                QK_NORM_ROPE_KV_F8 => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(|s| s.head_dim > 0 && s.head_dim <= 256 && s.head_dim % 2 == 0 && s.out == Some(DType::Bf16))
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(10),
                ATTN_DECODE_F8_SPLITKV => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(crate::attn::splitkv_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(10)
                    .scratch(crate::attn::splitkv_scratch),
                ATTN_PREFILL_F8_FLASH => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(crate::flash::flash_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(10),
                ATTN_PREFILL_F8_FLASH_NAX => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(crate::flash_nax::nax_flash_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(12),
                ATTN_PREFILL_F8_SPLITKV_ROWS => k
                    .acts(bf)
                    .weights([DType::Fp8E4m3])
                    .when(crate::attn::rows_ok)
                    .exact(Exactness::Deterministic)
                    .tol(BF16_TOL)
                    .priority(-1)
                    .rows_exact()
                    .scratch(crate::attn::rows_scratch),
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
        assert_eq!(m4.len(), 2 * crate::gemm_nax::NAX_VARIANTS.len() + crate::flash_nax::NAX_FLASH_VARIANTS.len());
        assert!(!shader_names().iter().any(|n| m4.contains(n)));
        let set: HashSet<_> = m4.iter().collect();
        assert_eq!(set.len(), m4.len());
    }

    #[test]
    fn attention_is_registered_for_both_ops() {
        // decode has the single-pass kernel (any even head_dim) and the split-KV one (ATTN_DECODE_BF16_SPLITKV)
        for (op, n) in [(Op::AttnPrefill, 7), (Op::AttnDecode, 3)] {
            assert_eq!(TABLE.iter().filter(|(_, o)| *o == op).count(), n, "{op}");
        }
        assert!(shader_names().contains(&ATTN_PREFILL_BF16) && shader_names().contains(&ATTN_DECODE_BF16));
        for s in ATTN_DECODE_SPLITKV_SHADERS {
            assert!(shader_names().contains(&s), "{s}");
        }
    }
}

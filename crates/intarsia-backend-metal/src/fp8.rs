//! Launchers of `matmul_fp8_gemv` / `matmul_fp8_mma` (`Op::MatmulFp8`, kernels/metal/fp8.metal; the port of kernels/hip/fp8.hip).
//!
//! `out[M,N] = q(x)[M,K] W^T * ws * is`, W = e4m3 bytes `[N,K]`, `scales [2] f32 = (ws, is)` read on the GPU: `is > 0` is W8A8
//! static (q(x) = e4m3(x / is)), else W8A16. The decode GEMV (M <= 8) is one dispatch that stages q(x) in threadgroup memory;
//! the GEMM (any M) first QDQs x once into the plan-reserved scratch (bf16 `[M,K]`, exact) and then runs a tiled
//! `simdgroup_matrix` bf16 GEMM that dequantises the weight tiles.

use crate::dense::OpP;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// `KernelImpl::name` of the kernels.
pub const MATMUL_FP8_GEMV: &str = "matmul_fp8_gemv";
pub const MATMUL_FP8_MMA: &str = "matmul_fp8_mma";
/// `matmul_fp8_mma`'s shape class on the neural accelerators (kernels/metal/gemm_nax_fp8.metal, `INTARSIA_METAL_FP8_NAX`).
pub const MATMUL_FP8_NAX: &str = "matmul_fp8_nax";
/// `rows_exact` multi-row GEMV (continuous batching of the hybrid 35B, docs/results/2026-10-23-mb-metal-hybrid-batching.md):
/// row r of an M-row launch (M = 2..=8) is bitwise the M = 1 `matmul_fp8_gemv` launch on row r alone (tests/rows_exact.rs).
pub const MATMUL_FP8_GEMV_ROWS: &str = "matmul_fp8_gemv_rows";

const XPREP: &str = "fp8_xprep_bf16";
const XPREP_T: &str = "fp8_xprep_t";
/// Shader entry points (kernels/metal/fp8.metal).
pub const SHADERS: [&str; 19] = [
    XPREP,
    "fp8_gemv_bf16_m1",
    "fp8_gemv_bf16_m8",
    "fp8_gemv_f32_m1",
    "fp8_gemv_f32_m8",
    "fp8_mma_bf16",
    "fp8_mma_f32",
    "fp8_gemv_bf16_m4",
    "fp8_gemv_f32_m4",
    "fp8_gemv_bf16_m4_xd",
    "fp8_gemv_f32_m4_xd",
    "fp8_gemv_bf16_m4_xd_neg",
    "fp8_gemv_f32_m4_xd_neg",
    XPREP_T,
    "fp8_rows_mmt_bf16",
    "fp8_rows_mmt_f32",
    "fp8_rows_mmt_bf16_neg",
    "fp8_rows_mmt_f32_neg",
    W_TO_BF16,
];

const W_TO_BF16: &str = "fp8_w_to_bf16";

/// Entry points of gemm_nax_fp8.metal (metal4.0 library): `[bf16 out, f32 out]` per `gemm_nax::NAX_VARIANTS` tile, same order.
pub const NAX_FP8_SHADERS: [[&str; 2]; 5] = [
    ["gemm_nax_fp8_bf16_t64x64w4", "gemm_nax_fp8_f32_t64x64w4"],
    ["gemm_nax_fp8_bf16_t128x64w4", "gemm_nax_fp8_f32_t128x64w4"],
    ["gemm_nax_fp8_bf16_t32x32w1", "gemm_nax_fp8_f32_t32x32w1"],
    ["gemm_nax_fp8_bf16_t32x128w2", "gemm_nax_fp8_f32_t32x128w2"],
    ["gemm_nax_fp8_bf16_t64x64w2", "gemm_nax_fp8_f32_t64x64w2"],
];

/// Simdgroups per GEMV threadgroup (`FP8_SG`), output rows per simdgroup of `_m1` / `_m8` (`RS`), activation rows of `_m8`.
const GEMV_SG: usize = 8;
const GEMV_RS: [usize; 2] = [4, 2];
const GEMV_MB: usize = 8;
/// Largest threadgroup memory a GEMV may stage its activations in (Metal's per-threadgroup limit).
pub const GEMV_TG_BYTES: usize = 32768;
/// GEMM tile (`FP8_BM` / `FP8_BN`) and threads per threadgroup.
const TILE: usize = 64;
const MMA_THREADS: usize = 128;

/// Largest M of the decode GEMV for K: the staged activations (`M * K` bf16) must fit threadgroup memory.
pub fn gemv_ok(s: &ShapeInfo) -> bool {
    (1..=GEMV_MB).contains(&s.m) && s.k % 16 == 0 && s.m * s.k * 2 <= GEMV_TG_BYTES
}

/// The row-exact multi-row GEMV serves 2..=8 rows; 4 rows of activations must fit threadgroup memory (K <= 4096), 8 rows
/// are staged in one pass up to K = 2048 and in two passes of 4 (the `_m4` shader) above.
pub fn rows_ok(s: &ShapeInfo) -> bool {
    (2..=crate::CHUNK_ROWS_MAX).contains(&s.m) && s.k % 16 == 0 && 4 * s.k * 2 <= GEMV_TG_BYTES
}

/// GEMM scratch: the QDQ'd activations as bf16 `[M,K]`.
pub fn scratch(s: &ShapeInfo) -> usize {
    s.m * s.k * 2 + 16
}

/// Largest weight (`N * K` elements) the NAX form decodes into its scratch per call (bf16: 256 MB); larger ones keep `mma`.
const NAX_MAX_W: usize = 128 << 20;

/// `INTARSIA_METAL_FP8_NAX=0` turns the NAX form off (`matmul_fp8_mma` at every M > 8 again, bitwise the previous build).
pub fn nax_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_FP8_NAX").map_or(true, |v| v != "0"))
}

/// The NAX form serves the calls above the decode GEMV's rows (M > 8) with K % 16 == 0 and a weight that fits [`NAX_MAX_W`].
pub fn nax_ok(s: &ShapeInfo) -> bool {
    nax_enabled() && s.m > GEMV_MB && s.k % 16 == 0 && s.k >= 16 && s.n * s.k <= NAX_MAX_W
}

thread_local! {
    static NAX_TILE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Test hook: run `f` with every `matmul_fp8_nax` launch of this thread on tile `gemm_nax::NAX_VARIANTS[tile]` instead of
/// `pick_variant`'s (the tiles must give the same bits, tests/fp8_nax.rs).
pub fn with_nax_tile<R>(tile: usize, f: impl FnOnce() -> R) -> R {
    NAX_TILE.with(|t| t.set(Some(tile)));
    let r = f();
    NAX_TILE.with(|t| t.set(None));
    r
}

/// NAX scratch: the QDQ'd activations (bf16 `[M,K]`), then the weights decoded to bf16 (`[N,K]`), 16 bytes of alignment slack.
pub fn nax_scratch(s: &ShapeInfo) -> usize {
    (s.m + s.n) * s.k * 2 + 16
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal matmul_fp8: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_fp8(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (x, w, sc, out) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w, sc], [out]) => (x, w, sc, out),
            _ => return Err(Error::invalid("metal matmul_fp8: expected 3 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 || out.shape.len() != 2 {
            return Err(Error::invalid("metal matmul_fp8: x, w and out must be 2-D"));
        }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        if k % 16 != 0 || w.shape[1] != k || sc.numel() != 2 || out.shape[0] != m || out.shape[1] != n {
            return Err(Error::invalid("metal matmul_fp8: shape mismatch (K % 16 == 0, scales [2])"));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let f32_out = match out.dtype {
            DType::Bf16 => false,
            DType::F32 => true,
            d => return Err(Error::invalid(format!("metal matmul_fp8: output dtype {d}"))),
        };
        // the shaders index with 32-bit math in places
        u32_of(m * k, "M*K")?;
        u32_of(n * k, "N*K")?;
        u32_of(m * n, "M*N")?;
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let wb = self.bind(w, DType::Fp8E4m3, false, "w")?;
        let sb = self.bind(sc, DType::F32, false, "scales")?;
        let ob = self.bind(out, if f32_out { DType::F32 } else { DType::Bf16 }, true, "out")?;
        if wb.skip != 0 || wb.off % 8 != 0 {
            return Err(Error::invalid(format!(
                "metal matmul_fp8: the weights must be 8-byte aligned (bound at byte {}, skip {})",
                wb.off, wb.skip
            )));
        }
        let pdims = [u32_of(m, "M")?, u32_of(n, "N")?, u32_of(k, "K")?, 0, 0, 0, 0, 0];
        if kernel == MATMUL_FP8_GEMV_ROWS && call.scratch.is_some() && xd_on() {
            return self.launch_fp8_rows_xd(call, (&xb, &wb, &sb, &ob), pdims, f32_out);
        }
        if kernel == MATMUL_FP8_GEMV || kernel == MATMUL_FP8_GEMV_ROWS {
            let rows = kernel == MATMUL_FP8_GEMV_ROWS;
            if m > GEMV_MB || m * k * 2 > GEMV_TG_BYTES && !(rows && 4 * k * 2 <= GEMV_TG_BYTES) {
                return Err(Error::invalid(format!("metal {kernel}: M = {m}, K = {k} does not fit")));
            }
            if rows && m < 2 {
                return Err(Error::invalid(format!("metal {kernel}: M = {m} (2..=8 rows)")));
            }
            // activation rows per threadgroup: the serving GEMV 1 / 8; the row-exact form 8, or 4 where 8 rows do not fit
            let mb = if m == 1 {
                1
            } else if rows && m * k * 2 > GEMV_TG_BYTES {
                4
            } else {
                GEMV_MB
            };
            let (name, rs) = match (f32_out, mb) {
                (false, 1) => (SHADERS[1], GEMV_RS[0]),
                (false, 4) => (SHADERS[7], GEMV_RS[1]),
                (false, _) => (SHADERS[2], GEMV_RS[1]),
                (true, 1) => (SHADERS[3], GEMV_RS[0]),
                (true, 4) => (SHADERS[8], GEMV_RS[1]),
                (true, _) => (SHADERS[4], GEMV_RS[1]),
            };
            let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
            if pso.maxTotalThreadsPerThreadgroup() < GEMV_SG * 32 {
                return Err(Error::backend(format!("metal: '{name}' cannot run {} threads", GEMV_SG * 32)));
            }
            let p = OpP { dims: pdims, skips: [xb.skip, ob.skip, 0, 0, 0], ..Default::default() };
            let tg_bytes = (m.min(mb) * k * 2).next_multiple_of(16); // the staged rows of one threadgroup
            return self.with_encoder(|enc| {
                // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`).
                unsafe {
                    enc.setComputePipelineState(pso);
                    for (i, b) in [&xb, &wb, &sb, &ob].into_iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                    }
                    enc.setThreadgroupMemoryLength_atIndex(tg_bytes, 0);
                }
                set_bytes(enc, &p, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n.div_ceil(GEMV_SG * rs), height: m.div_ceil(mb), depth: 1 },
                    MTLSize { width: GEMV_SG * 32, height: 1, depth: 1 },
                );
            });
        }
        if kernel == MATMUL_FP8_NAX {
            return self.launch_fp8_nax(call, (&xb, &wb, &sb, &ob), pdims, f32_out);
        }
        if kernel != MATMUL_FP8_MMA {
            return Err(Error::NoKernel(format!("metal kernel '{kernel}' for op 'matmul_fp8'")));
        }
        let need = m * k * 2;
        let scratch = call
            .scratch
            .as_ref()
            .ok_or_else(|| Error::invalid("metal matmul_fp8_mma: the scratch workspace (OpCall::scratch) is missing"))?;
        if scratch.numel() * scratch.dtype.size_bytes() < need {
            return Err(Error::invalid(format!(
                "metal matmul_fp8_mma: scratch of {} bytes, need {need}",
                scratch.numel() * scratch.dtype.size_bytes()
            )));
        }
        let xs = self.bind(scratch, DType::U8, true, "scratch")?;
        if xs.skip != 0 {
            return Err(Error::invalid("metal matmul_fp8_mma: the scratch must be 4-byte aligned"));
        }
        let gemm = if f32_out { SHADERS[6] } else { SHADERS[5] };
        let pso_x = self.pso(XPREP).ok_or_else(|| Error::backend(format!("metal: no pipeline '{XPREP}'")))?;
        let pso_g = self.pso(gemm).ok_or_else(|| Error::backend(format!("metal: no pipeline '{gemm}'")))?;
        if pso_g.maxTotalThreadsPerThreadgroup() < MMA_THREADS {
            return Err(Error::backend(format!("metal: '{gemm}' cannot run {MMA_THREADS} threads")));
        }
        let px = OpP { dims: pdims, skips: [xb.skip, 0, 0, 0, 0], ..Default::default() };
        let pg = OpP { skips: [0, ob.skip, 0, 0, 0], ..px };
        let tx = pso_x.maxTotalThreadsPerThreadgroup().min(256);
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`, the scratch size above).
            unsafe {
                enc.setComputePipelineState(pso_x);
                for (i, b) in [&xb, &sb, &xs].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &px, 3);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: (m * k).div_ceil(8), height: 1, depth: 1 },
                MTLSize { width: tx, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso_g);
                for (i, b) in [&xs, &wb, &sb, &ob].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &pg, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n.div_ceil(TILE), height: m.div_ceil(TILE), depth: 1 },
                MTLSize { width: MMA_THREADS, height: 1, depth: 1 },
            );
        })
    }

    /// `matmul_fp8_nax` in three dispatches: q(x) of all the rows (`fp8_xprep_bf16`) and the weights decoded to bf16
    /// (`fp8_w_to_bf16`) into the scratch, then the NAX GEMM (`gemm_nax_fp8_*`, tile by `gemm_nax::pick_variant`) that
    /// scales and stores bf16 / f32. Scratch: `[xs: M*K bf16][Wb: N*K bf16]` from the next 16-byte boundary.
    fn launch_fp8_nax(
        &self,
        call: &OpCall,
        (xb, wb, sb, ob): (&crate::dense::Bind, &crate::dense::Bind, &crate::dense::Bind, &crate::dense::Bind),
        pdims: [u32; 8],
        f32_out: bool,
    ) -> Result<()> {
        use crate::gemm_nax::{grid, pick_variant, NAX_VARIANTS};
        let (m, n, k) = (pdims[0] as usize, pdims[1] as usize, pdims[2] as usize);
        let scratch = call
            .scratch
            .as_ref()
            .ok_or_else(|| Error::invalid("metal matmul_fp8_nax: the scratch workspace (OpCall::scratch) is missing"))?;
        let scratch_bytes = scratch.numel() * scratch.dtype.size_bytes();
        let sb_bind = self.bind(scratch, DType::U8, true, "scratch")?;
        if sb_bind.skip != 0 {
            return Err(Error::invalid("metal matmul_fp8_nax: the scratch must be 4-byte aligned"));
        }
        // the GEMM reads both operands from the next 16-byte boundary inside the scratch's 16 bytes of slack
        let pad = sb_bind.off.next_multiple_of(16) - sb_bind.off;
        let (xbytes, wbytes) = (m * k * 2, n * k * 2);
        if scratch_bytes < pad + xbytes + wbytes {
            return Err(Error::invalid(format!(
                "metal matmul_fp8_nax: scratch of {scratch_bytes} bytes at offset {}, need {}",
                sb_bind.off,
                pad + xbytes + wbytes
            )));
        }
        let xs = crate::dense::Bind { mtl: sb_bind.mtl.clone(), off: sb_bind.off + pad, skip: 0 };
        let wbf = crate::dense::Bind { mtl: sb_bind.mtl.clone(), off: sb_bind.off + pad + xbytes, skip: 0 };
        let v = match NAX_TILE.with(|t| t.get()) {
            Some(i) => &NAX_VARIANTS[i],
            None => pick_variant(m, n),
        };
        let vi = NAX_VARIANTS.iter().position(|x| x.shader == v.shader).expect("NAX variant table");
        let gemm = NAX_FP8_SHADERS[vi][f32_out as usize];
        let pso_x = self.pso(XPREP).ok_or_else(|| Error::backend(format!("metal: no pipeline '{XPREP}'")))?;
        let pso_w = self.pso(W_TO_BF16).ok_or_else(|| Error::backend(format!("metal: no pipeline '{W_TO_BF16}'")))?;
        let pso_g = self.pso(gemm).ok_or_else(|| {
            Error::backend(format!("metal: no pipeline '{gemm}' (metal4.0 library not loaded)"))
        })?;
        let px = OpP { dims: pdims, skips: [xb.skip, 0, 0, 0, 0], ..Default::default() };
        let pw = OpP { dims: pdims, ..Default::default() };
        let pg = NaxFp8Params { m: pdims[0], n: pdims[1], k: pdims[2], swz_log: 0, s_c: ob.skip, pad: [0; 3] };
        let tx = pso_x.maxTotalThreadsPerThreadgroup().min(256);
        let tw = pso_w.maxTotalThreadsPerThreadgroup().min(256);
        let (gx, gy) = grid(v, pg.m, pg.n, 0);
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds checked by `OpCall::check`, the scratch size above).
            unsafe {
                enc.setComputePipelineState(pso_x);
                for (i, b) in [xb, sb, &xs].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &px, 3);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: (m * k).div_ceil(8), height: 1, depth: 1 },
                MTLSize { width: tx, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso_w);
                for (i, b) in [wb, &wbf].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &pw, 2);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: (n * k).div_ceil(16), height: 1, depth: 1 },
                MTLSize { width: tw, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso_g);
                for (i, b) in [&xs, &wbf, sb, ob].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &pg, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: gx, height: gy, depth: 1 },
                MTLSize { width: v.threads() as usize, height: 1, depth: 1 },
            );
        })
    }

    /// `matmul_fp8_gemv_rows` in two dispatches: q(x) of all the rows once into the scratch (`fp8_xprep_bf16`, the
    /// expression the staged GEMV evaluates per threadgroup), then the `_xd` multi-row GEMV reading it from device memory.
    /// Every row's bits are the one-row kernel's (tests/rows_exact.rs).
    fn launch_fp8_rows_xd(
        &self,
        call: &OpCall,
        (xb, wb, sb, ob): (&crate::dense::Bind, &crate::dense::Bind, &crate::dense::Bind, &crate::dense::Bind),
        pdims: [u32; 8],
        f32_out: bool,
    ) -> Result<()> {
        let (m, n, k) = (pdims[0] as usize, pdims[1] as usize, pdims[2] as usize);
        if !(2..=GEMV_MB).contains(&m) {
            return Err(Error::invalid(format!("metal {MATMUL_FP8_GEMV_ROWS}: M = {m} (2..=8 rows)")));
        }
        let scratch = call.scratch.as_ref().expect("checked by the caller");
        let xs = self.bind(scratch, DType::U8, true, "scratch")?;
        // the GEMV reads q(x) as uint4: start it at the next 16-byte boundary inside the scratch's 16 bytes of slack
        let pad = xs.off.next_multiple_of(16) - xs.off;
        let need = rows_scratch_q(m, k);
        if xs.skip != 0 || scratch.numel() * scratch.dtype.size_bytes() < need + pad {
            return Err(Error::invalid(format!(
                "metal {MATMUL_FP8_GEMV_ROWS}: scratch of {} bytes at offset {}, need {} (4-byte aligned)",
                scratch.numel() * scratch.dtype.size_bytes(),
                xs.off,
                need + pad
            )));
        }
        let xs = crate::dense::Bind { mtl: xs.mtl.clone(), off: xs.off + pad, skip: 0 };
        if m >= mmt_min() {
            return self.launch_fp8_rows_mmt(&xs, (xb, wb, sb, ob), pdims, f32_out);
        }
        let neg = xd_neg();
        let (name, rs, mb) = match (f32_out, neg) {
            (false, false) => (SHADERS[9], GEMV_RS[1], XD_MB),
            (true, false) => (SHADERS[10], GEMV_RS[1], XD_MB),
            (false, true) => (SHADERS[11], GEMV_RS[1], XD_MB),
            (true, true) => (SHADERS[12], GEMV_RS[1], XD_MB),
        };
        let pso_x = self.pso(XPREP).ok_or_else(|| Error::backend(format!("metal: no pipeline '{XPREP}'")))?;
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        if pso.maxTotalThreadsPerThreadgroup() < GEMV_SG * 32 {
            return Err(Error::backend(format!("metal: '{name}' cannot run {} threads", GEMV_SG * 32)));
        }
        let px = OpP { dims: pdims, skips: [xb.skip, 0, 0, 0, 0], ..Default::default() };
        let pg = OpP { dims: pdims, skips: [0, ob.skip, 0, 0, 0], ..Default::default() };
        let tx = pso_x.maxTotalThreadsPerThreadgroup().min(256);
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds checked by `OpCall::check`, the scratch size above).
            unsafe {
                enc.setComputePipelineState(pso_x);
                for (i, b) in [xb, sb, &xs].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &px, 3);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: (m * k).div_ceil(8), height: 1, depth: 1 },
                MTLSize { width: tx, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in [&xs, wb, sb, ob].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &pg, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n.div_ceil(GEMV_SG * rs), height: m.div_ceil(mb), depth: 1 },
                MTLSize { width: GEMV_SG * 32, height: 1, depth: 1 },
            );
        })
    }

    /// `matmul_fp8_gemv_rows` on simdgroup matrices (M >= [`mmt_min`]): the block-transposed q(x) of all rows, padded to
    /// 8, into the scratch (`fp8_xprep_t`), then one threadgroup per 8 outputs (`fp8_rows_mmt_*`). Every row's bits are
    /// the one-row kernel's (tests/rows_exact.rs).
    fn launch_fp8_rows_mmt(
        &self,
        xs: &crate::dense::Bind,
        (xb, wb, sb, ob): (&crate::dense::Bind, &crate::dense::Bind, &crate::dense::Bind, &crate::dense::Bind),
        mut pdims: [u32; 8],
        f32_out: bool,
    ) -> Result<()> {
        let (m, n, k) = (pdims[0] as usize, pdims[1] as usize, pdims[2] as usize);
        debug_assert!(m <= MMT_ROWS);
        pdims[3] = MMT_ROWS as u32;
        let name = match (f32_out, mmt_neg()) {
            (false, false) => SHADERS[14],
            (true, false) => SHADERS[15],
            (false, true) => SHADERS[16],
            (true, true) => SHADERS[17],
        };
        let pso_x = self.pso(XPREP_T).ok_or_else(|| Error::backend(format!("metal: no pipeline '{XPREP_T}'")))?;
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        if pso.maxTotalThreadsPerThreadgroup() < GEMV_SG * 32 {
            return Err(Error::backend(format!("metal: '{name}' cannot run {} threads", GEMV_SG * 32)));
        }
        let px = OpP { dims: pdims, skips: [xb.skip, 0, 0, 0, 0], ..Default::default() };
        let pg = OpP { dims: pdims, skips: [0, ob.skip, 0, 0, 0], ..Default::default() };
        let tx = pso_x.maxTotalThreadsPerThreadgroup().min(256);
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds checked by `OpCall::check`, the scratch size by the caller).
            unsafe {
                enc.setComputePipelineState(pso_x);
                for (i, b) in [xb, sb, xs].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &px, 3);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: k * MMT_ROWS / 2, height: 1, depth: 1 },
                MTLSize { width: tx, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in [xs, wb, sb, ob].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &pg, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n.div_ceil(8), height: 1, depth: 1 },
                MTLSize { width: GEMV_SG * 32, height: 1, depth: 1 },
            );
        })
    }
}

/// Kernel arguments of `gemm_nax_fp8_t` (`NaxFp8P` in gemm_nax_fp8.metal; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NaxFp8Params {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub swz_log: u32,
    /// Leading elements (0 or 1) of C past its 4-byte aligned binding.
    pub s_c: u32,
    pub pad: [u32; 3],
}

/// Rows of the simdgroup-matrix form (`fp8_rows_mmt_*`; its q(x) is padded to this many).
const MMT_ROWS: usize = 8;

/// Fewest rows that take the simdgroup-matrix form: 5 (it costs the same at every M, the `_xd` form one 4-row tile more
/// from M = 5 on; endor timings in docs/results/2026-10-25-mm-metal-fp8-mmt.md). `INTARSIA_METAL_FP8_ROWS_MMT=0` turns
/// it off (the `_xd` form at every M), `=N` (2..=8) moves the threshold (tests use 2).
fn mmt_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("INTARSIA_METAL_FP8_ROWS_MMT") {
        Ok(v) => match v.parse::<usize>() {
            Ok(0) => usize::MAX,
            Ok(n) if (2..=MMT_ROWS).contains(&n) => n,
            _ => 5,
        },
        Err(_) => 5,
    })
}

/// `INTARSIA_METAL_NEGCTL=fp8_mmt`: the simdgroup-matrix GEMV with the two MMAs of a block swapped (tests only: NOT the
/// solo bits).
fn mmt_neg() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_NEGCTL").is_ok_and(|v| v == "fp8_mmt"))
}

/// q(x) bytes the `matmul_fp8_gemv_rows` form for M rows reads from the scratch (without the 16 bytes of alignment slack).
fn rows_scratch_q(m: usize, k: usize) -> usize {
    if m >= mmt_min() { MMT_ROWS * k * 2 } else { m * k * 2 }
}

/// The `_xd` form of `matmul_fp8_gemv_rows` (on unless `INTARSIA_METAL_FP8_ROWS_XD=0`, which takes the staged form back).
fn xd_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_FP8_ROWS_XD").map_or(true, |v| v != "0"))
}

/// Scratch of `matmul_fp8_gemv_rows`: the q(x) rows (`_xd` form: M rows; simdgroup-matrix form: padded to 8).
pub fn rows_scratch(s: &ShapeInfo) -> usize {
    s.m.max(MMT_ROWS) * s.k * 2 + 16
}

/// `INTARSIA_METAL_NEGCTL=fp8_xd`: the `_xd` GEMV with another association on odd blocks (tests only: NOT the solo bits).
fn xd_neg() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_NEGCTL").is_ok_and(|v| v == "fp8_xd"))
}

/// Rows per threadgroup of the `_xd` GEMV.
const XD_MB: usize = 4;

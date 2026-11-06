//! Launcher of `matmul_mxfp4_gemv` (`Op::MatmulMxfp4`, kernels/metal/mxfp4.metal; the port of kernels/hip/mxfp4.hip).
//!
//! `out[M,N] = q(x)[M,K] @ W^T`, W = packed e2m1 `u8 [N,K/2]` with one e8m0 scale per 32 along K (`u8 [N,K/32]`), `act
//! [1] f32` picking W4A16 (0, x as is) or W4A4 (1, x QDQ'd to MXFP4 per group of 32, Quark "even" scale). Two dispatches
//! in one launch: the activations are QDQ'd (or copied) once into the plan-reserved scratch (`bf16 [M,K]`, exact), then a
//! simdgroup-per-output-row GEMV reads them. Any M: 1 row uses the `_m1` shader, more take 8 activation rows per
//! threadgroup row of the grid.

use crate::dense::OpP;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// `KernelImpl::name` of the kernel.
pub const MATMUL_MXFP4_GEMV: &str = "matmul_mxfp4_gemv";
/// The prefill form (M > 8): the weights decoded to bf16 into the scratch, then the fp8 NAX GEMM on the neural accelerators
/// (needs the metal4.0 library, `INTARSIA_METAL_MXFP4_NAX`).
pub const MATMUL_MXFP4_NAX: &str = "matmul_mxfp4_nax";

const XPREP: &str = "mxfp4_xprep_bf16";
/// Weight decode of the NAX form (mxfp4.metal).
const W_TO_BF16: &str = "mxfp4_w_to_bf16";
/// Shader entry points (kernels/metal/mxfp4.metal).
pub const SHADERS: [&str; 6] =
    [XPREP, "mxfp4_gemv_bf16_m1", "mxfp4_gemv_bf16_m8", "mxfp4_gemv_f32_m1", "mxfp4_gemv_f32_m8", W_TO_BF16];

/// Lane df: the M-row GEMV with P output rows per simdgroup (`mxfp4_gemv_mr_impl<M, P>`, M = 2..=8, P = 2 or 4), bitwise the `_m1`
/// shader per row (tests/mxfp4_gemv.rs). Order: P 2 then 4, M 2..=8, bf16 then f32 output.
pub const MR_SHADERS: [&str; 28] = [
    "mxfp4_gemv_bf16_mr2p2",
    "mxfp4_gemv_f32_mr2p2",
    "mxfp4_gemv_bf16_mr3p2",
    "mxfp4_gemv_f32_mr3p2",
    "mxfp4_gemv_bf16_mr4p2",
    "mxfp4_gemv_f32_mr4p2",
    "mxfp4_gemv_bf16_mr5p2",
    "mxfp4_gemv_f32_mr5p2",
    "mxfp4_gemv_bf16_mr6p2",
    "mxfp4_gemv_f32_mr6p2",
    "mxfp4_gemv_bf16_mr7p2",
    "mxfp4_gemv_f32_mr7p2",
    "mxfp4_gemv_bf16_mr8p2",
    "mxfp4_gemv_f32_mr8p2",
    "mxfp4_gemv_bf16_mr2p4",
    "mxfp4_gemv_f32_mr2p4",
    "mxfp4_gemv_bf16_mr3p4",
    "mxfp4_gemv_f32_mr3p4",
    "mxfp4_gemv_bf16_mr4p4",
    "mxfp4_gemv_f32_mr4p4",
    "mxfp4_gemv_bf16_mr5p4",
    "mxfp4_gemv_f32_mr5p4",
    "mxfp4_gemv_bf16_mr6p4",
    "mxfp4_gemv_f32_mr6p4",
    "mxfp4_gemv_bf16_mr7p4",
    "mxfp4_gemv_f32_mr7p4",
    "mxfp4_gemv_bf16_mr8p4",
    "mxfp4_gemv_f32_mr8p4",
];

/// Lane dm: M = 2..=8 on the simdgroup matrix instructions (`mxfp4_gemv_*_mma`, knob value 8; bitwise only on archs where fp32
/// `simdgroup_multiply_accumulate` is a sequential k-ascending chain, see mxfp4.metal), bf16 then f32 out.
pub const MMA_SHADERS: [&str; 2] = ["mxfp4_gemv_bf16_mma", "mxfp4_gemv_f32_mma"];

/// Lane dm: `MR_SHADERS` with the codes decoded through a threadgroup byte table (knob values 12 / 14 = P 2 / 4; same order as `MR_SHADERS`).
pub const MR_LUT_SHADERS: [&str; 28] = [
    "mxfp4_gemv_bf16_mr2p2l",
    "mxfp4_gemv_f32_mr2p2l",
    "mxfp4_gemv_bf16_mr3p2l",
    "mxfp4_gemv_f32_mr3p2l",
    "mxfp4_gemv_bf16_mr4p2l",
    "mxfp4_gemv_f32_mr4p2l",
    "mxfp4_gemv_bf16_mr5p2l",
    "mxfp4_gemv_f32_mr5p2l",
    "mxfp4_gemv_bf16_mr6p2l",
    "mxfp4_gemv_f32_mr6p2l",
    "mxfp4_gemv_bf16_mr7p2l",
    "mxfp4_gemv_f32_mr7p2l",
    "mxfp4_gemv_bf16_mr8p2l",
    "mxfp4_gemv_f32_mr8p2l",
    "mxfp4_gemv_bf16_mr2p4l",
    "mxfp4_gemv_f32_mr2p4l",
    "mxfp4_gemv_bf16_mr3p4l",
    "mxfp4_gemv_f32_mr3p4l",
    "mxfp4_gemv_bf16_mr4p4l",
    "mxfp4_gemv_f32_mr4p4l",
    "mxfp4_gemv_bf16_mr5p4l",
    "mxfp4_gemv_f32_mr5p4l",
    "mxfp4_gemv_bf16_mr6p4l",
    "mxfp4_gemv_f32_mr6p4l",
    "mxfp4_gemv_bf16_mr7p4l",
    "mxfp4_gemv_f32_mr7p4l",
    "mxfp4_gemv_bf16_mr8p4l",
    "mxfp4_gemv_f32_mr8p4l",
];

/// `INTARSIA_METAL_MXFP4_MR`: unset or `1` (default, user decision 2026-11-06) = the M = 2..=8 calls of `matmul_mxfp4_gemv` take
/// `MR_SHADERS` where they measured faster ([`mr_auto`]); `0` = the `_m8` shader (the pre-2026-11-06 path); `2` or `4` = every M = 2..=8
/// call with that P (A/B); `8` = every M = 2..=8 call on `MMA_SHADERS` (lane dm, opt-in); `12` / `14` = P 2 / 4 with the table decode (`MR_LUT_SHADERS`, lane dm). Bitwise the same output in every setting (tests/mxfp4_gemv.rs).
pub fn mr_knob() -> usize {
    static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *P.get_or_init(|| match std::env::var("INTARSIA_METAL_MXFP4_MR").ok().as_deref().map(str::trim) {
        Some("0") => 0,
        Some("2") => 2,
        Some("4") => 4,
        Some("8") => 8,
        Some("12") => 12,
        Some("14") => 14,
        _ => 1,
    })
}

/// The P of the auto setting for an M x N call, 0 = the `_m8` shader. From the 27B-shape bench (docs/results/2026-11-06-27b-dflash-gap.md,
/// M5 Max): M 2..=4 and 8 gain (M = 3: 2.1-2.3x of M = 1 down to 1.25-1.45x); M 5..=7 measured slower than `_m8` and keep it; P = 4
/// wins for outputs of at most 8192 rows at M <= 3 (the down and out projections), P = 2 elsewhere.
pub fn mr_auto(m: usize, n: usize) -> usize {
    match m {
        2 | 3 if n <= 8192 => 4,
        2..=4 | 8 => 2,
        _ => 0,
    }
}

thread_local! {
    static MR_FORCE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Test hook: run `f` with this thread's `matmul_mxfp4_gemv` launches as the knob value `p` (0 = the `_m8` shader, 1 = auto, 2 / 4 = that P) whatever the env says.
pub fn with_mr<R>(p: usize, f: impl FnOnce() -> R) -> R {
    MR_FORCE.with(|t| t.set(Some(p)));
    let r = f();
    MR_FORCE.with(|t| t.set(None));
    r
}

fn mr_shader(f32_out: bool, m: usize, p: usize) -> &'static str {
    MR_SHADERS[((usize::from(p == 4) * 7 + (m - 2)) * 2) + usize::from(f32_out)]
}

/// Output rows (= simdgroups) per threadgroup (`MX_SG` in the shader) and activation rows per grid row of `_m8`.
const ROWS_PER_TG: usize = 4;
const M_BLOCK: usize = 8;

/// Scratch: the QDQ'd activations as bf16 `[M,K]` (same bound as the hip entry).
pub fn scratch(s: &ShapeInfo) -> usize {
    s.m * s.k * 2 + 16
}

/// Largest weight (`N * K` elements) the NAX form decodes into its scratch per call (bf16: 256 MB); larger ones keep the GEMV.
const NAX_MAX_W: usize = 128 << 20;

/// Rows of the decode GEMV's launch; the NAX form takes the calls above it.
const GEMV_MB: usize = 8;

/// `INTARSIA_METAL_MXFP4_NAX=0` turns the NAX form off (`matmul_mxfp4_gemv` at every M again, bitwise the previous build).
pub fn nax_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_MXFP4_NAX").map_or(true, |v| v != "0"))
}

/// `INTARSIA_METAL_MXFP4_XFUSE=0` keeps the activation pre-pass dispatch before every GEMV (the GEMV then reads the copy, as before the fusion;
/// bitwise the same result). The fusion is for W4A16 calls only (`OpAttrs::mxfp4_w4a16`): a QDQ in every output row's simdgroup measured 1.9x to 3.4x slower.
pub fn xfuse_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_MXFP4_XFUSE").map_or(true, |v| v != "0"))
}

/// The NAX form serves the calls above the GEMV's 8 rows with K % 32 == 0 and a weight that fits [`NAX_MAX_W`].
pub fn nax_ok(s: &ShapeInfo) -> bool {
    nax_enabled() && !s.mxfp4_a8 && s.m > GEMV_MB && s.k >= 32 && s.k % 32 == 0 && s.n * s.k <= NAX_MAX_W
}

/// NAX scratch: the QDQ'd activations (bf16 `[M,K]`), then, from the next 16-byte boundary, the decoded weights (bf16 `[N,K]`)
/// and the GEMM's scale pair (2 f32); 16 + 16 bytes of slack for the two alignments.
pub fn nax_scratch(s: &ShapeInfo) -> usize {
    (s.m * s.k * 2).next_multiple_of(16) + s.n * s.k * 2 + 8 + 32
}

thread_local! {
    static NAX_TILE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Test hook: run `f` with every `matmul_mxfp4_nax` launch of this thread on tile `gemm_nax::NAX_VARIANTS[tile]` instead of
/// `pick_variant`'s (the tiles must give the same bits).
pub fn with_nax_tile<R>(tile: usize, f: impl FnOnce() -> R) -> R {
    NAX_TILE.with(|t| t.set(Some(tile)));
    let r = f();
    NAX_TILE.with(|t| t.set(None));
    r
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal matmul_mxfp4_gemv: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_mxfp4(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (x, w, sc, act, out) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w, sc, act], [out]) => (x, w, sc, act, out),
            _ => return Err(Error::invalid("metal matmul_mxfp4_gemv: expected 4 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 || sc.shape.len() != 2 || out.shape.len() != 2 {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: x, w, scale and out must be 2-D"));
        }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        if k % 32 != 0
            || w.shape[1] != k / 2
            || sc.shape[0] != n
            || sc.shape[1] != k / 32
            || act.numel() != 1
            || out.shape[0] != m
            || out.shape[1] != n
        {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: shape mismatch"));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let f32_out = match out.dtype {
            DType::Bf16 => false,
            DType::F32 => true,
            d => return Err(Error::invalid(format!("metal matmul_mxfp4_gemv: output dtype {d}"))),
        };
        // the shaders index with 32-bit math in places
        u32_of(m * k, "M*K")?;
        u32_of(n * (k / 2), "N*K/2")?;
        u32_of(m * n, "M*N")?;
        if k == 0 {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: K = 0"));
        }
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let wb = self.bind(w, DType::U8, false, "w")?;
        let sb = self.bind(sc, DType::U8, false, "scale")?;
        let ab = self.bind(act, DType::F32, false, "act")?;
        let ob = self.bind(out, if f32_out { DType::F32 } else { DType::Bf16 }, true, "out")?;
        if wb.skip != 0 {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: the weights must be 4-byte aligned"));
        }
        if kernel == MATMUL_MXFP4_NAX {
            return self.launch_mxfp4_nax(call, (&xb, &wb, &sb, &ab, &ob), [m, n, k], f32_out);
        }
        let need = m * k * 2 + 15;
        let scratch = call.scratch.as_ref().ok_or_else(|| {
            Error::invalid("metal matmul_mxfp4_gemv: the scratch workspace (OpCall::scratch) is missing")
        })?;
        if scratch.numel() * scratch.dtype.size_bytes() < need {
            return Err(Error::invalid(format!(
                "metal matmul_mxfp4_gemv: scratch of {} bytes, need {need}",
                scratch.numel() * scratch.dtype.size_bytes()
            )));
        }
        let xs = self.bind(scratch, DType::U8, true, "scratch")?;
        if xs.skip != 0 {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: the scratch must be 4-byte aligned"));
        }
        // the GEMV loads the activations 16 bytes at a time: they start at the next 16-byte boundary of the scratch
        let xs = crate::dense::Bind { mtl: xs.mtl.clone(), off: xs.off.next_multiple_of(16), skip: 0 };

        let mr = match MR_FORCE.with(|t| t.get()).unwrap_or_else(mr_knob) {
            1 => mr_auto(m, n),
            p => p,
        };
        let mr = if (2..=M_BLOCK).contains(&m) && matches!(mr, 2 | 4 | 8 | 12 | 14) { mr } else { 0 };
        let gemv = match (f32_out, m == 1) {
            _ if mr == 8 => MMA_SHADERS[usize::from(f32_out)],
            _ if mr > 8 => MR_LUT_SHADERS[((usize::from(mr == 14) * 7 + (m - 2)) * 2) + usize::from(f32_out)],
            _ if mr > 0 => mr_shader(f32_out, m, mr),
            (false, true) => SHADERS[1],
            (false, false) => SHADERS[2],
            (true, true) => SHADERS[3],
            (true, false) => SHADERS[4],
        };
        let pso_x = self.pso(XPREP).ok_or_else(|| Error::backend(format!("metal: no pipeline '{XPREP}'")))?;
        let pso_g = self.pso(gemv).ok_or_else(|| Error::backend(format!("metal: no pipeline '{gemv}'")))?;
        let px = OpP {
            dims: [u32_of(m, "M")?, u32_of(n, "N")?, u32_of(k, "K")?, 0, 0, 0, 0, 0],
            skips: [xb.skip, 0, 0, 0, 0],
            ..Default::default()
        };
        // a W4A16 call (act is 0, the pre-pass is a copy) with 16-byte aligned x: the GEMV reads x itself, no pre-pass dispatch
        let fused = xfuse_enabled() && call.attrs.mxfp4_w4a16 && xb.skip == 0 && xb.off % 16 == 0;
        let mut pg = OpP { skips: [sb.skip, ob.skip, 0, 0, 0], ..px };
        pg.dims[3] = fused as u32;
        let groups = m * (k / 32);
        let tx = pso_x.maxTotalThreadsPerThreadgroup().min(256);
        let m_blocks = if m == 1 { 1 } else { m.div_ceil(M_BLOCK) };
        self.with_encoder(|enc| {
            if !fused {
                // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`, the scratch size above).
                unsafe {
                    enc.setComputePipelineState(pso_x);
                    for (i, b) in [&xb, &ab, &xs].into_iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                    }
                }
                set_bytes(enc, &px, 3);
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: groups, height: 1, depth: 1 },
                    MTLSize { width: tx, height: 1, depth: 1 },
                );
            }
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso_g);
                for (i, b) in [&xs, &wb, &sb, &ob].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &pg, 4);
            // SAFETY: as above; x is the call's own input (the GEMV reads it when `fused`, else never touches it).
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&xb.mtl), xb.off, 5);
            }
            // the M-row shader: P output rows per simdgroup, one grid row (M <= 8)
            // the MMA shader: 8 output rows per simdgroup, MMA_SG = 2 simdgroups per threadgroup
            let (gx, gy, tw) = match mr {
                8 => (n.div_ceil(16), 1, 64),
                12 | 14 => (n.div_ceil(ROWS_PER_TG * (mr - 10)), 1, ROWS_PER_TG * 32),
                1.. => (n.div_ceil(ROWS_PER_TG * mr), 1, ROWS_PER_TG * 32),
                0 => (n.div_ceil(ROWS_PER_TG), m_blocks, ROWS_PER_TG * 32),
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: gx, height: gy, depth: 1 },
                MTLSize { width: tw, height: 1, depth: 1 },
            );
        })
    }
    /// `matmul_mxfp4_nax` in three dispatches: the activations QDQ'd (or copied) as in the GEMV form (`mxfp4_xprep_bf16`),
    /// the weights decoded to bf16 and the scale pair `[1, 0]` (`mxfp4_w_to_bf16`), then the fp8 NAX GEMM (`gemm_nax_fp8_*`,
    /// tile by `gemm_nax::pick_variant`) on those two, which stores bf16 / f32. Scratch: `[xs: M*K bf16]` then, from the next
    /// 16-byte boundary, `[Wb: N*K bf16][scale pair: 2 f32]`.
    fn launch_mxfp4_nax(
        &self,
        call: &OpCall,
        (xb, wb, sb, ab, ob): (
            &crate::dense::Bind,
            &crate::dense::Bind,
            &crate::dense::Bind,
            &crate::dense::Bind,
            &crate::dense::Bind,
        ),
        [m, n, k]: [usize; 3],
        f32_out: bool,
    ) -> Result<()> {
        use crate::gemm_nax::{grid, pick_variant, NAX_VARIANTS};
        let scratch = call.scratch.as_ref().ok_or_else(|| {
            Error::invalid("metal matmul_mxfp4_nax: the scratch workspace (OpCall::scratch) is missing")
        })?;
        let scratch_bytes = scratch.numel() * scratch.dtype.size_bytes();
        let sbind = self.bind(scratch, DType::U8, true, "scratch")?;
        if sbind.skip != 0 {
            return Err(Error::invalid("metal matmul_mxfp4_nax: the scratch must be 4-byte aligned"));
        }
        // the GEMM reads A and Wb as 16-byte aligned tensors: xs from the next 16-byte boundary, Wb from the one after xs
        let pad = sbind.off.next_multiple_of(16) - sbind.off;
        let xbytes = (m * k * 2).next_multiple_of(16);
        let wbytes = n * k * 2;
        if scratch_bytes < pad + xbytes + wbytes + 8 {
            return Err(Error::invalid(format!(
                "metal matmul_mxfp4_nax: scratch of {scratch_bytes} bytes at offset {}, need {}",
                sbind.off,
                pad + xbytes + wbytes + 8
            )));
        }
        let xs = crate::dense::Bind { mtl: sbind.mtl.clone(), off: sbind.off + pad, skip: 0 };
        let wbf = crate::dense::Bind { mtl: sbind.mtl.clone(), off: sbind.off + pad + xbytes, skip: 0 };
        let pair = crate::dense::Bind { mtl: sbind.mtl.clone(), off: sbind.off + pad + xbytes + wbytes, skip: 0 };
        let v = match NAX_TILE.with(|t| t.get()) {
            Some(i) => &NAX_VARIANTS[i],
            None => pick_variant(m, n),
        };
        let vi = NAX_VARIANTS.iter().position(|x| x.shader == v.shader).expect("NAX variant table");
        let gemm = crate::fp8::NAX_FP8_SHADERS[vi][f32_out as usize];
        let pso_x = self.pso(XPREP).ok_or_else(|| Error::backend(format!("metal: no pipeline '{XPREP}'")))?;
        let pso_w = self.pso(W_TO_BF16).ok_or_else(|| Error::backend(format!("metal: no pipeline '{W_TO_BF16}'")))?;
        let pso_g = self
            .pso(gemm)
            .ok_or_else(|| Error::backend(format!("metal: no pipeline '{gemm}' (metal4.0 library not loaded)")))?;
        let dims = [u32_of(m, "M")?, u32_of(n, "N")?, u32_of(k, "K")?, 0, 0, 0, 0, 0];
        let px = OpP { dims, skips: [xb.skip, 0, 0, 0, 0], ..Default::default() };
        let pw = OpP { dims, skips: [sb.skip, 0, 0, 0, 0], ..Default::default() };
        let pg = crate::fp8::NaxFp8Params { m: dims[0], n: dims[1], k: dims[2], swz_log: 0, s_c: ob.skip, pad: [0; 3] };
        let tx = pso_x.maxTotalThreadsPerThreadgroup().min(256);
        let tw = pso_w.maxTotalThreadsPerThreadgroup().min(256);
        let (gx, gy) = grid(v, pg.m, pg.n, 0);
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds checked by `OpCall::check`, the scratch size above).
            unsafe {
                enc.setComputePipelineState(pso_x);
                for (i, b) in [xb, ab, &xs].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &px, 3);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: m * (k / 32), height: 1, depth: 1 },
                MTLSize { width: tx, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso_w);
                for (i, b) in [wb, sb, &wbf, &pair].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &pw, 4);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: n * (k / 32), height: 1, depth: 1 },
                MTLSize { width: tw, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso_g);
                for (i, b) in [&xs, &wbf, &pair, ob].into_iter().enumerate() {
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
}

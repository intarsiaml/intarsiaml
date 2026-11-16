//! Launchers of `matmul_nvfp4_gemv` / `matmul_nvfp4_mma` (`Op::MatmulNvfp4`, kernels/metal/nvfp4.metal; the port of
//! kernels/hip/nvfp4.hip).
//!
//! `out[M,N] = x[M,K] W^T`, W = e2m1 `u8 [N,K/2]` * e4m3 block scales `[N,K/16]` * global scale `[1] f32` (read on the GPU).
//! The decode GEMV (M <= 8) is one dispatch without scratch; the GEMM (any M) is a tiled `simdgroup_matrix` bf16 GEMM that
//! dequantises the weight tiles (code * block scale, exact in bf16) and applies the global scale in the epilogue.
//! `matmul_nvfp4_nax` (M > 8, prefill) decodes the same bf16 weights into the plan scratch (`nv_w_to_bf16`) and runs the fp8
//! NAX GEMM on them (gemm_nax_fp8.metal: float accumulate on the neural accelerators, scaled by the global scale in
//! registers, rounded once); only the order of the K sum differs from `mma` (docs/results/2026-10-29-mpre-metal-prefill.md).

use crate::dense::OpP;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// `KernelImpl::name` of the kernels.
pub const MATMUL_NVFP4_GEMV: &str = "matmul_nvfp4_gemv";
pub const MATMUL_NVFP4_MMA: &str = "matmul_nvfp4_mma";
/// `rows_exact` multi-row GEMV (continuous batching of the hybrid 35B, docs/results/2026-10-23-mb-metal-hybrid-batching.md):
/// row r of an M-row launch (M = 2..=8) is bitwise the M = 1 `matmul_nvfp4_gemv` launch on row r alone (tests/rows_exact.rs).
pub const MATMUL_NVFP4_GEMV_ROWS: &str = "matmul_nvfp4_gemv_rows";
/// `matmul_nvfp4_mma`'s class on the neural accelerators (needs the metal4.0 library, `INTARSIA_METAL_NVFP4_NAX`).
pub const MATMUL_NVFP4_NAX: &str = "matmul_nvfp4_nax";

/// Shader entry points (kernels/metal/nvfp4.metal).
pub const SHADERS: [&str; 7] =
    ["nv_gemv_bf16_m1", "nv_gemv_bf16_m8", "nv_gemv_f32_m1", "nv_gemv_f32_m8", "nv_mma_bf16", "nv_mma_f32", W_TO_BF16];
/// Weight decode of the NAX form (nvfp4.metal).
const W_TO_BF16: &str = "nv_w_to_bf16";

/// The widest matrix whose multi-row GEMV runs one row per threadgroup row (lane mm): an N-row matrix with N at most this
/// gets the one-row shader over a grid of M threadgroup rows (each row is then the one-row launch itself; the weights come
/// again from cache), wider ones the 8-row shader. The 35B's shared-expert matrices (512 / 2048 rows) give the 8-row shader
/// only 32 / 128 threadgroups. `INTARSIA_METAL_NV_ROWS_SPLIT=0` keeps the 8-row shader everywhere.
const NARROW_N: usize = 4096;

fn nv_rows_mb(n: usize) -> usize {
    static SPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let split = *SPLIT.get_or_init(|| std::env::var("INTARSIA_METAL_NV_ROWS_SPLIT").map_or(true, |v| v != "0"));
    if split && n <= NARROW_N {
        1
    } else {
        GEMV_MB
    }
}

/// Simdgroups per GEMV threadgroup (`NV_SG`), output rows per simdgroup of `_m1` / `_m8` (`RS`), activation rows of `_m8`.
const GEMV_SG: usize = 8;
const GEMV_RS: [usize; 2] = [4, 2];
const GEMV_MB: usize = 8;
/// GEMM tile (`NV_BM` / `NV_BN`) and threads per threadgroup.
const TILE: usize = 64;
const MMA_THREADS: usize = 128;

/// The decode GEMV serves 1..=8 rows.
pub fn gemv_ok(s: &ShapeInfo) -> bool {
    (1..=GEMV_MB).contains(&s.m) && s.k % 16 == 0
}

/// The row-exact multi-row GEMV serves 2..=8 rows.
pub fn rows_ok(s: &ShapeInfo) -> bool {
    (2..=crate::CHUNK_ROWS_MAX).contains(&s.m) && s.k % 16 == 0
}

/// Largest weight (`N * K` elements) the NAX form decodes into its scratch per call (bf16: 256 MB); larger ones keep `mma`.
const NAX_MAX_W: usize = 128 << 20;

/// `INTARSIA_METAL_NVFP4_NAX=0` turns the NAX form off (`matmul_nvfp4_mma` at every M > 8 again, bitwise the previous build).
pub fn nax_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_NVFP4_NAX").map_or(true, |v| v != "0"))
}

/// The NAX form serves the calls above the decode GEMV's rows (M > 8) with K % 16 == 0 and a weight that fits [`NAX_MAX_W`].
pub fn nax_ok(s: &ShapeInfo) -> bool {
    nax_enabled() && s.m > GEMV_MB && s.k % 16 == 0 && s.k >= 16 && s.n * s.k <= NAX_MAX_W
}

/// NAX scratch: the decoded weights (bf16 `[N,K]`) from the next 16-byte boundary, then the GEMM's scale pair.
pub fn nax_scratch(s: &ShapeInfo) -> usize {
    s.n * s.k * 2 + 16 + 8
}

thread_local! {
    static NAX_TILE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Test hook: run `f` with every `matmul_nvfp4_nax` launch of this thread on tile `gemm_nax::NAX_VARIANTS[tile]` instead of
/// `pick_variant`'s (the tiles must give the same bits).
pub fn with_nax_tile<R>(tile: usize, f: impl FnOnce() -> R) -> R {
    NAX_TILE.with(|t| t.set(Some(tile)));
    let r = f();
    NAX_TILE.with(|t| t.set(None));
    r
}

static FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `matmul_nvfp4_nax` launches that ran `matmul_nvfp4_mma` instead (x or out not where the GEMM can read / write them).
pub fn nax_fallbacks() -> u64 {
    FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal matmul_nvfp4: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_nvfp4(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (x, w, bs, gs, out) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w, bs, gs], [out]) => (x, w, bs, gs, out),
            _ => return Err(Error::invalid("metal matmul_nvfp4: expected 4 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 || bs.shape.len() != 2 || out.shape.len() != 2 {
            return Err(Error::invalid("metal matmul_nvfp4: x, w, block scales and out must be 2-D"));
        }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        if k % 16 != 0
            || w.shape[1] != k / 2
            || bs.shape[0] != n
            || bs.shape[1] != k / 16
            || gs.numel() != 1
            || out.shape[0] != m
            || out.shape[1] != n
        {
            return Err(Error::invalid("metal matmul_nvfp4: shape mismatch (K % 16 == 0)"));
        }
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let f32_out = match out.dtype {
            DType::Bf16 => false,
            DType::F32 => true,
            d => return Err(Error::invalid(format!("metal matmul_nvfp4: output dtype {d}"))),
        };
        // the shaders index with 32-bit math in places
        u32_of(m * k, "M*K")?;
        u32_of(n * (k / 2), "N*K/2")?;
        u32_of(m * n, "M*N")?;
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let wb = self.bind(w, DType::U8, false, "w")?;
        let sb = self.bind(bs, DType::Fp8E4m3, false, "block scales")?;
        let gb = self.bind(gs, DType::F32, false, "global scale")?;
        let ob = self.bind(out, if f32_out { DType::F32 } else { DType::Bf16 }, true, "out")?;
        // rows are K / 2 = 8 * (K / 16) bytes, so an 8-byte aligned base keeps every 8-byte weight load aligned
        if wb.skip != 0 || wb.off % 8 != 0 {
            return Err(Error::invalid(format!(
                "metal matmul_nvfp4: the weights must be 8-byte aligned (bound at byte {}, skip {})",
                wb.off, wb.skip
            )));
        }
        let p = OpP {
            dims: [u32_of(m, "M")?, u32_of(n, "N")?, u32_of(k, "K")?, 0, 0, 0, 0, 0],
            skips: [xb.skip, sb.skip, ob.skip, 0, 0],
            ..Default::default()
        };
        if kernel == MATMUL_NVFP4_NAX {
            // the GEMM reads x as a 16-byte aligned [M,K] tensor without a skip; anything else runs the tiled kernel
            let xa = self.mbuf(&x.buf)?.base + x.offset;
            if xb.skip == 0 && xa % 16 == 0 {
                return self.launch_nvfp4_nax(call, (&xb, &wb, &sb, &gb, &ob), p, f32_out);
            }
            if FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
                eprintln!(
                    "metal {MATMUL_NVFP4_NAX}: x at byte {xa} (skip {}); running {MATMUL_NVFP4_MMA} (reported once)",
                    xb.skip
                );
            }
            return self.launch_nvfp4(MATMUL_NVFP4_MMA, call);
        }
        let (pso, grid, threads) = match kernel {
            MATMUL_NVFP4_GEMV | MATMUL_NVFP4_GEMV_ROWS => {
                if m > GEMV_MB || (kernel == MATMUL_NVFP4_GEMV_ROWS && m < 2) {
                    return Err(Error::invalid(format!("metal {kernel}: M = {m} outside the kernel's rows")));
                }
                let mb = if m == 1 {
                    1
                } else if kernel == MATMUL_NVFP4_GEMV_ROWS {
                    nv_rows_mb(n)
                } else {
                    GEMV_MB
                };
                let (name, rs) = match (f32_out, mb) {
                    (false, 1) => (SHADERS[0], GEMV_RS[0]),
                    (false, _) => (SHADERS[1], GEMV_RS[1]),
                    (true, 1) => (SHADERS[2], GEMV_RS[0]),
                    (true, _) => (SHADERS[3], GEMV_RS[1]),
                };
                let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
                (pso, (n.div_ceil(GEMV_SG * rs), m.div_ceil(mb)), GEMV_SG * 32)
            }
            MATMUL_NVFP4_MMA => {
                let name = if f32_out { SHADERS[5] } else { SHADERS[4] };
                let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
                (pso, (n.div_ceil(TILE), m.div_ceil(TILE)), MMA_THREADS)
            }
            _ => return Err(Error::NoKernel(format!("metal kernel '{kernel}' for op 'matmul_nvfp4'"))),
        };
        if pso.maxTotalThreadsPerThreadgroup() < threads {
            return Err(Error::backend(format!("metal: '{kernel}' cannot run {threads} threads")));
        }
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`).
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in [&xb, &wb, &sb, &gb, &ob].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &p, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: grid.0, height: grid.1, depth: 1 },
                MTLSize { width: threads, height: 1, depth: 1 },
            );
        })
    }

    /// `matmul_nvfp4_nax` in two dispatches: the weights decoded to bf16 and the scale pair `[gs, 0]` (`nv_w_to_bf16`) into
    /// the scratch, then the fp8 NAX GEMM (`gemm_nax_fp8_*`, tile by `gemm_nax::pick_variant`) on x and those weights, which
    /// scales by gs and stores bf16 / f32. Scratch: `[Wb: N*K bf16][scale pair: 2 f32]` from the next 16-byte boundary.
    fn launch_nvfp4_nax(
        &self,
        call: &OpCall,
        (xb, wb, sb, gb, ob): (
            &crate::dense::Bind,
            &crate::dense::Bind,
            &crate::dense::Bind,
            &crate::dense::Bind,
            &crate::dense::Bind,
        ),
        p: OpP,
        f32_out: bool,
    ) -> Result<()> {
        use crate::gemm_nax::{grid, pick_variant, NAX_VARIANTS};
        let (m, n, k) = (p.dims[0] as usize, p.dims[1] as usize, p.dims[2] as usize);
        let scratch = call.scratch.as_ref().ok_or_else(|| {
            Error::invalid("metal matmul_nvfp4_nax: the scratch workspace (OpCall::scratch) is missing")
        })?;
        let scratch_bytes = scratch.numel() * scratch.dtype.size_bytes();
        let sbind = self.bind(scratch, DType::U8, true, "scratch")?;
        if sbind.skip != 0 {
            return Err(Error::invalid("metal matmul_nvfp4_nax: the scratch must be 4-byte aligned"));
        }
        let pad = sbind.off.next_multiple_of(16) - sbind.off;
        let wbytes = n * k * 2;
        if scratch_bytes < pad + wbytes + 8 {
            return Err(Error::invalid(format!(
                "metal matmul_nvfp4_nax: scratch of {scratch_bytes} bytes at offset {}, need {}",
                sbind.off,
                pad + wbytes + 8
            )));
        }
        let wbf = crate::dense::Bind { mtl: sbind.mtl.clone(), off: sbind.off + pad, skip: 0 };
        let scb = crate::dense::Bind { mtl: sbind.mtl.clone(), off: sbind.off + pad + wbytes, skip: 0 };
        let v = match NAX_TILE.with(|t| t.get()) {
            Some(i) => &NAX_VARIANTS[i],
            None => pick_variant(m, n),
        };
        let vi = NAX_VARIANTS.iter().position(|x| x.shader == v.shader).expect("NAX variant table");
        let gemm = crate::fp8::NAX_FP8_SHADERS[vi][f32_out as usize];
        let pso_w = self.pso(W_TO_BF16).ok_or_else(|| Error::backend(format!("metal: no pipeline '{W_TO_BF16}'")))?;
        let pso_g = self
            .pso(gemm)
            .ok_or_else(|| Error::backend(format!("metal: no pipeline '{gemm}' (metal4.0 library not loaded)")))?;
        let pg = crate::fp8::NaxFp8Params {
            m: p.dims[0],
            n: p.dims[1],
            k: p.dims[2],
            swz_log: 0,
            s_c: ob.skip,
            pad: [0; 3],
        };
        let tw = pso_w.maxTotalThreadsPerThreadgroup().min(256);
        let (gx, gy) = grid(v, pg.m, pg.n, 0);
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds checked by `OpCall::check`, the scratch size above).
            unsafe {
                enc.setComputePipelineState(pso_w);
                for (i, b) in [wb, sb, gb, &wbf, &scb].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &p, 5);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: (n * k / 16).max(1), height: 1, depth: 1 },
                MTLSize { width: tw, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso_g);
                for (i, b) in [xb, &wbf, &scb, ob].into_iter().enumerate() {
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

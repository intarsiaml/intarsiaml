//! Inkling-2-Mini depth-mix, conv and input-projection ops on Metal (lane ink-m-mix; kernels/metal/inkling2_mix.metal and
//! the metal4.0 inkling2_mix_nax.metal). Contract: `intarsia_core::inkling2::ops` (module doc); f64 reference
//! `intarsia-backend-cpu/src/ink_ref.rs`; results docs/results/2026-11-21-inkling2-metal-mix.md.
//!
//! | kernel | op | rows | form |
//! |---|---|---|---|
//! | `ink_in_proj_bf16_gemv` | `InkInProj` | 1 | one GEMV dispatch over the stacked weight, the segment picks its x block |
//! | `ink_in_proj_bf16_gemv_rows` | `InkInProj` | 1..=16, `rows_exact` | the same shader family with R rows: row r bitwise the 1-row launch |
//! | `ink_in_proj_bf16_nax` | `InkInProj` | >= 2 | NAX GEMM over the three segments in one grid (prefill) |
//!
//! The GEMV shaders mirror `gemv_multirow.metal`'s `rt` kernels but with a lane count fitted to D = 2304 (see
//! [`ProjCfg`]); R only adds accumulators and P only shares an x load between P weight rows, so neither changes a bit.

use crate::MetalBackend;
use intarsia_core::backend::TensorRef;
use intarsia_core::{DType, Error, Exactness, KernelImpl, Op, OpAttrs, OpCall, Result, Tol};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

pub const INK_IN_PROJ_BF16_GEMV: &str = "ink_in_proj_bf16_gemv";
pub const INK_IN_PROJ_BF16_GEMV_ROWS: &str = "ink_in_proj_bf16_gemv_rows";
pub const INK_IN_PROJ_BF16_NAX: &str = "ink_in_proj_bf16_nax";

/// inkling2_mix_nax.metal entry points (metal4.0 library), one per `gemm_nax::NAX_VARIANTS` tile, in that order.
pub const NAX_SHADERS: [&str; 5] = [
    "ink_in_proj_nax_t64x64w4",
    "ink_in_proj_nax_t128x64w4",
    "ink_in_proj_nax_t32x32w1",
    "ink_in_proj_nax_t32x128w2",
    "ink_in_proj_nax_t64x64w2",
];

/// Every kernel of this module, with its op (the `kernels::TABLE` rows).
pub const KERNELS: &[(&str, Op)] = &[
    (INK_IN_PROJ_BF16_GEMV, Op::InkInProj),
    (INK_IN_PROJ_BF16_GEMV_ROWS, Op::InkInProj),
    (INK_IN_PROJ_BF16_NAX, Op::InkInProj),
    (CONV_MIX_BF16, Op::ConvMix),
    (ATTN_RES_MIX_BF16, Op::AttnResMix),
    (SCONV_RES_BF16, Op::SconvRes),
    (LOOP_BLEND_BF16, Op::LoopBlend),
];

/// bf16-rounded outputs: a one-ulp flip is allowed (kernels.rs `BF16_TOL`, the budget of the other bf16 Metal kernels).
const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
/// f32 outputs (the unnormed mixes): accumulation-order noise only (kernels.rs `F32_TOL`).
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Registration of a kernel of [`KERNELS`] (`kernels::all`).
pub fn register(k: KernelImpl) -> KernelImpl {
    let bf = [DType::Bf16];
    match k.name.as_str() {
        // ordinary calls up to PROJ_GEMV_ORDINARY_MAX rows too, above the NAX GEMM (priority 5): at the real shape the
        // GEMV takes 70-83 us for 1-8 rows, NAX a flat 106 us for 2-16 (docs/results/2026-11-21-inkling2-metal-mix.md)
        INK_IN_PROJ_BF16_GEMV => k
            .acts(bf)
            .weights(bf)
            .when(|s| (1..=PROJ_GEMV_ORDINARY_MAX).contains(&s.m))
            .exact(Exactness::Deterministic)
            .tol(BF16_TOL)
            .priority(6),
        // the verify / batched twin: bitwise the 1-row launch per row (tests/inkmix.rs); only `exact_rows` calls select it
        INK_IN_PROJ_BF16_GEMV_ROWS => k
            .acts(bf)
            .weights(bf)
            .when(|s| (1..=PROJ_ROWS_MAX).contains(&s.m))
            .exact(Exactness::Deterministic)
            .tol(BF16_TOL)
            .priority(-1)
            .rows_exact(),
        // prefill (the `matmul_bf16_nax` rule: M >= 2, above the GEMV); a segment is bitwise `matmul_bf16_nax` on it
        INK_IN_PROJ_BF16_NAX => k
            .acts(bf)
            .weights(bf)
            .when(|s| s.m >= crate::gemm_nax::NAX_MIN_M && s.k >= 16)
            .exact(Exactness::Deterministic)
            .tol(BF16_TOL)
            .priority(5),
        // one threadgroup per row, the same code for every row count: rows-exact by construction (tests/inkmix.rs), so no
        // separate verify twin; chunk-invariant for prefill too
        CONV_MIX_BF16 => k
            .acts(bf)
            .weights(bf)
            .when(|s| s.k > 0 && s.k <= MIX_D_MAX)
            .exact(Exactness::Deterministic)
            .tol(BF16_TOL)
            .tol_f32(F32_TOL)
            .scratch(blend_scratch),
        ATTN_RES_MIX_BF16 => k
            .acts(bf)
            .when(|s| s.k > 0 && s.k <= MIX_D_MAX && s.kv_len <= MIX_SOURCES_MAX)
            .exact(Exactness::Deterministic)
            .tol(BF16_TOL)
            .tol_f32(F32_TOL),
        SCONV_RES_BF16 => k.acts(bf).weights(bf).when(|s| s.k > 0 && s.k <= MIX_D_MAX).exact(Exactness::Deterministic).tol(BF16_TOL),
        LOOP_BLEND_BF16 => k
            .acts(bf)
            .weights(bf)
            .when(|s| s.k > 0 && s.k <= MIX_D_MAX && s.n <= BLEND_STREAMS_MAX)
            .exact(Exactness::Deterministic)
            .tol(BF16_TOL)
            .scratch(blend_scratch),
        other => unreachable!("inkmix kernel {other} has no registration"),
    }
}

/// Most rows an ordinary (not `exact_rows`) InkInProj call runs on the GEMV; above it the NAX GEMM is faster.
pub const PROJ_GEMV_ORDINARY_MAX: usize = 8;

/// Most rows the GEMV forms take in one launch.
pub const PROJ_ROWS_MAX: usize = 16;

/// The (simdgroups, 16-byte chunks per lane per step) shader families compiled in (`IP_CFG` in inkling2_mix.metal).
pub const PROJ_CFGS: [(u32, u32); 4] = [(8, 2), (9, 1), (3, 3), (1, 9)];

/// (R, P, RB) instances of every family (`IP_CFG`).
pub const PROJ_INSTS: [(usize, usize, usize); 24] = [
    (1, 1, 1),
    (1, 2, 1),
    (1, 4, 1),
    (2, 1, 2),
    (2, 2, 2),
    (3, 1, 3),
    (3, 2, 3),
    (4, 1, 4),
    (4, 2, 4),
    (5, 1, 5),
    (6, 1, 6),
    (7, 1, 7),
    (8, 1, 8),
    (8, 2, 4),
    (9, 1, 9),
    (10, 1, 10),
    (11, 1, 11),
    (12, 1, 12),
    (13, 1, 13),
    (14, 1, 14),
    (15, 1, 15),
    (16, 1, 16),
    (16, 1, 8),
    (16, 2, 8),
];

/// The serving GEMV family: lanes and chunks per lane. Fixed for every row count (changing it changes bits).
#[derive(Clone, Copy, Debug)]
pub struct ProjCfg {
    pub sg: u32,
    pub u: u32,
}

/// Default family (docs/results/2026-11-21-inkling2-metal-mix.md, InkInProj sweep). `INTARSIA_INK_PROJ_CFG=s<SG>u<U>` overrides it
/// (measurement knob; every family is bitwise rows-exact on its own, but they differ from each other).
pub const PROJ_DEFAULT: ProjCfg = ProjCfg { sg: 8, u: 2 };

fn proj_cfg() -> ProjCfg {
    static C: std::sync::OnceLock<ProjCfg> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        let Ok(v) = std::env::var("INTARSIA_INK_PROJ_CFG") else { return PROJ_DEFAULT };
        PROJ_CFGS
            .iter()
            .find(|&&(sg, u)| v == format!("s{sg}u{u}"))
            .map(|&(sg, u)| ProjCfg { sg, u })
            .unwrap_or(PROJ_DEFAULT)
    })
}

/// (P, RB) the launcher uses for `r` rows. `INTARSIA_INK_PROJ_P=<p>` / `INTARSIA_INK_PROJ_RB=<rb>` pick another compiled
/// instance (measurement knobs, same bits).
fn proj_inst(r: usize) -> (usize, usize) {
    static OV: std::sync::OnceLock<(Option<usize>, Option<usize>)> = std::sync::OnceLock::new();
    let (p_ov, rb_ov) = *OV.get_or_init(|| {
        let g = |k: &str| std::env::var(k).ok().and_then(|v| v.parse().ok());
        (g("INTARSIA_INK_PROJ_P"), g("INTARSIA_INK_PROJ_RB"))
    });
    let want_p = p_ov.unwrap_or(1);
    let want_rb = rb_ov.unwrap_or(r);
    let pick = PROJ_INSTS
        .iter()
        .find(|&&(rr, p, rb)| rr == r && p == want_p && rb == want_rb)
        .or_else(|| PROJ_INSTS.iter().find(|&&(rr, p, _)| rr == r && p == want_p))
        .or_else(|| PROJ_INSTS.iter().find(|&&(rr, _, _)| rr == r))
        .expect("every row count 1..=16 is compiled");
    (pick.1, pick.2)
}

/// Threadgroups the GEMV grid is capped at (the rest loop, grid-stride). `INTARSIA_INK_PROJ_GRID=<n>` (measurement knob, same bits).
fn proj_grid_cap() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("INTARSIA_INK_PROJ_GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX))
}

fn proj_shader_name(r: usize, p: usize, sg: u32, u: u32, rb: usize) -> &'static str {
    static T: std::sync::OnceLock<std::collections::HashMap<(usize, usize, u32, u32, usize), &'static str>> =
        std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        for &(sg, u) in &PROJ_CFGS {
            for &(r, p, rb) in &PROJ_INSTS {
                let n: &'static str = Box::leak(format!("ink_in_proj_gemv_r{r}p{p}s{sg}u{u}b{rb}").into_boxed_str());
                m.insert((r, p, sg, u, rb), n);
            }
        }
        m
    })
    .get(&(r, p, sg, u, rb))
    .copied()
    .expect("compiled InkInProj instance")
}

/// Host names of every shader in inkling2_mix.metal (the pipeline table).
pub fn shader_names() -> Vec<&'static str> {
    let mut v = Vec::new();
    for &(sg, u) in &PROJ_CFGS {
        for &(r, p, rb) in &PROJ_INSTS {
            v.push(proj_shader_name(r, p, sg, u, rb));
        }
    }
    v.extend(row_shaders());
    v
}

/// `InkProjP` of the shader (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct InkProjP {
    k: u32,
    n: [u32; 3],
    xo1: u32,
    xo2: u32,
    ys: [u32; 3],
    g0: u32,
    g1: u32,
    gtot: u32,
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal ink: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_inkmix(&self, kernel: &str, call: &OpCall) -> Result<()> {
        match (kernel, call.op) {
            (INK_IN_PROJ_BF16_GEMV | INK_IN_PROJ_BF16_GEMV_ROWS, Op::InkInProj) => self.launch_in_proj_gemv(call),
            (INK_IN_PROJ_BF16_NAX, Op::InkInProj) => self.launch_in_proj_nax(call),
            (CONV_MIX_BF16 | ATTN_RES_MIX_BF16 | SCONV_RES_BF16 | LOOP_BLEND_BF16, _) => self.launch_mix(kernel, call),
            _ => Err(Error::invalid(format!("metal: {kernel} does not run {}", call.op))),
        }
    }

    /// `InkInProj` with T = 1..=16 rows: one dispatch of `ink_in_proj_gemv_r<T>..` over the three weight segments. Operands
    /// the 16-byte shader cannot take (K % 8 != 0, W or x not 16-byte aligned) run as three multi-row `Matmul`s instead.
    fn launch_in_proj_gemv(&self, call: &OpCall) -> Result<()> {
        let (i, o) = (&call.inputs, &call.outputs);
        let bf = DType::Bf16;
        if i[0].dtype != bf || i[1].dtype != bf || o.iter().any(|y| y.dtype != bf) {
            return Err(Error::invalid("ink_in_proj_bf16_gemv: bf16 only"));
        }
        let t = i[0].shape[1];
        if t > PROJ_ROWS_MAX {
            return Err(Error::invalid(format!("ink_in_proj_bf16_gemv: {t} rows (at most {PROJ_ROWS_MAX})")));
        }
        self.in_proj_rows(&i[0], &i[1], o, i[0].shape[0])
    }

    /// `InkInProj` prefill: one NAX grid over the three segments, the tile of `gemm_nax::pick_variant(T, N)`.
    fn launch_in_proj_nax(&self, call: &OpCall) -> Result<()> {
        let (i, o) = (&call.inputs, &call.outputs);
        let bf = DType::Bf16;
        let (s, t, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        if t == 0 {
            return Ok(());
        }
        let ns = [o[0].shape[1], o[1].shape[1], o[2].shape[1]];
        let v = crate::gemm_nax::pick_variant(t, ns.iter().sum());
        let vi = crate::gemm_nax::NAX_VARIANTS.iter().position(|x| x.shader == v.shader).expect("NAX tile");
        let name = NAX_SHADERS[vi];
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}' (metal4.0 library not loaded)")))?;
        let (x, w) = (self.bind(&i[0], bf, false, "x")?, self.bind(&i[1], bf, false, "w")?);
        let ys: Vec<_> = o.iter().enumerate().map(|(j, y)| self.bind(y, bf, true, ["pq", "pk", "pv"][j])).collect::<Result<_>>()?;
        let xoff = |j: usize| -> Result<u32> { dim(j.min(s - 1) * t * d, "x offset") };
        let p = [
            dim(t, "T")?,
            dim(d, "D")?,
            dim(ns[0], "Nq")?,
            dim(ns[1], "Nk")?,
            dim(ns[2], "Nv")?,
            xoff(1)?,
            xoff(2)?,
            x.skip,
            w.skip,
            ys[0].skip,
            ys[1].skip,
            ys[2].skip,
        ];
        let tiles_n: usize = ns.iter().map(|n| n.div_ceil(v.bn as usize)).sum();
        let tiles_m = t.div_ceil(v.bm as usize);
        self.with_encoder(|enc| {
            enc.setComputePipelineState(pso);
            // SAFETY: offsets inside the buffers (OpCall::check), 4-byte aligned with the skips in `p`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&x.mtl), x.off, 0);
                enc.setBuffer_offset_atIndex(Some(&w.mtl), w.off, 1);
                for (j, y) in ys.iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&y.mtl), y.off, 2 + j);
                }
            }
            crate::set_bytes(enc, &p, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: tiles_n, height: tiles_m, depth: 1 },
                MTLSize { width: v.threads() as usize, height: 1, depth: 1 },
            );
        })
    }

    /// `y_j = x[min(j, s-1)] . W_j^T` (x `[s, T, D]`, W the three segments stacked, y_j `[T, n_j]`), T <= 16, one dispatch.
    pub(crate) fn in_proj_rows(&self, xt: &TensorRef, wt: &TensorRef, o: &[TensorRef], s: usize) -> Result<()> {
        let bf = DType::Bf16;
        let (t, d) = (xt.shape[1], xt.shape[2]);
        let i = [xt.clone(), wt.clone()];
        if t == 0 {
            return Ok(());
        }
        let ns = [o[0].shape[1], o[1].shape[1], o[2].shape[1]];
        let (xb, wb) = (self.mbuf(&i[0].buf)?, self.mbuf(&i[1].buf)?);
        let (xo, wo) = (xb.base + i[0].offset, wb.base + i[1].offset);
        if d % 8 != 0 || xo % 16 != 0 || wo % 16 != 0 {
            // multi-row GEMVs on views, 16 rows at a time (each row bitwise its own 1-row Matmul GEMV)
            let mut row0 = 0;
            for (j, y) in o.iter().enumerate() {
                if ns[j] == 0 {
                    continue;
                }
                let x = i[0].rows(j.min(s - 1), 1).reshaped(&[t, d]);
                let w = i[1].rows(row0, ns[j]);
                for r0 in (0..t).step_by(PROJ_ROWS_MAX) {
                    let n = (t - r0).min(PROJ_ROWS_MAX);
                    let call = OpCall::new(Op::Matmul, vec![x.rows(r0, n), w.clone()], vec![y.rows(r0, n)], OpAttrs::default());
                    self.launch_matmul_rows(&call)?;
                }
                row0 += ns[j];
            }
            return Ok(());
        }
        // whole chunks of 16 rows on the grid's y axis, then the tail rows as one launch of their own row count: every row
        // is computed exactly as by a launch of 16 (or of the tail's count), so bitwise its 1-row launch
        let (full, tail) = if t <= PROJ_ROWS_MAX { (0, t) } else { (t / PROJ_ROWS_MAX, t % PROJ_ROWS_MAX) };
        let ys: Vec<_> = o.iter().enumerate().map(|(j, y)| self.bind(y, bf, true, ["pq", "pk", "pv"][j])).collect::<Result<_>>()?;
        let xoff = |j: usize| -> Result<u32> { dim(j.min(s - 1) * t * d, "x offset") };
        let cfg = proj_cfg();
        for (r, r0, chunks) in [(PROJ_ROWS_MAX, 0, full), (tail, full * PROJ_ROWS_MAX, 1)] {
            if chunks == 0 || r == 0 {
                continue;
            }
            let (p, rb) = proj_inst(r);
            let name = proj_shader_name(r, p, cfg.sg, cfg.u, rb);
            let g = ns.map(|n| n.div_ceil(p));
            // the tail: x and y bound at its first row (x keeps the full-chunk segment offsets)
            let yskip: Vec<usize> = ys.iter().zip(&ns).map(|(y, &n)| y.skip as usize + r0 * n).collect();
            let prm = InkProjP {
                k: dim(d, "D")?,
                n: [dim(ns[0], "Nq")?, dim(ns[1], "Nk")?, dim(ns[2], "Nv")?],
                xo1: xoff(1)?,
                xo2: xoff(2)?,
                ys: [dim(yskip[0], "y skip")?, dim(yskip[1], "y skip")?, dim(yskip[2], "y skip")?],
                g0: dim(g[0], "groups")?,
                g1: dim(g[1], "groups")?,
                gtot: dim(g.iter().sum::<usize>(), "groups")?,
            };
            let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
            let groups = g.iter().sum::<usize>().min(proj_grid_cap());
            let threads = cfg.sg as usize * 32;
            let xo_r = xo + r0 * d * 2;
            self.with_encoder(|enc| {
                enc.setComputePipelineState(pso);
                // SAFETY: offsets inside the buffers (OpCall::check), W and x 16-byte aligned (checked above; r0 * D * 2 is a
                // multiple of 16 as D % 8 == 0).
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&wb.mtl), wo, 0);
                    enc.setBuffer_offset_atIndex(Some(&xb.mtl), xo_r, 1);
                    for (j, y) in ys.iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&y.mtl), y.off, 2 + j);
                    }
                }
                crate::set_bytes(enc, &prm, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: groups, height: chunks, depth: 1 },
                    MTLSize { width: threads, height: 1, depth: 1 },
                );
            })?;
        }
        Ok(())
    }
}

/// `x [S,T,D]` rows view helper for tests: the x block a segment reads.
pub fn segment_x(x: &TensorRef, j: usize) -> TensorRef {
    let (s, t, d) = (x.shape[0], x.shape[1], x.shape[2]);
    x.rows(j.min(s - 1), 1).reshaped(&[t, d])
}

// ---------------------------------------------------------------------------------------------------------------------
// The row ops: SconvRes, ConvMix, AttnResMix, LoopBlend
// ---------------------------------------------------------------------------------------------------------------------

pub const CONV_MIX_BF16: &str = "conv_mix_bf16";
pub const ATTN_RES_MIX_BF16: &str = "attn_res_mix_bf16";
pub const SCONV_RES_BF16: &str = "sconv_res_bf16";
pub const LOOP_BLEND_BF16: &str = "loop_blend_bf16";

/// Elements per thread compiled into `ink_mix_row_e<NE>..` (`MX_NS`); a row of D elements runs on ceil(D / NE) threads.
pub const MIX_NE: [usize; 4] = [1, 2, 3, 4];
/// Widest row the row kernels take (NE 4 x 1024 threads).
pub const MIX_D_MAX: usize = 4096;
/// Stored sources + last the depth mix holds in registers (`MX_NMAX`).
pub const MIX_SOURCES_MAX: usize = 8;
/// Rows per blend chunk (its scratch holds that many rows: 21 MB at D 2304, S 3), streams and controller rank bounds.
pub const BLEND_ROWS: usize = 256;
/// Rows per `ink_blend_gate` threadgroup. With [`BLEND_CS`] a launch-shape choice only: a channel's gate value and the 32-channel
/// partial sums do not depend on it (`INTARSIA_INK_BLEND_RB` / `_CS` measurement knobs).
pub const BLEND_RB: usize = 16;

fn blend_geom() -> (usize, usize) {
    static G: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        let g = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        let rb = g("INTARSIA_INK_BLEND_RB").filter(|&v| v >= 1).unwrap_or(BLEND_RB);
        let cs = g("INTARSIA_INK_BLEND_CS").filter(|&v| v % 32 == 0 && (32..=1024).contains(&v)).unwrap_or(BLEND_CS);
        (rb, cs)
    })
}
pub const BLEND_STREAMS_MAX: usize = 3;
pub const BLEND_G_MAX: usize = 128;
/// Channels per slice of `ink_blend_gate` (one thread each).
pub const BLEND_CS: usize = 64;

const MX_CONV: u32 = 1;
const MX_RES: u32 = 2;
const MX_ADD: u32 = 4;
const MX_SUM: u32 = 8;
const MX_NORM: u32 = 16;
const MX_RING: u32 = 32;
const MX_PAIR: u32 = 64;

/// `InkMixP` of the shader (112 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct InkMixP {
    d: u32,
    t: u32,
    t0: u32,
    nrows: u32,
    k: u32,
    rr: u32,
    pos0: u32,
    use_dev: u32,
    n: u32,
    q: u32,
    flags: u32,
    orow0: u32,
    ocount: u32,
    nstream: u32,
    pad: [u32; 2],
    eps: f32,
    cscale: f32,
    fpad: [f32; 2],
    sk: [u32; 12],
}

/// `InkBlendP` of the shader (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct InkBlendP {
    d: u32,
    rows: u32,
    s: u32,
    sd: u32,
    g: u32,
    nsl: u32,
    cs: u32,
    t0: u32,
    ocount: u32,
    dcount: u32,
    mcount: u32,
    pad0: u32,
    dt0: u32,
    mt0: u32,
    pad1: [u32; 2],
    eps: f32,
    fpad: [f32; 3],
    sk: [u32; 8],
}

/// (NE, threads per threadgroup) of the row kernels for rows of `d` elements: the fewest elements per thread that fit in
/// 1024 threads (`INTARSIA_INK_MIX_NE=<ne>` forces another compiled NE: a measurement knob that changes the reduction order,
/// fixed per D, so still rows-exact).
fn mix_geom(d: usize) -> (usize, usize) {
    static OV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let ov = *OV.get_or_init(|| std::env::var("INTARSIA_INK_MIX_NE").ok().and_then(|v| v.parse().ok()).filter(|n| MIX_NE.contains(n)));
    let ne = ov.filter(|&n| n * 1024 >= d).unwrap_or_else(|| MIX_NE.iter().copied().find(|&n| n * 1024 >= d).unwrap_or(4));
    (ne, d.div_ceil(ne).next_multiple_of(32).clamp(32, 1024))
}

/// `ink_mix_row_e<NE>n<NS>q<QB>`: NS = stored sources + 1 (1 when the call mixes nothing), QB = 1 for one query, else 4.
fn mix_row_shader(ne: usize, ns: usize, qb: usize) -> &'static str {
    static T: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    let t = T.get_or_init(row_names);
    let (ei, qi) = (MIX_NE.iter().position(|&e| e == ne).expect("compiled NE"), usize::from(qb > 1));
    t[(ei * 2 + qi) * MIX_SOURCES_MAX + ns - 1]
}

fn row_names() -> Vec<&'static str> {
    let mut v = Vec::new();
    for ne in MIX_NE {
        for qb in [1, 4] {
            for ns in 1..=MIX_SOURCES_MAX {
                v.push(&*Box::leak(format!("ink_mix_row_e{ne}n{ns}q{qb}").into_boxed_str()));
            }
        }
    }
    v
}

/// The row-kernel entry points of inkling2_mix.metal (`MX_NS`), and the ring / blend helpers.
pub fn row_shaders() -> Vec<&'static str> {
    let mut v = row_names();
    v.extend(["ink_ring_write", "ink_blend_pair", "ink_blend_gate_g1", "ink_blend_gate_g8", "ink_blend_gate_g16", "ink_blend_norm"]);
    v
}

/// Scratch of the blend path (ConvMix with msg, LoopBlend) for rows of `d` channels, regions 256-byte aligned:
/// dest f32 [3, rows, D], pair bf16 [3, rows, 2D], h bf16 [3, rows, G <= 128], y f32 [3, rows, D], partial f32 [3, rows, nsl],
/// rows = the call's rows up to [`BLEND_ROWS`].
fn blend_layout(d: usize, rows: usize) -> [usize; 6] {
    let r = |b: usize| b.next_multiple_of(256);
    let (s, rows) = (BLEND_STREAMS_MAX, rows.clamp(1, BLEND_ROWS));
    let dest = r(s * rows * d * 4);
    let pair = r(s * rows * 2 * d * 2);
    let h = r(s * rows * BLEND_G_MAX * 2);
    let y = r(s * rows * d * 4);
    let part = r(s * rows * d.div_ceil(1024).max(1) * 32 * 4);
    [0, dest, dest + pair, dest + pair + h, dest + pair + h + y, dest + pair + h + y + part]
}

pub fn blend_scratch(s: &intarsia_core::ShapeInfo) -> usize {
    blend_layout(s.k.max(1), s.m)[5]
}

/// A `[n]` view of `bytes` bytes at `off` of the scratch, as `dt`.
fn sview(scratch: &TensorRef, off: usize, dt: DType, n: usize) -> TensorRef {
    TensorRef { buf: scratch.buf.clone(), offset: scratch.offset + off, dtype: dt, shape: vec![n] }
}

fn absent(t: &TensorRef) -> bool {
    t.numel() == 0
}

impl MetalBackend {
    /// Bind `t` as `dt`, or (absent) `dummy` in its place: the kernel never touches an absent operand.
    fn bind_or(&self, t: &TensorRef, dt: DType, write: bool, what: &str, dummy: &crate::dense::Bind) -> Result<crate::dense::Bind> {
        if absent(t) {
            Ok(crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 })
        } else {
            self.bind(t, dt, write, what)
        }
    }

    /// The row kernel over rows `t0 .. t0 + nrows` of a SconvRes / ConvMix / AttnResMix call. `dest` set: the mix (unnormed,
    /// f32) goes to that scratch view as `[Q, nrows, D]` instead of the call's output (the blend path). Encodes the deferred
    /// ring write after it when a row of the launch reads a ring row another row writes.
    fn mix_rows(&self, call: &OpCall, t0: usize, nrows: usize, dest: Option<&TensorRef>, pair: Option<&TensorRef>) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        let f32t = DType::F32;
        let op = call.op;
        let conv = op != Op::AttnResMix;
        // operands by role
        let (x, taps, res, add) = if conv { (&i[0], Some(&i[1]), Some(&i[2]), &i[3]) } else { (&i[1], None, None, &i[2]) };
        let (src, qry, nrm) = match op {
            Op::ConvMix => (Some(&i[4]), Some(&i[5]), Some(&i[6])),
            Op::AttnResMix => (Some(&i[0]), Some(&i[3]), Some(&i[4])),
            _ => (None, None, None),
        };
        let (t, d) = (x.shape[0], x.shape[1]);
        let xb = self.bind(x, bf, false, "x")?;
        let dummy = crate::dense::Bind { mtl: xb.mtl.clone(), off: xb.off, skip: 0 };
        let taps_b = match taps {
            Some(tp) => self.bind(tp, bf, false, "taps")?,
            None => crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 },
        };
        let none = TensorRef { buf: x.buf.clone(), offset: 0, dtype: bf, shape: vec![0] };
        let res = res.unwrap_or(&none);
        let res_b = self.bind_or(res, bf, false, "res", &dummy)?;
        let add_b = self.bind_or(add, bf, false, "addend", &dummy)?;
        let (y_b, ring_b, sum_b) = if conv {
            (self.bind(&o[0], bf, true, "y")?, self.bind(&o[1], bf, true, "ring")?, self.bind_or(&o[2], bf, true, "sum", &dummy)?)
        } else {
            let d0 = || crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 };
            (d0(), d0(), d0())
        };
        let nq = qry.map_or(0, |q| q.shape[0]);
        let ns = src.map_or(0, |s| s.shape[0]);
        if nq > 0 && ns + 1 > MIX_SOURCES_MAX {
            return Err(Error::invalid(format!("metal {}: {} stored sources (at most {})", call.op, ns, MIX_SOURCES_MAX - 1)));
        }
        let src_b = match src {
            Some(s) if ns > 0 && nq > 0 => self.bind(s, bf, false, "src")?,
            _ => crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 },
        };
        let qry_b = match qry {
            Some(q) if nq > 0 => self.bind(q, f32t, false, "queries")?,
            _ => crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 },
        };
        let has_norm = dest.is_none() && nrm.is_some_and(|n| !absent(n)) && nq > 0;
        let nrm_b = match nrm {
            Some(n) if has_norm => self.bind(n, bf, false, "norm")?,
            _ => crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 },
        };
        let out_t = match op {
            Op::ConvMix => Some(&o[3]),
            Op::AttnResMix => Some(&o[0]),
            _ => None,
        };
        let (outb, outf) = if nq == 0 {
            let d0 = || crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 };
            (d0(), d0())
        } else if let Some(dv) = dest {
            let b = self.bind(dv, f32t, true, "dest scratch")?;
            (crate::dense::Bind { mtl: b.mtl.clone(), off: b.off, skip: 0 }, b)
        } else {
            let ot = out_t.expect("mix output");
            if has_norm {
                let b = self.bind(ot, bf, true, "mix")?;
                (crate::dense::Bind { mtl: b.mtl.clone(), off: b.off, skip: b.skip }, b)
            } else {
                let b = self.bind(ot, f32t, true, "mix (f32)")?;
                (crate::dense::Bind { mtl: b.mtl.clone(), off: b.off, skip: 0 }, b)
            }
        };
        let (pos_b, use_dev) = self.pos_bind(call)?;
        let pos_b = pos_b.unwrap_or(crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 });
        let (k, rr) = if conv { (i[1].shape[0], o[1].shape[0]) } else { (0, 0) };
        let hres = conv && !absent(res);
        let mut flags = 0;
        if conv {
            flags |= MX_CONV;
        }
        if hres {
            flags |= MX_RES;
        }
        if !absent(add) {
            flags |= MX_ADD;
        }
        if conv && !absent(&o[2]) {
            flags |= MX_SUM;
        }
        if has_norm {
            flags |= MX_NORM;
        }
        // in-kernel ring write when no row of this launch reads a ring row a row of it writes
        let readers = conv && t0 < k.saturating_sub(1);
        let writers = conv && t0 + nrows > t.saturating_sub(rr);
        let safe = !readers || !writers || t + k.saturating_sub(1) <= rr;
        if conv && safe {
            flags |= MX_RING;
        }
        // the blend's pair stage, folded in (ConvMix with msg): msg rows [S, T, D] f32
        let (msg_b, pair_b, nstream) = match pair {
            Some(pv) if op == Op::ConvMix => {
                flags |= MX_PAIR;
                (self.bind(&i[7], f32t, false, "msg")?, self.bind(pv, bf, true, "pair")?, i[7].shape[0])
            }
            _ => (crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 }, crate::dense::Bind { mtl: dummy.mtl.clone(), off: dummy.off, skip: 0 }, 0),
        };
        let (ne, tpg) = mix_geom(d);
        let p = InkMixP {
            d: dim(d, "D")?,
            t: dim(t, "T")?,
            t0: dim(t0, "row")?,
            nrows: dim(nrows, "rows")?,
            k: dim(k, "K")?,
            rr: dim(rr, "ring rows")?,
            pos0: dim(a.pos0, "pos0")?,
            use_dev,
            n: dim(ns, "N")?,
            q: dim(nq, "Q")?,
            flags,
            orow0: if dest.is_some() { t0 as u32 } else { 0 },
            ocount: dim(if dest.is_some() { nrows } else { t }, "rows")?,
            nstream: dim(nstream, "streams")?,
            pad: [0; 2],
            eps: a.eps,
            cscale: ((k + 1) as f32).powf(-0.5),
            fpad: [0.0; 2],
            sk: [xb.skip, taps_b.skip, res_b.skip, add_b.skip, y_b.skip, ring_b.skip, sum_b.skip, src_b.skip, nrm_b.skip, outb.skip, 0, 0],
        };
        let binds = [&xb, &taps_b, &res_b, &add_b, &y_b, &ring_b, &sum_b, &src_b, &qry_b, &nrm_b, &outb, &outf, &pos_b, &msg_b, &pair_b];
        let (ns_i, qb) = if nq == 0 { (1, 1) } else { (ns + 1, if nq == 1 { 1 } else { 4 }) };
        self.dispatch(mix_row_shader(ne, ns_i, qb), &binds, &p, crate::dense::Grid::Groups { x: nrows, y: 1, threads: tpg })?;
        if conv && !safe {
            // ink_ring_write reads `pad0` (here `nrows`) as the launch's row count
            let pw = p;
            let binds = [&xb, &ring_b, &pos_b];
            self.dispatch("ink_ring_write", &binds, &pw, crate::dense::Grid::Threads(nrows * d))?;
        }
        Ok(())
    }

    /// The blend of rows `t0 .. t0 + n` (n <= 16): pair, h GEMV over the stacked `down`, gate, norm.
    #[allow(clippy::too_many_arguments)]
    fn blend_rows(
        &self,
        dest: &TensorRef,
        dest_rows: (usize, usize),
        msg: &TensorRef,
        down: &TensorRef,
        gain: &TensorRef,
        forget: &TensorRef,
        norm: &TensorRef,
        out: &TensorRef,
        t0: usize,
        n: usize,
        scratch: &TensorRef,
        eps: f32,
        pair_done: bool,
    ) -> Result<()> {
        let bf = DType::Bf16;
        let f32t = DType::F32;
        let (s, mt, d) = (msg.shape[0], msg.shape[1], msg.shape[2]);
        let g = down.shape[1];
        let sd = dest.shape[0];
        if s > BLEND_STREAMS_MAX || g > BLEND_G_MAX || g == 0 {
            return Err(Error::invalid(format!("metal blend: {s} streams, rank {g} (at most {BLEND_STREAMS_MAX} / {BLEND_G_MAX})")));
        }
        let lay = blend_layout(d, out.shape[1]);
        let pair = sview(scratch, lay[1], bf, s * n * 2 * d);
        let h = sview(scratch, lay[2], bf, s * n * g);
        let yv = sview(scratch, lay[3], f32t, s * n * d);
        let (brb, bcs) = blend_geom();
        let slices = d.div_ceil(bcs);
        let nsl = slices * (bcs / 32);
        let part = sview(scratch, lay[4], f32t, s * n * nsl);
        if g % 8 != 0 {
            return Err(Error::invalid(format!("metal blend: controller rank {g} is not a multiple of 8")));
        }
        for (tn, tt) in [("gain", gain), ("forget", forget)] {
            let mb = self.mbuf(&tt.buf)?;
            if (mb.base + tt.offset) % 16 != 0 {
                return Err(Error::invalid(format!("metal blend: {tn} must be 16-byte aligned (the loader places the controllers)")));
            }
        }
        let p = InkBlendP {
            d: dim(d, "D")?,
            rows: dim(n, "rows")?,
            s: dim(s, "S")?,
            sd: dim(sd, "Sd")?,
            g: dim(g, "G")?,
            nsl: dim(nsl, "slices")?,
            cs: bcs as u32,
            t0: dim(t0, "row")?,
            ocount: dim(out.shape[1], "rows")?,
            dcount: dim(dest_rows.1, "rows")?,
            mcount: dim(mt, "rows")?,
            dt0: dim(dest_rows.0, "row")?,
            mt0: dim(t0, "row")?,
            pad0: dim(brb, "gate rows")?,
            eps,
            ..Default::default()
        };
        let (dest_b, msg_b, pair_b) = (self.bind(dest, f32t, false, "dest")?, self.bind(msg, f32t, false, "msg")?, self.bind(&pair, bf, true, "pair")?);
        if !pair_done {
            self.dispatch("ink_blend_pair", &[&dest_b, &msg_b, &pair_b], &p, crate::dense::Grid::Groups { x: s * n, y: 1, threads: 256 })?;
        }
        // h[j] = rA(pair[j] . down[j]^T): the InkInProj GEMV with the streams as its segments
        let wd = TensorRef { shape: vec![s * g, 2 * d], ..down.clone() };
        let xs = TensorRef { shape: vec![s, n, 2 * d], ..pair.clone() };
        let hs: Vec<TensorRef> = (0..3)
            .map(|j| {
                let jj = j.min(s - 1);
                let hv = sview(&h, jj * n * g * 2, bf, n * g);
                TensorRef { shape: vec![n, if j < s { g } else { 0 }], ..hv }
            })
            .collect();
        self.in_proj_rows(&xs, &wd, &hs, s)?;
        let (h_b, gain_b, forget_b, y_b, part_b) = (
            self.bind(&h, bf, false, "h")?,
            self.bind(gain, bf, false, "gain")?,
            self.bind(forget, bf, false, "forget")?,
            self.bind(&yv, f32t, true, "y")?,
            self.bind(&part, f32t, true, "partial")?,
        );
        let mut pg = p;
        pg.sk = [gain_b.skip, forget_b.skip, 0, 0, 0, 0, 0, 0];
        let gate = match g / 8 {
            0..=1 => "ink_blend_gate_g1",
            2..=8 => "ink_blend_gate_g8",
            _ => "ink_blend_gate_g16",
        };
        self.dispatch(
            gate,
            &[&dest_b, &msg_b, &h_b, &gain_b, &forget_b, &y_b, &part_b],
            &pg,
            crate::dense::Grid::Groups { x: s * slices, y: n.div_ceil(brb), threads: bcs },
        )?;
        let (nrm_b, out_b) = (self.bind(norm, bf, false, "norm")?, self.bind(out, bf, true, "out")?);
        let mut pn = p;
        pn.sk = [0, 0, nrm_b.skip, out_b.skip, 0, 0, 0, 0];
        self.dispatch("ink_blend_norm", &[&y_b, &part_b, &nrm_b, &out_b], &pn, crate::dense::Grid::Groups { x: s * n, y: 1, threads: 256 })
    }

    pub(crate) fn launch_mix(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (i, o) = (&call.inputs, &call.outputs);
        let bf = DType::Bf16;
        match (kernel, call.op) {
            (SCONV_RES_BF16, Op::SconvRes) | (ATTN_RES_MIX_BF16, Op::AttnResMix) => {
                let x = if call.op == Op::SconvRes { &i[0] } else { &i[1] };
                if x.dtype != bf {
                    return Err(Error::invalid(format!("metal {kernel}: bf16 activations only")));
                }
                let t = x.shape[0];
                if t == 0 {
                    return Ok(());
                }
                self.mix_rows(call, 0, t, None, None)
            }
            (CONV_MIX_BF16, Op::ConvMix) => {
                if i[0].dtype != bf {
                    return Err(Error::invalid(format!("metal {kernel}: bf16 activations only")));
                }
                let t = i[0].shape[0];
                if t == 0 {
                    return Ok(());
                }
                if absent(&i[7]) || i[5].shape[0] == 0 {
                    return self.mix_rows(call, 0, t, None, None);
                }
                let scratch = call.scratch.as_ref().ok_or_else(|| Error::invalid("metal conv_mix_bf16: the blend needs the scratch (OpCall::scratch)"))?;
                let d = i[0].shape[1];
                let nq = i[5].shape[0];
                let lay = blend_layout(d, t);
                for t0 in (0..t).step_by(BLEND_ROWS) {
                    let n = (t - t0).min(BLEND_ROWS);
                    let dest = TensorRef { shape: vec![nq, n, d], ..sview(scratch, lay[0], DType::F32, nq * n * d) };
                    let s = i[7].shape[0];
                    let pair = TensorRef { shape: vec![s, n, 2 * d], ..sview(scratch, lay[1], DType::Bf16, s * n * 2 * d) };
                    self.mix_rows(call, t0, n, Some(&dest), Some(&pair))?;
                    self.blend_rows(&dest, (0, n), &i[7], &i[8], &i[9], &i[10], &i[6], &o[3], t0, n, scratch, call.attrs.eps, true)?;
                }
                Ok(())
            }
            (LOOP_BLEND_BF16, Op::LoopBlend) => {
                if o[0].dtype != bf {
                    return Err(Error::invalid(format!("metal {kernel}: bf16 output only")));
                }
                let r = o[0].shape[1];
                let scratch = call.scratch.as_ref().ok_or_else(|| Error::invalid("metal loop_blend_bf16: the scratch (OpCall::scratch) is missing"))?;
                for t0 in (0..r).step_by(BLEND_ROWS) {
                    let n = (r - t0).min(BLEND_ROWS);
                    self.blend_rows(&i[0], (t0, r), &i[1], &i[2], &i[3], &i[4], &i[5], &o[0], t0, n, scratch, call.attrs.eps, false)?;
                }
                Ok(())
            }
            _ => Err(Error::invalid(format!("metal: {kernel} does not run {}", call.op))),
        }
    }
}

//! Lossless 12-bit bf16 weights (P12, lane ink-pack12; kernels/metal/p12.metal, format `intarsia_core::quant::p12`).
//!
//! | kernel | op | rows | bitwise twin |
//! |---|---|---|---|
//! | `matmul_p12_gemv` | `MatmulP12`, bf16 out | 1..=16 | `matmul_bf16_gemv` (1 row) / `matmul_bf16_gemv_rows` |
//! | `matmul_p12_gemv_rows` | the same, `rows_exact` | 1..=16 | `matmul_bf16_gemv_rows` |
//! | `matmul_p12_logits_gemv` | `MatmulP12`, f32 out | 1..=16 | `logits_bf16_gemv` / `logits_bf16_gemv_rows` |
//! | `matmul_p12_logits_gemv_rows` | the same, `rows_exact` | 1..=16 | `logits_bf16_gemv_rows` |
//! | `ink_in_proj_p12_gemv` | `InkInProj` over P12 codes | 1..=16 | `ink_in_proj_bf16_gemv` (the default s8 family) |
//! | `ink_in_proj_p12_gemv_rows` | the same, `rows_exact` | 1..=16 | `ink_in_proj_bf16_gemv_rows` |
//!
//! Every kernel decodes the weights to their exact bf16 bits in registers and runs the arithmetic of its twin in the twin's
//! order (tests/p12.rs checks packed == bf16 bit for bit on the same weights). Operands: K % 16 == 0, codes 8-byte and x
//! 16-byte aligned, bf16 x. Prefill chunks (T > 16) keep the bf16 weights (the model's choice).

use crate::MetalBackend;
use intarsia_core::backend::TensorRef;
use intarsia_core::{DType, Error, Exactness, KernelImpl, Op, OpCall, Result, Tol};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

pub const MATMUL_P12_GEMV: &str = "matmul_p12_gemv";
pub const MATMUL_P12_GEMV_ROWS: &str = "matmul_p12_gemv_rows";
pub const MATMUL_P12_LOGITS_GEMV: &str = "matmul_p12_logits_gemv";
pub const MATMUL_P12_LOGITS_GEMV_ROWS: &str = "matmul_p12_logits_gemv_rows";
pub const INK_IN_PROJ_P12_GEMV: &str = "ink_in_proj_p12_gemv";
pub const INK_IN_PROJ_P12_GEMV_ROWS: &str = "ink_in_proj_p12_gemv_rows";

/// Every kernel of this module, with its op (the `kernels::TABLE` rows).
pub const KERNELS: &[(&str, Op)] = &[
    (MATMUL_P12_GEMV, Op::MatmulP12),
    (MATMUL_P12_GEMV_ROWS, Op::MatmulP12),
    (MATMUL_P12_LOGITS_GEMV, Op::MatmulP12),
    (MATMUL_P12_LOGITS_GEMV_ROWS, Op::MatmulP12),
    (INK_IN_PROJ_P12_GEMV, Op::InkInProj),
    (INK_IN_PROJ_P12_GEMV_ROWS, Op::InkInProj),
];

/// Most activation rows of a launch.
pub const ROWS_MAX: usize = 16;

const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Registration of a kernel of [`KERNELS`] (`kernels::all`). The plain kernel takes every row count 1..=16 (no other P12
/// kernel exists for ordinary multi-row calls); the `_rows` twin is the same shaders for `exact_rows` calls.
pub fn register(k: KernelImpl) -> KernelImpl {
    let name = k.name.clone();
    let base = k.acts([DType::Bf16]).weights([DType::U8]).exact(Exactness::Deterministic);
    match name.as_str() {
        MATMUL_P12_GEMV => base.when(|s| rows_ok(s) && s.out == Some(DType::Bf16)).tol(BF16_TOL).priority(10),
        MATMUL_P12_GEMV_ROWS => base.when(|s| rows_ok(s) && s.out == Some(DType::Bf16)).tol(BF16_TOL).priority(-1).rows_exact(),
        MATMUL_P12_LOGITS_GEMV => base.when(|s| rows_ok(s) && s.out == Some(DType::F32)).tol(F32_TOL).priority(10),
        MATMUL_P12_LOGITS_GEMV_ROWS => base.when(|s| rows_ok(s) && s.out == Some(DType::F32)).tol(F32_TOL).priority(-1).rows_exact(),
        INK_IN_PROJ_P12_GEMV => base.when(rows_ok).tol(BF16_TOL),
        INK_IN_PROJ_P12_GEMV_ROWS => base.when(rows_ok).tol(BF16_TOL).priority(-1).rows_exact(),
        other => unreachable!("p12: {other} has no registration"),
    }
}

/// 1..=16 activation rows, K a multiple of 16.
fn rows_ok(s: &intarsia_core::ShapeInfo) -> bool {
    (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0
}

/// (R, P, RB) instances compiled for every kernel family (`P12_ALL` in p12.metal).
const INSTS: [(usize, usize, usize); 21] = [
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
    (9, 1, 9),
    (10, 1, 5),
    (11, 1, 11),
    (12, 1, 6),
    (13, 1, 13),
    (14, 1, 7),
    (15, 1, 5),
    (16, 1, 8),
];

/// The three shader families and their compiled unroll factors (`P12_ALL` lines in p12.metal).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Fam {
    Proj,
    Rt,
    F32o,
}

impl Fam {
    fn stem(self) -> &'static str {
        match self {
            Fam::Proj => "p12_in_proj_gemv",
            Fam::Rt => "p12_gemv_rt",
            Fam::F32o => "p12_gemv_f32o",
        }
    }
    fn unrolls(self) -> &'static [usize] {
        match self {
            Fam::Proj | Fam::Rt => &[1, 2, 4],
            Fam::F32o => &[2, 4, 8],
        }
    }
    /// Default unroll (`INTARSIA_P12_U_PROJ` / `_RT` / `_F32O` override it: measurement knobs, same bits).
    fn unroll(self) -> usize {
        static C: std::sync::OnceLock<[usize; 3]> = std::sync::OnceLock::new();
        let c = C.get_or_init(|| {
            let g = |v: &str, d: usize, ok: &[usize]| std::env::var(v).ok().and_then(|s| s.parse().ok()).filter(|u| ok.contains(u)).unwrap_or(d);
            [
                g("INTARSIA_P12_U_PROJ", 2, Fam::Proj.unrolls()),
                g("INTARSIA_P12_U_RT", 2, Fam::Rt.unrolls()),
                g("INTARSIA_P12_U_F32O", 4, Fam::F32o.unrolls()),
            ]
        });
        c[self as usize]
    }
    /// P (weight rows per threadgroup, or per simdgroup for F32o) for `r` rows (`INTARSIA_P12_P`: measurement knob, same bits).
    fn inst(self, r: usize) -> (usize, usize) {
        static P: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let want = P.get_or_init(|| std::env::var("INTARSIA_P12_P").ok().and_then(|v| v.parse().ok())).unwrap_or(1);
        let pick = INSTS.iter().find(|&&(rr, p, _)| rr == r && p == want).or_else(|| INSTS.iter().find(|&&(rr, _, _)| rr == r)).expect("rows 1..=16");
        let _ = self;
        (pick.1, pick.2)
    }
}

fn shader(f: Fam, r: usize, p: usize, u: usize, rb: usize) -> &'static str {
    static T: std::sync::OnceLock<std::collections::HashMap<(Fam, usize, usize, usize, usize), &'static str>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        for f in [Fam::Proj, Fam::Rt, Fam::F32o] {
            for &u in f.unrolls() {
                for &(r, p, rb) in &INSTS {
                    m.insert((f, r, p, u, rb), &*Box::leak(format!("{}_r{r}p{p}u{u}b{rb}", f.stem()).into_boxed_str()));
                }
            }
        }
        m
    })
    .get(&(f, r, p, u, rb))
    .copied()
    .expect("compiled p12 instance")
}

/// Every pipeline of p12.metal (`kernels::shader_names`).
pub fn shader_names() -> Vec<&'static str> {
    let mut v = Vec::new();
    for f in [Fam::Proj, Fam::Rt, Fam::F32o] {
        for &u in f.unrolls() {
            for &(r, p, rb) in &INSTS {
                v.push(shader(f, r, p, u, rb));
            }
        }
    }
    v
}

#[repr(C)]
#[derive(Clone, Copy)]
struct P12GemvP {
    k: u32,
    n: u32,
    y_skip: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct P12ProjP {
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
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal p12: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_p12(&self, kernel: &str, call: &OpCall) -> Result<()> {
        match (kernel, call.op) {
            (MATMUL_P12_GEMV | MATMUL_P12_GEMV_ROWS | MATMUL_P12_LOGITS_GEMV | MATMUL_P12_LOGITS_GEMV_ROWS, Op::MatmulP12) => self.launch_p12_gemv(call),
            (INK_IN_PROJ_P12_GEMV | INK_IN_PROJ_P12_GEMV_ROWS, Op::InkInProj) => self.launch_p12_in_proj(call),
            _ => Err(Error::invalid(format!("metal: {kernel} does not run {}", call.op))),
        }
    }

    /// Codes (8-byte aligned), x (16-byte aligned, bf16, K % 16 == 0) and aux bindings shared by both launchers.
    fn p12_operands(&self, x: &TensorRef, codes: &TensorRef, aux: &TensorRef, k: usize, what: &str) -> Result<((usize, usize), crate::dense::Bind)> {
        if x.dtype != DType::Bf16 || codes.dtype != DType::U8 || aux.dtype != DType::I32 {
            return Err(Error::invalid(format!("{what}: bf16 x, u8 codes, u32 aux")));
        }
        let (xb, cb) = (self.mbuf(&x.buf)?, self.mbuf(&codes.buf)?);
        let (xo, co) = (xb.base + x.offset, cb.base + codes.offset);
        if k % 16 != 0 || xo % 16 != 0 || co % 8 != 0 {
            return Err(Error::invalid(format!("{what}: needs K % 16 == 0, x 16- and codes 8-byte aligned (K {k}, x {xo}, codes {co})")));
        }
        let ab = self.bind(aux, DType::I32, false, "aux")?;
        Ok(((xo, co), ab))
    }

    /// `MatmulP12`, 1..=16 rows: bf16 out on the `rt` family (gemv_bf16_rt_u2s8 order), f32 out on the `f32o` family.
    fn launch_p12_gemv(&self, call: &OpCall) -> Result<()> {
        let (i, y) = (&call.inputs, &call.outputs[0]);
        let (m, k, n) = (i[0].shape[0], i[0].shape[1], y.shape[1]);
        if m == 0 || n == 0 {
            return Ok(());
        }
        if m > ROWS_MAX {
            return Err(Error::invalid(format!("matmul_p12_gemv: {m} rows (at most {ROWS_MAX})")));
        }
        let ((xo, co), ab) = self.p12_operands(&i[0], &i[1], &i[2], k, "matmul_p12_gemv")?;
        let f32o = y.dtype == DType::F32;
        let yb = self.bind(y, y.dtype, true, "y")?;
        let fam = if f32o { Fam::F32o } else { Fam::Rt };
        let (p, rb) = fam.inst(m);
        let name = shader(fam, m, p, fam.unroll(), rb);
        let prm = P12GemvP { k: dim(k, "K")?, n: dim(n, "N")?, y_skip: yb.skip, pad: 0 };
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        let (groups, threads) = if f32o { (n.div_ceil(4 * p), 128) } else { (n.div_ceil(p), 256) };
        let (xb, cb) = (self.mbuf(&i[0].buf)?, self.mbuf(&i[1].buf)?);
        self.with_encoder(|enc| {
            enc.setComputePipelineState(pso);
            // SAFETY: offsets inside the buffers (OpCall::check), codes 8- and x 16-byte aligned (checked above).
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&cb.mtl), co, 0);
                enc.setBuffer_offset_atIndex(Some(&xb.mtl), xo, 1);
                enc.setBuffer_offset_atIndex(Some(&yb.mtl), yb.off, 2);
                enc.setBuffer_offset_atIndex(Some(&ab.mtl), ab.off, 3);
            }
            crate::set_bytes(enc, &prm, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: groups, height: 1, depth: 1 },
                MTLSize { width: threads, height: 1, depth: 1 },
            );
        })
    }

    /// `InkInProj` over P12 codes, T = 1..=16 rows: one dispatch over the three segments (ink_in_proj_gemv's s8 order).
    fn launch_p12_in_proj(&self, call: &OpCall) -> Result<()> {
        let (i, o) = (&call.inputs, &call.outputs);
        let (s, t, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        if t == 0 {
            return Ok(());
        }
        if t > ROWS_MAX || o.iter().any(|y| y.dtype != DType::Bf16) {
            return Err(Error::invalid(format!("ink_in_proj_p12_gemv: {t} rows (at most {ROWS_MAX}), bf16 outputs")));
        }
        let ((xo, co), ab) = self.p12_operands(&i[0], &i[1], &i[2], d, "ink_in_proj_p12_gemv")?;
        let ns = [o[0].shape[1], o[1].shape[1], o[2].shape[1]];
        let ys: Vec<_> = o.iter().enumerate().map(|(j, y)| self.bind(y, DType::Bf16, true, ["pq", "pk", "pv"][j])).collect::<Result<_>>()?;
        let (p, rb) = Fam::Proj.inst(t);
        let name = shader(Fam::Proj, t, p, Fam::Proj.unroll(), rb);
        let g = ns.map(|n| n.div_ceil(p));
        let xoff = |j: usize| -> Result<u32> { dim(j.min(s - 1) * t * d, "x offset") };
        let prm = P12ProjP {
            k: dim(d, "D")?,
            n: [dim(ns[0], "Nq")?, dim(ns[1], "Nk")?, dim(ns[2], "Nv")?],
            xo1: xoff(1)?,
            xo2: xoff(2)?,
            ys: [ys[0].skip, ys[1].skip, ys[2].skip],
            g0: dim(g[0], "groups")?,
            g1: dim(g[1], "groups")?,
            gtot: dim(g.iter().sum::<usize>(), "groups")?,
        };
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        let (xb, cb) = (self.mbuf(&i[0].buf)?, self.mbuf(&i[1].buf)?);
        let groups = g.iter().sum::<usize>();
        self.with_encoder(|enc| {
            enc.setComputePipelineState(pso);
            // SAFETY: offsets inside the buffers (OpCall::check), codes 8- and x 16-byte aligned (checked above).
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&cb.mtl), co, 0);
                enc.setBuffer_offset_atIndex(Some(&xb.mtl), xo, 1);
                for (j, y) in ys.iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&y.mtl), y.off, 2 + j);
                }
                enc.setBuffer_offset_atIndex(Some(&ab.mtl), ab.off, 5);
            }
            crate::set_bytes(enc, &prm, 6);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: groups, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        })
    }
}

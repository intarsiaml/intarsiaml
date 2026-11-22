//! Launchers of the Inkling-2-Mini MoE kernels (kernels/metal/inkling2_moe.metal, the grouped prefill in
//! kernels/metal/inkling2_moe_nax.metal): `Op::MoeRoute`, `Op::MoeExperts` with the SiTU activation (NVFP4, `.situ()` and
//! only situ calls) and `Op::MoeExpertsMxfp8` (silu / gelu / situ).
//!
//! Experts, decode and verify rows (T <= [`DEC_MAX_ROWS`], or any T without the metal4.0 library): gate+up into scratch `h`
//! (by pair), then either the fused down + k-order token sum (`dn_f`, one launch) or down by pair into scratch `y` + the
//! gather. Gate+up and the by-pair down run union-grouped by default (each expert's weights read once per call for all the
//! pairs routed to it, [`Policy`]). Every path gives a pair the same bits (fixed per-row chunk order and reduction, the token
//! sum in k order), so a T-row call is bitwise T one-row calls and the `_rows` kernels are the same launcher
//! (tests/inkmoe.rs). Longer calls group the pairs by expert (moe.metal's `moe_group`) and run the grouped NAX GEMMs.
//! Scratch: the core `moe_experts_scratch` layout (pairs, offsets, tile prefix, h by slot or pair, y by pair).

use crate::dense::{Bind, OpP};
use crate::{set_bytes, MetalBackend};
use intarsia_core::dsv41::ops::ScoreFunc;
use intarsia_core::{DType, Error, Exactness, KernelImpl, OpCall, Result, Tol};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use std::sync::atomic::{AtomicU8, Ordering};

/// `Op::MoeRoute`, f32 logits (any E <= 1024, K <= 32, every score function).
pub const MOE_ROUTE_F32: &str = "moe_route_f32";
/// `Op::MoeExperts` with `ink.situ`, NVFP4 experts, bf16 x: decode rows, verify rows, prefill.
pub const MOE_EXPERTS_SITU_BF16: &str = "moe_experts_situ_bf16";
/// The same launcher as a row-exact verify kernel (1..=16 rows): row t bitwise the one-row call on token t.
pub const MOE_EXPERTS_SITU_BF16_ROWS: &str = "moe_experts_situ_bf16_rows";
/// `Op::MoeExpertsMxfp8`, bf16 x, silu / gelu / situ.
pub const MOE_EXPERTS_MXFP8_BF16: &str = "moe_experts_mxfp8_bf16";
pub const MOE_EXPERTS_MXFP8_BF16_ROWS: &str = "moe_experts_mxfp8_bf16_rows";

/// (kernel, op) entries of `kernels::TABLE`.
pub const KERNELS: [(&str, intarsia_core::Op); 5] = [
    (MOE_ROUTE_F32, intarsia_core::Op::MoeRoute),
    (MOE_EXPERTS_SITU_BF16, intarsia_core::Op::MoeExperts),
    (MOE_EXPERTS_SITU_BF16_ROWS, intarsia_core::Op::MoeExperts),
    (MOE_EXPERTS_MXFP8_BF16, intarsia_core::Op::MoeExpertsMxfp8),
    (MOE_EXPERTS_MXFP8_BF16_ROWS, intarsia_core::Op::MoeExpertsMxfp8),
];

const ROUTE: &str = "ink_moe_route";
const GATHER: [&str; 2] = ["ink_moe_gather_bf16", "ink_moe_gather_f32"];

/// Shader entry points of kernels/metal/inkling2_moe.metal.
pub const SHADERS: [&str; 23] = [
    ROUTE,
    "ink_moe_gu_nv_a_x",
    "ink_moe_gu_nv_a",
    "ink_moe_gu_nv_u_x",
    "ink_moe_gu_nv_u",
    "ink_moe_gu_mx_a_x",
    "ink_moe_gu_mx_a",
    "ink_moe_gu_mx_u_x",
    "ink_moe_gu_mx_u",
    "ink_moe_dn_u_nv_a",
    "ink_moe_dn_u_nv_u",
    "ink_moe_dn_u_mx_a",
    "ink_moe_dn_u_mx_u",
    "ink_moe_dn_f_nv_a_bf16",
    "ink_moe_dn_f_nv_u_bf16",
    "ink_moe_dn_f_mx_a_bf16",
    "ink_moe_dn_f_mx_u_bf16",
    "ink_moe_dn_f_nv_a_f32",
    "ink_moe_dn_f_nv_u_f32",
    "ink_moe_dn_f_mx_a_f32",
    "ink_moe_dn_f_mx_u_f32",
    GATHER[0],
    GATHER[1],
];

/// The grouped prefill GEMMs on the neural accelerators (kernels/metal/inkling2_moe_nax.metal, metal4.0 library), by
/// format (NVFP4, MXFP8) and rows per tile (8, 16, 32, 64).
pub const NAX_GU: [[&str; 4]; 2] = [
    ["ink_moe_nax_gu_nv_m8", "ink_moe_nax_gu_nv_m16", "ink_moe_nax_gu_nv_m32", "ink_moe_nax_gu_nv_m64"],
    ["ink_moe_nax_gu_mx_m8", "ink_moe_nax_gu_mx_m16", "ink_moe_nax_gu_mx_m32", "ink_moe_nax_gu_mx_m64"],
];
pub const NAX_DN: [[&str; 4]; 2] = [
    ["ink_moe_nax_dn_nv_m8", "ink_moe_nax_dn_nv_m16", "ink_moe_nax_dn_nv_m32", "ink_moe_nax_dn_nv_m64"],
    ["ink_moe_nax_dn_mx_m8", "ink_moe_nax_dn_mx_m16", "ink_moe_nax_dn_mx_m32", "ink_moe_nax_dn_mx_m64"],
];

/// Entry points of the metal4.0 library (`kernels::metal4_shaders`).
pub fn metal4_shaders() -> impl Iterator<Item = &'static str> {
    NAX_GU.iter().chain(NAX_DN.iter()).flatten().copied()
}

/// `INTARSIA_INK_MOE_NAX=0` keeps every call on the decode kernels (no grouped prefill GEMMs).
fn nax_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_INK_MOE_NAX").map_or(true, |v| v != "0"))
}

/// Rows per grouped tile for `pairs` pairs over `e` experts, as (index into `NAX_GU` / `NAX_DN`, rows): the smallest of
/// 8 / 16 / 32 / 64 at or above the mean rows per expert (moe.rs's rule).
fn tile_rows(pairs: usize, e: usize) -> (usize, usize) {
    let mean = pairs.div_ceil(e.max(1));
    let ti = (0..3).find(|&t| mean <= 8 << t).unwrap_or(3);
    (ti, 8 << ti)
}

/// Rows at or below which the experts take the decode kernels (above: grouped GEMMs when the metal4.0 library loaded).
pub const DEC_MAX_ROWS: usize = 16;
const MAX_EXPERTS: usize = 1024;
const MAX_K: usize = 32;
/// gate+up rows per threadgroup (4 simdgroups x 4 rows), down rows per threadgroup of `dn_u` (4 x 8) and of `dn_f`.
const GU_ROWS: usize = 16;
const DNU_ROWS: usize = 32;
const DNF_ROWS: usize = 8;

/// bf16 expert outputs (the Metal MoE budget, `kernels::MOE_TOL`).
const MOE_TOL: Tol = Tol::new(1.5e-2, 1e-3);
/// f32 router weights.
const ROUTE_TOL: Tol = Tol::new(1e-3, 2e-4);

/// How the decode experts run (all give the same bits). Default from `INTARSIA_INK_MOE` (`union`, `pair`, `fused`,
/// `union-fused`: gate+up union-grouped or per pair; down fused with the token sum or by pair + gather), else
/// [`Policy::Auto`]: union gate+up, fused down at one row, union down + gather above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Policy {
    Auto = 0,
    /// union gate+up, union down + gather
    Union = 1,
    /// per-pair gate+up, per-pair down + gather
    Pair = 2,
    /// per-pair gate+up, fused down
    Fused = 3,
    /// union gate+up, fused down
    UnionFused = 4,
}

static POLICY: AtomicU8 = AtomicU8::new(255);

fn policy() -> Policy {
    let v = POLICY.load(Ordering::Relaxed);
    let p = if v == 255 {
        let p = match std::env::var("INTARSIA_INK_MOE").as_deref() {
            Ok("union") => Policy::Union,
            Ok("pair") => Policy::Pair,
            Ok("fused") => Policy::Fused,
            Ok("union-fused") => Policy::UnionFused,
            _ => Policy::Auto,
        };
        POLICY.store(p as u8, Ordering::Relaxed);
        p
    } else {
        [Policy::Auto, Policy::Union, Policy::Pair, Policy::Fused, Policy::UnionFused][v as usize]
    };
    p
}

static PROBE: AtomicU8 = AtomicU8::new(0);

/// Benchmarks only: 1 launches only the gate+up kernel of a decode call, 2 only its down kernel(s) (the output is then
/// garbage); 0 (default) the whole call.
pub fn set_probe(v: u8) {
    PROBE.store(v, Ordering::Relaxed);
}

static DNF_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(128);

/// Threads per threadgroup of the fused down kernel (a multiple of 32; benchmarks: every choice gives the same bits).
pub fn set_dnf_threads(n: usize) {
    DNF_THREADS.store(n, Ordering::Relaxed);
}

fn dnf_threads() -> usize {
    DNF_THREADS.load(Ordering::Relaxed)
}

/// Set the decode policy for this process (benchmarks and tests).
pub fn set_policy(p: Policy) {
    POLICY.store(p as u8, Ordering::Relaxed);
}

/// The registrations (`kernels::all` hands each entry of [`KERNELS`] here).
pub fn register(name: &str, k: KernelImpl) -> KernelImpl {
    let bf = [DType::Bf16];
    match name {
        MOE_ROUTE_F32 => k
            .acts([DType::F32])
            .when(|s| s.experts <= MAX_EXPERTS && (1..=MAX_K).contains(&s.top_k) && s.top_k <= s.experts)
            .exact(Exactness::Deterministic)
            .tol(ROUTE_TOL),
        // situ calls only: the 35B / Gemma NVFP4 calls keep `moe_experts_bf16`
        MOE_EXPERTS_SITU_BF16 => k
            .acts(bf)
            .weights([DType::U8])
            .situ()
            .when(|s| s.situ && s.experts <= MAX_EXPERTS && s.top_k <= MAX_K && s.k % 32 == 0 && s.n % 32 == 0)
            .exact(Exactness::Deterministic)
            .tol(MOE_TOL)
            .scratch(intarsia_core::quant::ops::moe_experts_scratch),
        MOE_EXPERTS_SITU_BF16_ROWS => k
            .acts(bf)
            .weights([DType::U8])
            .situ()
            .when(|s| {
                s.situ
                    && (1..=DEC_MAX_ROWS).contains(&s.m)
                    && s.experts <= MAX_EXPERTS
                    && s.top_k <= MAX_K
                    && s.k % 32 == 0
                    && s.n % 32 == 0
            })
            .exact(Exactness::Deterministic)
            .tol(MOE_TOL)
            .priority(-1)
            .rows_exact()
            .scratch(intarsia_core::quant::ops::moe_experts_scratch),
        MOE_EXPERTS_MXFP8_BF16 => k
            .acts(bf)
            .weights([DType::Fp8E4m3])
            .ext()
            .situ()
            .when(|s| s.experts <= MAX_EXPERTS && s.top_k <= MAX_K && s.k % 32 == 0 && s.n % 32 == 0)
            .exact(Exactness::Deterministic)
            .tol(MOE_TOL)
            .scratch(intarsia_core::quant::ops::moe_experts_scratch),
        MOE_EXPERTS_MXFP8_BF16_ROWS => k
            .acts(bf)
            .weights([DType::Fp8E4m3])
            .ext()
            .situ()
            .when(|s| {
                (1..=DEC_MAX_ROWS).contains(&s.m)
                    && s.experts <= MAX_EXPERTS
                    && s.top_k <= MAX_K
                    && s.k % 32 == 0
                    && s.n % 32 == 0
            })
            .exact(Exactness::Deterministic)
            .tol(MOE_TOL)
            .priority(-1)
            .rows_exact()
            .scratch(intarsia_core::quant::ops::moe_experts_scratch),
        other => unreachable!("inkmoe: no registration for {other}"),
    }
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal inkling2 moe: {what} = {v} exceeds u32")))
}

fn at(b: &Bind, off: usize) -> Bind {
    Bind { mtl: b.mtl.clone(), off: b.off + off, skip: 0 }
}

impl MetalBackend {
    fn ink_dispatch(&self, name: &str, binds: &[&Bind], p: &OpP, groups: MTLSize, threads: usize) -> Result<()> {
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        self.with_encoder(|enc| {
            enc.setComputePipelineState(pso);
            for (i, b) in binds.iter().enumerate() {
                // SAFETY: offsets lie inside the buffers (TensorRef bounds checked by `OpCall::check`, scratch by the launcher).
                unsafe { enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i) };
            }
            set_bytes(enc, p, binds.len());
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, MTLSize { width: threads, height: 1, depth: 1 });
        })
    }

    /// The Inkling-2 MoE kernels; `None` when `kernel` is not one of them.
    pub(crate) fn launch_inkmoe(&self, kernel: &str, call: &OpCall) -> Option<Result<()>> {
        match kernel {
            MOE_ROUTE_F32 => Some(self.launch_ink_route(call)),
            MOE_EXPERTS_SITU_BF16 | MOE_EXPERTS_SITU_BF16_ROWS => Some(self.launch_ink_experts(call, false)),
            MOE_EXPERTS_MXFP8_BF16 | MOE_EXPERTS_MXFP8_BF16_ROWS => Some(self.launch_ink_experts(call, true)),
            _ => None,
        }
    }

    fn launch_ink_route(&self, call: &OpCall) -> Result<()> {
        let ([lg, bias], [w, ids]) = (call.inputs.as_slice(), call.outputs.as_slice()) else {
            return Err(Error::invalid("metal moe_route: expected 2 inputs / 2 outputs"));
        };
        let (t, e, k) = (lg.shape[0], lg.shape[1], call.attrs.top_k);
        if e > MAX_EXPERTS || k == 0 || k > MAX_K || k > e {
            return Err(Error::invalid(format!(
                "metal moe_route: E = {e}, K = {k} (needs E <= {MAX_EXPERTS}, K in 1..={MAX_K})"
            )));
        }
        if t == 0 {
            return Ok(());
        }
        let lb = self.bind(lg, DType::F32, false, "logits")?;
        let bb = self.bind(bias, DType::F32, false, "bias")?;
        let wb = self.bind(w, DType::F32, true, "weights")?;
        let ib = self.bind(ids, DType::I32, true, "ids")?;
        let sf = match call.attrs.dsv.score_func {
            ScoreFunc::Softmax => 0,
            ScoreFunc::Sigmoid => 1,
            ScoreFunc::SqrtSoftplus => 2,
        };
        let a = &call.attrs;
        let p = OpP {
            dims: [u32_of(t, "T")?, e as u32, k as u32, a.norm_topk as u32, sf, 0, 0, 0],
            f0: a.scale,
            f1: a.temperature,
            ..Default::default()
        };
        self.ink_dispatch(ROUTE, &[&lb, &bb, &wb, &ib], &p, MTLSize { width: t.div_ceil(4), height: 1, depth: 1 }, 128)
    }

    /// `MoeExperts` (situ, NVFP4: `mx` false) or `MoeExpertsMxfp8` (`mx` true).
    fn launch_ink_experts(&self, call: &OpCall, mx: bool) -> Result<()> {
        let n_in = if mx { 7 } else { 8 };
        if call.inputs.len() != n_in || call.outputs.len() != 1 {
            return Err(Error::invalid(format!("metal inkling2 experts: expected {n_in} inputs / 1 output")));
        }
        let i = &call.inputs;
        let out = &call.outputs[0];
        let (x, ids, wts) = (&i[0], &i[1], &i[2]);
        let (t, h, k) = (x.shape[0], x.shape[1], ids.shape[1]);
        let (e, inter) = (i[3].shape[0], i[3].shape[1] / 2);
        if e > MAX_EXPERTS || k > MAX_K || h % 32 != 0 || inter % 32 != 0 {
            return Err(Error::invalid(format!(
                "metal inkling2 experts: E = {e} (<= {MAX_EXPERTS}), K = {k} (<= {MAX_K}), H = {h}, I = {inter} (multiples of 32)"
            )));
        }
        let a = &call.attrs;
        if !mx && !a.ink.situ {
            return Err(Error::invalid("metal moe_experts_situ: a call without ink.situ"));
        }
        let epi: u32 = if a.ink.situ {
            2
        } else if a.gelu {
            1
        } else {
            0
        };
        let pairs = t * k;
        if pairs == 0 || h == 0 {
            return Ok(());
        }
        let o32 = match out.dtype {
            DType::Bf16 => 0,
            DType::F32 => 1,
            d => return Err(Error::invalid(format!("metal inkling2 experts: output dtype {d}"))),
        };
        u32_of(pairs * h.max(inter), "T*K*max(H,I)")?;
        let (wdt, sdt) = if mx { (DType::Fp8E4m3, DType::U8) } else { (DType::U8, DType::Fp8E4m3) };
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let ib = self.bind(ids, DType::I32, false, "ids")?;
        let wb = self.bind(wts, DType::F32, false, "weights")?;
        let w13 = self.bind(&i[3], wdt, false, "w13")?;
        let s13 = self.bind(&i[4], sdt, false, "w13 scales")?;
        let w2 = self.bind(&i[5], wdt, false, "w2")?;
        let s2 = self.bind(&i[6], sdt, false, "w2 scales")?;
        // the global scales (NVFP4); MXFP8 has none, the kernels then never read binding 5
        let gb = if mx { at(&wb, 0) } else { self.bind(&i[7], DType::F32, false, "global scales")? };
        let ob = self.bind(out, out.dtype, true, "out")?;
        if w13.skip != 0 || w2.skip != 0 {
            return Err(Error::invalid("metal inkling2 experts: the expert weights must be 4-byte aligned"));
        }
        let a16 = w13.off % 16 == 0 && w2.off % 16 == 0;
        let ax = (xb.off + 2 * xb.skip as usize) % 16 == 0;
        let scratch = call.scratch.as_ref().ok_or_else(|| {
            Error::invalid("metal inkling2 experts: the scratch workspace (OpCall::scratch) is missing")
        })?;
        let r = |v: usize| (v + 255) & !255;
        let (o_off, o_tiles, o_h) = (r(pairs * 4), r(pairs * 4) + r((e + 1) * 4), r(pairs * 4) + 2 * r((e + 1) * 4));
        let o_y = o_h + r(pairs * inter * 2);
        let need = o_y + r(pairs * h * 2);
        let have = scratch.numel() * scratch.dtype.size_bytes();
        if have < need {
            return Err(Error::invalid(format!("metal inkling2 experts: scratch of {have} bytes, need {need}")));
        }
        let sc = self.bind(scratch, scratch.dtype, true, "scratch")?;
        if sc.skip != 0 {
            return Err(Error::invalid("metal inkling2 experts: the scratch must be 4-byte aligned"));
        }
        let mut p = OpP {
            dims: [u32_of(t, "T")?, e as u32, k as u32, u32_of(h, "H")?, u32_of(inter, "I")?, epi, 0, 0],
            skips: [xb.skip, s13.skip, s2.skip, ob.skip, 0],
            f0: a.ink.gate_cap,
            f1: a.ink.up_cap,
            ..Default::default()
        };
        let hb = at(&sc, o_h);
        let fi = mx as usize;
        if t > DEC_MAX_ROWS && nax_enabled() && self.has_metal4() && self.pso(NAX_GU[fi][0]).is_some() {
            // grouped by expert (moe.metal's moe_group reads the rows per tile from p.f), h by slot, y by pair, then the gather
            let (pb, offb, tb, yb) = (at(&sc, 0), at(&sc, o_off), at(&sc, o_tiles), at(&sc, o_y));
            let (ti, tm) = tile_rows(pairs, e);
            // lane ink-prefill: the second-generation forms when one is selected (inkprefill.rs)
            let form = crate::inkprefill::moe_nax(mx, tm).filter(|f| {
                self.pso(f.gu).is_some()
                    && self.pso(f.dn).is_some()
                    && (f.bn3.is_none() || self.pso(crate::inkprefill::NAX3_GATHER).is_some())
            });
            let (gu_k, dn_k, tm) = form.map_or((NAX_GU[fi][ti], NAX_DN[fi][ti], tm), |f| (f.gu, f.dn, f.tm));
            let bn3 = form.and_then(|f| f.bn3);
            let mut pg = p;
            pg.dims[5] = tm as u32;
            let gt = self.pso("moe_group").map_or(256, |s| {
                use objc2_metal::MTLComputePipelineState;
                s.maxTotalThreadsPerThreadgroup().min(1024)
            });
            self.ink_dispatch(
                "moe_group",
                &[&ib, &pb, &offb, &tb],
                &pg,
                MTLSize { width: 1, height: 1, depth: 1 },
                gt,
            )?;
            let max_tiles = pairs.div_ceil(tm) + pairs.min(e);
            let (cg, cd) = bn3.map_or((32, 64), |bn| (bn / 2, bn));
            let ggu = MTLSize { width: inter.div_ceil(cg), height: max_tiles, depth: 1 };
            let skip = crate::inkprefill::nax_skip();
            // third generation: the x rows into slot order first (the y region, free until the down writes it)
            let ab = if bn3.is_some() {
                let gx = MTLSize { width: (h / 8).div_ceil(64), height: pairs, depth: 1 };
                self.ink_dispatch(crate::inkprefill::NAX3_GATHER, &[&xb, &pb, &offb, &yb], &p, gx, 64)?;
                at(&yb, 0)
            } else {
                at(&xb, 0)
            };
            if skip & 1 == 0 {
                self.ink_dispatch(gu_k, &[&ab, &pb, &offb, &tb, &w13, &s13, &gb, &wb, &hb], &p, ggu, 128)?;
            }
            let gdn = MTLSize { width: h.div_ceil(cd), height: max_tiles, depth: 1 };
            if skip & 2 == 0 {
                self.ink_dispatch(dn_k, &[&hb, &pb, &offb, &tb, &w2, &s2, &gb, &wb, &yb], &p, gdn, 128)?;
            }
            if skip & 4 != 0 {
                return Ok(());
            }
            let gg = MTLSize { width: (t * h).div_ceil(256), height: 1, depth: 1 };
            return self.ink_dispatch(GATHER[o32], &[&yb, &ib, &wb, &ob], &p, gg, 256);
        }
        let fmt = if mx { "mx" } else { "nv" };
        let al = if a16 { "a" } else { "u" };
        let pol = policy();
        let (uni_gu, fused) = match pol {
            Policy::Auto => (true, t == 1),
            Policy::Union => (true, false),
            Policy::Pair => (false, false),
            Policy::Fused => (false, true),
            Policy::UnionFused => (true, true),
        };
        // gate + up
        p.dims[5] = epi | if uni_gu { 4 } else { 0 };
        let gu = format!("ink_moe_gu_{fmt}_{al}{}", if ax { "_x" } else { "" });
        let rb = inter.div_ceil(GU_ROWS);
        let g1 = if uni_gu {
            MTLSize { width: rb, height: pairs, depth: 1 }
        } else {
            MTLSize { width: pairs, height: rb, depth: 1 }
        };
        let probe = PROBE.load(Ordering::Relaxed);
        if probe != 2 {
            self.ink_dispatch(&gu, &[&xb, &ib, &wb, &w13, &s13, &gb, &hb], &p, g1, 128)?;
        }
        if probe == 1 {
            return Ok(());
        }
        if fused {
            p.dims[5] = epi;
            let dn = format!("ink_moe_dn_f_{fmt}_{al}_{}", if o32 == 1 { "f32" } else { "bf16" });
            let g2 = MTLSize { width: t, height: h.div_ceil(DNF_ROWS), depth: 1 };
            return self.ink_dispatch(&dn, &[&hb, &ib, &wb, &w2, &s2, &gb, &ob], &p, g2, dnf_threads());
        }
        let uni_dn = uni_gu;
        p.dims[5] = epi | if uni_dn { 4 } else { 0 };
        let yb = at(&sc, o_y);
        let dn = format!("ink_moe_dn_u_{fmt}_{al}");
        let nb = h.div_ceil(DNU_ROWS);
        let g2 = if uni_dn {
            MTLSize { width: nb, height: pairs, depth: 1 }
        } else {
            MTLSize { width: pairs, height: nb, depth: 1 }
        };
        self.ink_dispatch(&dn, &[&hb, &ib, &w2, &s2, &gb, &yb], &p, g2, 128)?;
        p.dims[5] = epi;
        let gg = MTLSize { width: (t * h).div_ceil(256), height: 1, depth: 1 };
        self.ink_dispatch(GATHER[o32], &[&yb, &ib, &wb, &ob], &p, gg, 256)
    }
}

//! Launchers of the sparse-MoE ops (`Op::MoeTopk`, `Op::MoeExperts`, `Op::MoeCombine`; kernels/metal/moe.metal, the port
//! of kernels/hip/moe.hip and the grouped structure of kernels/cuda/moe_grouped.cu).
//!
//! `MoeExperts` takes one of two paths per call: T*K <= `DECODE_MAX_PAIRS` (HIP's decode cut) runs per-pair GEMVs (gate +
//! up into scratch `h`, then down + the weighted per-token sum in k order); larger calls group the pairs by expert on the
//! GPU (`moe_group`) into tiles of `tile_rows` (per call, from the mean rows per expert), run each tile against 64
//! weight rows at a time on bf16 simdgroup matrices (each tile's weights decoded once) and gather per token in k order. Scratch is the core `moe_experts_scratch` layout (pairs, offsets, the "active" region
//! holding the per-expert tile prefix, h by slot, y by pair).

use crate::dense::{Bind, OpP};
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

pub const MOE_TOPK_BF16: &str = "moe_topk_bf16";
pub const MOE_EXPERTS_BF16: &str = "moe_experts_bf16";
pub const MOE_COMBINE_BF16: &str = "moe_combine_bf16";
/// `rows_exact` expert-grouped decode for 2..=8 tokens (continuous batching of the hybrid 35B,
/// docs/results/2026-10-23-mb-metal-hybrid-batching.md): token t is bitwise the T = 1 `moe_experts_bf16` call on token t alone
/// (tests/rows_exact.rs), and each expert's weights are read once for all the tokens routed to it.
pub const MOE_EXPERTS_BF16_ROWS: &str = "moe_experts_bf16_rows";

const TOPK_BF16: &str = "moe_topk_bf16";
const TOPK_F32: &str = "moe_topk_f32";
const COMBINE: &str = "moe_combine_bf16";
const DEC_GU: &str = "moe_dec_gu";
const DEC_DN: [&str; 2] = ["moe_dec_dn_bf16", "moe_dec_dn_f32"];
const GROUP: &str = "moe_group";
const DEC_GU_X: &str = "moe_dec_gu_x";
const DEC_DN_X: &str = "moe_dec_dn_x";
/// Grouped GEMMs by rows per tile (8, 16, 32, 64).
const MM_GU: [&str; 4] = ["moe_mm_gu_m8", "moe_mm_gu_m16", "moe_mm_gu_m32", "moe_mm_gu_m64"];
const MM_DN: [&str; 4] = ["moe_mm_dn_m8", "moe_mm_dn_m16", "moe_mm_dn_m32", "moe_mm_dn_m64"];
const GATHER: [&str; 2] = ["moe_gather_bf16", "moe_gather_f32"];

/// The row-exact decode's fat-threadgroup gate+up / down kernels (`MOE_PR` = 2 rows per simdgroup, no grouping by expert;
/// lane mm, docs/results/2026-10-25-mm-metal-moe-grouped.md) and their negative-control forms.
const GU_P: &str = "moe_dec_gu_p";
const DN_P: &str = "moe_dec_dn_p";
const GU_P_NEG: &str = "moe_dec_gu_p_neg";
const DN_P_NEG: &str = "moe_dec_dn_p_neg";
const PR: usize = 2;

/// Shader entry points (kernels/metal/moe.metal).
pub const SHADERS: [&str; 23] = [
    TOPK_BF16, TOPK_F32, COMBINE, DEC_GU, DEC_DN[0], DEC_DN[1], GROUP, MM_GU[0], MM_GU[1], MM_GU[2], MM_GU[3],
    MM_DN[0], MM_DN[1], MM_DN[2], MM_DN[3], GATHER[0], GATHER[1], DEC_GU_X, DEC_DN_X, GU_P, DN_P, GU_P_NEG, DN_P_NEG,
];

/// The rows path's gate+up and down shaders with their rows per simdgroup: the fat-threadgroup kernels unless
/// `INTARSIA_METAL_MOE_ROWS_FAT=0` (the expert-grouped `_x` kernels of lane mb); `INTARSIA_METAL_NEGCTL=moe_gu` / `moe_dn`
/// swaps in a negative-control kernel (tests only: its output is NOT the solo call's).
fn rows_shaders() -> ((&'static str, usize), (&'static str, usize)) {
    static PICK: std::sync::OnceLock<((&'static str, usize), (&'static str, usize))> = std::sync::OnceLock::new();
    *PICK.get_or_init(|| {
        if std::env::var("INTARSIA_METAL_MOE_ROWS_FAT").is_ok_and(|v| v == "0") {
            return ((DEC_GU_X, 1), (DEC_DN_X, 1));
        }
        let neg = std::env::var("INTARSIA_METAL_NEGCTL").unwrap_or_default();
        let gu = if neg == "moe_gu" { GU_P_NEG } else { GU_P };
        let dn = if neg == "moe_dn" { DN_P_NEG } else { DN_P };
        ((gu, PR), (dn, PR))
    })
}

/// Tokens the row-exact decode takes (`MOE_XR` pairs per tile: an expert has at most one pair per token).
pub const ROWS_MAX: usize = 8;

/// T*K at or below which `MoeExperts` takes the per-pair decode path (the hip decode kernels' cut).
pub const DECODE_MAX_PAIRS: usize = 16;
/// Tokens per threadgroup of the top-k shader (`MOE_TOPK_SG`), rows per threadgroup of the decode GEMVs (`MOE_DEC_SG`),
/// gate (= up) / down columns of a grouped tile (`MOE_WR` / 2, `MOE_WR`).
const TOPK_SG: usize = 4;
const DEC_SG: usize = 4;
const GU_COLS: usize = 32;
const DN_COLS: usize = 64;
/// The largest expert count `moe_group` takes (its threadgroup counters).
pub const MAX_EXPERTS: usize = 1024;

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal moe: {what} = {v} exceeds u32")))
}

fn at(b: &Bind, off: usize) -> Bind {
    Bind { mtl: b.mtl.clone(), off: b.off + off, skip: 0 }
}

impl MetalBackend {
    fn moe_dispatch(&self, name: &str, binds: &[&Bind], p: &OpP, groups: MTLSize, threads: usize) -> Result<()> {
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

    pub(crate) fn launch_moe_topk(&self, call: &OpCall) -> Result<()> {
        let (lg, w, ids) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([lg], [w, ids]) => (lg, w, ids),
            _ => return Err(Error::invalid("metal moe_topk: expected 1 input / 2 outputs")),
        };
        let (t, e, k) = (lg.shape[0], lg.shape[1], call.attrs.top_k);
        if e > 512 || k == 0 || k > 32 || k > e {
            return Err(Error::invalid(format!("metal moe_topk: E = {e}, K = {k} (needs E <= 512, K in 1..=32)")));
        }
        if t == 0 {
            return Ok(());
        }
        let shader = match lg.dtype {
            DType::Bf16 => TOPK_BF16,
            DType::F32 => TOPK_F32,
            d => return Err(Error::invalid(format!("metal moe_topk: logits dtype {d}"))),
        };
        let lb = self.bind(lg, lg.dtype, false, "logits")?;
        let wb = self.bind(w, DType::F32, true, "weights")?;
        let ib = self.bind(ids, DType::I32, true, "ids")?;
        let p = OpP {
            dims: [u32_of(t, "T")?, e as u32, k as u32, call.attrs.norm_topk as u32, 0, 0, 0, 0],
            skips: [lb.skip, 0, 0, 0, 0],
            ..Default::default()
        };
        let groups = MTLSize { width: t.div_ceil(TOPK_SG), height: 1, depth: 1 };
        self.moe_dispatch(shader, &[&lb, &wb, &ib], &p, groups, TOPK_SG * 32)
    }

    pub(crate) fn launch_moe_combine(&self, call: &OpCall) -> Result<()> {
        let (r, s, g, out) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([r, s, g], [out]) => (r, s, g, out),
            _ => return Err(Error::invalid("metal moe_combine: expected 3 inputs / 1 output")),
        };
        let (t, h) = (r.shape[0], r.shape[1]);
        if t * h == 0 {
            return Ok(());
        }
        let rb = self.bind(r, DType::Bf16, false, "routed")?;
        let sb = self.bind(s, DType::Bf16, false, "shared")?;
        let gb = self.bind(g, DType::Bf16, false, "gate")?;
        let ob = self.bind(out, DType::Bf16, true, "out")?;
        let p = OpP {
            dims: [u32_of(t, "T")?, u32_of(h, "H")?, 0, 0, 0, 0, 0, 0],
            skips: [rb.skip, sb.skip, gb.skip, ob.skip, 0],
            ..Default::default()
        };
        u32_of(t * h, "T*H")?;
        let groups = MTLSize { width: (t * h).div_ceil(256), height: 1, depth: 1 };
        self.moe_dispatch(COMBINE, &[&rb, &sb, &gb, &ob], &p, groups, 256)
    }

    /// `rows`: the row-exact expert-grouped decode of 2..=`ROWS_MAX` tokens ([`MOE_EXPERTS_BF16_ROWS`]).
    pub(crate) fn launch_moe_experts(&self, call: &OpCall, rows: bool) -> Result<()> {
        let [x, ids, wts, guw, gus, dw, ds, gsc] = call.inputs.as_slice() else {
            return Err(Error::invalid("metal moe_experts: expected 8 inputs"));
        };
        let [out] = call.outputs.as_slice() else {
            return Err(Error::invalid("metal moe_experts: expected 1 output"));
        };
        let (t, h, k) = (x.shape[0], x.shape[1], ids.shape[1]);
        let (e, i) = (guw.shape[0], guw.shape[1] / 2);
        if e > MAX_EXPERTS || h % 16 != 0 || i % 16 != 0 {
            return Err(Error::invalid(format!("metal moe_experts: E = {e} (<= {MAX_EXPERTS}), H = {h}, I = {i} (multiples of 16)")));
        }
        let pairs = t * k;
        if pairs == 0 || h == 0 {
            return Ok(());
        }
        if rows && !(2..=ROWS_MAX).contains(&t) {
            return Err(Error::invalid(format!("metal moe_experts_bf16_rows: T = {t} (2..={ROWS_MAX} tokens)")));
        }
        let o32 = match out.dtype {
            DType::Bf16 => 0,
            DType::F32 => 1,
            d => return Err(Error::invalid(format!("metal moe_experts: output dtype {d}"))),
        };
        // the shaders index weights with 64-bit math, everything else with 32-bit
        u32_of(pairs * h.max(i), "T*K*max(H,I)")?;
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let ib = self.bind(ids, DType::I32, false, "ids")?;
        let wb = self.bind(wts, DType::F32, false, "weights")?;
        let guwb = self.bind(guw, DType::U8, false, "gate_up w")?;
        let gusb = self.bind(gus, DType::Fp8E4m3, false, "gate_up scales")?;
        let dwb = self.bind(dw, DType::U8, false, "down w")?;
        let dsb = self.bind(ds, DType::Fp8E4m3, false, "down scales")?;
        let gb = self.bind(gsc, DType::F32, false, "global scales")?;
        let ob = self.bind(out, out.dtype, true, "out")?;
        // packed_uint2 weight loads need 4-byte alignment (rows are multiples of 8 bytes since H, I % 16 == 0)
        if guwb.skip != 0 || dwb.skip != 0 {
            return Err(Error::invalid("metal moe_experts: the expert weights must be 4-byte aligned"));
        }
        let scratch = call
            .scratch
            .as_ref()
            .ok_or_else(|| Error::invalid("metal moe_experts: the scratch workspace (OpCall::scratch) is missing"))?;
        let r = |v: usize| (v + 255) & !255;
        let (o_off, o_tiles, o_h) = (r(pairs * 4), r(pairs * 4) + r((e + 1) * 4), r(pairs * 4) + 2 * r((e + 1) * 4));
        let o_y = o_h + r(pairs * i * 2);
        let need = o_y + r(pairs * h * 2);
        let have = scratch.numel() * scratch.dtype.size_bytes();
        if have < need {
            return Err(Error::invalid(format!("metal moe_experts: scratch of {have} bytes, need {need}")));
        }
        let sc = self.bind(scratch, scratch.dtype, true, "scratch")?;
        if sc.skip != 0 {
            return Err(Error::invalid("metal moe_experts: the scratch must be 4-byte aligned"));
        }
        let p = OpP {
            dims: [u32_of(t, "T")?, e as u32, u32_of(k, "K")?, u32_of(h, "H")?, u32_of(i, "I")?, 0, 0, 0],
            skips: [xb.skip, gusb.skip, dsb.skip, ob.skip, 0],
            ..Default::default()
        };
        let hb = at(&sc, o_h);
        if rows {
            // no sort pass: a threadgroup row per pair, the first pair of each expert serves all the pairs of that expert (two
            // kernels read each expert's weights once for all of them), moe_gather sums per token in k order (as moe_dec_dn does)
            let yb = at(&sc, o_y);
            let ((gu, rg), (dn, rd)) = rows_shaders();
            let g1 = MTLSize { width: i.div_ceil(DEC_SG * rg), height: pairs, depth: 1 };
            self.moe_dispatch(gu, &[&xb, &ib, &guwb, &gusb, &gb, &hb], &p, g1, DEC_SG * 32)?;
            let g2 = MTLSize { width: h.div_ceil(DEC_SG * rd), height: pairs, depth: 1 };
            self.moe_dispatch(dn, &[&hb, &ib, &dwb, &dsb, &gb, &yb], &p, g2, DEC_SG * 32)?;
            let gg = MTLSize { width: (t * h).div_ceil(256), height: 1, depth: 1 };
            return self.moe_dispatch(GATHER[o32], &[&yb, &ib, &wb, &ob], &p, gg, 256);
        }
        if pairs <= DECODE_MAX_PAIRS {
            let g1 = MTLSize { width: i.div_ceil(DEC_SG), height: pairs, depth: 1 };
            self.moe_dispatch(DEC_GU, &[&xb, &ib, &guwb, &gusb, &gb, &hb], &p, g1, DEC_SG * 32)?;
            let g2 = MTLSize { width: h.div_ceil(DEC_SG), height: t, depth: 1 };
            return self.moe_dispatch(DEC_DN[o32], &[&hb, &ib, &wb, &dwb, &dsb, &gb, &ob], &p, g2, DEC_SG * 32);
        }
        let (pb, offb, tb, yb) = (at(&sc, 0), at(&sc, o_off), at(&sc, o_tiles), at(&sc, o_y));
        let (ti, tm) = tile_rows(pairs, e);
        let mut p = p;
        p.dims[5] = tm as u32;
        let gt = self.pso(GROUP).map_or(256, |s| {
            use objc2_metal::MTLComputePipelineState;
            s.maxTotalThreadsPerThreadgroup().min(1024)
        });
        self.moe_dispatch(GROUP, &[&ib, &pb, &offb, &tb], &p, MTLSize { width: 1, height: 1, depth: 1 }, gt)?;
        // tile slots: at most ceil(P / tm) full tiles plus one partial tile per non-empty expert (the rest exit early)
        let max_tiles = pairs.div_ceil(tm) + pairs.min(e);
        let ggu = MTLSize { width: i.div_ceil(GU_COLS), height: max_tiles, depth: 1 };
        self.moe_dispatch(MM_GU[ti], &[&xb, &pb, &offb, &tb, &guwb, &gusb, &gb, &hb], &p, ggu, 128)?;
        let gdn = MTLSize { width: h.div_ceil(DN_COLS), height: max_tiles, depth: 1 };
        self.moe_dispatch(MM_DN[ti], &[&hb, &pb, &offb, &tb, &dwb, &dsb, &gb, &yb], &p, gdn, 128)?;
        let gg = MTLSize { width: (t * h).div_ceil(256), height: 1, depth: 1 };
        self.moe_dispatch(GATHER[o32], &[&yb, &ib, &wb, &ob], &p, gg, 256)
    }
}

/// Rows per grouped tile for `pairs` (token, slot) pairs over `e` experts, as (index into `MM_GU` / `MM_DN`, rows): the
/// smallest of 8 / 16 / 32 / 64 at or above the mean rows per expert (64 at most).
fn tile_rows(pairs: usize, e: usize) -> (usize, usize) {
    let mean = pairs.div_ceil(e.max(1));
    let ti = (0..3).find(|&t| mean <= 8 << t).unwrap_or(3);
    (ti, 8 << ti)
}

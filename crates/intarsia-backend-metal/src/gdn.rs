//! Launchers of the gated-attention / GatedDeltaNet helper ops (`GateSplit`, `SigmoidMul`, `GdnConv`, `GdnDeltaRule`;
//! kernels/metal/gdn.metal, semantics crates/intarsia-core/src/gdn.rs, hip twin kernels/hip/gdn.hip). Operands are bound
//! as in dense.rs (4-byte aligned offsets, the leftover bf16 element goes to the shader as a `skip`).

use crate::dense::{Grid, OpP};
use crate::kernels::*;
use crate::MetalBackend;
use intarsia_core::{DType, Error, Op, OpCall, Result, ShapeInfo};

/// Tokens per thread of `gdn_conv_bf16_k*` (`GDN_CONV_TILE` in gdn.metal) and its threadgroup width (`GDN_CONV_TG`).
const CONV_TILE: usize = 32;
pub(crate) const CONV_TG: usize = 128;
/// Simdgroups (value-head rows) per threadgroup of the delta-rule kernels (`GDN_DELTA_SG`).
pub(crate) const DELTA_SG: usize = 4;
/// Value rows per threadgroup of `gdn_delta_rule_pre_dk*` (`GDN_PRE_ROWS`: `GDN_DELTA_SG` simdgroups of `GDN_PRE_RPS` rows).
pub(crate) const PRE_ROWS: usize = DELTA_SG * 4;

/// Conv kernels by tap count K = 2..=8.
pub const CONV_SHADERS: [&str; 7] = [
    "gdn_conv_bf16_k2",
    "gdn_conv_bf16_k3",
    "gdn_conv_bf16_k4",
    "gdn_conv_bf16_k5",
    "gdn_conv_bf16_k6",
    "gdn_conv_bf16_k7",
    "gdn_conv_bf16_k8",
];
const CONV_STATE: &str = "gdn_conv_state";
/// `ShortConv` (LFM2) kernels by tap count K = 2..=8: the conv kernels above with the gated read of B * x, the activation off and
/// the C gate on the output (`SHORT` in gdn.metal), and their multi-tile state pass.
pub const SHORT_CONV_SHADERS: [&str; 7] = [
    "short_conv_bf16_k2",
    "short_conv_bf16_k3",
    "short_conv_bf16_k4",
    "short_conv_bf16_k5",
    "short_conv_bf16_k6",
    "short_conv_bf16_k7",
    "short_conv_bf16_k8",
];
const SHORT_CONV_STATE: &str = "short_conv_state";
/// Delta-rule kernels by dk.
pub const DELTA_SHADERS: [(usize, &str); 4] = [
    (32, "gdn_delta_rule_bf16_dk32"),
    (64, "gdn_delta_rule_bf16_dk64"),
    (128, "gdn_delta_rule_bf16_dk128"),
    (256, "gdn_delta_rule_bf16_dk256"),
];

/// Prefill form of the delta rule by dk: (prep pass, recurrence).
pub const DELTA_PRE_SHADERS: [(usize, &str, &str); 4] = [
    (32, "gdn_delta_prep_dk32", "gdn_delta_rule_pre_dk32"),
    (64, "gdn_delta_prep_dk64", "gdn_delta_rule_pre_dk64"),
    (128, "gdn_delta_prep_dk128", "gdn_delta_rule_pre_dk128"),
    (256, "gdn_delta_prep_dk256", "gdn_delta_rule_pre_dk256"),
];

/// Fewest tokens of a call the prep form serves (shorter ones, decode and the MTP verify rows included, run the original).
pub const PRE_MIN_T: usize = 64;

pub fn shader_names() -> Vec<&'static str> {
    let mut v = vec![GATE_SPLIT_BF16, SIGMOID_MUL_BF16, CONV_STATE, SHORT_CONV_STATE];
    v.extend(CONV_SHADERS);
    v.extend(SHORT_CONV_SHADERS);
    v.extend(DELTA_SHADERS.iter().map(|d| d.1));
    v.extend(DELTA_PRE_SHADERS.iter().flat_map(|d| [d.1, d.2]));
    v
}

/// `INTARSIA_METAL_GDN_PRE=0` hides the prep form (`gdn_delta_rule_bf16` serves every T again; same bits).
pub fn pre_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_GDN_PRE").map_or(true, |v| v != "0"))
}

/// The prep form's facts: the original's, T >= [`PRE_MIN_T`], and the scratch indexed with 32-bit math.
pub fn pre_ok(s: &ShapeInfo) -> bool {
    pre_enabled() && delta_ok(s) && s.n % PRE_ROWS == 0 && s.m >= PRE_MIN_T && pre_scratch(s) <= (u32::MAX as usize) * 4
}

/// Scratch floats: normalised q and k `[T, Hk, 2, dk]`, then beta and decay `[T, Hv, 2]`.
fn pre_floats(t: usize, hk: usize, hv: usize, dk: usize) -> (usize, usize) {
    (t * hk * 2 * dk, t * hv * 2)
}

/// Scratch bytes of the prep form at `s` (T tokens), rounded up to 256.
pub fn pre_scratch(s: &ShapeInfo) -> usize {
    let (a, b) = pre_floats(s.m, s.n_kv_heads, s.n_heads, s.head_dim);
    (4 * (a + b)).next_multiple_of(256)
}

/// `GdnConv` and `ShortConv`: 2..=8 taps (`ShapeInfo::k`).
pub fn conv_ok(s: &ShapeInfo) -> bool {
    (2..=CONV_SHADERS.len() + 1).contains(&s.k)
}

/// `GdnDeltaRule`: dk in 32 / 64 / 128 / 256, value heads a multiple of the key heads.
pub fn delta_ok(s: &ShapeInfo) -> bool {
    DELTA_SHADERS.iter().any(|d| d.0 == s.head_dim) && s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0
}

/// Kernel arguments of `gdn_delta_rule_*` (`GdnDeltaP` in gdn.metal; 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DeltaP {
    t: u32,
    hk: u32,
    hv: u32,
    dv: u32,
    /// Leading bf16 elements of qkv, a, b, A_log, dt_bias, o.
    skips: [u32; 6],
    eps: f32,
    pad: u32,
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_gdn(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (i, o) = (&call.inputs, &call.outputs);
        let bf = DType::Bf16;
        match (kernel, call.op) {
            (GATE_SPLIT_BF16, Op::GateSplit) => {
                let d = i[0].shape[2] / 2;
                let n = o[0].numel();
                if n == 0 {
                    return Ok(());
                }
                let (x, q, g) =
                    (self.bind(&i[0], bf, false, "x")?, self.bind(&o[0], bf, true, "q")?, self.bind(&o[1], bf, true, "gate")?);
                let p = OpP {
                    dims: [dim(n, "T*N*D")?, dim(d, "D")?, 0, 0, 0, 0, 0, 0],
                    skips: [x.skip, q.skip, g.skip, 0, 0],
                    ..Default::default()
                };
                dim(2 * n, "T*N*2D")?;
                self.dispatch(kernel, &[&x, &q, &g], &p, Grid::Threads(n))
            }
            (SIGMOID_MUL_BF16, Op::SigmoidMul) => {
                let n = i[0].numel();
                if n == 0 {
                    return Ok(());
                }
                let (a, g, out) =
                    (self.bind(&i[0], bf, false, "a")?, self.bind(&i[1], bf, false, "g")?, self.bind(&o[0], bf, true, "out")?);
                let p = OpP {
                    dims: [dim(n, "elements")?, 0, 0, 0, 0, 0, 0, 0],
                    skips: [a.skip, g.skip, out.skip, 0, 0],
                    ..Default::default()
                };
                self.dispatch(kernel, &[&a, &g, &out], &p, Grid::Threads(n))
            }
            (GDN_CONV_BF16, Op::GdnConv) | (SHORT_CONV_BF16, Op::ShortConv) => {
                // ShortConv reads the [T,3C] projection (B | C | x per row): the channel count is the taps' leading dim
                let short = call.op == Op::ShortConv;
                let (t, c, k) = (i[0].shape[0], i[1].shape[0], i[1].shape[1]);
                let (shaders, state_pass) = if short { (&SHORT_CONV_SHADERS, SHORT_CONV_STATE) } else { (&CONV_SHADERS, CONV_STATE) };
                let Some(&shader) = shaders.get(k.wrapping_sub(2)) else {
                    return Err(Error::invalid(format!("metal {kernel}: {k} taps (2..=8 supported)")));
                };
                if t * c == 0 {
                    return Ok(());
                }
                let (x, w, y) =
                    (self.bind(&i[0], bf, false, "x")?, self.bind(&i[1], bf, false, "taps")?, self.bind(&o[0], bf, true, "y")?);
                let st = self.bind(&o[1], DType::F32, true, "conv state")?;
                dim(t * c * if short { 3 } else { 1 }, "T*C")?;
                dim(c * k, "C*K")?;
                let tiles = t.div_ceil(CONV_TILE);
                // one tile: each thread updates its own channel's state at the end; else a second dispatch, after all
                // tiles have read the old state
                let p = OpP {
                    dims: [dim(t, "T")?, dim(c, "C")?, dim(k, "K")?, (tiles == 1) as u32, 0, 0, 0, 0],
                    skips: [x.skip, w.skip, y.skip, 0, 0],
                    ..Default::default()
                };
                self.dispatch(shader, &[&x, &w, &y, &st], &p, Grid::Groups { x: c.div_ceil(CONV_TG), y: tiles, threads: CONV_TG })?;
                if tiles > 1 {
                    self.dispatch(state_pass, &[&x, &st], &p, Grid::Threads(c))?;
                }
                Ok(())
            }
            (GDN_DELTA_RULE_BF16_PRE, Op::GdnDeltaRule) => self.launch_delta_pre(call),
            (GDN_DELTA_RULE_BF16, Op::GdnDeltaRule) => {
                let d = intarsia_core::gdn::DeltaDims::from_call(call);
                let Some(&(_, shader)) = DELTA_SHADERS.iter().find(|s| s.0 == d.dk) else {
                    return Err(Error::invalid(format!("metal {kernel}: dk = {} (32/64/128/256 supported)", d.dk)));
                };
                if d.t * d.hv * d.dv == 0 {
                    return Ok(());
                }
                if d.hk == 0 || d.hv % d.hk != 0 {
                    return Err(Error::invalid(format!("metal {kernel}: {} value heads, {} key heads", d.hv, d.hk)));
                }
                let (qkv, a, b, al, db) = (
                    self.bind(&i[0], bf, false, "qkv")?,
                    self.bind(&i[1], bf, false, "a")?,
                    self.bind(&i[2], bf, false, "b")?,
                    self.bind(&i[3], bf, false, "A_log")?,
                    self.bind(&i[4], bf, false, "dt_bias")?,
                );
                let (out, st) = (self.bind(&o[0], bf, true, "o")?, self.bind(&o[1], DType::F32, true, "state")?);
                dim(d.t * d.conv_dim(), "T*C")?;
                let p = DeltaP {
                    t: dim(d.t, "T")?,
                    hk: dim(d.hk, "Hk")?,
                    hv: dim(d.hv, "Hv")?,
                    dv: dim(d.dv, "dv")?,
                    skips: [qkv.skip, a.skip, b.skip, al.skip, db.skip, out.skip],
                    eps: call.attrs.eps,
                    pad: 0,
                };
                self.dispatch(
                    shader,
                    &[&qkv, &a, &b, &al, &db, &out, &st],
                    &p,
                    Grid::Groups { x: d.dv.div_ceil(DELTA_SG), y: d.hv, threads: 32 * DELTA_SG },
                )
            }
            _ => Err(Error::NoKernel(format!("metal kernel '{kernel}' for op '{}'", call.op))),
        }
    }

    /// `gdn_delta_rule_bf16_pre`: the prep pass, then the recurrence, both on the plan's scratch workspace.
    fn launch_delta_pre(&self, call: &OpCall) -> Result<()> {
        let (i, o) = (&call.inputs, &call.outputs);
        let bf = DType::Bf16;
        let d = intarsia_core::gdn::DeltaDims::from_call(call);
        let Some(&(_, prep, rec)) = DELTA_PRE_SHADERS.iter().find(|s| s.0 == d.dk) else {
            return Err(Error::invalid(format!("metal gdn_delta_rule_bf16_pre: dk = {} (32/64/128/256 supported)", d.dk)));
        };
        if d.t * d.hv * d.dv == 0 {
            return Ok(());
        }
        if d.hk == 0 || d.hv % d.hk != 0 || d.dv % PRE_ROWS != 0 {
            return Err(Error::invalid(format!(
                "metal gdn_delta_rule_bf16_pre: {} value heads, {} key heads, dv = {} (a multiple of {PRE_ROWS})",
                d.hv, d.hk, d.dv
            )));
        }
        let (qk_floats, gate_floats) = pre_floats(d.t, d.hk, d.hv, d.dk);
        let need = 4 * (qk_floats + gate_floats);
        let scratch = call
            .scratch
            .as_ref()
            .ok_or_else(|| Error::invalid("metal gdn_delta_rule_bf16_pre: the scratch workspace (OpCall::scratch) is missing"))?;
        if scratch.numel() * scratch.dtype.size_bytes() < need {
            return Err(Error::invalid(format!(
                "metal gdn_delta_rule_bf16_pre: scratch of {} bytes, need {need}",
                scratch.numel() * scratch.dtype.size_bytes()
            )));
        }
        let sc = self.bind(scratch, DType::U8, true, "scratch")?;
        if sc.skip != 0 || (self.mbuf(&scratch.buf)?.base + scratch.offset) % 16 != 0 {
            return Err(Error::invalid("metal gdn_delta_rule_bf16_pre: scratch must be 16-byte aligned"));
        }
        let (qkv, a, b, al, db) = (
            self.bind(&i[0], bf, false, "qkv")?,
            self.bind(&i[1], bf, false, "a")?,
            self.bind(&i[2], bf, false, "b")?,
            self.bind(&i[3], bf, false, "A_log")?,
            self.bind(&i[4], bf, false, "dt_bias")?,
        );
        let (out, st) = (self.bind(&o[0], bf, true, "o")?, self.bind(&o[1], DType::F32, true, "state")?);
        dim(d.t * d.conv_dim(), "T*C")?;
        let p = DeltaP {
            t: dim(d.t, "T")?,
            hk: dim(d.hk, "Hk")?,
            hv: dim(d.hv, "Hv")?,
            dv: dim(d.dv, "dv")?,
            skips: [qkv.skip, a.skip, b.skip, al.skip, db.skip, out.skip],
            eps: call.attrs.eps,
            pad: dim(qk_floats, "qk floats")?,
        };
        self.dispatch(prep, &[&qkv, &a, &b, &al, &db, &sc], &p, Grid::Groups { x: d.t, y: d.hk, threads: 32 })?;
        self.dispatch(
            rec,
            &[&qkv, &sc, &out, &st],
            &p,
            Grid::Groups { x: d.dv / PRE_ROWS, y: d.hv, threads: 32 * DELTA_SG },
        )
    }
}

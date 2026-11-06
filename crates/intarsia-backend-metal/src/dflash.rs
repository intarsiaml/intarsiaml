//! DFlash2 draft ops on Metal (lane dm, docs/results/2026-11-06-metal-dflash2.md): `DynConv` (kernels/metal/dflash.metal, the hip
//! twin kernels/hip/dflash.hip) and the sliding-window `AttnDecode` the draft runs past its window (`OpAttrs::window > 0`).
//!
//! The window form runs the existing decode shaders unchanged: the query at position `p` with window `w` sees keys
//! `s ..= p`, `s = max(0, p + 1 - w)`, so the K / V cache operands are bound `s` rows further in (same per-head stride `cap`)
//! and the call runs as a plain decode at position `p - s`. Plain calls (`window == 0`) never reach it (`.ext()`, priority -1).

use crate::dense::{Grid, OpP};
use crate::kernels::*;
use crate::MetalBackend;
use intarsia_core::backend::TensorRef;
use intarsia_core::{DType, Error, Op, OpCall, Result, ShapeInfo};

pub const DYN_CONV_BF16: &str = "dyn_conv_bf16";
/// Registry name of the windowed decode (no shader of its own: `attn_decode_bf16_splitkv` or `attn_decode_bf16(_v8)` run it).
pub const ATTN_DECODE_BF16_WINDOW: &str = "attn_decode_bf16_window";

/// `logits_rows_scalar_m<M>` (kernels/metal/dflash.metal), M = 2..=16: the M-row twin of `gemv_f32o_scalar`.
pub const LRS_SHADERS: [&str; 15] = [
    "logits_rows_scalar_m2",
    "logits_rows_scalar_m3",
    "logits_rows_scalar_m4",
    "logits_rows_scalar_m5",
    "logits_rows_scalar_m6",
    "logits_rows_scalar_m7",
    "logits_rows_scalar_m8",
    "logits_rows_scalar_m9",
    "logits_rows_scalar_m10",
    "logits_rows_scalar_m11",
    "logits_rows_scalar_m12",
    "logits_rows_scalar_m13",
    "logits_rows_scalar_m14",
    "logits_rows_scalar_m15",
    "logits_rows_scalar_m16",
];
/// Output rows per threadgroup of `logits_rows_scalar_m*` (`LRS_SG * LRS_P`) and its threadgroup width.
const LRS_ROWS_PER_TG: usize = 32;
const LRS_THREADS: usize = 256;

pub fn shader_names() -> Vec<&'static str> {
    let mut v = vec![DYN_CONV_BF16];
    v.extend(LRS_SHADERS);
    v
}

/// `INTARSIA_METAL_LOGITS_ROWS=1` (lane dm, default off): a multi-row `Logits` verify call whose operands the 16-byte row kernel
/// cannot take (the 27B lm_head sits at byte 8 mod 16) runs `logits_rows_scalar_m<M>` (one weight pass, bitwise the 1-row
/// `gemv_f32o_scalar` per row) instead of M one-row launches. Read at every call (tests flip it).
pub fn logits_rows_on() -> bool {
    std::env::var("INTARSIA_METAL_LOGITS_ROWS").is_ok_and(|v| v == "1")
}

/// The hip `window_decode_ok`: one query row, even head_dim up to 256, whole GQA groups.
pub fn window_ok(s: &ShapeInfo) -> bool {
    s.m == 1 && s.head_dim > 0 && s.head_dim <= 256 && s.head_dim % 2 == 0 && s.n_kv_heads > 0 && s.n_heads % s.n_kv_heads == 0
}

/// Scratch of the split-KV form where it applies (the launcher falls back to the single-pass kernel otherwise).
pub fn window_scratch(s: &ShapeInfo) -> usize {
    if crate::attn::splitkv_ok(s) {
        crate::attn::splitkv_scratch(s)
    } else {
        0
    }
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_dflash(&self, kernel: &str, call: &OpCall) -> Result<()> {
        match (kernel, call.op) {
            (DYN_CONV_BF16, Op::DynConv) => self.launch_dyn_conv(call),
            (ATTN_DECODE_BF16_WINDOW, Op::AttnDecode) => self.launch_attn_window(call),
            _ => Err(Error::invalid(format!("metal: {kernel} does not run {}", call.op))),
        }
    }

    /// `Logits` of `m` = 2..=16 rows on `logits_rows_scalar_m<m>`: `x [m, k]`, `w [n, k]` bf16 at any 2-byte alignment, `y [m, n]` f32.
    pub(crate) fn launch_logits_rows_scalar(&self, x: &TensorRef, w: &TensorRef, y: &TensorRef) -> Result<()> {
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        let Some(&name) = LRS_SHADERS.get(m.wrapping_sub(2)) else {
            return Err(Error::invalid(format!("metal logits_rows_scalar: {m} rows (2..=16)")));
        };
        let (xb, wb, yb) = (self.bind(x, DType::Bf16, false, "x")?, self.bind(w, DType::Bf16, false, "w")?, self.bind(y, DType::F32, true, "y")?);
        dim(m * k, "M*K")?;
        dim(m * n, "M*N")?;
        let params = crate::GemvParams { k: dim(k, "K")?, n: dim(n, "N")?, w_skip: wb.skip, x_skip: xb.skip, y_skip: yb.skip, pad: [0; 3] };
        self.dispatch(name, &[&wb, &xb, &yb], &params, Grid::Groups { x: n.div_ceil(LRS_ROWS_PER_TG), y: 1, threads: LRS_THREADS })
    }

    fn launch_dyn_conv(&self, call: &OpCall) -> Result<()> {
        let (i, o) = (&call.inputs, &call.outputs);
        let bf = DType::Bf16;
        let (t, h) = (i[0].shape[0], i[0].shape[1]);
        let taps = i[2].shape[0];
        if i[2].shape[1] != h || taps == 0 || taps > 64 || i[1].shape[0] != t || i[1].shape[1] % taps != 0 {
            return Err(Error::invalid(format!(
                "metal {DYN_CONV_BF16}: x {:?}, k {:?}, base {:?}",
                i[0].shape, i[1].shape, i[2].shape
            )));
        }
        let g = i[1].shape[1] / taps;
        if g == 0 || h % g != 0 || o[0].shape != [t, h] {
            return Err(Error::invalid(format!("metal {DYN_CONV_BF16}: {g} groups of H = {h}, out {:?}", o[0].shape)));
        }
        let n = t * h;
        if n == 0 {
            return Ok(());
        }
        let (x, k, base, out) = (
            self.bind(&i[0], bf, false, "x")?,
            self.bind(&i[1], bf, false, "k")?,
            self.bind(&i[2], bf, false, "base")?,
            self.bind(&o[0], bf, true, "out")?,
        );
        dim(t * taps * g, "T*K*G")?;
        let p = OpP {
            dims: [dim(n, "T*H")?, dim(h, "H")?, dim(taps, "K")?, dim(g, "G")?, 0, 0, 0, 0],
            skips: [x.skip, k.skip, base.skip, out.skip, 0],
            ..Default::default()
        };
        self.dispatch(DYN_CONV_BF16, &[&x, &k, &base, &out], &p, Grid::Threads(n))
    }

    fn launch_attn_window(&self, call: &OpCall) -> Result<()> {
        let a = &call.attrs;
        if call.pos.is_some() {
            return Err(Error::invalid(format!("metal {ATTN_DECODE_BF16_WINDOW}: a device-side position is not supported")));
        }
        let (d, cap) = (call.inputs[0].shape[2], call.inputs[1].shape[1]);
        let p = a.pos0.min(cap.saturating_sub(1));
        let start = if a.window > 0 && p + 1 > a.window { p + 1 - a.window } else { 0 };
        let mut c = call.clone();
        for kv in &mut c.inputs[1..3] {
            kv.offset += start * d * kv.dtype.size_bytes();
        }
        c.attrs.pos0 = p - start;
        c.attrs.window = 0;
        let s = ShapeInfo {
            m: 1,
            n_heads: call.inputs[0].shape[1],
            n_kv_heads: call.inputs[1].shape[0],
            head_dim: d,
            ..Default::default()
        };
        if crate::attn::splitkv_ok(&s) && c.scratch.is_some() {
            self.launch_attn_splitkv(&c)
        } else {
            self.launch_dense(ATTN_DECODE_BF16, &c)
        }
    }
}

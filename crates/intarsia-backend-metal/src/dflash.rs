//! DFlash2 draft ops on Metal (lane dm, docs/results/2026-11-06-metal-dflash2.md): `DynConv` (kernels/metal/dflash.metal, the hip
//! twin kernels/hip/dflash.hip) and the sliding-window `AttnDecode` the draft runs past its window (`OpAttrs::window > 0`).
//!
//! The window form runs the existing decode shaders unchanged: the query at position `p` with window `w` sees keys
//! `s ..= p`, `s = max(0, p + 1 - w)`, so the K / V cache operands are bound `s` rows further in (same per-head stride `cap`)
//! and the call runs as a plain decode at position `p - s`. Plain calls (`window == 0`) never reach it (`.ext()`, priority -1).

use crate::dense::{Grid, OpP};
use crate::kernels::*;
use crate::MetalBackend;
use intarsia_core::{DType, Error, Op, OpCall, Result, ShapeInfo};

pub const DYN_CONV_BF16: &str = "dyn_conv_bf16";
/// Registry name of the windowed decode (no shader of its own: `attn_decode_bf16_splitkv` or `attn_decode_bf16(_v8)` run it).
pub const ATTN_DECODE_BF16_WINDOW: &str = "attn_decode_bf16_window";

pub fn shader_names() -> Vec<&'static str> {
    vec![DYN_CONV_BF16]
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

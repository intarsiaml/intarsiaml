//! The DSpine draft's code-bridge ops on Metal (kernels/metal/bridge.metal; semantics `intarsia_core::ops`, the cpu oracle
//! `intarsia-backend-cpu/src/bridge_ref.rs`; docs/plans/2026-11-07-dspine-port.md section 5). bf16 only, one threadgroup per
//! block row (a thread per element for the gate).

use crate::dense::{Grid, OpP};
use crate::MetalBackend;
use intarsia_core::{DType, Error, Op, OpCall, Result};

pub const BRIDGE_NORM_BF16: &str = "bridge_norm_bf16";
pub const BRIDGE_CODE_BF16: &str = "bridge_code_bf16";
pub const BRIDGE_GATE_BF16: &str = "bridge_gate_bf16";
pub const BRIDGE_WRITE_BF16: &str = "bridge_write_bf16";

pub const KERNELS: &[(&str, Op)] = &[
    (BRIDGE_NORM_BF16, Op::BridgeNorm),
    (BRIDGE_CODE_BF16, Op::BridgeCode),
    (BRIDGE_GATE_BF16, Op::BridgeGate),
    (BRIDGE_WRITE_BF16, Op::BridgeWrite),
];

pub fn shader_names() -> Vec<&'static str> {
    KERNELS.iter().map(|k| k.0).collect()
}

/// Threads per threadgroup of the row kernels: a whole number of simdgroups, at most one per 4 elements of the row.
fn row_threads(width: usize) -> usize {
    let t = (width / 4).clamp(32, 256);
    t.next_multiple_of(32).min(256)
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal bridge: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_bridge(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        match (kernel, call.op) {
            (BRIDGE_NORM_BF16, Op::BridgeNorm) => {
                let (t, d) = (i[0].shape[0], i[0].shape[1]);
                if t * d == 0 {
                    return Ok(());
                }
                let (x, y, rms) = (self.bind(&i[0], bf, false, "x")?, self.bind(&o[0], bf, true, "y")?, self.bind(&o[1], DType::F32, true, "rms")?);
                let p = OpP { dims: [dim(d, "D")?, dim(t, "T")?, 0, 0, 0, 0, 0, 0], skips: [x.skip, y.skip, rms.skip, 0, 0], f0: a.eps, ..Default::default() };
                self.dispatch(kernel, &[&x, &y, &rms], &p, Grid::Groups { x: t, y: 1, threads: row_threads(d) })
            }
            (BRIDGE_CODE_BF16, Op::BridgeCode) => {
                let (t, d) = (i[0].shape[0], i[0].shape[1]);
                if t * d == 0 {
                    return Ok(());
                }
                let (x, y) = (self.bind(&i[0], bf, false, "x")?, self.bind(&o[0], bf, true, "y")?);
                let p = OpP { dims: [dim(d, "C")?, dim(t, "T")?, 0, 0, 0, 0, 0, 0], skips: [x.skip, y.skip, 0, 0, 0], f0: a.eps, f1: a.scale, ..Default::default() };
                self.dispatch(kernel, &[&x, &y], &p, Grid::Groups { x: t, y: 1, threads: row_threads(d) })
            }
            (BRIDGE_GATE_BF16, Op::BridgeGate) => {
                let (t, r) = (i[0].shape[0], i[1].shape[0]);
                if t * r == 0 {
                    return Ok(());
                }
                let (pk, bias, s0, msg) =
                    (self.bind(&i[0], bf, false, "packed")?, self.bind(&i[1], bf, false, "bias")?, self.bind(&i[2], bf, false, "s0")?, self.bind(&o[0], bf, true, "msg")?);
                let p = OpP { dims: [dim(r, "R")?, dim(t, "T")?, 0, 0, 0, 0, 0, 0], skips: [pk.skip, bias.skip, s0.skip, msg.skip, 0], ..Default::default() };
                self.dispatch(kernel, &[&pk, &bias, &s0, &msg], &p, Grid::Threads(t * r))
            }
            (BRIDGE_WRITE_BF16, Op::BridgeWrite) => {
                let (t, h) = (i[0].shape[0], i[0].shape[1]);
                if t * h == 0 {
                    return Ok(());
                }
                let (raw, rms, delta, w, out, normed) = (
                    self.bind(&i[0], bf, false, "raw")?,
                    self.bind(&i[1], DType::F32, false, "rms")?,
                    self.bind(&i[2], bf, false, "delta")?,
                    self.bind(&i[3], bf, false, "w")?,
                    self.bind(&o[0], bf, true, "out")?,
                    self.bind(&o[1], bf, true, "normed")?,
                );
                let p = OpP {
                    dims: [dim(h, "H")?, dim(t, "T")?, 0, 0, 0, 0, 0, normed.skip],
                    skips: [raw.skip, rms.skip, delta.skip, w.skip, out.skip],
                    f0: a.eps,
                    ..Default::default()
                };
                self.dispatch(kernel, &[&raw, &rms, &delta, &w, &out, &normed], &p, Grid::Groups { x: t, y: 1, threads: row_threads(h) })
            }
            _ => Err(Error::invalid(format!("metal: {kernel} does not run {}", call.op))),
        }
    }
}

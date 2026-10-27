//! cpu implementations of the gated-delta-net and gated-attention helper ops. They wrap the f64
//! reference math in `intarsia_core::gdn` (the same functions the unit tests check against HF).
//! Both kernel sets (`ref_*` and `cpu_f32_*`) run this code.

use crate::mem::{read_f32, write_f32};
use intarsia_core::gdn::{self, DeltaDims};
use intarsia_core::{Op, OpCall, Result};

fn rd(t: &intarsia_core::TensorRef) -> Result<Vec<f64>> {
    Ok(read_f32(t)?.into_iter().map(|v| v as f64).collect())
}

fn wr(t: &intarsia_core::TensorRef, v: &[f64]) -> Result<()> {
    let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
    write_f32(t, &f)
}

pub fn run(call: &OpCall) -> Result<()> {
    let (i, o) = (&call.inputs, &call.outputs);
    match call.op {
        Op::GateSplit => {
            let (t, n, d2) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let (q, g) = gdn::gate_split(&rd(&i[0])?, t, n, d2 / 2);
            wr(&o[0], &q)?;
            wr(&o[1], &g)
        }
        Op::SigmoidMul => wr(&o[0], &gdn::sigmoid_mul(&rd(&i[0])?, &rd(&i[1])?)),
        Op::GdnConv => {
            let (t, c, k) = (i[0].shape[0], i[0].shape[1], i[1].shape[1]);
            let mut state = rd(&o[1])?;
            let y = gdn::conv_silu(&rd(&i[0])?, &rd(&i[1])?, &mut state, t, c, k);
            wr(&o[0], &y)?;
            wr(&o[1], &state)
        }
        Op::ShortConv => {
            let (t, c, k) = (i[0].shape[0], i[1].shape[0], i[1].shape[1]);
            let mut state = rd(&o[1])?;
            let y = gdn::short_conv(&rd(&i[0])?, &rd(&i[1])?, &mut state, t, c, k);
            wr(&o[0], &y)?;
            wr(&o[1], &state)
        }
        Op::GdnDeltaRule => {
            let d = DeltaDims::from_call(call);
            let mut state = rd(&o[1])?;
            let out = gdn::delta_rule(
                &d,
                call.attrs.eps as f64,
                &rd(&i[0])?,
                &rd(&i[1])?,
                &rd(&i[2])?,
                &rd(&i[3])?,
                &rd(&i[4])?,
                &mut state,
            );
            wr(&o[0], &out)?;
            wr(&o[1], &state)
        }
        other => Err(intarsia_core::Error::invalid(format!("gdn_ref cannot run {other}"))),
    }
}

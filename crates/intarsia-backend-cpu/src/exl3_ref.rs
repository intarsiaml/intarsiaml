//! cpu implementation of `Op::MatmulExl3` (both kernel sets, `ref_*` and `cpu_f32_*`): the f64 math of
//! `intarsia_core::quant::exl3::forward` (rotate the input, multiply by the decoded `W_q`, rotate the output), run in
//! parallel over 128-column output blocks and rounded once to the output dtype.

use crate::mem::{bytes, read_f32, write_f32};
use intarsia_core::quant::exl3::{self, Codebook, HAD};
use intarsia_core::{Error, OpCall, Result};
use rayon::prelude::*;

pub fn run(call: &OpCall) -> Result<()> {
    let (i, o) = (&call.inputs, &call.outputs);
    let (m, k) = (i[0].shape[0], i[0].shape[1]);
    let n = 16 * i[1].shape[1];
    let k2 = exl3::k2_of(&i[1].shape).ok_or_else(|| Error::invalid(format!("matmul_exl3: bad trellis {:?}", i[1].shape)))?;
    let cbv = i32::from_le_bytes(bytes(&i[4])?[..4].try_into().unwrap());
    let cb = Codebook::from_i32(cbv).ok_or_else(|| Error::invalid(format!("matmul_exl3: codebook {cbv} is not 0, 1 or 2")))?;
    if !exl3::valid_k2(k2, cb) {
        return Err(Error::invalid(format!("matmul_exl3: {} bits need the mul1 codebook", k2 as f64 / 2.0)));
    }
    let x = read_f32(&i[0])?;
    let trellis = bytes(&i[1])?;
    let (suh, svh) = (exl3::f16_bytes(bytes(&i[2])?), exl3::f16_bytes(bytes(&i[3])?));
    let xh = exl3::rotate_input(&x, m, k, &suh);
    let blocks: Vec<Vec<f64>> =
        (0..n / HAD).into_par_iter().map(|nb| exl3::forward_block(&xh, m, k, n, trellis, &svh, nb, k2, cb)).collect();
    let mut out = vec![0f32; m * n];
    for (nb, blk) in blocks.iter().enumerate() {
        for mi in 0..m {
            for j in 0..HAD {
                out[mi * n + nb * HAD + j] = blk[mi * HAD + j] as f32;
            }
        }
    }
    write_f32(&o[0], &out)
}

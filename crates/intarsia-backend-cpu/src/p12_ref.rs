//! cpu side of the lossless P12 weights (lane ink-pack12, `intarsia_core::quant::p12`): the reference of every P12 form is
//! "decode to the bf16 weight, then the bf16 op" (both kernel sets: `ref_*` runs the f64 reference, `cpu_f32_*` the fast path).

use crate::mem::{bytes, CpuBuffer};
use intarsia_core::quant::p12;
use intarsia_core::{Buffer, DType, Error, Op, OpCall, Result, TensorRef};

/// The decoded bf16 weight `[n, k]` of a P12 pair (`codes` u8 `[n, 3k/2]`, `aux` u32) as a cpu tensor.
pub(crate) fn decode(codes: &TensorRef, aux: &TensorRef, n: usize, k: usize) -> Result<TensorRef> {
    let w = p12::decode_parts(bytes(codes)?, bytes(aux)?, n, k)?;
    let out = CpuBuffer::zeroed(n * k * 2);
    let b: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.write(0, &b)?;
    Ok(TensorRef::new(Buffer::new(out), DType::Bf16, &[n, k]))
}

/// The bf16 call a P12 call is defined as (`MatmulP12` -> `Matmul` / `Logits`; `InkInProj` with u8 codes -> its float form).
pub(crate) fn as_bf16(call: &OpCall) -> Result<OpCall> {
    let i = &call.inputs;
    let mut c = call.clone();
    match call.op {
        Op::MatmulP12 => {
            let (k, n) = (i[0].shape[1], call.outputs[0].shape[1]);
            let w = decode(&i[1], &i[2], n, k)?;
            c.op = if call.outputs[0].dtype == DType::F32 { Op::Logits } else { Op::Matmul };
            c.inputs = vec![i[0].clone(), w];
        }
        Op::InkInProj if i.len() == 3 && i[1].dtype == DType::U8 => {
            let d = i[0].shape[2];
            let n: usize = call.outputs.iter().map(|o| o.shape[1]).sum();
            c.inputs = vec![i[0].clone(), decode(&i[1], &i[2], n, d)?];
        }
        _ => return Err(Error::invalid(format!("p12: {} is not a P12 call", call.op))),
    }
    c.check()?;
    Ok(c)
}

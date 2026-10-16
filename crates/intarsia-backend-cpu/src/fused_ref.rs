//! cpu implementations of the fused decode ops (`MatmulPair`, `MatmulFp8Pair`, `MatmulNvfp4Glu`, `GatedRmsnorm`,
//! `QkNormRopeKv`).
//! Each is defined as the composition of the ops it replaces, so the oracle of a fused kernel is exactly "the
//! unfused path, including the output rounding of every intermediate tensor". `sub` is the executor of the
//! constituent ops: the f64 reference for `ref_*`, the f32 kernels for `cpu_f32_*`.

use crate::mem::CpuBuffer;
use intarsia_core::{Buffer, Error, Op, OpAttrs, OpCall, Result, TensorRef};

fn tmp(like: &TensorRef, shape: &[usize]) -> TensorRef {
    let bytes = shape.iter().product::<usize>() * like.dtype.size_bytes();
    TensorRef::new(Buffer::new(CpuBuffer::zeroed(bytes)), like.dtype, shape)
}

fn step(sub: fn(&OpCall) -> Result<()>, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: OpAttrs) -> Result<()> {
    let c = OpCall::new(op, inputs, outputs, attrs);
    c.check()?;
    sub(&c)
}

pub fn run(call: &OpCall, sub: fn(&OpCall) -> Result<()>) -> Result<()> {
    let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
    let d = OpAttrs::default;
    match call.op {
        Op::MatmulPair => {
            step(sub, Op::Matmul, vec![i[0].clone(), i[1].clone()], vec![o[0].clone()], d())?;
            step(sub, Op::Matmul, vec![i[0].clone(), i[2].clone()], vec![o[1].clone()], d())
        }
        Op::MatmulFp8Pair => {
            step(sub, Op::MatmulFp8, vec![i[0].clone(), i[1].clone(), i[2].clone()], vec![o[0].clone()], d())?;
            step(sub, Op::MatmulFp8, vec![i[0].clone(), i[3].clone(), i[4].clone()], vec![o[1].clone()], d())
        }
        Op::MatmulNvfp4Glu => {
            let shape = o[0].shape.clone();
            let (g, u) = (tmp(&o[0], &shape), tmp(&o[0], &shape));
            step(sub, Op::MatmulNvfp4, vec![i[0].clone(), i[1].clone(), i[2].clone(), i[3].clone()], vec![g.clone()], d())?;
            step(sub, Op::MatmulNvfp4, vec![i[0].clone(), i[4].clone(), i[5].clone(), i[6].clone()], vec![u.clone()], d())?;
            step(sub, Op::SiluGlu, vec![g, u], vec![o[0].clone()], d())
        }
        Op::GatedRmsnorm => {
            let normed = tmp(&o[0], &o[0].shape);
            step(sub, Op::RmsNorm, vec![i[0].clone(), i[2].clone()], vec![normed.clone()], a.clone())?;
            step(sub, Op::SiluGlu, vec![i[1].clone(), normed], vec![o[0].clone()], d())
        }
        Op::QkNormRopeKv => {
            // intermediates are in the activation dtype (the unfused path stores each op's output in its tensor), the
            // caches get their own dtype from `KvAppend`
            let (qn, kn, kr) = (tmp(&i[0], &i[0].shape), tmp(&i[1], &i[1].shape), tmp(&i[1], &i[1].shape));
            step(sub, Op::QkNorm, vec![i[0].clone(), i[3].clone()], vec![qn.clone()], a.clone())?;
            step(sub, Op::Rope, vec![qn], vec![o[0].clone()], a.clone())?;
            step(sub, Op::QkNorm, vec![i[1].clone(), i[4].clone()], vec![kn.clone()], a.clone())?;
            step(sub, Op::Rope, vec![kn], vec![kr.clone()], a.clone())?;
            step(sub, Op::KvAppend, vec![kr], vec![o[1].clone()], a.clone())?;
            step(sub, Op::KvAppend, vec![i[2].clone()], vec![o[2].clone()], a.clone())
        }
        other => Err(Error::invalid(format!("fused_ref cannot run {other}"))),
    }
}

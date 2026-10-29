//! cpu implementations of the fused decode ops (`MatmulPair`, `MatmulFp8Pair`, `MatmulFp8Triple`, `MatmulMxfp4Pair`, `MatmulNvfp4Glu`, `GatedRmsnorm`,
//! `QkNormRopeKv`, `RmsNormAddMatmulPair`, `GatedRmsnormMatmulFp8`, `RmsNormAddMatmulFp8Pair`, `SigmoidMulMatmulFp8`,
//! `SharedDownCombine`, `GateSplitQkNormRopeKv`, `GdnConvDelta`, `MoeBlock`).
//! Each is defined as the composition of the ops it replaces, so the oracle of a fused kernel is exactly "the
//! unfused path, including the output rounding of every intermediate tensor". `sub` is the executor of the
//! constituent ops: the f64 reference for `ref_*`, the f32 kernels for `cpu_f32_*`.

use crate::mem::CpuBuffer;
use intarsia_core::{Buffer, DType, Error, Op, OpAttrs, OpCall, Result, TensorRef};

fn tmp(like: &TensorRef, shape: &[usize]) -> TensorRef {
    let bytes = shape.iter().product::<usize>() * like.dtype.size_bytes();
    TensorRef::new(Buffer::new(CpuBuffer::zeroed(bytes)), like.dtype, shape)
}

fn tmp_dt(dtype: DType, shape: &[usize]) -> TensorRef {
    let bytes = shape.iter().product::<usize>() * dtype.size_bytes();
    TensorRef::new(Buffer::new(CpuBuffer::zeroed(bytes)), dtype, shape)
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
        Op::MatmulFp8Triple => {
            for j in 0..3 {
                step(sub, Op::MatmulFp8, vec![i[0].clone(), i[1 + 2 * j].clone(), i[2 + 2 * j].clone()], vec![o[j].clone()], d())?;
            }
            Ok(())
        }
        Op::MatmulMxfp4Pair => {
            step(sub, Op::MatmulMxfp4, vec![i[0].clone(), i[1].clone(), i[2].clone(), i[5].clone()], vec![o[0].clone()], d())?;
            step(sub, Op::MatmulMxfp4, vec![i[0].clone(), i[3].clone(), i[4].clone(), i[5].clone()], vec![o[1].clone()], d())
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
        Op::RmsNormAddMatmulPair => {
            // the normed row is stored in the activation dtype between the norm and the GEMVs, like the unfused path
            step(sub, Op::RmsNormAdd, vec![i[0].clone(), i[1].clone(), i[2].clone()], vec![o[0].clone(), o[1].clone()], a.clone())?;
            step(sub, Op::Matmul, vec![o[0].clone(), i[3].clone()], vec![o[2].clone()], d())?;
            step(sub, Op::Matmul, vec![o[0].clone(), i[4].clone()], vec![o[3].clone()], d())
        }
        Op::GatedRmsnormMatmulFp8 => {
            // gated norm over the `R` rows, then the fp8 matmul of the result read as one row `[1, R * D]`
            let k = i[0].shape[0] * i[0].shape[1];
            let normed = tmp(&i[0], &i[0].shape);
            step(sub, Op::RmsNorm, vec![i[0].clone(), i[2].clone()], vec![normed.clone()], a.clone())?;
            let gated = tmp(&i[0], &i[0].shape);
            step(sub, Op::SiluGlu, vec![i[1].clone(), normed], vec![gated.clone()], d())?;
            step(sub, Op::MatmulFp8, vec![gated.reshaped(&[1, k]), i[3].clone(), i[4].clone()], vec![o[0].clone()], d())
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
        Op::RmsNormAddMatmulFp8Pair => {
            // the normed rows are stored in the activation dtype between the norm and the GEMVs, like the unfused path
            step(sub, Op::RmsNormAdd, vec![i[0].clone(), i[1].clone(), i[2].clone()], vec![o[0].clone(), o[1].clone()], a.clone())?;
            step(sub, Op::MatmulFp8, vec![o[0].clone(), i[3].clone(), i[4].clone()], vec![o[2].clone()], d())?;
            step(sub, Op::MatmulFp8, vec![o[0].clone(), i[5].clone(), i[6].clone()], vec![o[3].clone()], d())
        }
        Op::SigmoidMulMatmulFp8 => {
            // the gated product is stored in the activation dtype (the unfused path's `SigmoidMul` output tensor)
            let gated = tmp(&i[0], &i[0].shape);
            step(sub, Op::SigmoidMul, vec![i[0].clone(), i[1].clone()], vec![gated.clone()], d())?;
            step(sub, Op::MatmulFp8, vec![gated, i[2].clone(), i[3].clone()], vec![o[0].clone()], d())
        }
        Op::SharedDownCombine => {
            // the down projection is stored in the output dtype (the unfused path's shared-expert output tensor)
            let down = tmp(&o[0], &o[0].shape);
            step(sub, Op::MatmulNvfp4, vec![i[0].clone(), i[1].clone(), i[2].clone(), i[3].clone()], vec![down.clone()], d())?;
            step(sub, Op::MoeCombine, vec![i[4].clone(), down, i[5].clone()], vec![o[0].clone()], d())
        }
        Op::GateSplitQkNormRopeKv => {
            // the split-off query is stored in the activation dtype (the unfused path's `GateSplit` output), the gate goes
            // straight to its output; then QkNormRopeKv's own composition
            let (t, n, d2) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
            let q = tmp(&i[0], &[t, n, d2 / 2]);
            step(sub, Op::GateSplit, vec![i[0].clone()], vec![q.clone(), o[1].clone()], d())?;
            let prologue = OpCall::new(
                Op::QkNormRopeKv,
                vec![q, i[1].clone(), i[2].clone(), i[3].clone(), i[4].clone()],
                vec![o[0].clone(), o[2].clone(), o[3].clone()],
                a.clone(),
            );
            prologue.check()?;
            run(&prologue, sub)
        }
        Op::GdnConvDelta => {
            // the conv output is stored in the activation dtype (the unfused path's `GdnConv` output tensor) and read back by
            // the delta rule; the conv state is updated in place by the conv, the delta-rule state by the delta rule
            let y = tmp(&i[0], &i[0].shape);
            let (conv, delta) = intarsia_core::gdn::conv_delta_parts(call, &y);
            conv.check()?;
            sub(&conv)?;
            delta.check()?;
            sub(&delta)
        }
        Op::MoeBlock => {
            // exactly the chain `MoeWeights::forward_folded` runs without the op: MoeTopk -> MoeExperts (routed output in the
            // activation dtype) -> shared gate / up NVFP4 matmuls -> SiluGlu -> SharedDownCombine's own composition
            let (t, h, k) = (i[0].shape[0], i[0].shape[1], a.top_k);
            let (is, act) = (i[8].shape[0], o[0].dtype);
            let (tw, ti) = (tmp_dt(DType::F32, &[t, k]), tmp_dt(DType::I32, &[t, k]));
            step(sub, Op::MoeTopk, vec![i[1].clone()], vec![tw.clone(), ti.clone()], a.clone())?;
            let routed = tmp_dt(act, &[t, h]);
            let experts = vec![i[0].clone(), ti, tw, i[3].clone(), i[4].clone(), i[5].clone(), i[6].clone(), i[7].clone()];
            step(sub, Op::MoeExperts, experts, vec![routed.clone()], a.clone())?;
            let (g, u, act_sh) = (tmp_dt(act, &[t, is]), tmp_dt(act, &[t, is]), tmp_dt(act, &[t, is]));
            step(sub, Op::MatmulNvfp4, vec![i[0].clone(), i[8].clone(), i[9].clone(), i[10].clone()], vec![g.clone()], d())?;
            step(sub, Op::MatmulNvfp4, vec![i[0].clone(), i[11].clone(), i[12].clone(), i[13].clone()], vec![u.clone()], d())?;
            step(sub, Op::SiluGlu, vec![g, u], vec![act_sh.clone()], d())?;
            let down = OpCall::new(
                Op::SharedDownCombine,
                vec![act_sh, i[14].clone(), i[15].clone(), i[16].clone(), routed, i[2].clone()],
                vec![o[0].clone()],
                d(),
            );
            down.check()?;
            run(&down, sub)
        }
        other => Err(Error::invalid(format!("fused_ref cannot run {other}"))),
    }
}

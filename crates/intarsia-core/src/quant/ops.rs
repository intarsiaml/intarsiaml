//! Validation and shape facts of the quantised-weight and MoE ops (`Op::MatmulFp8` ... `Op::MoeCombine`) and of the
//! fused decode ops (`Op::MatmulPair`, `MatmulFp8Pair`, `MatmulMxfp4Pair`, `MatmulNvfp4Glu`, `GatedRmsnorm`, `RmsNormAddMatmulPair`,
//! `GatedRmsnormMatmulFp8`, `RmsNormAddMatmulFp8Pair`, `SigmoidMulMatmulFp8`, `SharedDownCombine`, `MoeBlock`).
//! Called from `OpCall::check` / `ShapeInfo::from_call`; the table of signatures is in `ops.rs`.

use crate::backend::TensorRef;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Op, OpCall, ShapeInfo};

fn bad<T>(op: Op, m: String) -> Result<T> {
    Err(Error::invalid(format!("{op}: {m}")))
}

fn act(t: &TensorRef) -> bool {
    matches!(t.dtype, DType::Bf16 | DType::F16 | DType::F32)
}

/// The five routed-expert tensors of `MoeExperts` (inputs 3..8) / `MoeBlock` (inputs 3..8): gate_up w / scales, down w /
/// scales, global scales, against `e` experts, hidden `h` and routed intermediate `inter`.
fn expert_weights(op: Op, w: [&TensorRef; 5], e: usize, h: usize, inter: usize) -> Result<()> {
    let want: [(&TensorRef, DType, Vec<usize>, &str); 5] = [
        (w[0], DType::U8, vec![e, 2 * inter, h / 2], "gate_up w"),
        (w[1], DType::Fp8E4m3, vec![e, 2 * inter, h / 16], "gate_up scales"),
        (w[2], DType::U8, vec![e, h, inter / 2], "down w"),
        (w[3], DType::Fp8E4m3, vec![e, h, inter / 16], "down scales"),
        (w[4], DType::F32, vec![e, 3], "global scales"),
    ];
    for (tn, dt, sh, what) in want {
        if tn.dtype != dt || tn.shape != sh {
            return bad(op, format!("{what} must be {dt} {sh:?}, got {} {:?}", tn.dtype, tn.shape));
        }
    }
    Ok(())
}

/// One NVFP4 projection `[n, k]` (weight u8 `[N,K/2]`, block scales fp8 `[N,K/16]`, global scale f32 `[1]`).
fn nvfp4_parts(op: Op, what: &str, w: &TensorRef, bs: &TensorRef, gs: &TensorRef, n: usize, k: usize) -> Result<()> {
    if w.dtype != DType::U8 || w.shape != [n, k / 2] {
        return bad(op, format!("{what} w must be u8 [N,K/2]=[{n},{}], got {} {:?}", k / 2, w.dtype, w.shape));
    }
    if bs.dtype != DType::Fp8E4m3 || bs.shape != [n, k / 16] {
        return bad(op, format!("{what} block scales must be fp8_e4m3 [N,K/16]=[{n},{}], got {:?}", k / 16, bs.shape));
    }
    if gs.dtype != DType::F32 || gs.shape != [1] {
        return bad(op, format!("{what} global scale must be f32 [1]"));
    }
    Ok(())
}

/// Validate one of the quantised/MoE ops (arity and bounds were already checked by the caller).
pub fn check(call: &OpCall) -> Result<()> {
    let (i, o, a, op) = (&call.inputs, &call.outputs, &call.attrs, call.op);
    let rank = |t: &TensorRef, r: usize, what: &str| -> Result<()> {
        if t.shape.len() != r {
            return bad(op, format!("{what} must be rank {r}, got {:?}", t.shape));
        }
        Ok(())
    };
    match op {
        Op::MatmulMxfp4 => {
            rank(&i[0], 2, "x")?;
            rank(&i[1], 2, "w")?;
            rank(&o[0], 2, "out")?;
            let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
            if !act(&i[0]) || !act(&o[0]) {
                return bad(op, "x and out must be bf16/f16/f32".into());
            }
            if o[0].shape != [m, n] {
                return bad(op, format!("out {:?} != [{m},{n}]", o[0].shape));
            }
            let g = crate::quant::mxfp4::GROUP;
            if k % g != 0 {
                return bad(op, format!("K={k} must be a multiple of {g}"));
            }
            if i[1].dtype != DType::U8 || i[1].shape != [n, k / 2] {
                return bad(op, format!("w must be u8 [N,K/2]=[{n},{}], got {} {:?}", k / 2, i[1].dtype, i[1].shape));
            }
            if i[2].dtype != DType::U8 || i[2].shape != [n, k / g] {
                return bad(op, format!("scales must be u8 (e8m0) [N,K/32]=[{n},{}], got {} {:?}", k / g, i[2].dtype, i[2].shape));
            }
            if i[3].dtype != DType::F32 || i[3].shape != [1] {
                return bad(op, "act must be f32 [1] (0 = 16-bit activations, 1 = dynamic MXFP4)".into());
            }
        }
        Op::MatmulMxfp4Pair => {
            rank(&i[0], 2, "x")?;
            let (m, k) = (i[0].shape[0], i[0].shape[1]);
            if !act(&i[0]) {
                return bad(op, "x must be bf16/f16/f32".into());
            }
            let g = crate::quant::mxfp4::GROUP;
            if k % g != 0 {
                return bad(op, format!("K={k} must be a multiple of {g}"));
            }
            for j in 0..2 {
                let (w, sc, out) = (&i[1 + 2 * j], &i[2 + 2 * j], &o[j]);
                rank(w, 2, "w")?;
                let n = w.shape[0];
                if w.dtype != DType::U8 || w.shape != [n, k / 2] {
                    return bad(op, format!("w{j} must be u8 [N,K/2]=[{n},{}], got {} {:?}", k / 2, w.dtype, w.shape));
                }
                if sc.dtype != DType::U8 || sc.shape != [n, k / g] {
                    return bad(op, format!("scales{j} must be u8 (e8m0) [N,K/32]=[{n},{}], got {} {:?}", k / g, sc.dtype, sc.shape));
                }
                if !act(out) || out.shape != [m, n] {
                    return bad(op, format!("out{j} {:?} != [{m},{n}]", out.shape));
                }
            }
            if i[5].dtype != DType::F32 || i[5].shape != [1] {
                return bad(op, "act must be f32 [1] (0 = 16-bit activations, 1 = dynamic MXFP4)".into());
            }
        }
        Op::MatmulFp8 | Op::MatmulNvfp4 => {
            rank(&i[0], 2, "x")?;
            rank(&i[1], 2, "w")?;
            rank(&o[0], 2, "out")?;
            let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
            if !act(&i[0]) || !act(&o[0]) {
                return bad(op, "x and out must be bf16/f16/f32".into());
            }
            if o[0].shape != [m, n] {
                return bad(op, format!("out {:?} != [{m},{n}]", o[0].shape));
            }
            if op == Op::MatmulFp8 {
                if i[1].dtype != DType::Fp8E4m3 || i[1].shape != [n, k] {
                    return bad(op, format!("w must be fp8_e4m3 [N,K]=[{n},{k}], got {} {:?}", i[1].dtype, i[1].shape));
                }
                if i[2].dtype != DType::F32 || i[2].shape != [2] {
                    return bad(op, "scales must be f32 [2] = (weight_scale, input_scale)".into());
                }
            } else {
                if k % 16 != 0 {
                    return bad(op, format!("K={k} must be a multiple of 16"));
                }
                if i[1].dtype != DType::U8 || i[1].shape != [n, k / 2] {
                    return bad(op, format!("w must be u8 [N,K/2]=[{n},{}], got {} {:?}", k / 2, i[1].dtype, i[1].shape));
                }
                if i[2].dtype != DType::Fp8E4m3 || i[2].shape != [n, k / 16] {
                    return bad(op, format!("block scales must be fp8_e4m3 [N,K/16]=[{n},{}], got {:?}", k / 16, i[2].shape));
                }
                if i[3].dtype != DType::F32 || i[3].shape != [1] {
                    return bad(op, "global scale must be f32 [1]".into());
                }
            }
        }
        Op::MoeTopk => {
            rank(&i[0], 2, "router logits")?;
            let (t, e) = (i[0].shape[0], i[0].shape[1]);
            let k = a.top_k;
            if !act(&i[0]) {
                return bad(op, "logits must be bf16/f16/f32".into());
            }
            if k == 0 || k > e {
                return bad(op, format!("top_k={k} must be in 1..={e}"));
            }
            if o[0].dtype != DType::F32 || o[0].shape != [t, k] || o[1].dtype != DType::I32 || o[1].shape != [t, k] {
                return bad(op, "outputs must be weights f32 [T,K] and ids i32 [T,K]".into());
            }
        }
        Op::MoeExperts => {
            rank(&i[0], 2, "x")?;
            rank(&i[1], 2, "ids")?;
            rank(&i[3], 3, "gate_up w")?;
            rank(&i[5], 3, "down w")?;
            let (t, h) = (i[0].shape[0], i[0].shape[1]);
            let (e, two_i) = (i[3].shape[0], i[3].shape[1]);
            let k = i[1].shape[1];
            if !act(&i[0]) || !act(&o[0]) {
                return bad(op, "x and out must be bf16/f16/f32".into());
            }
            if two_i % 2 != 0 || h % 16 != 0 {
                return bad(op, format!("need even 2I and H % 16 == 0 (2I={two_i}, H={h})"));
            }
            let inter = two_i / 2;
            if inter % 16 != 0 {
                return bad(op, format!("I={inter} must be a multiple of 16"));
            }
            for (tn, dt, sh, what) in [
                (&i[1], DType::I32, vec![t, k], "ids"),
                (&i[2], DType::F32, vec![t, k], "weights"),
            ] {
                if tn.dtype != dt || tn.shape != sh {
                    return bad(op, format!("{what} must be {dt} {sh:?}, got {} {:?}", tn.dtype, tn.shape));
                }
            }
            expert_weights(op, [&i[3], &i[4], &i[5], &i[6], &i[7]], e, h, inter)?;
            if o[0].shape != [t, h] {
                return bad(op, format!("out {:?} != [{t},{h}]", o[0].shape));
            }
            if k == 0 || k > e {
                return bad(op, format!("{k} experts per token with {e} experts"));
            }
        }
        Op::MoeCombine => {
            rank(&i[0], 2, "routed")?;
            let (t, h) = (i[0].shape[0], i[0].shape[1]);
            if i[1].shape != [t, h] || i[2].shape != [t, 1] || o[0].shape != [t, h] {
                return bad(op, format!("shapes routed {:?} shared {:?} gate {:?} out {:?}", i[0].shape, i[1].shape, i[2].shape, o[0].shape));
            }
            if !act(&i[0]) || !act(&i[1]) || !act(&i[2]) || !act(&o[0]) {
                return bad(op, "tensors must be bf16/f16/f32".into());
            }
        }
        Op::MatmulPair => {
            rank(&i[0], 2, "x")?;
            let (m, k) = (i[0].shape[0], i[0].shape[1]);
            if !act(&i[0]) {
                return bad(op, "x must be bf16/f16/f32".into());
            }
            for (j, (w, out)) in [(&i[1], &o[0]), (&i[2], &o[1])].into_iter().enumerate() {
                rank(w, 2, "w")?;
                let n = w.shape[0];
                if !act(w) || w.shape[1] != k {
                    return bad(op, format!("w{j} must be a bf16/f16/f32 [N,{k}] matrix, got {} {:?}", w.dtype, w.shape));
                }
                if !act(out) || out.shape != [m, n] {
                    return bad(op, format!("out{j} {:?} != [{m},{n}]", out.shape));
                }
            }
        }
        Op::MatmulFp8Pair => {
            rank(&i[0], 2, "x")?;
            let (m, k) = (i[0].shape[0], i[0].shape[1]);
            if !act(&i[0]) {
                return bad(op, "x must be bf16/f16/f32".into());
            }
            for j in 0..2 {
                let (w, sc, out) = (&i[1 + 2 * j], &i[2 + 2 * j], &o[j]);
                rank(w, 2, "w")?;
                let n = w.shape[0];
                if w.dtype != DType::Fp8E4m3 || w.shape[1] != k {
                    return bad(op, format!("w{j} must be fp8_e4m3 [N,{k}], got {} {:?}", w.dtype, w.shape));
                }
                if sc.dtype != DType::F32 || sc.shape != [2] {
                    return bad(op, format!("scales{j} must be f32 [2] = (weight_scale, input_scale)"));
                }
                if !act(out) || out.shape != [m, n] {
                    return bad(op, format!("out{j} {:?} != [{m},{n}]", out.shape));
                }
            }
        }
        Op::MatmulNvfp4Glu => {
            rank(&i[0], 2, "x")?;
            let (m, k) = (i[0].shape[0], i[0].shape[1]);
            if !act(&i[0]) || !act(&o[0]) {
                return bad(op, "x and out must be bf16/f16/f32".into());
            }
            if k % 16 != 0 {
                return bad(op, format!("K={k} must be a multiple of 16"));
            }
            rank(&i[1], 2, "gate w")?;
            let n = i[1].shape[0];
            if o[0].shape != [m, n] {
                return bad(op, format!("out {:?} != [{m},{n}]", o[0].shape));
            }
            for (j, what) in ["gate", "up"].into_iter().enumerate() {
                let (w, bs, gs) = (&i[1 + 3 * j], &i[2 + 3 * j], &i[3 + 3 * j]);
                if w.dtype != DType::U8 || w.shape != [n, k / 2] {
                    return bad(op, format!("{what} w must be u8 [N,K/2]=[{n},{}], got {} {:?}", k / 2, w.dtype, w.shape));
                }
                if bs.dtype != DType::Fp8E4m3 || bs.shape != [n, k / 16] {
                    return bad(op, format!("{what} block scales must be fp8_e4m3 [N,K/16]=[{n},{}], got {:?}", k / 16, bs.shape));
                }
                if gs.dtype != DType::F32 || gs.shape != [1] {
                    return bad(op, format!("{what} global scale must be f32 [1]"));
                }
            }
        }
        Op::GatedRmsnorm => {
            rank(&i[0], 2, "x")?;
            let (r, d) = (i[0].shape[0], i[0].shape[1]);
            if i[1].shape != [r, d] || i[2].shape != [d] || o[0].shape != [r, d] {
                return bad(op, format!("shapes x {:?} z {:?} w {:?} out {:?}", i[0].shape, i[1].shape, i[2].shape, o[0].shape));
            }
            if !act(&i[0]) || !act(&i[1]) || !act(&i[2]) || !act(&o[0]) {
                return bad(op, "tensors must be bf16/f16/f32".into());
            }
        }
        Op::RmsNormAddMatmulPair => {
            rank(&i[0], 2, "x")?;
            let (m, k) = (i[0].shape[0], i[0].shape[1]);
            if i[1].shape != [m, k] || i[2].shape != [k] || o[0].shape != [m, k] || o[1].shape != [m, k] {
                return bad(
                    op,
                    format!("shapes x {:?} res {:?} norm w {:?} normed {:?} sum {:?}", i[0].shape, i[1].shape, i[2].shape, o[0].shape, o[1].shape),
                );
            }
            if !act(&i[0]) || !act(&i[1]) || !act(&i[2]) || !act(&o[0]) || !act(&o[1]) {
                return bad(op, "x, res, norm w, normed and sum must be bf16/f16/f32".into());
            }
            for (j, (w, out)) in [(&i[3], &o[2]), (&i[4], &o[3])].into_iter().enumerate() {
                rank(w, 2, "w")?;
                let n = w.shape[0];
                if !act(w) || w.shape[1] != k {
                    return bad(op, format!("w{j} must be a bf16/f16/f32 [N,{k}] matrix, got {} {:?}", w.dtype, w.shape));
                }
                if !act(out) || out.shape != [m, n] {
                    return bad(op, format!("out{j} {:?} != [{m},{n}]", out.shape));
                }
            }
        }
        Op::GatedRmsnormMatmulFp8 => {
            rank(&i[0], 2, "x")?;
            let (r, d) = (i[0].shape[0], i[0].shape[1]);
            let k = r * d;
            if i[1].shape != [r, d] || i[2].shape != [d] {
                return bad(op, format!("shapes x {:?} z {:?} norm w {:?}", i[0].shape, i[1].shape, i[2].shape));
            }
            if !act(&i[0]) || !act(&i[1]) || !act(&i[2]) || !act(&o[0]) {
                return bad(op, "x, z, norm w and out must be bf16/f16/f32".into());
            }
            rank(&i[3], 2, "w")?;
            let n = i[3].shape[0];
            if i[3].dtype != DType::Fp8E4m3 || i[3].shape[1] != k {
                return bad(op, format!("w must be fp8_e4m3 [N,{k}], got {} {:?}", i[3].dtype, i[3].shape));
            }
            if i[4].dtype != DType::F32 || i[4].shape != [2] {
                return bad(op, "scales must be f32 [2] = (weight_scale, input_scale)".into());
            }
            if o[0].shape != [1, n] {
                return bad(op, format!("out {:?} != [1,{n}]", o[0].shape));
            }
        }
        Op::RmsNormAddMatmulFp8Pair => {
            rank(&i[0], 2, "x")?;
            let (m, k) = (i[0].shape[0], i[0].shape[1]);
            if i[1].shape != [m, k] || i[2].shape != [k] || o[0].shape != [m, k] || o[1].shape != [m, k] {
                return bad(
                    op,
                    format!("shapes x {:?} res {:?} norm w {:?} normed {:?} sum {:?}", i[0].shape, i[1].shape, i[2].shape, o[0].shape, o[1].shape),
                );
            }
            if !act(&i[0]) || !act(&i[1]) || !act(&i[2]) || !act(&o[0]) || !act(&o[1]) {
                return bad(op, "x, res, norm w, normed and sum must be bf16/f16/f32".into());
            }
            for j in 0..2 {
                let (w, sc, out) = (&i[3 + 2 * j], &i[4 + 2 * j], &o[2 + j]);
                rank(w, 2, "w")?;
                let n = w.shape[0];
                if w.dtype != DType::Fp8E4m3 || w.shape[1] != k {
                    return bad(op, format!("w{j} must be fp8_e4m3 [N,{k}], got {} {:?}", w.dtype, w.shape));
                }
                if sc.dtype != DType::F32 || sc.shape != [2] {
                    return bad(op, format!("scales{j} must be f32 [2] = (weight_scale, input_scale)"));
                }
                if !act(out) || out.shape != [m, n] {
                    return bad(op, format!("out{j} {:?} != [{m},{n}]", out.shape));
                }
            }
        }
        Op::SigmoidMulMatmulFp8 => {
            rank(&i[0], 2, "a")?;
            rank(&i[2], 2, "w")?;
            rank(&o[0], 2, "out")?;
            let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[2].shape[0]);
            if i[1].shape != [m, k] {
                return bad(op, format!("gate {:?} != a {:?}", i[1].shape, i[0].shape));
            }
            if !act(&i[0]) || !act(&i[1]) || !act(&o[0]) {
                return bad(op, "a, gate and out must be bf16/f16/f32".into());
            }
            if i[2].dtype != DType::Fp8E4m3 || i[2].shape != [n, k] {
                return bad(op, format!("w must be fp8_e4m3 [N,{k}], got {} {:?}", i[2].dtype, i[2].shape));
            }
            if i[3].dtype != DType::F32 || i[3].shape != [2] {
                return bad(op, "scales must be f32 [2] = (weight_scale, input_scale)".into());
            }
            if o[0].shape != [m, n] {
                return bad(op, format!("out {:?} != [{m},{n}]", o[0].shape));
            }
        }
        Op::SharedDownCombine => {
            rank(&i[0], 2, "x")?;
            rank(&i[1], 2, "w")?;
            rank(&o[0], 2, "out")?;
            let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
            if !act(&i[0]) || !act(&i[4]) || !act(&i[5]) || !act(&o[0]) {
                return bad(op, "x, routed, gate logit and out must be bf16/f16/f32".into());
            }
            if k % 16 != 0 {
                return bad(op, format!("K={k} must be a multiple of 16"));
            }
            if i[1].dtype != DType::U8 || i[1].shape != [n, k / 2] {
                return bad(op, format!("w must be u8 [N,K/2]=[{n},{}], got {} {:?}", k / 2, i[1].dtype, i[1].shape));
            }
            if i[2].dtype != DType::Fp8E4m3 || i[2].shape != [n, k / 16] {
                return bad(op, format!("block scales must be fp8_e4m3 [N,K/16]=[{n},{}], got {:?}", k / 16, i[2].shape));
            }
            if i[3].dtype != DType::F32 || i[3].shape != [1] {
                return bad(op, "global scale must be f32 [1]".into());
            }
            if i[4].shape != [m, n] || i[5].shape != [m, 1] || o[0].shape != [m, n] {
                return bad(op, format!("shapes routed {:?} gate {:?} out {:?} for [{m},{n}]", i[4].shape, i[5].shape, o[0].shape));
            }
        }
        Op::MoeBlock => {
            // the whole MoE block after the router logits: MoeTopk(logits) -> MoeExperts(n, ..) -> shared gate / up / down
            // (NVFP4) -> SharedDownCombine-style combine with the shared-gate logit
            rank(&i[0], 2, "n")?;
            rank(&i[1], 2, "router logits")?;
            rank(&i[2], 2, "shared-gate logit")?;
            rank(&i[3], 3, "gate_up w")?;
            rank(&i[5], 3, "down w")?;
            rank(&o[0], 2, "out")?;
            let (t, h) = (i[0].shape[0], i[0].shape[1]);
            let (e, two_i) = (i[3].shape[0], i[3].shape[1]);
            let k = a.top_k;
            if !act(&i[0]) || !act(&i[1]) || !act(&i[2]) || !act(&o[0]) {
                return bad(op, "n, logits, gate logit and out must be bf16/f16/f32".into());
            }
            if i[1].shape != [t, e] {
                return bad(op, format!("logits {:?} != [{t},{e}]", i[1].shape));
            }
            if k == 0 || k > e {
                return bad(op, format!("top_k={k} must be in 1..={e}"));
            }
            if two_i % 2 != 0 || h % 16 != 0 {
                return bad(op, format!("need even 2I and H % 16 == 0 (2I={two_i}, H={h})"));
            }
            let inter = two_i / 2;
            if inter % 16 != 0 {
                return bad(op, format!("I={inter} must be a multiple of 16"));
            }
            expert_weights(op, [&i[3], &i[4], &i[5], &i[6], &i[7]], e, h, inter)?;
            rank(&i[8], 2, "shared gate w")?;
            let is = i[8].shape[0];
            if is % 16 != 0 {
                return bad(op, format!("shared intermediate IS={is} must be a multiple of 16"));
            }
            nvfp4_parts(op, "shared gate", &i[8], &i[9], &i[10], is, h)?;
            nvfp4_parts(op, "shared up", &i[11], &i[12], &i[13], is, h)?;
            nvfp4_parts(op, "shared down", &i[14], &i[15], &i[16], h, is)?;
            if i[2].shape != [t, 1] || o[0].shape != [t, h] {
                return bad(op, format!("shapes gate logit {:?} out {:?} for [{t},{h}]", i[2].shape, o[0].shape));
            }
        }
        _ => unreachable!("quant::ops::check called for {op}"),
    }
    Ok(())
}

/// Fill `s` for one of these ops (see `ShapeInfo` fields; MoE ops use `experts` / `top_k`).
pub fn shape(call: &OpCall, s: &mut ShapeInfo) {
    let i = &call.inputs;
    match call.op {
        Op::MatmulFp8 | Op::MatmulNvfp4 | Op::MatmulMxfp4 => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
            s.n = i[1].shape[0];
            s.weight = Some(i[1].dtype);
        }
        Op::MoeTopk => {
            s.m = i[0].shape[0];
            s.n = i[0].shape[1];
            s.k = s.n;
            s.experts = s.n;
            s.top_k = call.attrs.top_k;
        }
        Op::MoeExperts => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1]; // hidden
            s.n = i[3].shape[1] / 2; // moe intermediate
            s.experts = i[3].shape[0];
            s.top_k = i[1].shape[1];
            s.weight = Some(i[3].dtype);
        }
        Op::MoeCombine => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
        }
        // the fused decode ops: m rows, k reduction (or row width), n the first (or only) output width
        Op::MatmulPair | Op::MatmulFp8Pair | Op::MatmulMxfp4Pair => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
            s.n = i[1].shape[0];
            s.weight = Some(i[1].dtype);
        }
        // the first GEMV's weight gives n and the weight dtype (the norm is over k)
        Op::RmsNormAddMatmulPair => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
            s.n = i[3].shape[0];
            s.weight = Some(i[3].dtype);
        }
        // one output row: the norm is over `d = x.shape[1]` and the matmul over `k = rows * d`
        Op::GatedRmsnormMatmulFp8 => {
            s.m = 1;
            s.k = i[0].shape[0] * i[0].shape[1];
            s.n = i[3].shape[0];
            s.head_dim = i[0].shape[1];
            s.weight = Some(i[3].dtype);
        }
        Op::MatmulNvfp4Glu => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
            s.n = i[1].shape[0];
            s.weight = Some(i[1].dtype);
        }
        Op::GatedRmsnorm => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
        }
        // the norm is over k, the first GEMV's weight gives n and the weight dtype (like RmsNormAddMatmulPair)
        Op::RmsNormAddMatmulFp8Pair => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
            s.n = i[3].shape[0];
            s.weight = Some(i[3].dtype);
        }
        Op::SigmoidMulMatmulFp8 => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
            s.n = i[2].shape[0];
            s.weight = Some(i[2].dtype);
        }
        Op::SharedDownCombine => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
            s.n = i[1].shape[0];
            s.weight = Some(i[1].dtype);
        }
        // m tokens, k hidden, n the routed intermediate I, experts E; head_dim carries the shared intermediate IS (no ShapeInfo
        // field is added for it: a new field could change the plan JSON)
        Op::MoeBlock => {
            s.m = i[0].shape[0];
            s.k = i[0].shape[1];
            s.n = i[3].shape[1] / 2;
            s.experts = i[3].shape[0];
            s.top_k = call.attrs.top_k;
            s.head_dim = i[8].shape[0];
            s.weight = Some(i[3].dtype);
        }
        _ => {}
    }
}

/// Scratch bytes the `MoeExperts` kernels need (kernels/cuda/moe.cu, `moe_scratch_layout`): sorted pair list
/// `[T*K]` i32, expert offsets `[E+1]` i32, active-expert list `[E+1]` i32 (last = count), bf16 activations
/// `[T*K, I]`, bf16 per-pair outputs `[T*K, H]`; each region rounded up to 256 bytes.
pub fn moe_experts_scratch(s: &ShapeInfo) -> usize {
    let r = |x: usize| (x + 255) & !255;
    let pairs = s.m * s.top_k;
    r(pairs * 4) + r((s.experts + 1) * 4) + r((s.experts + 1) * 4) + r(pairs * s.n * 2) + r(pairs * s.k * 2)
}

/// Scratch bytes of the grouped tensor-core experts kernel (kernels/cuda/moe_grouped.cu, `scratch_layout`): pairs
/// `[T*K]` i32, offsets `[E+1]` i32, tile count (256 bytes), tile table `[ceil(P/16) + min(E,P)]` of (expert, row)
/// i32 pairs (sized for the smallest tile), bf16 sorted activations `[T*K, I]`, bf16 per-pair outputs `[T*K, H]`;
/// each region rounded up to 256 bytes.
pub fn moe_experts_mma_scratch(s: &ShapeInfo) -> usize {
    let r = |x: usize| (x + 255) & !255;
    let pairs = s.m * s.top_k;
    let max_tiles = pairs.div_ceil(16) + pairs.min(s.experts);
    r(pairs * 4) + r((s.experts + 1) * 4) + 256 + r(max_tiles * 8) + r(pairs * s.n * 2) + r(pairs * s.k * 2)
}

/// Scratch bytes of the two-launch decode experts kernel (kernels/cuda/moe_decode.cu): bf16 activations
/// `[T*K, I]`, rounded up to 256 bytes.
pub fn moe_experts_decode_scratch(s: &ShapeInfo) -> usize {
    ((s.m * s.top_k * s.n * 2) + 255) & !255
}

/// Scratch bytes of the `MoeBlock` kernel from its exact dims: bf16 routed-expert activations `h` `[T*K, I]` and bf16
/// shared-expert activations `sa` `[T, IS]` (`silu(gate) * up`), each region rounded up to 256 bytes.
pub fn moe_block_scratch_bytes(t: usize, top_k: usize, inter: usize, shared_inter: usize) -> usize {
    let r = |x: usize| (x + 255) & !255;
    r(t * top_k * inter * 2) + r(t * shared_inter * 2)
}

/// [`moe_block_scratch_bytes`] for the `ScratchFn` signature (`KernelImpl::scratch`): `m` tokens, `top_k`, the routed
/// intermediate `I` = `n` and, for this op, the shared intermediate `IS` carried in `head_dim` (see [`shape`]).
pub fn moe_block_scratch(s: &ShapeInfo) -> usize {
    moe_block_scratch_bytes(s.m, s.top_k, s.n, s.head_dim)
}

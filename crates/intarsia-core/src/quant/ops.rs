//! Validation and shape facts of the quantised-weight and MoE ops (`Op::MatmulFp8` ... `Op::MoeCombine`).
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
            let want: [(&TensorRef, DType, Vec<usize>, &str); 7] = [
                (&i[1], DType::I32, vec![t, k], "ids"),
                (&i[2], DType::F32, vec![t, k], "weights"),
                (&i[3], DType::U8, vec![e, two_i, h / 2], "gate_up w"),
                (&i[4], DType::Fp8E4m3, vec![e, two_i, h / 16], "gate_up scales"),
                (&i[5], DType::U8, vec![e, h, inter / 2], "down w"),
                (&i[6], DType::Fp8E4m3, vec![e, h, inter / 16], "down scales"),
                (&i[7], DType::F32, vec![e, 3], "global scales"),
            ];
            for (tn, dt, sh, what) in want {
                if tn.dtype != dt || tn.shape != sh {
                    return bad(op, format!("{what} must be {dt} {sh:?}, got {} {:?}", tn.dtype, tn.shape));
                }
            }
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
        _ => unreachable!("quant::ops::check called for {op}"),
    }
    Ok(())
}

/// Fill `s` for one of these ops (see `ShapeInfo` fields; MoE ops use `experts` / `top_k`).
pub fn shape(call: &OpCall, s: &mut ShapeInfo) {
    let i = &call.inputs;
    match call.op {
        Op::MatmulFp8 | Op::MatmulNvfp4 => {
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

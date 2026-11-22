//! The row-table form of the Inkling-2 stateful ops on cpu (`SconvRes`, `InkKvPrologue`, `RelSinkAttn`, `ConvMix`,
//! `InkAttnDecode` with `OpCall::rows`; contract `intarsia_core::inkling2::ops`, "Row tables"; lane ink-batch).
//!
//! The reference of that form: the call is run as its definition, `rows` one-row (`T = 1`) calls, row `r` on row `r`'s
//! slices (`Row` tensors as views, `Planes` gathered into temporaries and scattered back), row `r`'s state tensors as views
//! of the buffers its table entries name (plus the carrier's offset), at `pos0 = pos[r]`; each sub-call is the ordinary
//! cpu kernel. So row `r` is the solo one-row call on sequence `r` by construction.

use crate::mem::{self, CpuBuffer};
use intarsia_core::backend::Buffer;
use intarsia_core::inkling2::ops::{row_roles, RowRole};
use intarsia_core::{Error, OpCall, Result, TensorRef};

/// Row `r` of a `[N, T, D]` tensor as an owned `[N, 1, D]` copy.
fn gather_planes(t: &TensorRef, r: usize) -> Result<TensorRef> {
    let (n, tt) = (t.shape[0], t.shape[1]);
    let row: usize = t.shape[2..].iter().product::<usize>() * t.dtype.size_bytes();
    let src = mem::bytes(t)?;
    let mut out = vec![0u8; n * row];
    for p in 0..n {
        let o = (p * tt + r) * row;
        out[p * row..(p + 1) * row].copy_from_slice(&src[o..o + row]);
    }
    let b = CpuBuffer::zeroed(out.len());
    b.write(0, &out)?;
    let mut shape = t.shape.clone();
    shape[1] = 1;
    Ok(TensorRef { buf: Buffer::new(b), offset: 0, dtype: t.dtype, shape })
}

/// Write the `[N, 1, D]` `part` back as row `r` of the `[N, T, D]` tensor `t`.
fn scatter_planes(t: &TensorRef, part: &TensorRef, r: usize) -> Result<()> {
    let (n, tt) = (t.shape[0], t.shape[1]);
    let row: usize = t.shape[2..].iter().product::<usize>() * t.dtype.size_bytes();
    let src = mem::bytes(part)?.to_vec();
    let dst = mem::cpu_buf(t)?;
    for p in 0..n {
        dst.write(t.offset + (p * tt + r) * row, &src[p * row..(p + 1) * row])?;
    }
    Ok(())
}

/// Run a row-table call as its `rows` one-row calls, each by `one`.
pub fn run(call: &OpCall, one: &dyn Fn(&OpCall) -> Result<()>) -> Result<()> {
    let rt = call.rows.as_ref().ok_or_else(|| Error::invalid("cpu row-table run without a row table"))?;
    let (ri, ro) = row_roles(call.op).ok_or_else(|| Error::invalid(format!("{}: no row-table form", call.op)))?;
    let pos = call.pos.as_ref().ok_or_else(|| Error::invalid(format!("{}: a row-table call needs `pos`", call.op)))?;
    let pos: Vec<i32> = mem::bytes(pos)?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let tab = mem::bytes(&rt.ptrs)?.to_vec();
    let entry = |r: usize, j: usize| -> u64 {
        let o = (r * rt.stride + rt.index + j) * 8;
        u64::from_le_bytes(tab[o..o + 8].try_into().unwrap())
    };
    for r in 0..rt.rows {
        if pos[r] < 0 {
            return Err(Error::invalid(format!("{}: negative position {} of row {r}", call.op, pos[r])));
        }
        let view = |t: &TensorRef, role: RowRole| -> Result<TensorRef> {
            if t.numel() == 0 {
                return Ok(t.clone());
            }
            Ok(match role {
                RowRole::Fixed => t.clone(),
                RowRole::Row => t.rows(r, 1),
                RowRole::Planes => gather_planes(t, r)?,
                RowRole::State(j) => {
                    let a = entry(r, j);
                    if a == 0 {
                        return Err(Error::invalid(format!("{}: row {r} has no state address (entry {})", call.op, rt.index + j)));
                    }
                    // SAFETY: the entry is the `address` of a live owned buffer of the row's state, laid out like the carrier's
                    let b = unsafe { CpuBuffer::raw(a as usize, t.buf.bytes()) };
                    TensorRef { buf: Buffer::new(b), offset: t.offset, dtype: t.dtype, shape: t.shape.clone() }
                }
            })
        };
        let mut sub = call.clone();
        sub.rows = None;
        sub.pos = None;
        sub.attrs.pos0 = pos[r] as usize;
        for (k, t) in call.inputs.iter().enumerate() {
            sub.inputs[k] = view(t, ri[k])?;
        }
        for (k, t) in call.outputs.iter().enumerate() {
            sub.outputs[k] = view(t, ro[k])?;
        }
        sub.check()?;
        one(&sub)?;
        for (k, t) in call.outputs.iter().enumerate() {
            if ro[k] == RowRole::Planes && t.numel() > 0 {
                scatter_planes(t, &sub.outputs[k], r)?;
            }
        }
    }
    Ok(())
}

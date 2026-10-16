//! Launcher of the GQA-grouped split-KV decode attention (`attn_decode_bf16_splitkv`, kernels/metal/attention_splitkv.metal;
//! design notes in the shader header). Stage 1 runs `nkv x nsplit` threadgroups, each reading its slice of one kv head's
//! keys / values once for all query heads that share it; with more than one split a combine pass (one threadgroup per
//! query head) merges the partials, which live in the plan-reserved scratch workspace (`KernelImpl::scratch`,
//! `OpCall::scratch`; nothing is allocated here). Both stages are encoded as one launch.
//!
//! The split count comes from the number of visible keys only (never from the data), so a call is run-to-run identical.
//! `decode_splits` is the host copy of the shader's `ask_nsplit`; the shader recomputes it from the clamped position, so
//! a host `pos0` and a device-side position (`OpCall::pos`) use the same split count (they may differ in simdgroups per
//! threadgroup, so their results agree up to rounding order, not bit for bit).

use crate::dense::OpP;
use crate::kernels::*;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use std::sync::OnceLock;

/// Most splits a launch uses (the scratch bound `splitkv_scratch` and the shader's grid are sized by it).
pub const MAX_SPLIT: usize = 64;
/// Split policy defaults: a split holds at least `MIN_KEYS` keys, and splitting stops once `nkv x nsplit` reaches
/// `TARGET_TGS` threadgroups (docs/results/2026-10-14-metal-phase-b.md, "Phase B part 2a"): 120 = 3 threadgroups of 4 simdgroups per core of a 40-core GPU, more than a wave costs 20 percent.
const MIN_KEYS: usize = 128;
const TARGET_TGS: usize = 120;
/// Largest query-head group per kv head the kernels take (`GP` of the shader).
const MAX_GROUP: usize = 8;
/// Threadgroup accumulator floats of the shader (`ASK_TG_FLOATS`): `nsg * GP * D` must fit.
const TG_FLOATS: usize = 4096;
const MAX_SG: usize = 8;
/// At most this many threadgroups in the grid: use `MAX_SG` simdgroups per threadgroup, else `MAX_SG / 2`.
const FEW_TGS: usize = 64;

/// `(min_keys, target_tgs)`; `INTARSIA_METAL_SPLITKV=<min_keys>,<target>` overrides them (diagnostic for sweeps, it
/// changes the rounding order of the result like any other split count).
fn policy() -> (usize, usize) {
    static P: OnceLock<(usize, usize)> = OnceLock::new();
    *P.get_or_init(|| {
        let parsed = std::env::var("INTARSIA_METAL_SPLITKV").ok().and_then(|v| {
            let (a, b) = v.split_once(',')?;
            Some((a.trim().parse::<usize>().ok()?.max(1), b.trim().parse::<usize>().ok()?.max(1)))
        });
        parsed.unwrap_or((MIN_KEYS, TARGET_TGS))
    })
}

/// Number of KV splits for `visible` keys over `nkv` kv heads, at most `max_split` (shader: `ask_nsplit`).
pub fn decode_splits(visible: usize, nkv: usize, max_split: usize) -> usize {
    let (min_keys, target) = policy();
    visible.div_ceil(min_keys).min(target.div_ceil(nkv.max(1))).clamp(1, max_split.max(1))
}

/// Shapes the split-KV kernels take: one query row, head_dim a power of two in 8..=256, query heads a multiple of the
/// kv heads with a group of at most 8. (K / V cache alignment is not part of the shape: the launcher falls back.)
pub fn splitkv_ok(s: &ShapeInfo) -> bool {
    s.m == 1
        && matches!(s.head_dim, 8 | 16 | 32 | 64 | 128 | 256)
        && s.n_kv_heads > 0
        && s.n_heads % s.n_kv_heads == 0
        && s.n_heads / s.n_kv_heads <= MAX_GROUP
}

/// Scratch the kernels need: float part[N][nsplit][D] + float2 ml[N][nsplit]. An upper bound that does not depend on the
/// context length (plans probe at short contexts; a long one must never need more).
pub fn splitkv_scratch(s: &ShapeInfo) -> usize {
    s.n_heads * MAX_SPLIT * (s.head_dim + 2) * 4
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    /// Launch `attn_decode_bf16_splitkv`. Operands the kernels cannot take (q or K / V caches not 16-byte aligned, or a shape
    /// `splitkv_ok` rejects) run the single-pass `attn_decode_bf16` instead, which needs no scratch.
    pub(crate) fn launch_attn_splitkv(&self, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let (nkv, cap) = (i[1].shape[0], i[1].shape[1]);
        if t * n * d == 0 {
            return Ok(());
        }
        let shape_ok = t == 1
            && nkv > 0
            && n % nkv == 0
            && n / nkv <= MAX_GROUP
            && matches!(d, 8 | 16 | 32 | 64 | 128 | 256);
        let (ka, va) = (self.mbuf(&i[1].buf)?.base + i[1].offset, self.mbuf(&i[2].buf)?.base + i[2].offset);
        let qa = self.mbuf(&i[0].buf)?.base + i[0].offset;
        if !shape_ok || ka % 16 != 0 || va % 16 != 0 || qa % 16 != 0 {
            return self.launch_dense(ATTN_DECODE_BF16, call);
        }

        let (q, kc, vc, out) = (
            self.bind(&i[0], bf, false, "q")?,
            self.bind(&i[1], bf, false, "k cache")?,
            self.bind(&i[2], bf, false, "v cache")?,
            self.bind(&o[0], bf, true, "out")?,
        );
        let (pos, use_dev) = self.pos_bind(call)?;
        let grp = n / nkv;
        // host position: the exact split count of this call; device position: the bound for a full cache (the shader
        // derives the real count from the position it reads and idles the surplus threadgroups)
        let max_split = if use_dev == 1 {
            decode_splits(cap, nkv, MAX_SPLIT)
        } else {
            decode_splits((a.pos0 + 1).min(cap), nkv, MAX_SPLIT)
        };
        let need = n * max_split * (d + 2) * 4;
        let scratch = call.scratch.as_ref().ok_or_else(|| {
            Error::invalid("metal attn_decode_bf16_splitkv: the scratch workspace (OpCall::scratch) is missing")
        })?;
        if scratch.numel() * scratch.dtype.size_bytes() < need {
            return Err(Error::invalid(format!(
                "metal attn_decode_bf16_splitkv: scratch of {} bytes, need {need}",
                scratch.numel() * scratch.dtype.size_bytes()
            )));
        }
        let sc = self.bind(scratch, DType::U8, true, "scratch")?;
        if sc.skip != 0 || (self.mbuf(&scratch.buf)?.base + scratch.offset) % 16 != 0 {
            return Err(Error::invalid("metal attn_decode_bf16_splitkv: scratch must be 16-byte aligned"));
        }

        let (min_keys, target) = policy();
        let p = OpP {
            dims: [
                u32_of(target, "target")?,
                u32_of(n, "N")?,
                u32_of(nkv, "Nkv")?,
                u32_of(d, "D")?,
                u32_of(cap, "cap")?,
                u32_of(a.pos0, "pos0")?,
                use_dev,
                u32_of(max_split, "max_split")?,
            ],
            skips: [0, 0, 0, out.skip, 0],
            f0: a.scale,
            f1: 0.0,
            pad: u32_of(min_keys, "min_keys")?,
        };
        u32_of(nkv * cap * d, "Nkv*cap*D")?;
        u32_of(n * max_split * (d + 2), "N*max_split*(D+2)")?; // the shader indexes the scratch with 32-bit math
        let gp = grp.next_power_of_two();
        // few threadgroups (short context): 8 simdgroups each put more loads in flight; a full wave of them: 4, which
        // lets more threadgroups share a core (b2a sweeps: 4k 47 us vs 60 us, 1k 21 us vs 23 us)
        let nsg = if use_dev == 0 && nkv * max_split <= FEW_TGS { MAX_SG } else { MAX_SG / 2 };
        let nsg = nsg.min((TG_FLOATS / (gp * d)).max(1));
        let (name1, name2) = (ATTN_DECODE_SPLITKV_SHADERS[gp.trailing_zeros() as usize], ATTN_DECODE_SPLITKV_SHADERS[4]);
        let pso1 = self.pso(name1).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name1}'")))?;
        let pso2 = self.pso(name2).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name2}'")))?;
        // with one split stage 1 writes the output itself; a device position decides that on the GPU
        let combine = use_dev == 1 || max_split > 1;
        let pos_b = pos.as_ref().unwrap_or(&q);
        // one `with_encoder` for both stages: one launch is one profiled record and one dispatch toward the commit limit
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`; the scratch
            // size above).
            unsafe {
                enc.setComputePipelineState(pso1);
                for (idx, b) in [&q, &kc, &vc, &out, pos_b, &sc].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, idx);
                }
            }
            set_bytes(enc, &p, 6);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: nkv, height: max_split, depth: 1 },
                MTLSize { width: nsg * 32, height: 1, depth: 1 },
            );
            if combine {
                // SAFETY: as above.
                unsafe {
                    enc.setComputePipelineState(pso2);
                    for (idx, b) in [&out, pos_b, &sc].into_iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, idx);
                    }
                }
                set_bytes(enc, &p, 3);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n, height: 1, depth: 1 },
                    MTLSize { width: d, height: 1, depth: 1 },
                );
            }
        })
    }
}

/// Most verification rows `launch_attn_rows` takes (G0b).
pub const ROWS_MAX: usize = 8;

/// Shapes `attn_prefill_bf16_splitkv_rows` takes: 2..=8 query rows, the split-KV decode shapes otherwise.
pub fn rows_ok(s: &ShapeInfo) -> bool {
    (2..=ROWS_MAX).contains(&s.m) && splitkv_ok(&ShapeInfo { m: 1, ..s.clone() })
}

/// Scratch of the multi-row kernels: one `splitkv_scratch` slab per row.
pub fn rows_scratch(s: &ShapeInfo) -> usize {
    splitkv_scratch(s) * s.m.clamp(1, ROWS_MAX)
}

/// Simdgroups per threadgroup of the 1-row launch (`launch_attn_splitkv`) that holds `nsplit` splits: the multi-row
/// kernel uses the same rule per row (`askr_nsg`), the host picks the maximum over the rows (row 0, the rule is
/// non-increasing in `nsplit`).
fn rows_nsg(host_pos: bool, nkv: usize, nsplit: usize, gp: usize, d: usize) -> usize {
    let nsg = if host_pos && nkv * nsplit <= FEW_TGS { MAX_SG } else { MAX_SG / 2 };
    nsg.min((TG_FLOATS / (gp * d)).max(1))
}

impl MetalBackend {
    /// `attn_prefill_bf16_splitkv_rows`: `AttnPrefill` with `T = 2..=8` query rows (row `r` at position `pos0 + r`, causal),
    /// bitwise the 1-row `launch_attn_splitkv` call at position `pos0 + r` on the same cache, per row (G0b,
    /// kernels/metal/attention_splitkv.metal). Operands the split-KV kernels cannot take run `T` single-pass 1-row launches
    /// (what the 1-row launcher does for them).
    #[doc(hidden)]
    pub fn launch_attn_rows(&self, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let (nkv, cap) = (i[1].shape[0], i[1].shape[1]);
        if t * n * d == 0 {
            return Ok(());
        }
        let shape_ok = (2..=ROWS_MAX).contains(&t)
            && nkv > 0
            && n % nkv == 0
            && n / nkv <= MAX_GROUP
            && matches!(d, 8 | 16 | 32 | 64 | 128 | 256);
        let (ka, va) = (self.mbuf(&i[1].buf)?.base + i[1].offset, self.mbuf(&i[2].buf)?.base + i[2].offset);
        let qa = self.mbuf(&i[0].buf)?.base + i[0].offset;
        if !shape_ok || ka % 16 != 0 || va % 16 != 0 || qa % 16 != 0 {
            for r in 0..t {
                let mut at = a.clone();
                at.pos0 = a.pos0 + r;
                let mut c = OpCall::new(
                    intarsia_core::Op::AttnDecode,
                    vec![i[0].rows(r, 1), i[1].clone(), i[2].clone()],
                    vec![o[0].rows(r, 1)],
                    at,
                );
                c.pos = None;
                self.launch_dense(ATTN_DECODE_BF16, &c)?;
            }
            return Ok(());
        }
        let (q, kc, vc, out) = (
            self.bind(&i[0], bf, false, "q")?,
            self.bind(&i[1], bf, false, "k cache")?,
            self.bind(&i[2], bf, false, "v cache")?,
            self.bind(&o[0], bf, true, "out")?,
        );
        let (pos, use_dev) = self.pos_bind(call)?;
        let grp = n / nkv;
        let last = if use_dev == 1 { cap } else { (a.pos0 + t).min(cap) };
        let max_split = decode_splits(last, nkv, MAX_SPLIT);
        let need = t * n * max_split * (d + 2) * 4;
        let scratch = call.scratch.as_ref().ok_or_else(|| {
            Error::invalid("metal attn_prefill_bf16_splitkv_rows: the scratch workspace (OpCall::scratch) is missing")
        })?;
        if scratch.numel() * scratch.dtype.size_bytes() < need {
            return Err(Error::invalid(format!(
                "metal attn_prefill_bf16_splitkv_rows: scratch of {} bytes, need {need}",
                scratch.numel() * scratch.dtype.size_bytes()
            )));
        }
        let sc = self.bind(scratch, DType::U8, true, "scratch")?;
        if sc.skip != 0 || (self.mbuf(&scratch.buf)?.base + scratch.offset) % 16 != 0 {
            return Err(Error::invalid("metal attn_prefill_bf16_splitkv_rows: scratch must be 16-byte aligned"));
        }
        let (min_keys, target) = policy();
        let p = OpP {
            dims: [
                u32_of(target, "target")?,
                u32_of(n, "N")?,
                u32_of(nkv, "Nkv")?,
                u32_of(d, "D")?,
                u32_of(cap, "cap")?,
                u32_of(a.pos0, "pos0")?,
                use_dev,
                u32_of(max_split, "max_split")?,
            ],
            skips: [u32_of(t, "rows")?, 0, 0, out.skip, 0],
            f0: a.scale,
            f1: 0.0,
            pad: u32_of(min_keys, "min_keys")?,
        };
        u32_of(nkv * cap * d, "Nkv*cap*D")?;
        u32_of(t * n * max_split * (d + 2), "R*N*max_split*(D+2)")?;
        let gp = grp.next_power_of_two();
        // threadgroup shape: the largest simdgroup count any row's 1-row launch uses (row 0: the fewest splits)
        let first_splits = decode_splits(if use_dev == 1 { cap } else { (a.pos0 + 1).min(cap) }, nkv, MAX_SPLIT);
        let nsg = rows_nsg(use_dev == 0, nkv, if use_dev == 1 { usize::MAX / 2 } else { first_splits }, gp, d);
        let (name1, name2) = (ATTN_ROWS_SHADERS[gp.trailing_zeros() as usize], ATTN_ROWS_SHADERS[4]);
        let pso1 = self.pso(name1).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name1}'")))?;
        let pso2 = self.pso(name2).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name2}'")))?;
        let combine = use_dev == 1 || max_split > 1;
        let pos_b = pos.as_ref().unwrap_or(&q);
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`; the scratch
            // size above).
            unsafe {
                enc.setComputePipelineState(pso1);
                for (idx, b) in [&q, &kc, &vc, &out, pos_b, &sc].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, idx);
                }
            }
            set_bytes(enc, &p, 6);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: nkv * t, height: max_split, depth: 1 },
                MTLSize { width: nsg * 32, height: 1, depth: 1 },
            );
            if combine {
                // SAFETY: as above.
                unsafe {
                    enc.setComputePipelineState(pso2);
                    for (idx, b) in [&out, pos_b, &sc].into_iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, idx);
                    }
                }
                set_bytes(enc, &p, 3);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n, height: t, depth: 1 },
                    MTLSize { width: d, height: 1, depth: 1 },
                );
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_counts_are_bounded_and_monotone() {
        for nkv in [1, 2, 8, 40] {
            let mut prev = 1;
            for v in (1..40000).step_by(7) {
                let s = decode_splits(v, nkv, MAX_SPLIT);
                assert!((1..=MAX_SPLIT).contains(&s));
                assert!(s >= prev, "nkv {nkv} visible {v}");
                prev = s;
            }
        }
        assert_eq!(decode_splits(1, 8, MAX_SPLIT), 1);
    }

    #[test]
    fn scratch_bounds_every_context_length() {
        for (n, nkv, d) in [(32, 8, 128), (16, 2, 256), (6, 2, 32), (8, 8, 64), (8, 1, 128)] {
            let s = ShapeInfo { m: 1, n_heads: n, n_kv_heads: nkv, head_dim: d, ..Default::default() };
            let bound = splitkv_scratch(&s);
            for v in [1, 17, 128, 1000, 4096, 16384, 262144, usize::MAX / 2] {
                assert!(n * decode_splits(v, nkv, MAX_SPLIT) * (d + 2) * 4 <= bound);
            }
        }
    }

    #[test]
    fn threadgroup_accumulators_fit() {
        for gp in [1, 2, 4, 8] {
            for d in [8, 16, 32, 64, 128, 256] {
                assert!((TG_FLOATS / (gp * d)).clamp(1, MAX_SG) * gp * d <= TG_FLOATS, "gp {gp} d {d}");
            }
        }
    }
}

//! Launcher of the flash-style causal prefill attention (`attn_prefill_bf16_flash`, kernels/metal/attention_flash.metal;
//! the design notes are in the shader header). One threadgroup per (block of query positions, kv head): the K / V tile it
//! stages is shared by all query heads of the group, and the score / output products run on `simdgroup_matrix`
//! fragments with float accumulation.
//!
//! The kernel serves every call of the op for head_dim 64 / 128 / 256 and a group size (query heads per kv head) of 1, 2,
//! 4 or 8, whatever T and pos0 are, so that how a prompt is cut into chunks never changes which kernel a row goes through.
//! Operands it cannot take (a leading element skip on q / out, K / V caches not 16-byte aligned) run the
//! `attn_prefill_bf16` kernel instead; the fallbacks are counted (`fallbacks`) and the first one is reported on stderr.

use crate::dense::OpP;
use crate::kernels::*;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// One instantiation of `attn_flash<D, BK, NSG>`.
#[derive(Clone, Copy, Debug)]
pub struct FlashVariant {
    pub name: &'static str,
    pub head_dim: usize,
    /// Simdgroups per threadgroup (the group size must divide it): one query head of the group each, times
    /// `nsg / group` position slabs of 8 rows.
    pub nsg: usize,
}

macro_rules! fv {
    ($n:literal, $d:literal, $s:literal) => {
        FlashVariant { name: $n, head_dim: $d, nsg: $s }
    };
}

/// Every instantiation compiled into the metallib. The first with a fitting `nsg` is the default of its head_dim: four
/// simdgroups (groups 1, 2, 4; 8 rows of one position slab per query head), eight for group 8.
pub const FLASH_VARIANTS: &[FlashVariant] = &[
    fv!("attn_flash_d128_s4", 128, 4),
    fv!("attn_flash_d128_s8", 128, 8),
    fv!("attn_flash_d64_s4", 64, 4),
    fv!("attn_flash_d64_s8", 64, 8),
    fv!("attn_flash_d256_s4", 256, 4),
    fv!("attn_flash_d256_s8", 256, 8),
];

/// Group sizes (query heads per kv head) the kernel takes: they must divide the simdgroups of a threadgroup.
const GROUPS: [usize; 4] = [1, 2, 4, 8];

static FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Launches that ran `attn_prefill_bf16` instead of the flash kernel (misaligned operands).
pub fn fallbacks() -> u64 {
    FALLBACKS.load(Ordering::Relaxed)
}

/// Shapes the flash kernel takes: head_dim 64, 128 or 256, query heads a multiple of the kv heads with a group in 1, 2, 4, 8.
pub fn flash_ok(s: &ShapeInfo) -> bool {
    s.m >= 1
        && matches!(s.head_dim, 64 | 128 | 256)
        && s.n_kv_heads > 0
        && s.n_heads % s.n_kv_heads == 0
        && GROUPS.contains(&(s.n_heads / s.n_kv_heads))
}

const UNSET: usize = usize::MAX;
const NONE: usize = usize::MAX - 1;
/// Index into `FLASH_VARIANTS` forced by `INTARSIA_METAL_FLASH=<name>` or [`select_variant`]; `UNSET` until the first
/// launch reads the environment, `NONE` for "the default of each head_dim".
static FORCED: AtomicUsize = AtomicUsize::new(UNSET);

/// Force the variant `name` (or the default again with `None`) for later launches: diagnostic for sweeps and tests, a
/// variant changes the rounding order of the result. A forced variant whose `nsg` the group size does not divide, or of
/// another head_dim, is ignored.
pub fn select_variant(name: Option<&str>) {
    let ix = name.map_or(NONE, |n| FLASH_VARIANTS.iter().position(|v| v.name == n).expect("unknown flash variant"));
    FORCED.store(ix, Ordering::Relaxed);
}

/// The variant a launch with head_dim `d` and `g` query heads per kv head uses.
fn pick(d: usize, g: usize) -> &'static FlashVariant {
    let mut ix = FORCED.load(Ordering::Relaxed);
    if ix == UNSET {
        let env = std::env::var("INTARSIA_METAL_FLASH").ok();
        ix = env.and_then(|n| FLASH_VARIANTS.iter().position(|v| v.name == n)).unwrap_or(NONE);
        FORCED.store(ix, Ordering::Relaxed);
    }
    if let Some(v) = FLASH_VARIANTS.get(ix).filter(|v| v.head_dim == d && v.nsg % g == 0) {
        return v;
    }
    FLASH_VARIANTS.iter().find(|v| v.head_dim == d && v.nsg % g == 0).expect("a flash variant for head_dim and group")
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    /// Launch `attn_prefill_bf16_flash`.
    pub(crate) fn launch_attn_flash(&self, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let (nkv, cap) = (i[1].shape[0], i[1].shape[1]);
        if t * n * d == 0 {
            return Ok(());
        }
        if nkv == 0 || n % nkv != 0 || !GROUPS.contains(&(n / nkv)) || !matches!(d, 64 | 128 | 256) {
            return Err(Error::invalid(format!(
                "{ATTN_PREFILL_BF16_FLASH}: unsupported shape N {n} Nkv {nkv} head_dim {d}"
            )));
        }
        let (q, kc, vc, out) = (
            self.bind(&i[0], bf, false, "q")?,
            self.bind(&i[1], bf, false, "k cache")?,
            self.bind(&i[2], bf, false, "v cache")?,
            self.bind(&o[0], bf, true, "out")?,
        );
        let (ka, va) = (self.mbuf(&i[1].buf)?.base + i[1].offset, self.mbuf(&i[2].buf)?.base + i[2].offset);
        if q.skip != 0 || out.skip != 0 || ka % 16 != 0 || va % 16 != 0 {
            if FALLBACKS.fetch_add(1, Ordering::Relaxed) == 0 {
                eprintln!(
                    "metal {ATTN_PREFILL_BF16_FLASH}: misaligned operands (q skip {}, out skip {}, k cache at {ka}, v cache at {va}); \
                     running {ATTN_PREFILL_BF16} (reported once)",
                    q.skip, out.skip
                );
            }
            return self.launch_dense(ATTN_PREFILL_BF16, call);
        }
        let g = n / nkv;
        let v = pick(d, g);
        let bq = 8 * (v.nsg / g);
        let p = OpP {
            dims: [
                u32_of(t, "T")?,
                u32_of(n, "N")?,
                u32_of(nkv, "Nkv")?,
                u32_of(d, "D")?,
                u32_of(cap, "cap")?,
                u32_of(a.pos0, "pos0")?,
                0,
                0,
            ],
            skips: [0; 5],
            f0: a.scale,
            f1: 0.0,
            pad: 0,
        };
        u32_of(t * n * d, "T*N*D")?;
        u32_of(nkv * cap * d, "Nkv*cap*D")?;
        let pso = self.pso(v.name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{}'", v.name)))?;
        self.with_encoder(|enc| {
            enc.setComputePipelineState(pso);
            for (idx, b) in [&q, &kc, &vc, &out].into_iter().enumerate() {
                // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`).
                unsafe { enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, idx) };
            }
            set_bytes(enc, &p, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: t.div_ceil(bq), height: nkv, depth: 1 },
                MTLSize { width: v.nsg * 32, height: 1, depth: 1 },
            );
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_head_dim_and_group_has_a_variant() {
        for d in [64, 128, 256] {
            for g in GROUPS {
                assert!(FLASH_VARIANTS.iter().any(|v| v.head_dim == d && v.nsg % g == 0), "d {d} group {g}");
            }
        }
        assert!(!flash_ok(&ShapeInfo { m: 4, n_heads: 12, n_kv_heads: 4, head_dim: 128, ..Default::default() }));
        assert!(flash_ok(&ShapeInfo { m: 4, n_heads: 32, n_kv_heads: 8, head_dim: 128, ..Default::default() }));
    }
}

//! Launcher of the NAX (M5 neural accelerator) flash-style causal prefill attention (`attn_prefill_bf16_flash_nax`,
//! kernels/metal/attention_flash_nax.metal, compiled `-std=metal4.0` into `intarsia_metal4.metallib`; the design notes are
//! in the shader header). One threadgroup per (block of query positions, kv head); a simdgroup owns 16 query rows of one
//! query head, Q stays in registers and K / V fragments are read from device memory by the lanes.
//!
//! Same op contract as `attn_prefill_bf16_flash` (flash.rs): any T and pos0, so how a prompt is cut into chunks never
//! changes which kernel a row goes through. Served: head_dim 128 and a group size (query heads per kv head) of 1, 2, 4 or
//! 8; other head dims stay on the P2 kernel. Operands it cannot take (a leading element skip on q / out, K / V caches not
//! 16-byte aligned) run the P2 flash kernel (or `attn_prefill_bf16` where that cannot either); counted in `fallbacks`.

use crate::dense::OpP;
use crate::kernels::*;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// One instantiation of `attn_nax<D, BK, NSG>`.
#[derive(Clone, Copy, Debug)]
pub struct NaxFlashVariant {
    pub name: &'static str,
    pub head_dim: usize,
    /// Simdgroups per threadgroup (the group size must divide it): one query head of the group each, times
    /// `nsg / group` position slabs of 16 rows.
    pub nsg: usize,
    /// Keys per tile.
    pub bk: usize,
}

macro_rules! nv {
    ($n:literal, $d:literal, $s:literal, $k:literal) => {
        NaxFlashVariant { name: $n, head_dim: $d, nsg: $s, bk: $k }
    };
}

/// Every instantiation compiled into the metal4.0 library; the first with a fitting `nsg` is the default.
pub const NAX_FLASH_VARIANTS: &[NaxFlashVariant] = &[
    nv!("attn_nax_d128_s4", 128, 4, 32),
    nv!("attn_nax_d128_s8", 128, 8, 32),
];

const GROUPS: [usize; 4] = [1, 2, 4, 8];

static FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Launches that ran another kernel instead (misaligned operands).
pub fn fallbacks() -> u64 {
    FALLBACKS.load(Ordering::Relaxed)
}

/// Shapes the NAX flash kernel takes: head_dim 128, query heads a multiple of the kv heads with a group in 1, 2, 4, 8.
pub fn nax_flash_ok(s: &ShapeInfo) -> bool {
    s.m >= 1
        && s.head_dim == 128
        && s.n_kv_heads > 0
        && s.n_heads % s.n_kv_heads == 0
        && GROUPS.contains(&(s.n_heads / s.n_kv_heads))
}

const UNSET: usize = usize::MAX;
const NONE: usize = usize::MAX - 1;
/// Index into `NAX_FLASH_VARIANTS` forced by `INTARSIA_METAL_FLASH_NAX=<name>` or [`select_variant`].
static FORCED: AtomicUsize = AtomicUsize::new(UNSET);

/// Force the variant `name` (or the default again with `None`) for later launches: diagnostic for sweeps and tests (a
/// variant changes the rounding order of the result).
pub fn select_variant(name: Option<&str>) {
    let ix = name.map_or(NONE, |n| NAX_FLASH_VARIANTS.iter().position(|v| v.name == n).expect("unknown nax flash variant"));
    FORCED.store(ix, Ordering::Relaxed);
}

fn pick(d: usize, g: usize) -> &'static NaxFlashVariant {
    let mut ix = FORCED.load(Ordering::Relaxed);
    if ix == UNSET {
        let env = std::env::var("INTARSIA_METAL_FLASH_NAX").ok();
        ix = env.and_then(|n| NAX_FLASH_VARIANTS.iter().position(|v| v.name == n)).unwrap_or(NONE);
        FORCED.store(ix, Ordering::Relaxed);
    }
    if let Some(v) = NAX_FLASH_VARIANTS.get(ix).filter(|v| v.head_dim == d && v.nsg % g == 0) {
        return v;
    }
    NAX_FLASH_VARIANTS.iter().find(|v| v.head_dim == d && v.nsg % g == 0).expect("a nax flash variant for head_dim and group")
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    /// Launch `attn_prefill_bf16_flash_nax`.
    pub(crate) fn launch_attn_flash_nax(&self, call: &OpCall) -> Result<()> {
        let (i, o, a) = (&call.inputs, &call.outputs, &call.attrs);
        let bf = DType::Bf16;
        let (t, n, d) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let (nkv, cap) = (i[1].shape[0], i[1].shape[1]);
        if t * n * d == 0 {
            return Ok(());
        }
        if nkv == 0 || n % nkv != 0 || !GROUPS.contains(&(n / nkv)) || d != 128 {
            return Err(Error::invalid(format!(
                "{ATTN_PREFILL_BF16_FLASH_NAX}: unsupported shape N {n} Nkv {nkv} head_dim {d}"
            )));
        }
        let (q, kc, vc, out) = (
            self.bind(&i[0], bf, false, "q")?,
            self.bind(&i[1], bf, false, "k cache")?,
            self.bind(&i[2], bf, false, "v cache")?,
            self.bind(&o[0], bf, true, "out")?,
        );
        let (ka, va) = (self.mbuf(&i[1].buf)?.base + i[1].offset, self.mbuf(&i[2].buf)?.base + i[2].offset);
        let (qa, oa) = (self.mbuf(&i[0].buf)?.base + i[0].offset, self.mbuf(&o[0].buf)?.base + o[0].offset);
        // q / out are read and written in 8-byte pieces, K / V tensors need 16-byte aligned rows
        if q.skip != 0 || out.skip != 0 || qa % 16 != 0 || oa % 16 != 0 || ka % 16 != 0 || va % 16 != 0 {
            if FALLBACKS.fetch_add(1, Ordering::Relaxed) == 0 {
                eprintln!(
                    "metal {ATTN_PREFILL_BF16_FLASH_NAX}: misaligned operands (q skip {}, out skip {}, q at {qa}, out at {oa}, k cache at {ka}, v cache at {va}); \
                     running {ATTN_PREFILL_BF16_FLASH} (reported once)",
                    q.skip, out.skip
                );
            }
            return self.launch_attn_flash(call);
        }
        let g = n / nkv;
        let v = pick(d, g);
        let bq = 16 * (v.nsg / g);
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
    fn every_group_has_a_variant() {
        for g in GROUPS {
            assert!(NAX_FLASH_VARIANTS.iter().any(|v| v.head_dim == 128 && v.nsg % g == 0), "group {g}");
        }
        assert!(!nax_flash_ok(&ShapeInfo { m: 4, n_heads: 12, n_kv_heads: 4, head_dim: 128, ..Default::default() }));
        assert!(!nax_flash_ok(&ShapeInfo { m: 4, n_heads: 32, n_kv_heads: 8, head_dim: 64, ..Default::default() }));
        assert!(nax_flash_ok(&ShapeInfo { m: 4, n_heads: 32, n_kv_heads: 8, head_dim: 128, ..Default::default() }));
    }
}

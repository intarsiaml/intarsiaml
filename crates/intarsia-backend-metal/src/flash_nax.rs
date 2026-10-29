//! Launcher of the NAX (M5 neural accelerator) flash-style causal prefill attention (`attn_prefill_bf16_flash_nax`,
//! kernels/metal/attention_flash_nax.metal, compiled `-std=metal4.0` into `intarsia_metal4.metallib`; the design notes are
//! in the shader header). One threadgroup per (block of query positions, kv head); a simdgroup owns 16 query rows of one
//! query head, Q stays in registers and K / V fragments are read from device memory by the lanes.
//!
//! Same op contract as `attn_prefill_bf16_flash` (flash.rs): any T and pos0, so how a prompt is cut into chunks never
//! changes which kernel a row goes through and the attention of a row is bitwise chunk invariant. Served: head_dim 128 with
//! a group size (query heads per kv head) of 1 to 8, and head_dim 256 with an even group (`attn_nax_x`, each head split
//! over two simdgroups); other head dims stay on the P2 kernel. Operands it cannot take
//! (a leading element skip on q / out, q / out / K / V not 16-byte aligned) run the P2 flash kernel (or `attn_prefill_bf16`
//! where that cannot either); counted in `fallbacks`.

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
    /// head_dim 256 (`attn_nax_x<BK, NH>`): query heads per threadgroup, each split over two simdgroups that hold half
    /// of the head's columns; 0 for `attn_nax<D, BK, NSG>`.
    pub nh: usize,
}

macro_rules! nv {
    ($n:literal, $d:literal, $s:literal, $k:literal) => {
        NaxFlashVariant { name: $n, head_dim: $d, nsg: $s, bk: $k, nh: 0 }
    };
    ($n:literal, x $k:literal, $h:literal) => {
        NaxFlashVariant { name: $n, head_dim: 256, nsg: 2 * $h, bk: $k, nh: $h }
    };
}

/// Every instantiation compiled into the metal4.0 library; the first with a fitting `nsg` is the default.
pub const NAX_FLASH_VARIANTS: &[NaxFlashVariant] = &[
    nv!("attn_nax_d128_s4", 128, 4, 32),
    nv!("attn_nax_d128_s8", 128, 8, 32),
    // head_dim 256: h4 first (the default for groups 4 and 8), h2 for groups 2 and 6; equal bits (same tile)
    nv!("attn_nax_d256_h4", x 32, 4),
    nv!("attn_nax_d256_h2", x 32, 2),
];

/// Group sizes (query heads per kv head); 3, 5, 6, 7 run the next larger variant with surplus simdgroups owning no head (see flash.rs).
const GROUPS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

static FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Launches that ran another kernel instead (misaligned operands).
pub fn fallbacks() -> u64 {
    FALLBACKS.load(Ordering::Relaxed)
}

/// Smallest chunk (query rows) the NAX kernel is registered for: every prefill chunk (T = 1 is the decode op). A short
/// tail chunk is slower here than on the P2 kernel (8-row slabs instead of 16: at 4k context T = 8..32 runs 380 us per
/// layer against P2's 316; level at T = 48, faster from T = 64, docs/results/raw/2026-10-15-metal-prefill-f1/
/// bench-small-chunk.txt), about 2 ms per prompt at 4k, 9 ms at 16k. It is registered anyway so that how a prompt is cut
/// into chunks never changes which attention kernel a row goes through: with the kernel switching at 48 rows, a 4050-token
/// chunk with a 12-row tail flipped a greedy token (position 22 of 48) on the 4062-token prompt that every chunking
/// gave identically before.
pub const NAX_FLASH_MIN_M: usize = 2;

/// Whether head_dim 256 takes the NAX kernel (`attn_nax_x`, docs/results/2026-10-29-mpre-metal-prefill.md); on by default,
/// `INTARSIA_METAL_FLASH_NAX_D256=0` leaves it on the P2 kernel (bitwise the previous build).
pub fn d256_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_METAL_FLASH_NAX_D256").map_or(true, |v| v != "0"))
}

/// Shapes the NAX flash kernel is registered for: at least [`NAX_FLASH_MIN_M`] query rows, query heads a multiple of the
/// kv heads with a group of 1 to 8, and head_dim 128, or head_dim 256 with an even group (unless [`d256_enabled`] is
/// off). (A call to `launch` by name takes any T.)
pub fn nax_flash_ok(s: &ShapeInfo) -> bool {
    s.m >= NAX_FLASH_MIN_M
        && s.n_kv_heads > 0
        && s.n_heads % s.n_kv_heads == 0
        && GROUPS.contains(&(s.n_heads / s.n_kv_heads))
        && (s.head_dim == 128 || (s.head_dim == 256 && (s.n_heads / s.n_kv_heads) % 2 == 0 && d256_enabled()))
}

const UNSET: usize = usize::MAX;
const NONE: usize = usize::MAX - 1;
/// Index into `NAX_FLASH_VARIANTS` forced by `INTARSIA_METAL_FLASH_NAX=<name>` or [`select_variant`].
static FORCED: AtomicUsize = AtomicUsize::new(UNSET);

/// Force the variant `name` (or the default again with `None`) for later launches: diagnostic for sweeps and tests (a
/// variant changes the rounding order of the result).
pub fn select_variant(name: Option<&str>) {
    let ix =
        name.map_or(NONE, |n| NAX_FLASH_VARIANTS.iter().position(|v| v.name == n).expect("unknown nax flash variant"));
    FORCED.store(ix, Ordering::Relaxed);
}

/// Whether variant `v` serves head_dim `d` at group size `g`.
fn fits(v: &NaxFlashVariant, d: usize, g: usize) -> bool {
    v.head_dim == d && if v.nh > 0 { g % v.nh == 0 } else { v.nsg % g == 0 }
}

fn pick(d: usize, g: usize) -> &'static NaxFlashVariant {
    let mut ix = FORCED.load(Ordering::Relaxed);
    if ix == UNSET {
        let env = std::env::var("INTARSIA_METAL_FLASH_NAX").ok();
        ix = env.and_then(|n| NAX_FLASH_VARIANTS.iter().position(|v| v.name == n)).unwrap_or(NONE);
        FORCED.store(ix, Ordering::Relaxed);
    }
    if let Some(v) = NAX_FLASH_VARIANTS.get(ix).filter(|v| fits(v, d, g)) {
        return v;
    }
    NAX_FLASH_VARIANTS
        .iter()
        .find(|v| fits(v, d, g))
        .or_else(|| NAX_FLASH_VARIANTS.iter().find(|v| v.head_dim == d && v.nh == 0 && v.nsg >= g))
        .expect("a nax flash variant for head_dim and group")
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
        if nkv == 0 || n % nkv != 0 || !GROUPS.contains(&(n / nkv)) || !(d == 128 || (d == 256 && (n / nkv) % 2 == 0)) {
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
        let (bq, depth) = if v.nh > 0 { (16, g / v.nh) } else { (16 * (v.nsg / g), 1) };
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
                MTLSize { width: t.div_ceil(bq), height: nkv, depth },
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
            assert!(NAX_FLASH_VARIANTS.iter().any(|v| v.head_dim == 128 && v.nsg >= g), "group {g}");
        }
        let sh = |m, n, nkv, d| ShapeInfo { m, n_heads: n, n_kv_heads: nkv, head_dim: d, ..Default::default() };
        assert!(nax_flash_ok(&sh(64, 28, 4, 128)));
        assert!(!nax_flash_ok(&sh(64, 36, 4, 128)), "group 9");
        assert!(!nax_flash_ok(&sh(64, 32, 8, 64)));
        assert!(nax_flash_ok(&sh(64, 16, 2, 256)), "the 35B: head_dim 256, group 8");
        assert!(nax_flash_ok(&sh(64, 16, 4, 256)), "group 4");
        assert!(!nax_flash_ok(&sh(64, 12, 4, 256)), "head_dim 256 needs an even group");
        for g in [2, 4, 6, 8] {
            assert!(NAX_FLASH_VARIANTS.iter().any(|v| fits(v, 256, g)), "head_dim 256 group {g}");
        }
        assert!(!nax_flash_ok(&sh(1, 32, 8, 128)), "T = 1 is the decode op");
        assert!(nax_flash_ok(&sh(NAX_FLASH_MIN_M, 32, 8, 128)));
        assert!(nax_flash_ok(&sh(47, 32, 8, 128)));
    }
}

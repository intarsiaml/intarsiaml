//! Fault injection for checking that an accuracy gate can see a small change (docs/accuracy.md, "Choosing the
//! self-golden budgets"). Off unless `INTARSIA_CPU_FAULT` is set; never set it outside a gate-sensitivity check.
//!
//! `INTARSIA_CPU_FAULT=<op>[,<op>...]:<p>` flips the last mantissa bit of each bf16 output element of those ops,
//! up or down, with probability `p` (deterministic: a SplitMix64 stream keyed by the launch counter). A kernel that
//! sums in another order produces exactly this kind of output change: some elements end one bf16 ulp away
//! (rel. error 2^-8 .. 2^-9) and the rest are identical. `p` = 4.7e-3 gives rel_l2 of about 3.5e-4 on gaussian bf16 data (rel_l2 ~ 5.1e-3 * sqrt(p)), the
//! size measured between the old and the grouped MoE kernel (docs/quant-moe.md).
//!
//! `INTARSIA_CPU_MUTATE=<op>[,<op>...]:scale=<f>` is the opposite kind of hook, a deliberate BUG: every bf16 output
//! element of those ops is multiplied by `f` (`attn_prefill,attn_decode:scale=1.05` is an attention output 5 % too
//! large, `matmul:scale=1.05` a projection with a wrong global scale, the kind of error a mis-scaled NVFP4 tensor
//! makes). The noise-envelope gate must FAIL a run with this set while it passes a legitimate reordering
//! (docs/accuracy.md, "Noise envelope"). Same rules: off by default, gate-sensitivity checks only.

use intarsia_core::backend::Backend;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, OpCall, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

struct Fault {
    ops: Vec<String>,
    p: f64,
}

fn config() -> &'static Option<Fault> {
    static CFG: OnceLock<Option<Fault>> = OnceLock::new();
    CFG.get_or_init(|| {
        let v = std::env::var("INTARSIA_CPU_FAULT").ok()?;
        let (ops, p) = v.rsplit_once(':')?;
        let p: f64 = p.trim().parse().ok().filter(|p| (0.0..=1.0).contains(p))?;
        eprintln!("WARNING: INTARSIA_CPU_FAULT={v}: bf16 outputs of [{ops}] are perturbed on purpose (gate sensitivity check)");
        Some(Fault { ops: ops.split(',').map(|s| s.trim().to_string()).collect(), p })
    })
}

struct Mutation {
    ops: Vec<String>,
    scale: f32,
}

fn mutation() -> &'static Option<Mutation> {
    static CFG: OnceLock<Option<Mutation>> = OnceLock::new();
    CFG.get_or_init(|| {
        let v = std::env::var("INTARSIA_CPU_MUTATE").ok()?;
        let (ops, f) = v.rsplit_once(':')?;
        let scale: f32 = f.trim().strip_prefix("scale=")?.parse().ok().filter(|s: &f32| s.is_finite())?;
        eprintln!("WARNING: INTARSIA_CPU_MUTATE={v}: bf16 outputs of [{ops}] are scaled x{scale} on purpose (a deliberate bug, gate sensitivity check)");
        Some(Mutation { ops: ops.split(',').map(|s| s.trim().to_string()).collect(), scale })
    })
}

static LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Perturb the bf16 outputs of `call` in place when the fault is configured for its op.
pub fn apply(be: &dyn Backend, call: &OpCall) -> Result<()> {
    if let Some(m) = mutation() {
        if m.ops.iter().any(|o| o == call.op.as_str()) {
            for o in call.outputs.iter().filter(|o| o.dtype == DType::Bf16) {
                let mut bytes = vec![0u8; o.numel() * 2];
                be.download(&o.buf, o.offset, &mut bytes, 0)?;
                scale_bf16(&mut bytes, m.scale);
                be.upload(&o.buf, o.offset, &bytes, 0)?;
            }
        }
    }
    let Some(f) = config() else { return Ok(()) };
    if !f.ops.iter().any(|o| o == call.op.as_str()) {
        return Ok(());
    }
    let mut rng = SplitMix64::new(0xFA17 ^ LAUNCHES.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E37_79B9));
    for o in call.outputs.iter().filter(|o| o.dtype == DType::Bf16) {
        let mut bytes = vec![0u8; o.numel() * 2];
        be.download(&o.buf, o.offset, &mut bytes, 0)?;
        flip_ulps(&mut bytes, f.p, &mut rng);
        be.upload(&o.buf, o.offset, &bytes, 0)?;
    }
    Ok(())
}

/// Multiply each bf16 element (little-endian pairs in `bytes`) by `scale`, rounded back to bf16.
fn scale_bf16(bytes: &mut [u8], scale: f32) {
    for c in bytes.chunks_exact_mut(2) {
        let v = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]));
        c.copy_from_slice(&half::bf16::from_f32(v.to_f32() * scale).to_bits().to_le_bytes());
    }
}

/// Move each bf16 element (little-endian pairs in `bytes`) one ulp up or down in magnitude with probability `p`. Zeros,
/// infinities and NaNs are left alone, and the largest finite value is not pushed to infinity.
fn flip_ulps(bytes: &mut [u8], p: f64, rng: &mut SplitMix64) {
    for c in bytes.chunks_exact_mut(2) {
        if (rng.next_f32() as f64) < p {
            let v = u16::from_le_bytes([c[0], c[1]]);
            let (sign, mag) = (v & 0x8000, v & 0x7FFF);
            let up = rng.next_u64() & 1 == 0;
            let mag = match (up, mag) {
                (_, 0) => continue,
                (_, m) if m >= 0x7F80 => continue, // inf / NaN
                (true, 0x7F7F) => continue,
                (true, m) => m + 1,
                (false, m) => m - 1,
            };
            c.copy_from_slice(&(sign | mag).to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_l2_of(p: f64) -> f64 {
        let mut rng = SplitMix64::new(7);
        let n = 400_000;
        let x: Vec<half::bf16> = (0..n).map(|_| half::bf16::from_f32(rng.normal())).collect();
        let mut bytes: Vec<u8> = x.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        flip_ulps(&mut bytes, p, &mut rng);
        let (mut num, mut den) = (0f64, 0f64);
        for (c, v) in bytes.chunks_exact(2).zip(&x) {
            let y = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64();
            num += (y - v.to_f64()).powi(2);
            den += v.to_f64().powi(2);
        }
        (num / den).sqrt()
    }

    #[test]
    fn mutation_scales_every_element() {
        let x: Vec<half::bf16> = [1.0f32, -2.0, 0.5, 0.0].iter().map(|v| half::bf16::from_f32(*v)).collect();
        let mut bytes: Vec<u8> = x.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        scale_bf16(&mut bytes, 1.5);
        let y: Vec<f32> = bytes.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect();
        assert_eq!(y, [1.5, -3.0, 0.75, 0.0]);
    }

    /// The documented mapping from flip rate to relative L2 error (docs/accuracy.md uses it to size the injected faults).
    #[test]
    fn flip_rate_to_rel_l2() {
        assert_eq!(rel_l2_of(0.0), 0.0);
        let r = rel_l2_of(4.7e-3);
        assert!((2.8e-4..4.4e-4).contains(&r), "rel_l2 at p=4.7e-3 is {r:.3e}, documented as ~3.5e-4");
        let r1 = rel_l2_of(1.0);
        assert!((3.0e-3..7.0e-3).contains(&r1), "every element one ulp off: {r1:.3e}");
    }
}

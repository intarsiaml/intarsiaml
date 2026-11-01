//! CPU reference dequantization of Engram rows (the reference's `ParallelEngramEmbedding.forward`):
//! `bf16(f32(e4m3fn) * 2^(e8m0 - 127))`, one scale per 32 values. Engines dequantize on the GPU; this is for
//! tests and CPU backends.

use crate::layout::{SCALE_BYTES, WEIGHT_BYTES};

/// FP8 E4M3 (fn: no infinities, 0x7F/0xFF NaN) to f32.
pub fn e4m3fn_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    if e == 0xF && b & 0x7 == 0x7 {
        return f32::NAN;
    }
    if e == 0 {
        sign * m / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// E8M0 (unsigned exponent, bias 127, 0xFF NaN) to f32.
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        f32::NAN
    } else {
        2f32.powi(b as i32 - 127)
    }
}

/// Dequantize one 256-wide row to bf16, as the reference does (product in f32, then round to bf16).
pub fn dequant_row_bf16(weights: &[u8], scales: &[u8], out: &mut [half::bf16]) {
    assert!(weights.len() >= WEIGHT_BYTES && scales.len() >= SCALE_BYTES && out.len() >= WEIGHT_BYTES);
    let block = WEIGHT_BYTES / SCALE_BYTES;
    for i in 0..WEIGHT_BYTES {
        out[i] = half::bf16::from_f32(e4m3fn_to_f32(weights[i]) * e8m0_to_f32(scales[i / block]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_values() {
        assert_eq!(e4m3fn_to_f32(0x00), 0.0);
        assert_eq!(e4m3fn_to_f32(0x38), 1.0);
        assert_eq!(e4m3fn_to_f32(0xB8), -1.0);
        assert_eq!(e4m3fn_to_f32(0x7E), 448.0);
        assert_eq!(e4m3fn_to_f32(0x01), 2f32.powi(-9));
        assert!(e4m3fn_to_f32(0x7F).is_nan());
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(120), 2f32.powi(-7));
    }
}

//! Element types. The numeric values are part of the C kernel ABI
//! (`kernels/abi/ik_abi.h`, `IK_DTYPE_*`); a unit test keeps them in sync.

use crate::error::{Error, Result};
use half::{bf16, f16};

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
    Fp8E4m3 = 3,
    Fp8E5m2 = 4,
    I32 = 5,
    I8 = 6,
    U8 = 7,
}

impl DType {
    pub const ALL: [DType; 8] =
        [DType::F32, DType::F16, DType::Bf16, DType::Fp8E4m3, DType::Fp8E5m2, DType::I32, DType::I8, DType::U8];

    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::Bf16 => 2,
            DType::Fp8E4m3 | DType::Fp8E5m2 | DType::I8 | DType::U8 => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::Bf16 | DType::Fp8E4m3 | DType::Fp8E5m2)
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::Fp8E4m3 => "fp8_e4m3",
            DType::Fp8E5m2 => "fp8_e5m2",
            DType::I32 => "i32",
            DType::I8 => "i8",
            DType::U8 => "u8",
        }
    }

    /// Parse the names used in recipes and CLI flags ("bf16", "fp8_e4m3", ...).
    pub fn parse(s: &str) -> Result<DType> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" | "float32" | "fp32" => DType::F32,
            "f16" | "float16" | "fp16" => DType::F16,
            "bf16" | "bfloat16" => DType::Bf16,
            "fp8_e4m3" | "fp8" | "f8_e4m3" | "float8_e4m3fn" => DType::Fp8E4m3,
            "fp8_e5m2" | "f8_e5m2" | "float8_e5m2" => DType::Fp8E5m2,
            "i32" | "int32" => DType::I32,
            "i8" | "int8" => DType::I8,
            "u8" | "uint8" => DType::U8,
            other => return Err(Error::Model(format!("unknown dtype '{other}'"))),
        })
    }

    /// Parse a safetensors dtype string ("BF16", "F32", "F8_E4M3", ...).
    pub fn from_safetensors(s: &str) -> Result<DType> {
        Ok(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::Bf16,
            "F8_E4M3" => DType::Fp8E4m3,
            "F8_E5M2" => DType::Fp8E5m2,
            "I32" => DType::I32,
            "I8" => DType::I8,
            "U8" => DType::U8,
            other => return Err(Error::Unsupported(format!("safetensors dtype {other} is not supported"))),
        })
    }

    pub fn from_i32(v: i32) -> Option<DType> {
        DType::ALL.iter().copied().find(|d| *d as i32 == v)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// fp8 conversions (software, used by the oracle and host tools)
// ---------------------------------------------------------------------------

/// Decode an OCP FP8 E4M3 (finite, "fn") byte.
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    if e == 0xF && (b & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 {
        sign * (m / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

fn e4m3_positive_table() -> &'static [f32; 127] {
    use std::sync::OnceLock;
    static T: OnceLock<[f32; 127]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 127];
        for (i, v) in t.iter_mut().enumerate() {
            *v = fp8_e4m3_to_f32(i as u8);
        }
        t
    })
}

/// Round-to-nearest-even encode to FP8 E4M3, saturating at +-448 (NaN -> 0x7F).
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    let t = e4m3_positive_table();
    if a >= t[126] {
        return sign | 126;
    }
    // first index with t[i] >= a
    let hi = t.partition_point(|v| *v < a);
    if hi == 0 {
        return sign;
    }
    let lo = hi - 1;
    let dl = a - t[lo];
    let dh = t[hi] - a;
    let pick = if dl < dh {
        lo
    } else if dh < dl {
        hi
    } else if lo % 2 == 0 {
        lo
    } else {
        hi
    };
    sign | pick as u8
}

pub fn fp8_e5m2_to_f32(b: u8) -> f32 {
    f16::from_bits((b as u16) << 8).to_f32()
}

pub fn f32_to_fp8_e5m2(x: f32) -> u8 {
    let h = f16::from_f32(x).to_bits();
    if f16::from_bits(h).is_nan() {
        return 0x7F;
    }
    let lsb = (h >> 8) & 1;
    let r = h.wrapping_add(0x7F + lsb) >> 8;
    // overflow wraps into inf (0x7C); acceptable (saturating not required for e5m2)
    r as u8
}

// ---------------------------------------------------------------------------
// bulk conversions between raw little-endian bytes and f32
// ---------------------------------------------------------------------------

/// Decode `bytes` (little endian, `dtype` elements) into `out` as f32.
pub fn decode_to_f32(dtype: DType, bytes: &[u8], out: &mut [f32]) {
    let n = out.len();
    debug_assert!(bytes.len() >= n * dtype.size_bytes());
    match dtype {
        DType::F32 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        DType::Bf16 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        DType::F16 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            }
        }
        DType::Fp8E4m3 => {
            for (o, b) in out.iter_mut().zip(bytes.iter()) {
                *o = fp8_e4m3_to_f32(*b);
            }
        }
        DType::Fp8E5m2 => {
            for (o, b) in out.iter_mut().zip(bytes.iter()) {
                *o = fp8_e5m2_to_f32(*b);
            }
        }
        DType::I32 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32;
            }
        }
        DType::I8 => {
            for (o, b) in out.iter_mut().zip(bytes.iter()) {
                *o = *b as i8 as f32;
            }
        }
        DType::U8 => {
            for (o, b) in out.iter_mut().zip(bytes.iter()) {
                *o = *b as f32;
            }
        }
    }
}

/// Encode f32 values into `dtype` little-endian bytes (round-to-nearest-even).
pub fn encode_from_f32(dtype: DType, vals: &[f32], out: &mut [u8]) {
    debug_assert!(out.len() >= vals.len() * dtype.size_bytes());
    match dtype {
        DType::F32 => {
            for (v, c) in vals.iter().zip(out.chunks_exact_mut(4)) {
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::Bf16 => {
            for (v, c) in vals.iter().zip(out.chunks_exact_mut(2)) {
                c.copy_from_slice(&bf16::from_f32(*v).to_bits().to_le_bytes());
            }
        }
        DType::F16 => {
            for (v, c) in vals.iter().zip(out.chunks_exact_mut(2)) {
                c.copy_from_slice(&f16::from_f32(*v).to_bits().to_le_bytes());
            }
        }
        DType::Fp8E4m3 => {
            for (v, b) in vals.iter().zip(out.iter_mut()) {
                *b = f32_to_fp8_e4m3(*v);
            }
        }
        DType::Fp8E5m2 => {
            for (v, b) in vals.iter().zip(out.iter_mut()) {
                *b = f32_to_fp8_e5m2(*v);
            }
        }
        DType::I32 => {
            for (v, c) in vals.iter().zip(out.chunks_exact_mut(4)) {
                c.copy_from_slice(&(*v as i32).to_le_bytes());
            }
        }
        DType::I8 => {
            for (v, b) in vals.iter().zip(out.iter_mut()) {
                *b = (*v as i8) as u8;
            }
        }
        DType::U8 => {
            for (v, b) in vals.iter().zip(out.iter_mut()) {
                *b = *v as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_e4m3_roundtrip_all_codes() {
        for b in 0u8..=255 {
            let f = fp8_e4m3_to_f32(b);
            if f.is_nan() {
                continue;
            }
            let back = f32_to_fp8_e4m3(f);
            // -0 and +0 both fine; otherwise exact
            assert_eq!(fp8_e4m3_to_f32(back), f, "code {b:#x}");
        }
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert_eq!(f32_to_fp8_e4m3(1e9), 0x7E);
        assert_eq!(f32_to_fp8_e4m3(-1e9), 0xFE);
    }

    #[test]
    fn fp8_e5m2_roundtrip() {
        for b in 0u8..=255 {
            let f = fp8_e5m2_to_f32(b);
            if f.is_nan() || f.is_infinite() {
                continue;
            }
            assert_eq!(fp8_e5m2_to_f32(f32_to_fp8_e5m2(f)), f);
        }
    }

    #[test]
    fn bulk_roundtrip_bf16() {
        let v = [1.0f32, -2.5, 0.15625, 3.0e10];
        let mut bytes = vec![0u8; 8];
        encode_from_f32(DType::Bf16, &v, &mut bytes);
        let mut out = [0f32; 4];
        decode_to_f32(DType::Bf16, &bytes, &mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], -2.5);
        assert_eq!(out[2], 0.15625);
        assert!((out[3] - 3.0e10).abs() / 3.0e10 < 0.01);
    }

    #[test]
    fn abi_header_matches() {
        // kernels/abi/ik_abi.h carries IK_DTYPE_* constants equal to the repr values.
        let hdr = include_str!("../../../kernels/abi/ik_abi.h");
        for (name, d) in [
            ("IK_DTYPE_F32", DType::F32),
            ("IK_DTYPE_F16", DType::F16),
            ("IK_DTYPE_BF16", DType::Bf16),
            ("IK_DTYPE_FP8_E4M3", DType::Fp8E4m3),
            ("IK_DTYPE_FP8_E5M2", DType::Fp8E5m2),
            ("IK_DTYPE_I32", DType::I32),
            ("IK_DTYPE_I8", DType::I8),
            ("IK_DTYPE_U8", DType::U8),
        ] {
            let line = hdr
                .lines()
                .find(|l| l.contains(name) && l.contains('='))
                .unwrap_or_else(|| panic!("{name} missing in ik_abi.h"));
            let val: i32 = line
                .split('=')
                .nth(1)
                .unwrap()
                .trim()
                .trim_end_matches(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .split(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(val, d as i32, "{name}");
        }
    }
}

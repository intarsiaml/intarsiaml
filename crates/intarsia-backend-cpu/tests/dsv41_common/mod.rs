//! Shared helpers of the DeepSeek-V4.1 parity tests: a minimal safetensors reader, the golden directory, op runners and
//! error metrics. Included with `mod dsv41_common;` by `dsv41_parity.rs` and `dsv41_chunked.rs`.
#![allow(dead_code)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::{DType, HostTensor, Op, OpAttrs};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/dsv41")
}

/// One tensor of a safetensors file.
pub struct Raw {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Raw {
    pub fn f32(&self) -> Vec<f32> {
        match self.dtype.as_str() {
            "F32" | "BF16" | "F16" => self.host().to_f32(),
            "I32" => self.i32().into_iter().map(|v| v as f32).collect(),
            "U8" | "BOOL" => self.bytes.iter().map(|&b| b as f32).collect(),
            d => panic!("f32() of {d}"),
        }
    }
    /// The tensor as stored (F32 / BF16 / F16 / I32 / I16 / U8 only).
    pub fn host(&self) -> HostTensor {
        let dt = match self.dtype.as_str() {
            "F32" => DType::F32,
            "BF16" => DType::Bf16,
            "F16" => DType::F16,
            "I32" => DType::I32,
            "I16" => DType::I16,
            "U8" | "BOOL" => DType::U8,
            d => panic!("host() of {d}"),
        };
        HostTensor::new(dt, self.shape.clone(), self.bytes.clone()).unwrap()
    }
    pub fn i32(&self) -> Vec<i32> {
        match self.dtype.as_str() {
            "I32" => self.bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "I64" => self.bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32).collect(),
            d => panic!("i32() of {d}"),
        }
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    /// As an engine tensor of dtype `dt` (f32 values converted).
    pub fn tensor_as(&self, dt: DType) -> HostTensor {
        HostTensor::from_f32_as(dt, &self.shape, &self.f32())
    }
}

pub struct St {
    pub t: HashMap<String, Raw>,
}

impl St {
    pub fn open(p: &Path) -> St {
        let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        let h: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
        let mut t = HashMap::new();
        for (k, v) in h.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let o = v["data_offsets"].as_array().unwrap();
            let (s, e) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
            t.insert(
                k.clone(),
                Raw {
                    dtype: v["dtype"].as_str().unwrap().to_string(),
                    shape: v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                    bytes: b[8 + n + s..8 + n + e].to_vec(),
                },
            );
        }
        St { t }
    }
    pub fn golden(name: &str) -> St {
        St::open(&golden_dir().join(name))
    }
    pub fn get(&self, k: &str) -> &Raw {
        self.t.get(k).unwrap_or_else(|| panic!("fixture tensor {k} missing"))
    }
    pub fn has(&self, k: &str) -> bool {
        self.t.contains_key(k)
    }
}

pub fn run(kernel_prefix: &str, hc: &HostCall) -> Vec<HostTensor> {
    let b = CpuBackend::new();
    let name = format!("{kernel_prefix}{}", hc.op.as_str());
    run_host(&b, &name, hc).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Run on the reference kernel set (`ref_*`) and check the fast set (`cpu_f32_*`) agrees bit for bit (both run the same code).
pub fn go(hc: &HostCall) -> Vec<HostTensor> {
    run("ref_", hc)
}

pub fn call(op: Op, inputs: Vec<HostTensor>, outputs: Vec<OutSpec>, attrs: OpAttrs) -> HostCall {
    HostCall { op, inputs, outputs, attrs }
}

pub fn f32t(shape: &[usize], v: &[f32]) -> HostTensor {
    HostTensor::from_f32(shape, v)
}

/// Error of `got` against `want`: max abs error, max abs error relative to the largest `|want|`, and the relative rms error.
#[derive(Clone, Copy, Debug)]
pub struct Err3 {
    pub max_abs: f64,
    pub max_rel_max: f64,
    pub rel_rms: f64,
    pub n_diff: usize,
    pub n: usize,
    pub scale: f64,
}

pub fn err(got: &[f32], want: &[f32]) -> Err3 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs() as f64));
    let (mut ma, mut num, mut den, mut nd) = (0f64, 0f64, 0f64, 0usize);
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite() == w.is_finite() || !w.is_finite(), "non-finite got {g} want {w}");
        let d = (*g as f64 - *w as f64).abs();
        if d.is_nan() {
            continue;
        }
        ma = ma.max(d);
        num += d * d;
        den += (*w as f64) * (*w as f64);
        if g != w {
            nd += 1;
        }
    }
    Err3 { max_abs: ma, max_rel_max: if scale > 0.0 { ma / scale } else { ma }, rel_rms: if den > 0.0 { (num / den).sqrt() } else { num.sqrt() }, n_diff: nd, n: got.len(), scale }
}

/// Print a parity row and assert `max_abs / max|want| <= tol`.
pub fn check(what: &str, got: &[f32], want: &[f32], tol: f64) -> Err3 {
    let e = err(got, want);
    println!("PARITY {what:<46} n={:<8} max_abs={:.3e} max_abs/max|ref|={:.3e} rel_rms={:.3e} differing={} tol={tol:.0e}", e.n, e.max_abs, e.max_rel_max, e.rel_rms, e.n_diff);
    assert!(e.max_rel_max <= tol, "{what}: max_abs/max|ref| {:.3e} > tol {tol:.0e} (max_abs {:.3e}, scale {:.3e})", e.max_rel_max, e.max_abs, e.scale);
    e
}

pub trait Reshape {
    fn reshaped(self, shape: &[usize]) -> Self;
}
impl Reshape for HostTensor {
    fn reshaped(mut self, shape: &[usize]) -> Self {
        assert_eq!(self.numel(), shape.iter().product::<usize>());
        self.shape = shape.to_vec();
        self
    }
}

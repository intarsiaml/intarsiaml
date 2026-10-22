//! Quantisation formats as plugins (ARCHITECTURE 3.4) and the block ops that consume them.
//!
//! * [`QuantFormat`]: `detect` a format from checkpoint tensors, `load` it onto a backend as a [`QTensor`],
//!   `dequant_ref` back to f32 (the oracle: `dequant_ref` + a plain matmul is the meaning of every kernel),
//!   `quantize` (optional round-to-nearest quantiser for tests and on-the-fly use).
//! * Formats: `bf16` (plain dense), `fp8_e4m3` (per-tensor scale, [`fp8`]), `nvfp4` (ModelOpt, [`nvfp4`]),
//!   `mxfp4` (OCP MX, AMD Quark, [`mxfp4`]), `exl3` (ExLlamaV3 trellis, [`exl3`]).
//! * [`moe`]: the Qwen3.5/3.6 sparse MoE block (router + routed NVFP4 experts + gated shared expert).
//! * [`ops`]: shape/dtype validation of the quantised and MoE ops (`Op::MatmulFp8`, `MatmulNvfp4`, `MoeTopk`,
//!   `MoeExperts`, `MoeCombine`); [`cases`]: admission sweeps and plan-time case builders for them.
//!
//! Layout rule (v1): tensors keep the checkpoint layout (no repacking), so `dequant_ref` is checkpoint semantics.

pub mod cases;
pub mod exl3;
pub mod fp8;
pub mod moe;
pub mod mxfp4;
pub mod nvfp4;
pub mod ops;
pub mod reference;
pub mod scheme;

use crate::backend::{Backend, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{Op, OpAttrs, OpCall};
use crate::tensor::HostTensor;
use crate::weights::{WeightInfo, WeightSource};

pub type QuantId = &'static str;
pub const BF16: QuantId = "bf16";
pub const FP8_E4M3: QuantId = "fp8_e4m3";
pub const NVFP4: QuantId = "nvfp4";
pub const MXFP4: QuantId = "mxfp4";

pub use mxfp4::Mxfp4Act;
pub const EXL3: QuantId = "exl3";

/// How the FP8 format treats activations (per-recipe option, see `fp8.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fp8Act {
    /// W8A8: activations are quantised to e4m3 with the checkpoint's static `input_scale` (declared scheme).
    #[default]
    Static,
    /// W8A16: activations stay 16-bit.
    None,
}

impl Fp8Act {
    pub fn parse(s: &str) -> Result<Fp8Act> {
        match s {
            "static" | "w8a8" => Ok(Fp8Act::Static),
            "none" | "w8a16" => Ok(Fp8Act::None),
            other => Err(Error::model(format!("unknown fp8 activation mode '{other}' (static | none)"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QuantLoadOpts {
    pub fp8_act: Fp8Act,
    /// MXFP4: dynamic MXFP4 activations (the declared W4A4 scheme) or 16-bit activations (W4A16).
    pub mxfp4_act: Mxfp4Act,
    /// Tensor parallelism: keep only this rank's slice of the weight (see [`crate::weights::Shard`]). Formats that
    /// cannot slice their layout refuse with `Unsupported` instead of loading the whole tensor.
    pub shard: Option<crate::weights::Shard>,
}

impl QuantLoadOpts {
    /// Error for a format that has no sharded loader yet.
    pub fn refuse_shard(&self, what: &str, prefix: &str) -> Result<()> {
        match self.shard {
            Some(_) => Err(Error::Unsupported(format!(
                "tensor parallelism: '{prefix}' is {what}, whose weights cannot be sharded in this build"
            ))),
            None => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QuantCfg {
    /// FP8: static activation scale to record (None = W8A16).
    pub input_scale: Option<f32>,
    /// MXFP4: activation mode to record (None = the default, dynamic).
    pub mxfp4_act: Option<Mxfp4Act>,
}

/// The checkpoint tensors of one linear layer: `{prefix}.weight`, `{prefix}.weight_scale`, ...
pub struct WeightMeta<'a> {
    pub prefix: &'a str,
    pub src: &'a dyn WeightSource,
}

impl<'a> WeightMeta<'a> {
    pub fn new(src: &'a dyn WeightSource, prefix: &'a str) -> WeightMeta<'a> {
        WeightMeta { prefix, src }
    }
    pub fn info(&self, suffix: &str) -> Option<WeightInfo> {
        self.src.info(&format!("{}.{suffix}", self.prefix))
    }
    /// A one-element f32 tensor (`weight_scale`, `input_scale`, `weight_scale_2`).
    pub fn scalar_f32(&self, suffix: &str) -> Result<f32> {
        let name = format!("{}.{suffix}", self.prefix);
        let info = self.src.info(&name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
        if info.dtype != DType::F32 || info.shape.iter().product::<usize>() != 1 {
            return Err(Error::model(format!("tensor '{name}' must be a single f32, got {} {:?}", info.dtype, info.shape)));
        }
        let b = self.src.bytes(&name)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// A quantised weight matrix `[n, k]` resident on a backend. `parts` are the op's weight inputs in order
/// (after the activation): `[w]` (bf16), `[w, scales]` (fp8), `[w, block_scales, global]` (nvfp4),
/// `[w, e8m0_scales, act]` (mxfp4).
#[derive(Clone, Debug)]
pub struct QTensor {
    pub format: QuantId,
    pub n: usize,
    pub k: usize,
    pub parts: Vec<TensorRef>,
    /// FP8: the `scales` tensor carries a positive input scale (W8A8). MXFP4: the `act` input is 1 (W4A4).
    pub act_quant: bool,
}

/// Host copy of a [`QTensor`] (tests, admission, `dequant_ref`).
#[derive(Clone, Debug)]
pub struct HostQTensor {
    pub format: QuantId,
    pub n: usize,
    pub k: usize,
    pub parts: Vec<HostTensor>,
    pub act_quant: bool,
}

impl HostQTensor {
    /// Dequantised `[n, k]` f32 weights: what the format means.
    pub fn dequant(&self) -> Vec<f32> {
        match self.format {
            NVFP4 => {
                let gs = self.parts[2].to_f32()[0];
                nvfp4::dequant_rows(&self.parts[0].data, &self.parts[1].data, gs, self.n, self.k)
            }
            MXFP4 => mxfp4::dequant_rows(&self.parts[0].data, &self.parts[1].data, self.n, self.k),
            EXL3 => exl3::dequant_parts(&self.parts),
            FP8_E4M3 => {
                let ws = self.parts[1].to_f32()[0];
                self.parts[0].to_f32().into_iter().map(|v| v * ws).collect()
            }
            _ => self.parts[0].to_f32(),
        }
    }

    pub fn upload(&self, backend: &dyn Backend) -> Result<QTensor> {
        let mut parts = Vec::new();
        for p in &self.parts {
            let buf = crate::backend::upload_new(backend, &p.data)?;
            parts.push(TensorRef::new(buf, p.dtype, &p.shape));
        }
        Ok(QTensor { format: self.format, n: self.n, k: self.k, parts, act_quant: self.act_quant })
    }

    /// The op and its weight inputs as host tensors (for building admission cases).
    pub fn op(&self, f32_out: bool) -> Op {
        op_for(self.format, f32_out)
    }
}

fn op_for(format: QuantId, f32_out: bool) -> Op {
    match format {
        NVFP4 => Op::MatmulNvfp4,
        MXFP4 => Op::MatmulMxfp4,
        EXL3 => Op::MatmulExl3,
        FP8_E4M3 => Op::MatmulFp8,
        _ if f32_out => Op::Logits,
        _ => Op::Matmul,
    }
}

impl QTensor {
    /// The op that multiplies activations by this weight. `f32_out`: the bf16 format uses `Op::Logits`
    /// (f32 output); the quantised ops simply write the output tensor's dtype.
    pub fn op(&self, f32_out: bool) -> Op {
        op_for(self.format, f32_out)
    }

    /// `out = x @ W^T`. `out` dtype decides bf16/f32 output for the quantised ops.
    pub fn call(&self, x: TensorRef, out: TensorRef) -> OpCall {
        let f32_out = out.dtype == DType::F32;
        let mut inputs = vec![x];
        inputs.extend(self.parts.iter().cloned());
        OpCall::new(self.op(f32_out), inputs, vec![out], OpAttrs::default())
    }

    pub fn weight_bytes(&self) -> usize {
        self.parts.iter().map(|p| p.nbytes()).sum()
    }

    pub fn download(&self, backend: &dyn Backend) -> Result<HostQTensor> {
        let mut parts = Vec::new();
        for p in &self.parts {
            let mut data = vec![0u8; p.nbytes()];
            backend.download(&p.buf, p.offset, &mut data, 0)?;
            parts.push(HostTensor::new(p.dtype, p.shape.clone(), data)?);
        }
        Ok(HostQTensor { format: self.format, n: self.n, k: self.k, parts, act_quant: self.act_quant })
    }
}

pub trait QuantFormat: Send + Sync {
    fn id(&self) -> QuantId;
    /// Does the checkpoint hold this format under `meta.prefix`?
    fn detect(&self, meta: &WeightMeta) -> bool;
    /// Load onto `backend`. Zero-copy where the backend allows (mmap'd weights).
    fn load(&self, meta: &WeightMeta, backend: &dyn Backend, opts: &QuantLoadOpts) -> Result<QTensor>;
    /// Dequantised `[n, k]` f32 weights (the oracle).
    fn dequant_ref(&self, q: &QTensor, backend: &dyn Backend) -> Result<HostTensor>;
    /// Round-to-nearest quantiser (tests, on-the-fly); `None` when the format cannot hold `t`.
    fn quantize(&self, t: &HostTensor, cfg: &QuantCfg) -> Option<HostQTensor>;
}

/// Plain dense float weights.
pub struct Bf16Dense;

impl QuantFormat for Bf16Dense {
    fn id(&self) -> QuantId {
        BF16
    }
    fn detect(&self, meta: &WeightMeta) -> bool {
        meta.info("weight").is_some_and(|w| w.dtype.is_float() && w.dtype != DType::Fp8E4m3 && w.shape.len() == 2)
            && meta.info("weight_scale").is_none()
    }
    fn load(&self, meta: &WeightMeta, backend: &dyn Backend, opts: &QuantLoadOpts) -> Result<QTensor> {
        let name = format!("{}.weight", meta.prefix);
        let w = match opts.shard {
            Some(sh) => crate::weights::load_tensor_shard(backend, meta.src, &name, sh)?,
            None => crate::weights::load_tensor(backend, meta.src, &name, None)?,
        };
        if w.shape.len() != 2 {
            return Err(Error::model(format!("'{}.weight' must be 2-D, got {:?}", meta.prefix, w.shape)));
        }
        Ok(QTensor { format: BF16, n: w.shape[0], k: w.shape[1], parts: vec![w], act_quant: false })
    }
    fn dequant_ref(&self, q: &QTensor, backend: &dyn Backend) -> Result<HostTensor> {
        Ok(HostTensor::from_f32(&[q.n, q.k], &q.download(backend)?.dequant()))
    }
    fn quantize(&self, t: &HostTensor, _cfg: &QuantCfg) -> Option<HostQTensor> {
        (t.shape.len() == 2).then(|| HostQTensor {
            format: BF16,
            n: t.shape[0],
            k: t.shape[1],
            parts: vec![t.clone()],
            act_quant: false,
        })
    }
}

/// Every known format, most specific first.
pub fn formats() -> [&'static dyn QuantFormat; 5] {
    static NV: nvfp4::Nvfp4 = nvfp4::Nvfp4;
    static MX: mxfp4::Mxfp4 = mxfp4::Mxfp4;
    static EX: exl3::Exl3 = exl3::Exl3;
    static F8: fp8::Fp8E4m3 = fp8::Fp8E4m3;
    static BF: Bf16Dense = Bf16Dense;
    [&NV, &MX, &EX, &F8, &BF]
}

pub fn format_by_id(id: &str) -> Option<&'static dyn QuantFormat> {
    formats().into_iter().find(|f| f.id() == id)
}

/// The format that holds the linear layer `meta.prefix`, or an error naming what was found.
pub fn detect(meta: &WeightMeta) -> Result<&'static dyn QuantFormat> {
    formats().into_iter().find(|f| f.detect(meta)).ok_or_else(|| {
        let have: Vec<String> = ["weight", "weight_scale", "weight_scale_2", "input_scale", "trellis"]
            .iter()
            .filter_map(|s| meta.info(s).map(|i| format!("{s}:{}{:?}", i.dtype, i.shape)))
            .collect();
        Error::Unsupported(format!("no quantisation format matches '{}' (found {})", meta.prefix, have.join(", ")))
    })
}

/// Detect and load a linear layer in one call.
pub fn load_linear(
    src: &dyn WeightSource,
    prefix: &str,
    backend: &dyn Backend,
    opts: &QuantLoadOpts,
) -> Result<QTensor> {
    let meta = WeightMeta::new(src, prefix);
    detect(&meta)?.load(&meta, backend, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::MemWeights;

    #[test]
    fn detect_each_format() {
        let mut rng = crate::rng::SplitMix64::new(1);
        let mut w = MemWeights::new();
        let nv = nvfp4::random_rows(&mut rng, 4, 32);
        w.insert("a.weight", nv.parts[0].clone());
        w.insert("a.weight_scale", nv.parts[1].clone());
        w.insert("a.weight_scale_2", HostTensor::from_f32(&[], &[0.5]));
        let f8 = fp8::random_rows(&mut rng, 4, 32, true);
        w.insert("b.weight", f8.parts[0].clone());
        w.insert("b.weight_scale", HostTensor::from_f32(&[], &[0.01]));
        w.insert("b.input_scale", HostTensor::from_f32(&[], &[0.1]));
        w.insert("c.weight", HostTensor::from_f32_as(DType::Bf16, &[4, 32], &rng.normal_vec(128, 1.0)));
        let mx = mxfp4::random_rows(&mut rng, 4, 64, Mxfp4Act::Dynamic);
        w.insert("e.weight", mx.parts[0].clone());
        w.insert("e.weight_scale", mx.parts[1].clone());
        assert_eq!(detect(&WeightMeta::new(&w, "a")).unwrap().id(), NVFP4);
        assert_eq!(detect(&WeightMeta::new(&w, "b")).unwrap().id(), FP8_E4M3);
        assert_eq!(detect(&WeightMeta::new(&w, "c")).unwrap().id(), BF16);
        assert_eq!(detect(&WeightMeta::new(&w, "e")).unwrap().id(), MXFP4);
        let ex = exl3::random(&mut rng, 128, 128, 4, exl3::Codebook::Mcg);
        w.insert("x.trellis", ex.parts[0].clone());
        w.insert("x.suh", ex.parts[1].clone());
        w.insert("x.svh", ex.parts[2].clone());
        assert_eq!(detect(&WeightMeta::new(&w, "x")).unwrap().id(), EXL3);
        assert!(detect(&WeightMeta::new(&w, "d")).is_err());
    }
}

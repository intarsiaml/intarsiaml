//! compressed-tensors `nvfp4-pack-quantized` checkpoints (`quant_method = "compressed-tensors"`, llm-compressor's
//! `NVFP4A16` preset) read as ModelOpt-style NVFP4.
//!
//! The maths is the NVFP4 W4A16 the engine already has ([`super::nvfp4`]): e2m1 codes, one e4m3 scale per 16 consecutive
//! K elements, one fp32 tensor scale. Only the tensor names and the tensor-scale convention differ, which is checked
//! against the compressed-tensors reference (`scripts/ct/make_reference.py`, fixtures in
//! `crates/intarsia-models/tests/fixtures/ct/`):
//!
//! ```text
//! compressed-tensors (per Linear module)           ModelOpt name the loader reads       relation
//! weight_packed        u8   [N, K/2]               weight                               same bytes, low nibble = element 2i
//! weight_scale         e4m3 [N, K/16]              weight_scale                         same bytes
//! weight_global_scale  f32  [1]                    weight_scale_2                       weight_scale_2 = 1 / weight_global_scale
//! ```
//!
//! compressed-tensors dequantises `w = e2m1(code) * (weight_scale / weight_global_scale)` (`_dequantize` in
//! `quantization/lifecycle/forward_helpers.py`), ModelOpt `w = e2m1(code) * weight_scale * weight_scale_2`; the two
//! tensor scales are reciprocal (`weight_global_scale = 448 * 6 / amax`, `weight_scale_2 = amax / (6 * 448)`). The engine
//! multiplies by `1 / weight_global_scale`, computed once at open in f32, where compressed-tensors divides: they differ
//! by at most one or two f32 ulp per weight.
//!
//! [`CtNvfp4`] is a [`WeightSource`] view over the checkpoint that presents each such module under the ModelOpt names
//! (`<module>.weight`, `<module>.weight_scale`, `<module>.weight_scale_2`) and hides the compressed-tensors ones, so
//! format detection, the dense loader, tensor-parallel slicing and the per-expert MoE loader (`quant/moe.rs`) run unchanged
//! and no backend sees a different layout. A module is recognised by `weight_global_scale` next to `weight_packed`;
//! `pack-quantized` (INT4, `weight_packed` + `weight_shape`, no global scale) is not touched and still fails loudly.

use crate::backend::HostMemory;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::weights::{WeightInfo, WeightSource};
use std::collections::BTreeMap;
use std::sync::Arc;

const PACKED: &str = ".weight_packed";
const GLOBAL: &str = ".weight_global_scale";

/// What a name of the view resolves to in the underlying checkpoint.
enum Entry {
    /// The same bytes under another name (`<m>.weight` -> `<m>.weight_packed`).
    Alias(String),
    /// A scalar computed at open (`<m>.weight_scale_2`, f32 little-endian).
    Scalar([u8; 4]),
    /// A compressed-tensors name the view does not expose.
    Hidden,
}

/// A [`WeightSource`] that reads compressed-tensors NVFP4A16 modules of `inner` under their ModelOpt names.
pub struct CtNvfp4<S: WeightSource> {
    inner: S,
    map: BTreeMap<String, Entry>,
    modules: usize,
}

impl<S: WeightSource> CtNvfp4<S> {
    /// Scan `inner` for compressed-tensors NVFP4 modules and check them. Errors name the first module that is not
    /// `weight_packed` u8 `[N, K/2]` + `weight_scale` e4m3 `[N, K/16]` + `weight_global_scale` a finite positive f32.
    pub fn new(inner: S) -> Result<Self> {
        let mut map = BTreeMap::new();
        let mut modules = 0;
        for name in inner.names() {
            let Some(m) = name.strip_suffix(GLOBAL) else { continue };
            let (wp, ws, wg) = (format!("{m}{PACKED}"), format!("{m}.weight_scale"), name.clone());
            let bad = |what: String| Error::model(format!("compressed-tensors NVFP4 module '{m}': {what}"));
            let pi = inner.info(&wp).ok_or_else(|| bad(format!("'{wp}' is missing next to weight_global_scale")))?;
            if pi.dtype != DType::U8 || pi.shape.len() != 2 {
                return Err(bad(format!("weight_packed must be a 2-D u8 tensor, got {} {:?}", pi.dtype, pi.shape)));
            }
            let si = inner.info(&ws).ok_or_else(|| bad("weight_scale is missing".into()))?;
            if si.dtype != DType::Fp8E4m3 || si.shape != [pi.shape[0], pi.shape[1] * 2 / super::nvfp4::GROUP] {
                return Err(bad(format!(
                    "weight_scale must be fp8_e4m3 {:?} (one per {} elements), got {} {:?}",
                    [pi.shape[0], pi.shape[1] * 2 / super::nvfp4::GROUP],
                    super::nvfp4::GROUP,
                    si.dtype,
                    si.shape
                )));
            }
            if inner.info(&format!("{m}.weight")).is_some() {
                return Err(bad("has both weight and weight_packed".into()));
            }
            let gi = inner.info(&wg).ok_or_else(|| bad("weight_global_scale vanished".into()))?;
            if gi.dtype != DType::F32 || gi.shape.iter().product::<usize>() != 1 {
                return Err(bad(format!("weight_global_scale must be a single f32, got {} {:?}", gi.dtype, gi.shape)));
            }
            let gb = inner.bytes(&wg)?;
            let g = f32::from_le_bytes([gb[0], gb[1], gb[2], gb[3]]);
            if !(g.is_finite() && g > 0.0) {
                return Err(bad(format!("weight_global_scale is {g}, expected a finite positive value")));
            }
            map.insert(format!("{m}.weight"), Entry::Alias(wp.clone()));
            map.insert(format!("{m}.weight_scale_2"), Entry::Scalar((1.0 / g).to_le_bytes()));
            map.insert(wp, Entry::Hidden);
            map.insert(wg, Entry::Hidden);
            modules += 1;
        }
        Ok(CtNvfp4 { inner, map, modules })
    }

    /// Number of modules presented as NVFP4.
    pub fn modules(&self) -> usize {
        self.modules
    }
}

impl<S: WeightSource> WeightSource for CtNvfp4<S> {
    fn names(&self) -> Vec<String> {
        let mut out: Vec<String> = self.inner.names().into_iter().filter(|n| !matches!(self.map.get(n), Some(Entry::Hidden))).collect();
        out.extend(self.map.iter().filter(|(_, e)| !matches!(e, Entry::Hidden)).map(|(n, _)| n.clone()));
        out
    }
    fn info(&self, name: &str) -> Option<WeightInfo> {
        match self.map.get(name) {
            None => self.inner.info(name),
            Some(Entry::Alias(to)) => self.inner.info(to),
            Some(Entry::Scalar(_)) => Some(WeightInfo { dtype: DType::F32, shape: vec![1] }),
            Some(Entry::Hidden) => None,
        }
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        match self.map.get(name) {
            None => self.inner.bytes(name),
            Some(Entry::Alias(to)) => self.inner.bytes(to),
            Some(Entry::Scalar(b)) => Ok(b),
            Some(Entry::Hidden) => Err(Error::model(format!("tensor '{name}' not found"))),
        }
    }
    fn host_memory(&self, name: &str) -> Option<(Arc<dyn HostMemory>, usize, usize)> {
        match self.map.get(name) {
            None => self.inner.host_memory(name),
            Some(Entry::Alias(to)) => self.inner.host_memory(to),
            Some(_) => None,
        }
    }
    fn quant_scheme(&self) -> Option<&super::scheme::QuantScheme> {
        self.inner.quant_scheme()
    }
}

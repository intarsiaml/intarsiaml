//! Weight sources: where checkpoint tensors come from. `intarsia-models`
//! implements this for (sharded) safetensors; tests use [`MemWeights`].

use crate::backend::{upload_new, Backend, HostMemory, TensorRef};
use crate::dtype::DType;
use crate::error::{Error, Result};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct WeightInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

pub trait WeightSource: Send + Sync {
    fn names(&self) -> Vec<String>;
    fn info(&self, name: &str) -> Option<WeightInfo>;
    /// Raw little-endian bytes of a tensor.
    fn bytes(&self, name: &str) -> Result<&[u8]>;
    /// Backing memory for zero-copy import: (memory, offset, len).
    fn host_memory(&self, _name: &str) -> Option<(Arc<dyn HostMemory>, usize, usize)> {
        None
    }
    /// The quantisation scheme the checkpoint declares (`hf_quant_config.json`), when the source knows it.
    /// `Model::load` checks every linear layer against it.
    fn quant_scheme(&self) -> Option<&crate::quant::scheme::QuantScheme> {
        None
    }
}

/// Load one named tensor onto the backend (zero-copy when the backend allows it).
pub fn load_tensor(
    backend: &dyn Backend,
    src: &dyn WeightSource,
    name: &str,
    expect_shape: Option<&[usize]>,
) -> Result<TensorRef> {
    let info = src.info(name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
    if let Some(s) = expect_shape {
        if info.shape != s {
            return Err(Error::model(format!("tensor '{name}' has shape {:?}, expected {s:?}", info.shape)));
        }
    }
    let nbytes = info.shape.iter().product::<usize>() * info.dtype.size_bytes();
    if let Some((mem, off, len)) = src.host_memory(name) {
        if len == nbytes {
            if let Some(buf) = backend.wrap_host(mem, off, len) {
                return Ok(TensorRef::new(buf, info.dtype, &info.shape));
            }
        }
    }
    let bytes = src.bytes(name)?;
    if bytes.len() != nbytes {
        return Err(Error::model(format!("tensor '{name}': byte length mismatch")));
    }
    let buf = upload_new(backend, bytes)?;
    Ok(TensorRef::new(buf, info.dtype, &info.shape))
}

/// In-memory weights (tests, tiny synthetic models).
#[derive(Default)]
pub struct MemWeights {
    map: BTreeMap<String, (WeightInfo, Vec<u8>)>,
}

impl MemWeights {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, name: &str, t: crate::tensor::HostTensor) {
        self.map.insert(name.to_string(), (WeightInfo { dtype: t.dtype, shape: t.shape.clone() }, t.data));
    }
}

impl WeightSource for MemWeights {
    fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
    fn info(&self, name: &str) -> Option<WeightInfo> {
        self.map.get(name).map(|(i, _)| i.clone())
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.map.get(name).map(|(_, b)| b.as_slice()).ok_or_else(|| Error::model(format!("tensor '{name}' not found")))
    }
}

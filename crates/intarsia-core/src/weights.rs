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

/// Which axis of a 2-D `[rows, cols]` weight a tensor-parallel rank keeps a slice of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardAxis {
    /// Output features: rows `[rank * n / world, (rank + 1) * n / world)`. Contiguous in the file.
    Rows,
    /// Input features: columns `[rank * k / world, (rank + 1) * k / world)` of every row. Strided.
    Cols,
}

/// One rank's slice of a weight (`world` equal parts along `axis`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    pub axis: ShardAxis,
    pub rank: usize,
    pub world: usize,
}

impl Shard {
    /// Shape of the slice of a `[n, k]` weight.
    pub fn local_shape(&self, n: usize, k: usize) -> (usize, usize) {
        match self.axis {
            ShardAxis::Rows => (n / self.world, k),
            ShardAxis::Cols => (n, k / self.world),
        }
    }
}

/// Load this rank's slice of the 2-D tensor `name`. Only the slice is uploaded, so the device never holds the
/// other ranks' parts. `Rows` slices are a contiguous byte range (zero-copy where the backend allows);
/// `Cols` slices are gathered row by row into a host staging buffer first.
pub fn load_tensor_shard(backend: &dyn Backend, src: &dyn WeightSource, name: &str, shard: Shard) -> Result<TensorRef> {
    let info = src.info(name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
    if info.shape.len() != 2 {
        return Err(Error::model(format!("tensor '{name}' is {:?}; only 2-D weights can be sharded", info.shape)));
    }
    let (n, k) = (info.shape[0], info.shape[1]);
    let split = if shard.axis == ShardAxis::Rows { n } else { k };
    if shard.world == 0 || shard.rank >= shard.world || split % shard.world != 0 {
        return Err(Error::model(format!(
            "tensor '{name}' [{n}, {k}] cannot be split into {} equal parts along its {} (rank {})",
            shard.world,
            if shard.axis == ShardAxis::Rows { "rows" } else { "columns" },
            shard.rank
        )));
    }
    let esz = info.dtype.size_bytes();
    let (ln, lk) = shard.local_shape(n, k);
    let bytes = src.bytes(name)?;
    if bytes.len() != n * k * esz {
        return Err(Error::model(format!("tensor '{name}': byte length mismatch")));
    }
    let local = match shard.axis {
        ShardAxis::Rows => {
            let off = shard.rank * ln * k * esz;
            let len = ln * k * esz;
            if let Some((mem, moff, mlen)) = src.host_memory(name) {
                if mlen == bytes.len() {
                    if let Some(buf) = backend.wrap_host(mem, moff + off, len) {
                        return Ok(TensorRef::new(buf, info.dtype, &[ln, lk]));
                    }
                }
            }
            upload_new(backend, &bytes[off..off + len])?
        }
        ShardAxis::Cols => {
            let mut stage = Vec::with_capacity(ln * lk * esz);
            let (c0, c1) = (shard.rank * lk * esz, (shard.rank + 1) * lk * esz);
            for r in 0..n {
                stage.extend_from_slice(&bytes[r * k * esz + c0..r * k * esz + c1]);
            }
            upload_new(backend, &stage)?
        }
    };
    Ok(TensorRef::new(local, info.dtype, &[ln, lk]))
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

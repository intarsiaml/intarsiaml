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

/// Decode a float tensor's raw little-endian bytes (bf16 / f16 / f32) to f32 on the host.
pub fn to_f32_vec(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>> {
    if !matches!(dtype, DType::Bf16 | DType::F16 | DType::F32) {
        return Err(Error::invalid(format!("to_f32_vec of {dtype}")));
    }
    let mut out = vec![0f32; bytes.len() / dtype.size_bytes()];
    crate::dtype::decode_to_f32(dtype, bytes, &mut out);
    Ok(out)
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
    /// Output features stored as consecutive groups of `g[0]`, `g[1]`, `g[2]` rows (GatedDeltaNet's fused
    /// `[q | k | v]` projection and conv channels): each group is split into `world` parts on its own and the rank's
    /// parts are concatenated in group order. A zero-sized group is allowed.
    RowGroups([usize; 3]),
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
            ShardAxis::Rows | ShardAxis::RowGroups(_) => (n / self.world, k),
            ShardAxis::Cols => (n, k / self.world),
        }
    }

    /// `(first row, rows)` of this rank's pieces of an `n`-row weight, in the order they are concatenated. `None` for
    /// `Cols`, or when the rows do not split evenly (or the groups do not add up to `n`).
    pub fn row_ranges(&self, n: usize) -> Option<Vec<(usize, usize)>> {
        let w = self.world;
        if w == 0 || self.rank >= w {
            return None;
        }
        match self.axis {
            ShardAxis::Rows => (n % w == 0).then(|| vec![(self.rank * n / w, n / w)]),
            ShardAxis::RowGroups(g) => {
                if g.iter().sum::<usize>() != n || g.iter().any(|x| x % w != 0) {
                    return None;
                }
                let mut out = Vec::new();
                let mut base = 0;
                for x in g {
                    if x > 0 {
                        out.push((base + self.rank * x / w, x / w));
                    }
                    base += x;
                }
                Some(out)
            }
            ShardAxis::Cols => None,
        }
    }
}

/// This rank's slice of a row-major `[n, k]` matrix of `esz`-byte elements, as host bytes (`name` is for errors).
pub fn shard_bytes(bytes: &[u8], n: usize, k: usize, esz: usize, shard: Shard, name: &str) -> Result<Vec<u8>> {
    if bytes.len() != n * k * esz {
        return Err(Error::model(format!("tensor '{name}': byte length mismatch")));
    }
    let row = k * esz;
    if shard.axis == ShardAxis::Cols {
        if shard.world == 0 || shard.rank >= shard.world || k % shard.world != 0 {
            return Err(Error::model(format!(
                "tensor '{name}' [{n}, {k}] cannot be split into {} equal parts along its columns (rank {})",
                shard.world, shard.rank
            )));
        }
        let lk = k / shard.world;
        let (c0, c1) = (shard.rank * lk * esz, (shard.rank + 1) * lk * esz);
        let mut stage = Vec::with_capacity(n * lk * esz);
        for r in 0..n {
            stage.extend_from_slice(&bytes[r * row + c0..r * row + c1]);
        }
        return Ok(stage);
    }
    let ranges = shard.row_ranges(n).ok_or_else(|| {
        Error::model(format!(
            "tensor '{name}' with {n} rows cannot be split {:?} into {} equal parts (rank {})",
            shard.axis, shard.world, shard.rank
        ))
    })?;
    let mut stage = Vec::with_capacity(n / shard.world * row);
    for (r0, rn) in ranges {
        stage.extend_from_slice(&bytes[r0 * row..(r0 + rn) * row]);
    }
    Ok(stage)
}

/// Load this rank's slice of the tensor `name`. Only the slice is uploaded, so the device never holds the other ranks'
/// parts. `Cols` needs a 2-D tensor and gathers row by row through a host staging buffer. The row splits (`Rows`,
/// `RowGroups`) take any tensor of at least one dimension and split its first one, keeping the trailing ones (a 1-D
/// `[n]` vector, a conv `[c, 1, taps]`); a `Rows` slice is one contiguous byte range (zero-copy where the backend
/// allows), `RowGroups` pieces are staged.
pub fn load_tensor_shard(backend: &dyn Backend, src: &dyn WeightSource, name: &str, shard: Shard) -> Result<TensorRef> {
    let info = src.info(name).ok_or_else(|| Error::model(format!("tensor '{name}' not found in checkpoint")))?;
    let cols = shard.axis == ShardAxis::Cols;
    if info.shape.is_empty() || (cols && info.shape.len() != 2) {
        return Err(Error::model(format!("tensor '{name}' is {:?}; it cannot be split along {:?}", info.shape, shard.axis)));
    }
    let n = info.shape[0];
    let k: usize = info.shape[1..].iter().product();
    let ok = shard.world > 0 && shard.rank < shard.world && if cols { k % shard.world == 0 } else { shard.row_ranges(n).is_some() };
    if !ok {
        return Err(Error::model(format!(
            "tensor '{name}' {:?} cannot be split into {} equal parts along {:?} (rank {})",
            info.shape, shard.world, shard.axis, shard.rank
        )));
    }
    let esz = info.dtype.size_bytes();
    let mut shape = info.shape.clone();
    shape[usize::from(cols)] /= shard.world;
    let bytes = src.bytes(name)?;
    if bytes.len() != n * k * esz {
        return Err(Error::model(format!("tensor '{name}': byte length mismatch")));
    }
    if shard.axis == ShardAxis::Rows {
        let len = n / shard.world * k * esz;
        let off = shard.rank * len;
        if let Some((mem, moff, mlen)) = src.host_memory(name) {
            if mlen == bytes.len() {
                if let Some(buf) = backend.wrap_host(mem, moff + off, len) {
                    return Ok(TensorRef::new(buf, info.dtype, &shape));
                }
            }
        }
        return Ok(TensorRef::new(upload_new(backend, &bytes[off..off + len])?, info.dtype, &shape));
    }
    let stage = shard_bytes(bytes, n, k, esz, shard, name)?;
    Ok(TensorRef::new(upload_new(backend, &stage)?, info.dtype, &shape))
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

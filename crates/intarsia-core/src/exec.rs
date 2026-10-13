//! Run one op on a backend from host tensors (used by tests and the admission harness).

use crate::backend::{Backend, MemKind, TensorRef};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{Op, OpAttrs, OpCall};
use crate::tensor::HostTensor;

#[derive(Clone, Debug)]
pub struct OutSpec {
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Initial contents (in-place outputs such as the KV cache); zero otherwise.
    pub init: Option<HostTensor>,
}

impl OutSpec {
    pub fn new(dtype: DType, shape: &[usize]) -> OutSpec {
        OutSpec { dtype, shape: shape.to_vec(), init: None }
    }
}

/// A fully host-side description of an op call.
#[derive(Clone, Debug)]
pub struct HostCall {
    pub op: Op,
    pub inputs: Vec<HostTensor>,
    pub outputs: Vec<OutSpec>,
    pub attrs: OpAttrs,
}

fn put(backend: &dyn Backend, t: &HostTensor) -> Result<TensorRef> {
    // never allocate zero bytes: some drivers reject it
    let buf = backend.alloc(t.nbytes().max(16), MemKind::Device)?;
    backend.zero(&buf, 0)?;
    if t.nbytes() > 0 {
        backend.upload(&buf, 0, &t.data, 0)?;
    }
    Ok(TensorRef::new(buf, t.dtype, &t.shape))
}

/// Upload inputs, launch `kernel`, synchronise, download outputs.
pub fn run_host(backend: &dyn Backend, kernel: &str, hc: &HostCall) -> Result<Vec<HostTensor>> {
    let inputs: Vec<TensorRef> = hc.inputs.iter().map(|t| put(backend, t)).collect::<Result<_>>()?;
    let mut outputs = Vec::new();
    for o in &hc.outputs {
        let t = match &o.init {
            Some(init) => init.clone(),
            None => HostTensor::zeros(o.dtype, &o.shape),
        };
        outputs.push(put(backend, &t)?);
    }
    let call = OpCall::new(hc.op, inputs, outputs.clone(), hc.attrs.clone());
    call.check()?;
    backend.launch(kernel, &call, 0)?;
    backend.synchronize(0)?;
    let mut res = Vec::new();
    for (o, spec) in outputs.iter().zip(&hc.outputs) {
        let mut data = vec![0u8; o.nbytes()];
        backend.download(&o.buf, o.offset, &mut data, 0)?;
        res.push(HostTensor::new(spec.dtype, spec.shape.clone(), data)?);
    }
    Ok(res)
}

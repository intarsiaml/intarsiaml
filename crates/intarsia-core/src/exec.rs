//! Run one op on a backend from host tensors (used by tests and the admission harness).

use crate::backend::{Backend, MemKind, TensorRef};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{Op, OpAttrs, OpCall, ShapeInfo};
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
    run_host_in(backend, kernel, hc, false)
}

/// [`run_host`] with every dense fp8 weight the backend can store in fragment order (2-D e4m3 input,
/// `quant::fp8::frag_eligible`) rearranged first (`Backend::relayout_fp8_frag`): runs `kernel` on tagged weights (lane wg, HIP only).
pub fn run_host_frag(backend: &dyn Backend, kernel: &str, hc: &HostCall) -> Result<Vec<HostTensor>> {
    run_host_in(backend, kernel, hc, true)
}

fn run_host_in(backend: &dyn Backend, kernel: &str, hc: &HostCall, frag: bool) -> Result<Vec<HostTensor>> {
    let inputs: Vec<TensorRef> = hc.inputs.iter().map(|t| put(backend, t)).collect::<Result<_>>()?;
    if frag {
        for t in &inputs {
            if t.dtype == DType::Fp8E4m3 && t.shape.len() == 2 && crate::quant::fp8::frag_eligible(t.shape[0], t.shape[1]) {
                backend.relayout_fp8_frag(t)?;
            }
        }
    }
    let mut outputs = Vec::new();
    for o in &hc.outputs {
        let t = match &o.init {
            Some(init) => init.clone(),
            None => HostTensor::zeros(o.dtype, &o.shape),
        };
        outputs.push(put(backend, &t)?);
    }
    let mut call = OpCall::new(hc.op, inputs, outputs.clone(), hc.attrs.clone());
    call.check()?;
    // kernels that declare a scratch requirement get a fresh workspace of exactly that size
    if let Some(k) = backend.kernels().iter().find(|k| k.name == kernel && k.op == hc.op) {
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            let buf = backend.alloc(need.max(16), MemKind::Device)?;
            call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
        }
    }
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

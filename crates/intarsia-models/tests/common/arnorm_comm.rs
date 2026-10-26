//! A host-staged communicator that "fuses" the all-reduce with the residual add + rmsnorm after it (`Comm::all_reduce_add_rmsnorm`)
//! the way the semantics are defined: the reduction, then the backend's own `RmsNormAdd` kernel on its result. Bit for bit
//! the unfused pair by construction, so a model run with it must equal the run without it; what the tests check is the
//! model's wiring (which all-reduces defer and fuse, which flush, the plan, the fallbacks). It counts the calls it fused
//! and the ones it refused, so a test cannot pass vacuously.

use intarsia_core::backend::{Backend, StreamId, TensorRef};
use intarsia_core::comm::{ArNorm, Comm, CommStats, HostCommGroup};
use intarsia_core::{Error, Op, OpAttrs, OpCall, Registry, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct ArNormComm {
    inner: Arc<dyn Comm>,
    backend: Arc<dyn Backend>,
    registry: Registry,
    /// Refuse (`ar_norm_supported` false) a payload of more elements than this: the model must then issue the reduction and
    /// the norm separately, as for a prefill chunk above the HIP scratch slot.
    max_elems: usize,
    /// Fused collectives run.
    pub fused: AtomicUsize,
    /// Rows of the `[rows, H]` activations the fused collectives took, in call order.
    pub rows: std::sync::Mutex<Vec<usize>>,
}

impl ArNormComm {
    /// One fusing communicator per backend over a host-staged group; the `Arc<dyn Comm>`s go to `Model::load_tp`.
    pub fn group(backends: Vec<Arc<dyn Backend>>, max_elems: usize) -> (Vec<Arc<dyn Comm>>, Vec<Arc<ArNormComm>>) {
        let inner = HostCommGroup::new(backends.clone());
        let mine: Vec<Arc<ArNormComm>> = inner
            .into_iter()
            .zip(backends)
            .map(|(inner, backend)| {
                let registry = Registry::new(backend.name(), &backend.info(), backend.kernels());
                Arc::new(ArNormComm { inner, backend, registry, max_elems, fused: AtomicUsize::new(0), rows: Default::default() })
            })
            .collect();
        let comms = mine.iter().map(|c| c.clone() as Arc<dyn Comm>).collect();
        (comms, mine)
    }

    fn call(&self, a: &ArNorm<'_>) -> OpCall {
        let attrs = OpAttrs { eps: a.eps, w_offset: a.w_offset, ..Default::default() };
        OpCall::new(
            Op::RmsNormAdd,
            vec![a.out.clone(), a.residual.clone(), a.weight.clone()],
            vec![a.norm.clone(), a.sum.clone()],
            attrs,
        )
    }
}

impl Comm for ArNormComm {
    fn name(&self) -> &str {
        "host+arnorm"
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world(&self) -> usize {
        self.inner.world()
    }
    fn all_reduce_sum(&self, x: &TensorRef, stream: StreamId) -> Result<()> {
        self.inner.all_reduce_sum(x, stream)
    }
    fn all_reduce_sum_into(&self, partial: &TensorRef, out: &TensorRef, stream: StreamId) -> Result<()> {
        self.inner.all_reduce_sum_into(partial, out, stream)
    }
    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()> {
        self.inner.all_gather(send, recv, stream)
    }
    fn abort(&self) {
        self.inner.abort()
    }
    fn stats(&self) -> CommStats {
        self.inner.stats()
    }
    fn note_timed(&self, d: Duration) {
        self.inner.note_timed(d)
    }

    fn fuses_ar_norm(&self) -> bool {
        true
    }
    fn ar_norm_supported(&self, a: &ArNorm<'_>) -> bool {
        a.partial.numel() <= self.max_elems && a.out.numel() == a.partial.numel() && self.call(a).check().is_ok()
    }
    fn all_reduce_add_rmsnorm(&self, a: &ArNorm<'_>, stream: StreamId) -> Result<()> {
        if !self.ar_norm_supported(a) {
            return Err(Error::Unsupported("host+arnorm: unsupported operands".into()));
        }
        self.inner.all_reduce_sum_into(a.partial, a.out, stream)?;
        let call = self.call(a);
        let k = self.registry.select(&call)?;
        self.backend.launch(&k.name, &call, stream)?;
        self.fused.fetch_add(1, Ordering::Relaxed);
        self.rows.lock().unwrap().push(a.partial.numel() / a.weight.numel());
        Ok(())
    }
}

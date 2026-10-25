//! GPU-timestamp trace of the stream, for decode / prefill profiles. Off unless `INTARSIA_METAL_PROFILE` is set to
//! `kernel:<file>` or `cb:<file>`; the trace is JSON lines, one per committed command buffer, written when the buffers
//! are waited on (`synchronize`) and at drop. Times are the command buffer's `GPUStartTime` / `GPUEndTime` in seconds on
//! one clock, so gaps between records are GPU idle time. `ho` / `hc` / `hr` are the host times on that clock at which
//! the buffer's first dispatch started to be encoded and `commit` was entered / returned, to tell host stalls from GPU work.
//!
//!  * `kernel`: every launch is committed as its own command buffer and waited on before the next one is encoded, so
//!    records never overlap. Each is labelled with the kernel, the weight shape and weight bytes of matmul-like ops, and
//!    the leading dim of the first input (`t`: 1 for a decode step). A record's duration is the kernel plus Metal's
//!    fixed per-command-buffer start / end cost (a few us). Use it for the per-op table; host time between records is
//!    submit latency, not a property of the engine.
//!  * `cb`: the stream runs as normal (one command buffer per `MAX_DISPATCHES` / `synchronize`); each record is one
//!    committed command buffer with its dispatch count. Use it for step time and GPU idle gaps, which `kernel` distorts.
//!
//! This is a diagnostic: it changes how work is submitted (`kernel`) or only observes it (`cb`), never what runs.

use crate::CommandBuffer;
use intarsia_core::backend::TensorRef;
use intarsia_core::{Op, OpCall};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use std::io::Write;

/// What a launch is, attached to the command buffer that carries it.
#[derive(Clone, Debug, Default)]
pub struct Label {
    pub op: String,
    /// Leading dim of the first input (tokens of a row op, M of a matmul).
    pub t: usize,
    pub n: usize,
    pub k: usize,
    /// Weight bytes a matmul-like op streams.
    pub bytes: u64,
}

impl Label {
    pub fn of(kernel: &str, call: &OpCall) -> Label {
        let mut l = Label { op: kernel.to_string(), t: call.inputs.first().and_then(|t| t.shape.first().copied()).unwrap_or(0), ..Default::default() };
        if matches!(call.op, Op::Matmul | Op::Logits | Op::MatmulFp8 | Op::MatmulNvfp4) {
            if let Some(w) = call.inputs.get(1).filter(|w| w.shape.len() == 2) {
                (l.n, l.k, l.bytes) = (w.shape[0], w.shape[1], weight_bytes(w));
            }
        }
        if call.op == Op::MatmulPair {
            // both weight matrices: n is the sum of their rows
            if let [_, w0, w1] = call.inputs.as_slice() {
                (l.n, l.k, l.bytes) = (w0.shape[0] + w1.shape[0], w0.shape[1], weight_bytes(w0) + weight_bytes(w1));
            }
        }
        l
    }
}

fn weight_bytes(w: &TensorRef) -> u64 {
    (w.shape.iter().product::<usize>() * w.dtype.size_bytes()) as u64
}

struct Rec {
    seq: u64,
    label: Option<Label>,
    dispatches: usize,
    cb: CommandBuffer,
    host: Host,
}

/// Host-side times of one command buffer, seconds on the clock of `GPUStartTime` (see [`host_now`]): when its first
/// dispatch started to be encoded, and when `commit` was entered and returned.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Host {
    pub open: f64,
    pub commit: f64,
    pub committed: f64,
}

extern "C" {
    fn clock_gettime_nsec_np(clock_id: u32) -> u64;
}

/// Now on the clock command buffer `GPUStartTime` / `GPUEndTime` use (`CLOCK_UPTIME_RAW`, mach absolute time in seconds).
pub(crate) fn host_now() -> f64 {
    const CLOCK_UPTIME_RAW: u32 = 8;
    // SAFETY: plain libc call without pointers.
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) as f64 * 1e-9 }
}

pub(crate) struct Prof {
    pub per_kernel: bool,
    out: std::io::BufWriter<std::fs::File>,
    seq: u64,
    pending: Vec<Rec>,
}

impl Prof {
    pub fn from_env() -> Option<Prof> {
        let v = std::env::var("INTARSIA_METAL_PROFILE").ok()?;
        let (mode, path) = v.split_once(':')?;
        let per_kernel = match mode {
            "kernel" => true,
            "cb" => false,
            _ => {
                eprintln!("INTARSIA_METAL_PROFILE: expected kernel:<file> or cb:<file>, got '{v}'; profiling off");
                return None;
            }
        };
        match std::fs::File::create(path) {
            Ok(f) => Some(Prof { per_kernel, out: std::io::BufWriter::new(f), seq: 0, pending: Vec::new() }),
            Err(e) => {
                eprintln!("INTARSIA_METAL_PROFILE: cannot create {path}: {e}; profiling off");
                None
            }
        }
    }

    /// A command buffer was committed; its times are read by [`Self::resolve`] once it has completed.
    pub fn note(&mut self, cb: CommandBuffer, label: Option<Label>, dispatches: usize, host: Host) {
        self.pending.push(Rec { seq: self.seq, label, dispatches, cb, host });
        self.seq += 1;
    }

    /// Write the records of every completed command buffer; incomplete ones stay pending.
    pub fn resolve(&mut self) {
        let mut keep = Vec::new();
        for r in std::mem::take(&mut self.pending) {
            if r.cb.status() != MTLCommandBufferStatus::Completed {
                keep.push(r);
                continue;
            }
            let l = r.label.clone().unwrap_or_default();
            let _ = writeln!(
                self.out,
                "{{\"i\":{},\"op\":\"{}\",\"t\":{},\"n\":{},\"k\":{},\"bytes\":{},\"nd\":{},\"s\":{:.9},\"e\":{:.9},\"ho\":{:.9},\"hc\":{:.9},\"hr\":{:.9}}}",
                r.seq,
                if l.op.is_empty() { "other" } else { &l.op },
                l.t,
                l.n,
                l.k,
                l.bytes,
                r.dispatches,
                r.cb.GPUStartTime(),
                r.cb.GPUEndTime(),
                r.host.open,
                r.host.commit,
                r.host.committed
            );
        }
        self.pending = keep;
        let _ = self.out.flush();
    }
}

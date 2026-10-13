//! cpu-ref: the always-on reference backend.
//!
//! Two kernel sets are registered for every op:
//!  * `ref_<op>`: naive f64 implementations (`reference.rs`), flagged
//!    `is_reference`; the oracle all other implementations are graded against.
//!  * `cpu_f32_<op>`: rayon f32 implementations (`fast.rs`) used by the engine.
//!
//! Memory is plain host RAM; weights from mmap'd files are used zero-copy via
//! `Backend::wrap_host`.

mod fast;
mod fault;
mod fused_ref;
mod gdn_ref;
mod mem;
mod quant_ref;
mod reference;

use intarsia_core::backend::{Backend, Buffer, DeviceInfo, HostMemory, MemKind, StreamId};
use intarsia_core::{Error, Exactness, KernelImpl, Op, OpCall, Result, Tol};
use mem::CpuBuffer;
use std::sync::{Arc, Mutex};

pub struct CpuBackend {
    sys: Mutex<sysinfo::System>,
    brand: String,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    pub fn new() -> CpuBackend {
        let mut sys = sysinfo::System::new();
        sys.refresh_cpu_all();
        let brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|b| !b.is_empty())
            .unwrap_or_else(|| "host cpu".into());
        CpuBackend { sys: Mutex::new(sys), brand }
    }

    pub fn arc() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }
}

/// Memory a new allocation can use. Linux: sysinfo's `MemAvailable`, unchanged. macOS: sysinfo reports only the free
/// page list, which the OS keeps tiny on purpose (file cache and compressible memory fill the rest), so use
/// free + inactive + purgeable + speculative from `host_statistics64`, never below sysinfo's figure.
fn available_memory(sys: &sysinfo::System) -> u64 {
    let base = sys.available_memory();
    #[cfg(target_os = "macos")]
    {
        return base.max(macos_reclaimable_bytes().unwrap_or(0));
    }
    #[cfg(not(target_os = "macos"))]
    base
}

#[cfg(target_os = "macos")]
fn macos_reclaimable_bytes() -> Option<u64> {
    // SAFETY: host_statistics64 fills a vm_statistics64 of `count` natural_t words; zeroed is a valid initial value.
    unsafe {
        let mut vm: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::vm_statistics64>() / std::mem::size_of::<libc::integer_t>())
            as libc::mach_msg_type_number_t;
        #[allow(deprecated)]
        let host = libc::mach_host_self();
        let rc = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut vm as *mut _ as libc::host_info64_t,
            &mut count,
        );
        if rc != 0 {
            return None;
        }
        let pages = vm.free_count as u64 + vm.inactive_count as u64 + vm.purgeable_count as u64 + vm.speculative_count as u64;
        Some(pages * libc::vm_page_size as u64)
    }
}

fn tol_for(op: Op) -> Tol {
    match op {
        // pure data movement / integer results: must match bit for bit
        Op::Embedding | Op::KvAppend | Op::Argmax | Op::Sample => Tol::EXACT,
        // bf16-rounded outputs: one-ulp flips allowed
        _ => Tol::new(1.5e-2, 6e-3),
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn info(&self) -> DeviceInfo {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        DeviceInfo {
            name: format!("cpu-ref ({})", self.brand),
            backend: "cpu".into(),
            arch: "any".into(),
            total_mem: sys.total_memory(),
            // INTARSIA_CPU_FREE_MEM_GB overrides the measured value. On macOS sysinfo's available_memory() reports
            // only the tiny truly-free page list (under 2 GB on a 128 GB machine), so `available_memory` adds the
            // reclaimable pages there.
            free_mem: std::env::var("INTARSIA_CPU_FREE_MEM_GB")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .map(|gb| (gb * (1u64 << 30) as f64) as u64)
                .unwrap_or_else(|| available_memory(&sys)),
            unified: true,
            bandwidth_gbs: None,
        }
    }

    fn alloc(&self, bytes: usize, _kind: MemKind) -> Result<Buffer> {
        Ok(Buffer::new(CpuBuffer::zeroed(bytes)))
    }

    fn upload(&self, dst: &Buffer, dst_offset: usize, src: &[u8], _s: StreamId) -> Result<()> {
        dst.downcast::<CpuBuffer>().ok_or_else(|| Error::backend("foreign buffer"))?.write(dst_offset, src)
    }

    fn download(&self, src: &Buffer, src_offset: usize, dst: &mut [u8], _s: StreamId) -> Result<()> {
        src.downcast::<CpuBuffer>().ok_or_else(|| Error::backend("foreign buffer"))?.read(src_offset, dst)
    }

    fn copy(
        &self,
        dst: &Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
        _s: StreamId,
    ) -> Result<()> {
        let mut tmp = vec![0u8; bytes];
        self.download(src, src_offset, &mut tmp, 0)?;
        self.upload(dst, dst_offset, &tmp, 0)
    }

    fn wrap_host(&self, mem: Arc<dyn HostMemory>, offset: usize, len: usize) -> Option<Buffer> {
        Some(Buffer::new(CpuBuffer::host(mem, offset, len)))
    }

    fn kernels(&self) -> Vec<KernelImpl> {
        let mut v = Vec::new();
        for op in Op::ALL {
            v.push(
                KernelImpl::new(op, format!("ref_{}", op.as_str()), "cpu")
                    .exact(Exactness::Exact)
                    .tol(Tol::EXACT)
                    .reference(true)
                    .priority(-100),
            );
            v.push(
                KernelImpl::new(op, format!("cpu_f32_{}", op.as_str()), "cpu")
                    .exact(if tol_for(op).max_abs == 0.0 { Exactness::Exact } else { Exactness::Deterministic })
                    .tol(tol_for(op))
                    .priority(0),
            );
        }
        v
    }

    fn launch(&self, kernel: &str, call: &OpCall, _s: StreamId) -> Result<()> {
        call.check()?;
        let (reference, name) = if let Some(rest) = kernel.strip_prefix("ref_") {
            (true, rest)
        } else if let Some(rest) = kernel.strip_prefix("cpu_f32_") {
            (false, rest)
        } else {
            return Err(Error::NoKernel(format!("cpu kernel '{kernel}'")));
        };
        if name != call.op.as_str() {
            return Err(Error::invalid(format!("kernel '{kernel}' cannot run op '{}'", call.op)));
        }
        // device-side position: resolve it now (the cpu runs synchronously, so the value is current)
        let patched;
        let call = match &call.pos {
            Some(p) => {
                let mut bytes = [0u8; 4];
                self.download(&p.buf, p.offset, &mut bytes, 0)?;
                let pos0 = i32::from_le_bytes(bytes);
                if pos0 < 0 {
                    return Err(Error::invalid(format!("{}: negative device position {pos0}", call.op)));
                }
                let mut c = call.clone();
                c.attrs.pos0 = pos0 as usize;
                c.pos = None;
                c.check()?;
                patched = c;
                &patched
            }
            None => call,
        };
        if reference {
            reference::run(call)?;
        } else {
            fast::run(call)?;
        }
        fault::apply(self, call)
    }

    fn synchronize(&self, _s: StreamId) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod freemem_tests {
    use super::*;

    #[test]
    fn available_memory_is_positive_and_not_below_sysinfo() {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let old = sys.available_memory();
        let new = available_memory(&sys);
        eprintln!("sysinfo available_memory = {} MiB, planner figure = {} MiB", old >> 20, new >> 20);
        assert!(new > 0);
        assert!(new >= old);
        assert!(new <= sys.total_memory());
    }
}

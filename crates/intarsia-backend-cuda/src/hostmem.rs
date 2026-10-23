//! Free memory of a unified-memory (integrated) CUDA device.
//!
//! On the DGX Spark (GB10) the GPU allocates from the same LPDDR as the host, and `cudaMemGetInfo` reports the kernel's
//! `MemFree`: pages that hold nothing at all. Page cache (the weights of the last model read from disk, other models' files)
//! is not in it, although the kernel gives it back to a `cudaMalloc` that needs the room. Measured on spark1 with 60 GB of
//! cache: `cudaMemGetInfo` free 1.9 GB after loading the 35B (weights 20.8 GB), `MemAvailable` 33 GB. Memory planning from the
//! first figure caps a 16-slot batching engine at about 1.5k tokens of KV per slot for no reason.
//!
//! So for an integrated device the free figure is `max(cudaMemGetInfo, MemAvailable)`, capped at the device total. Memory
//! held by other GPU processes (a vLLM server's 34 GB) is neither free nor reclaimable cache, so it stays out of both.
//! `INTARSIA_CUDA_FREE_MEM=cuda` restores the raw `cudaMemGetInfo` figure.

/// `MemAvailable` of a `/proc/meminfo` text, in bytes.
pub fn mem_available_bytes(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let mut it = line.split_whitespace().skip(1);
    let kb: u64 = it.next()?.parse().ok()?;
    match it.next() {
        Some("kB") | None => Some(kb * 1024),
        Some(_) => None,
    }
}

/// The free figure to plan with: the CUDA runtime's, raised to the host's `MemAvailable` on an integrated device.
pub fn effective_free(cuda_free: u64, total: u64, integrated: bool, mem_available: Option<u64>, raw: bool) -> u64 {
    if !integrated || raw {
        return cuda_free;
    }
    match mem_available {
        Some(avail) => cuda_free.max(avail.min(total)),
        None => cuda_free,
    }
}

/// Whether `INTARSIA_CUDA_FREE_MEM=cuda` asks for the raw runtime figure.
pub fn raw_requested() -> bool {
    std::env::var("INTARSIA_CUDA_FREE_MEM").is_ok_and(|v| v.trim() == "cuda")
}

/// `MemAvailable` of this host (Linux), if it can be read.
pub fn host_mem_available() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|t| mem_available_bytes(&t))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEMINFO: &str = "MemTotal:       130663236 kB\nMemFree:         1894352 kB\nMemAvailable:   33263587 kB\nBuffers:          104 kB\n";

    #[test]
    fn parses_mem_available() {
        assert_eq!(mem_available_bytes(MEMINFO), Some(33_263_587 * 1024));
        assert_eq!(mem_available_bytes("MemFree: 1 kB\n"), None);
        assert_eq!(mem_available_bytes("MemAvailable: 12 MB\n"), None);
        assert_eq!(mem_available_bytes("MemAvailable: x kB\n"), None);
    }

    #[test]
    fn integrated_devices_plan_with_the_larger_figure() {
        let total = 130_663_236u64 * 1024;
        let cuda = 1_894_352u64 * 1024;
        let avail = mem_available_bytes(MEMINFO);
        assert_eq!(effective_free(cuda, total, true, avail, false), 33_263_587 * 1024);
        // discrete device, raw knob, unreadable meminfo: the runtime's figure
        assert_eq!(effective_free(cuda, total, false, avail, false), cuda);
        assert_eq!(effective_free(cuda, total, true, avail, true), cuda);
        assert_eq!(effective_free(cuda, total, true, None, false), cuda);
        // never above the device total, never below the runtime's figure
        assert_eq!(effective_free(cuda, total, true, Some(total * 2), false), total);
        assert_eq!(effective_free(cuda, total, true, Some(1), false), cuda);
    }
}

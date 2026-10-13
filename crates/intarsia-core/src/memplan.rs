//! Memory planning from *measured* free memory (ARCHITECTURE 3.6 step 4:
//! never from configured constants - wrong pins OOM'd radiance at startup and
//! GB10's nvidia-smi reports no GPU memory).

use crate::backend::DeviceInfo;
use crate::error::{Error, Result};

#[derive(Clone, Debug, serde::Serialize)]
pub struct MemoryPlan {
    pub device: String,
    pub total_bytes: u64,
    /// Free memory measured right before the KV allocation (after weights + workspace).
    pub free_bytes: u64,
    pub weights_bytes: u64,
    pub workspace_bytes: u64,
    pub kv_bytes_per_token: u64,
    /// Fraction of `free_bytes` the KV cache may take.
    pub kv_fraction: f32,
    pub kv_budget_bytes: u64,
    /// Planned context capacity in tokens.
    pub kv_cap_tokens: usize,
    pub kv_total_bytes: u64,
    pub notes: Vec<String>,
}

impl MemoryPlan {
    /// * `requested_max_seq`: user/default bound on context length.
    /// * `max_positions`: model limit.
    pub fn compute(
        info: &DeviceInfo,
        weights_bytes: u64,
        workspace_bytes: u64,
        kv_bytes_per_token: u64,
        requested_max_seq: usize,
        max_positions: usize,
        kv_fraction: f32,
    ) -> Result<MemoryPlan> {
        let mut notes = Vec::new();
        let budget = (info.free_mem as f64 * kv_fraction as f64) as u64;
        let by_mem = if kv_bytes_per_token == 0 { usize::MAX } else { (budget / kv_bytes_per_token) as usize };
        let mut cap = requested_max_seq.min(max_positions);
        if by_mem < cap {
            notes.push(format!("context clamped from {cap} to {by_mem} tokens by measured free memory"));
            cap = by_mem;
        }
        if cap < 16 {
            return Err(Error::Memory(format!(
                "only {} MiB free ({} MiB for KV at {:.0}%): cannot hold a 16-token context",
                info.free_mem >> 20,
                budget >> 20,
                kv_fraction * 100.0
            )));
        }
        Ok(MemoryPlan {
            device: info.name.clone(),
            total_bytes: info.total_mem,
            free_bytes: info.free_mem,
            weights_bytes,
            workspace_bytes,
            kv_bytes_per_token,
            kv_fraction,
            kv_budget_bytes: budget,
            kv_cap_tokens: cap,
            kv_total_bytes: cap as u64 * kv_bytes_per_token,
            notes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(free: u64) -> DeviceInfo {
        DeviceInfo {
            name: "t".into(),
            backend: "cpu".into(),
            arch: "any".into(),
            total_mem: free * 2,
            free_mem: free,
            unified: true,
            bandwidth_gbs: None,
        }
    }

    #[test]
    fn clamps_to_free_memory() {
        let p = MemoryPlan::compute(&info(1 << 30), 0, 0, 100_000, 100_000, 40960, 0.5).unwrap();
        assert_eq!(p.kv_cap_tokens, (1u64 << 29) as usize / 100_000);
        assert!(!p.notes.is_empty());
        let p = MemoryPlan::compute(&info(1 << 40), 0, 0, 100_000, 4096, 40960, 0.5).unwrap();
        assert_eq!(p.kv_cap_tokens, 4096);
        assert!(MemoryPlan::compute(&info(1000), 0, 0, 100_000, 4096, 40960, 0.5).is_err());
    }
}

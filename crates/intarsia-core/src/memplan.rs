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
        Self::compute_budgeted(
            info,
            weights_bytes,
            workspace_bytes,
            kv_bytes_per_token,
            requested_max_seq,
            max_positions,
            kv_fraction,
            None,
        )
    }

    /// Like [`MemoryPlan::compute`] with an optional `mem_budget`: the total bytes this process
    /// may use (weights + workspace + KV). The KV budget is also capped at what remains of it.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_budgeted(
        info: &DeviceInfo,
        weights_bytes: u64,
        workspace_bytes: u64,
        kv_bytes_per_token: u64,
        requested_max_seq: usize,
        max_positions: usize,
        kv_fraction: f32,
        mem_budget: Option<u64>,
    ) -> Result<MemoryPlan> {
        let mut notes = Vec::new();
        let mut budget = (info.free_mem as f64 * kv_fraction as f64) as u64;
        if let Some(total) = mem_budget {
            let left = total.saturating_sub(weights_bytes + workspace_bytes);
            if left < budget {
                notes.push(format!(
                    "KV budget capped at {} MiB by --mem-budget {} MiB (weights {} MiB + workspace {} MiB)",
                    left >> 20,
                    total >> 20,
                    weights_bytes >> 20,
                    workspace_bytes >> 20
                ));
                budget = left;
            }
        }
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

    #[test]
    fn mem_budget_caps_kv() {
        let big = info(1 << 40);
        let p = MemoryPlan::compute_budgeted(&big, 400_000, 100_000, 100, 1_000_000, 1_000_000, 0.5, Some(600_000))
            .unwrap();
        assert_eq!(p.kv_budget_bytes, 100_000);
        assert_eq!(p.kv_cap_tokens, 1000);
        assert!(p.notes.iter().any(|n| n.contains("--mem-budget")));
        // a budget below weights + workspace leaves no room for the KV
        assert!(MemoryPlan::compute_budgeted(&big, 400_000, 100_000, 100, 4096, 4096, 0.5, Some(450_000)).is_err());
        // a generous budget changes nothing
        let p = MemoryPlan::compute_budgeted(&big, 400_000, 100_000, 100, 4096, 4096, 0.5, Some(1 << 39)).unwrap();
        assert_eq!(p.kv_cap_tokens, 4096);
    }
}

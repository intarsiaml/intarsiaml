//! Deserialisable mirror of `intarsia_core::telemetry::TelemetrySnapshot` (docs/telemetry.md).
//! Every field defaults so an older or newer server still parses.

use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OpStat {
    pub calls: u64,
    pub timed_calls: u64,
    pub time_ms: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Device {
    pub name: String,
    pub backend: String,
    pub arch: String,
    pub total_mem_bytes: u64,
    pub free_mem_bytes: u64,
    pub unified: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Snapshot {
    pub schema_version: u32,
    pub timestamp_ms: u64,
    pub uptime_s: f64,
    pub model: String,
    pub device: Device,
    pub steps_total: u64,
    pub last_step_ms: u64,
    pub requests_total: u64,
    pub running: u32,
    pub waiting: u32,
    pub finished: BTreeMap<String, u64>,
    pub prompt_tokens_total: u64,
    pub generation_tokens_total: u64,
    pub cached_tokens_total: u64,
    pub last_prefill_tok_s: f64,
    pub last_decode_tok_s: f64,
    pub last_ttft_ms: f64,
    pub kv_used_bytes: u64,
    pub kv_total_bytes: u64,
    pub kv_used_tokens: u64,
    pub kv_capacity_tokens: u64,
    pub weights_bytes: u64,
    pub ops: BTreeMap<String, OpStat>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_doc_example_and_partial() {
        let s: Snapshot = serde_json::from_str(
            r#"{"schema_version":1,"model":"m","device":{"name":"GB10"},"running":1,
                "ops":{"matmul":{"calls":4,"timed_calls":2,"time_ms":3.5}},"future_field":1}"#,
        )
        .unwrap();
        assert_eq!(s.running, 1);
        assert_eq!(s.ops["matmul"].timed_calls, 2);
        assert_eq!(s.device.name, "GB10");
    }
}

//! `INTARSIA_CUDA_DISABLE=name1,name2` hides kernels from the registry (A/B and bisecting). Names that match no
//! registered kernel used to be ignored silently, so a typo recorded a "before" run that still used the new kernels
//! (review finding 1). Now [`check_env`] rejects them when the device is opened.

use crate::kernels;

/// Names in `list` (comma separated) that are not in `known`, in order.
pub fn unknown_names<'a>(list: &'a str, known: &[String]) -> Vec<&'a str> {
    list.split(',').map(str::trim).filter(|n| !n.is_empty() && !known.iter().any(|k| k == n)).collect()
}

/// Error text for a disable list with unknown names, or `None` when every name is a registered kernel.
pub fn explain(list: &str, known: &[String]) -> Option<String> {
    let bad = unknown_names(list, known);
    if bad.is_empty() {
        return None;
    }
    let mut names: Vec<&str> = known.iter().map(String::as_str).collect();
    names.sort_unstable();
    Some(format!(
        "INTARSIA_CUDA_DISABLE names no registered kernel: {} (a typo would silently leave the kernel enabled). Registered: {}",
        bad.join(", "),
        names.join(", ")
    ))
}

/// Kernels that are bitwise twins of another (the same arithmetic, faster): disabling the original also hides its twin,
/// so a disable list written before the twin existed (e.g. the accuracy envelope's `attn-simt` perturbation, which turns
/// the split-KV kernels off to run the SIMT ones) still means what it says. Disabling only the twin is possible too.
pub const TWINS: &[(&str, &str)] = &[
    ("attn_decode_bf16_splitkv", "attn_decode_bf16_splitkv_pipe"),
    ("attn_prefill_bf16_splitkv", "attn_prefill_bf16_splitkv_pipe"),
];

/// The names a disable list hides: its own names plus the bitwise twins of each.
pub fn expand(list: &str) -> Vec<String> {
    let mut off: Vec<String> = list.split(',').map(str::trim).filter(|x| !x.is_empty()).map(String::from).collect();
    for (orig, twin) in TWINS {
        if off.iter().any(|n| n == orig) && !off.iter().any(|n| n == twin) {
            off.push(twin.to_string());
        }
    }
    off
}

/// Check the environment variable against the kernel table.
pub fn check_env() -> intarsia_core::Result<()> {
    let Ok(list) = std::env::var("INTARSIA_CUDA_DISABLE") else { return Ok(()) };
    let known: Vec<String> = kernels::all().into_iter().map(|k| k.name).collect();
    match explain(&list, &known) {
        Some(msg) => Err(intarsia_core::Error::invalid(msg)),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typos_are_rejected_and_real_names_pass() {
        let known: Vec<String> = kernels::all().into_iter().map(|k| k.name).collect();
        assert!(known.len() > 10, "the table is not empty");
        let real = known[0].clone();
        assert!(explain(&real, &known).is_none());
        assert!(explain(&format!(" {real} , {real}"), &known).is_none(), "whitespace and repeats are fine");
        assert!(explain("", &known).is_none());
        let e = explain("moe_expert_mma_bf16", &known).expect("a typo must be refused");
        assert!(e.contains("moe_expert_mma_bf16") && e.contains("Registered:"), "{e}");
        assert_eq!(unknown_names("a,b,,c", &["b".to_string()]), ["a", "c"]);
    }

    /// The accuracy envelope's `attn-simt` list (crates/intarsia-cli/src/cmd/accuracy/envelope.rs, SIMT_ATTENTION_DISABLE)
    /// names the split-KV and flash kernels; with their pipelined twins also hidden, decode and short prefill fall back to
    /// the SIMT kernels as before the twins existed.
    #[test]
    fn disabling_a_kernel_hides_its_bitwise_twin() {
        let known: Vec<String> = kernels::all().into_iter().map(|k| k.name).collect();
        for (orig, twin) in TWINS {
            assert!(known.iter().any(|k| k == orig) && known.iter().any(|k| k == twin), "{orig} / {twin} registered");
        }
        let simt = "attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash";
        let off = expand(simt);
        let s = intarsia_core::ShapeInfo {
            m: 1,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            k: 256,
            kv_len: 16026,
            act: Some(intarsia_core::DType::Bf16),
            weight: Some(intarsia_core::DType::Bf16),
            ..Default::default()
        };
        let best = |op: intarsia_core::Op, m: usize| {
            let s = intarsia_core::ShapeInfo { m, ..s.clone() };
            let mut best: Option<intarsia_core::KernelImpl> = None;
            for k in kernels::all().into_iter().filter(|k| k.op == op && k.accepts(&s) && !off.contains(&k.name)) {
                if best.as_ref().is_none_or(|b| k.priority > b.priority) {
                    best = Some(k);
                }
            }
            best.map(|k| k.name)
        };
        assert_eq!(best(intarsia_core::Op::AttnDecode, 1).as_deref(), Some("attn_decode_bf16"));
        assert_eq!(best(intarsia_core::Op::AttnPrefill, 4).as_deref(), Some("attn_prefill_bf16"));
        assert_eq!(expand("attn_decode_bf16_splitkv_pipe"), ["attn_decode_bf16_splitkv_pipe"], "the twin alone");
        assert_eq!(expand(" a , attn_decode_bf16_splitkv"), ["a", "attn_decode_bf16_splitkv", "attn_decode_bf16_splitkv_pipe"]);
    }
}

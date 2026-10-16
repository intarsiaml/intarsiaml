//! `INTARSIA_HIP_DISABLE=name1,name2` hides kernels from the registry (A/B and bisecting). Names that match no
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
        "INTARSIA_HIP_DISABLE names no registered kernel: {} (a typo would silently leave the kernel enabled). Registered: {}",
        bad.join(", "),
        names.join(", ")
    ))
}

/// Check the environment variable against the kernel table.
pub fn check_env() -> intarsia_core::Result<()> {
    let Ok(list) = std::env::var("INTARSIA_HIP_DISABLE") else { return Ok(()) };
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
}

//! Lane gdn-chunk: the opt-in chunked delta rule (`INTARSIA_GDN_CHUNK=1`) has a `kernels test --quick` case too. With the knob unset its
//! predicate rejects every shape, so `quick_coverage.rs` leaves it out and THIS test checks it with the knob on. The knob is read once per
//! process, so it is set before the first registry call; this file has one test for that reason. Needs no GPU.

#[test]
fn the_chunked_delta_rule_has_a_quick_case_with_the_knob_on() {
    std::env::set_var("INTARSIA_GDN_CHUNK", "1");
    let kernels: Vec<_> = intarsia_backend_hip::kernels::all().into_iter().filter(|k| k.name == "gdn_delta_rule_bf16_chunk").collect();
    assert_eq!(kernels.len(), 1, "gdn_delta_rule_bf16_chunk is registered once");
    assert!(kernels[0].scratch.is_some(), "the chunked delta rule declares its scratch");
    let uncovered = intarsia_core::admission::kernels_without_quick_case(&kernels);
    assert!(uncovered.is_empty(), "no quick sweep case for the chunked delta rule (knob on): {uncovered:?}");
}

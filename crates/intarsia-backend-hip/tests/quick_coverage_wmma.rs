//! Lane wg: the opt-in WMMA family (`INTARSIA_HIP_WMMA_ROWS=1`) is covered by `kernels test --quick` too. With the knob unset every `*_wmma*`
//! kernel's shape predicate rejects (`wmma_takes` is false), so `quick_coverage.rs` leaves them out and THIS test checks them with the knob on and
//! every kind routed to the family (`fp8a8=all,bf16=all`). The knob is read once per process, so it is set before the first registry call; this
//! file has one test for that reason. Needs no GPU.

#[test]
fn every_wmma_kernel_has_a_quick_case_with_the_knob_on() {
    // set before the first registry call in this single-test binary (the knob is read once)
    std::env::set_var("INTARSIA_HIP_WMMA_ROWS", "1");
    std::env::set_var("INTARSIA_HIP_WMMA_TABLE", "fp8a8=all,bf16=all");
    let kernels: Vec<_> = intarsia_backend_hip::kernels::all().into_iter().filter(|k| k.name.contains("wmma")).collect();
    assert!(kernels.len() >= 10, "expected the wmma family, found {}", kernels.len());
    let uncovered = intarsia_core::admission::kernels_without_quick_case(&kernels);
    assert!(uncovered.is_empty(), "wmma kernels with no quick sweep case (knob on): {uncovered:?}");
}

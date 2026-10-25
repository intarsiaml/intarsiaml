//! Every kernel this backend registers has a case in the `kernels test --quick` sweep whose shape its
//! predicate accepts. A kernel without one reports `NO CASES (untested)` instead of being checked. Needs no
//! GPU, so it runs everywhere. To fix a failure add a quick case (see "Adding a kernel and its test" in
//! kernels/README.md).

#[test]
fn every_hip_kernel_has_a_quick_case() {
    // the opt-in WMMA family (INTARSIA_HIP_WMMA_ROWS=1) rejects every shape with the knob unset; tests/quick_coverage_wmma.rs checks it with the knob on
    let kernels: Vec<_> = intarsia_backend_hip::kernels::all().into_iter().filter(|k| !k.name.contains("wmma")).collect();
    assert!(!kernels.is_empty(), "no hip kernels registered");
    let uncovered = intarsia_core::admission::kernels_without_quick_case(&kernels);
    assert!(uncovered.is_empty(), "hip kernels with no quick sweep case: {uncovered:?}");
}

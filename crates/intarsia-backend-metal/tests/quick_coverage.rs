//! Every kernel this backend registers has a case in the `kernels test --quick` sweep whose shape its
//! predicate accepts. A kernel without one reports `NO CASES (untested)` instead of being checked. Needs no
//! GPU, so it runs everywhere. To fix a failure add a quick case (see "Adding a kernel and its test" in
//! kernels/README.md).

#[test]
fn every_metal_kernel_has_a_quick_case() {
    let kernels = intarsia_backend_metal::kernels::all();
    assert!(!kernels.is_empty(), "no metal kernels registered");
    let uncovered = intarsia_core::admission::kernels_without_quick_case(&kernels);
    assert!(uncovered.is_empty(), "metal kernels with no quick sweep case: {uncovered:?}");
}

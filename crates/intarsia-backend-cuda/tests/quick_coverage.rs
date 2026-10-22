//! Every kernel this backend registers has a case in the `kernels test --quick` sweep whose shape its
//! predicate accepts. A kernel without one reports `NO CASES (untested)` instead of being checked. Needs no
//! GPU, so it runs everywhere. To fix a failure add a quick case (see "Adding a kernel and its test" in
//! kernels/README.md).

#[test]
fn every_cuda_kernel_has_a_quick_case() {
    let kernels = intarsia_backend_cuda::kernels::all();
    assert!(!kernels.is_empty(), "no cuda kernels registered");
    let mut uncovered = intarsia_core::admission::kernels_without_quick_case(&kernels);
    // registered but never acceptable when the crate is built without sm_120a (`cfg!(ik_sm120a)` is part of its predicate)
    if !cfg!(ik_sm120a) {
        uncovered.retain(|n| n != "moe_experts_nvfp4_mma_bf16");
    }
    assert!(uncovered.is_empty(), "cuda kernels with no quick sweep case: {uncovered:?}");
}

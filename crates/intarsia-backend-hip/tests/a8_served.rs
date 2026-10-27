//! Whether this backend's kernel table serves W4A8 MXFP4 matmuls (`--mxfp4-act fp8`): the loader refuses the mode, with a message,
//! on a backend without such a kernel (`quant::mxfp4::backend_serves_a8`). Needs no GPU.

#[test]
fn the_hip_table_serves_w4a8_is_true() {
    let served = intarsia_core::quant::mxfp4::backend_serves_a8(&intarsia_backend_hip::kernels::all());
    assert_eq!(served, true, "hip: W4A8 kernels registered = {served}");
}

//! Whether this backend's kernel table serves W4A8 MXFP4 matmuls (`--mxfp4-act fp8`): the loader refuses the mode, with a message,
//! on a backend without such a kernel (`quant::mxfp4::backend_serves_a8`). Needs no GPU.

#[test]
fn the_metal_table_serves_w4a8_is_false() {
    let served = intarsia_core::quant::mxfp4::backend_serves_a8(&intarsia_backend_metal::kernels::all());
    assert_eq!(served, false, "metal: W4A8 kernels registered = {served}");
}

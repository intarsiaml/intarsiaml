//! Lane cvatflip: `INTARSIA_CUDA_VERIFY_ATTN_ROWS` `1`: on (the opt-in spelling of lane cvat still works); HIP, Metal and the CPU backend never take it. (Own file: the knob is read from
//! the environment, and the test binaries of one crate share no process.)

use intarsia_core::model::cuda_verify_attn_rows_env;

#[test]
fn knob_one() {
    std::env::set_var("INTARSIA_CUDA_VERIFY_ATTN_ROWS", "1");
    assert_eq!(cuda_verify_attn_rows_env("cuda"), true);
    for b in ["hip", "metal", "cpu"] {
        assert!(!cuda_verify_attn_rows_env(b), "{b}");
    }
}

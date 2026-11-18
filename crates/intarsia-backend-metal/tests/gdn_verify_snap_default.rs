#![cfg(target_os = "macos")]
//! Lane mgdnv: with `INTARSIA_METAL_GDN_VERIFY_FUSE` unset (the default since lane mgdnv-flip) `Backend::gdn_verify_rows` takes the call (`Ok(true)`) and encodes
//! the fused kernels. (Own file: the knob is read once per process.)

mod common;
use common::*;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::{DType, GdnVerifyRows};

#[test]
fn unset_takes_the_fused_path() {
    std::env::remove_var("INTARSIA_METAL_GDN_VERIFY_FUSE");
    let be = backend();
    let (t, c, k, hv, dv, dk) = (3usize, 2 * 128 + 4 * 32, 4usize, 4usize, 32usize, 128usize);
    let mk = |dt: DType, shape: &[usize]| {
        let n: usize = shape.iter().product::<usize>() * dt.size_bytes();
        let b = be.alloc(n + 16, MemKind::Device).unwrap();
        be.zero(&b, 0).unwrap();
        TensorRef::new(b, dt, shape)
    };
    let (x, taps, y) = (mk(DType::Bf16, &[t, c]), mk(DType::Bf16, &[c, k]), mk(DType::Bf16, &[t, c]));
    let (a, b) = (mk(DType::Bf16, &[t, hv]), mk(DType::Bf16, &[t, hv]));
    let (al, db) = (mk(DType::Bf16, &[hv]), mk(DType::Bf16, &[hv]));
    let o = mk(DType::Bf16, &[t, hv, dv]);
    let (cs, st) = (mk(DType::F32, &[c, k - 1]), mk(DType::F32, &[hv, dv, dk]));
    let sc: Vec<_> = (0..t - 1).map(|_| mk(DType::F32, &[c, k - 1])).collect();
    let ss: Vec<_> = (0..t - 1).map(|_| mk(DType::F32, &[hv, dv, dk])).collect();
    let call = GdnVerifyRows {
        x: &x, taps: &taps, y: &y, a: &a, b: &b, a_log: &al, dt_bias: &db, eps: 1e-6, o: &o, conv_state: &cs, state: &st,
        snap_conv: &sc, snap_s: &ss,
    };
    assert!(be.gdn_verify_rows(&call, 0).unwrap(), "knob unset: the fused path must take the call");
    be.synchronize(0).unwrap();
}

//! The metal4.0 fallback: with `INTARSIA_METAL_NO_METAL4=1` (or any reason the metal4.0 library cannot be loaded) the
//! NAX kernels are neither registered nor launchable and prefill matmuls run on the existing kernels. One test in this
//! file sets the process environment, so it must stay the only test here that constructs the env-driven backend.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::kernels::{MATMUL_BF16, MATMUL_BF16_NAX, MATMUL_BF16_NAX_RELAXED};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind};
use intarsia_core::registry::Registry;

fn select_and_run(be: &MetalBackend, raise_nax_priority: bool) -> String {
    let (m, n, k) = (205usize, 256usize, 512usize);
    let x = rand_bf16(m * k, 1);
    let w = rand_bf16(n * k, 2);
    let (xb, wb) = (upload(be, &x), upload(be, &w));
    let yb = be.alloc(m * n * 2, MemKind::Device).unwrap();
    let call = matmul_call(tref(&xb, 0, &[m, k]), tref(&wb, 0, &[n, k]), tref(&yb, 0, &[m, n]));
    // what the full P1 registration will look like: the NAX kernels outrank the tiled GEMM when they exist
    let ks: Vec<_> = be
        .kernels()
        .into_iter()
        .map(|k| if raise_nax_priority && k.name.starts_with("matmul_bf16_nax") { k.priority(5) } else { k })
        .collect();
    let reg = Registry::new("metal", &be.info(), ks);
    let name = reg.select(&call).unwrap().name.clone();
    be.launch(&name, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; m * n * 2];
    be.download(&yb, 0, &mut raw, 0).unwrap();
    let y: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    for &mi in &[0usize, 100, 204] {
        let refs = reference(&w, &x[mi * k..(mi + 1) * k], k, n);
        check(&y[mi * n..(mi + 1) * n], &refs, 0..n, &format!("{name} row {mi}"));
    }
    name
}

#[test]
fn prefill_runs_without_the_metal4_library() {
    // with the library: the raised-priority NAX kernel is chosen (so the next block really tests a fallback)
    let be = MetalBackend::new().unwrap();
    if be.has_metal4() {
        assert!(be.kernels().iter().any(|k| k.name == MATMUL_BF16_NAX));
        assert_eq!(select_and_run(&be, true), MATMUL_BF16_NAX);
    }
    drop(be);

    // forced fallback through the environment variable, the way a user or an old OS would hit it
    std::env::set_var("INTARSIA_METAL_NO_METAL4", "1");
    let be = MetalBackend::new().unwrap();
    std::env::remove_var("INTARSIA_METAL_NO_METAL4");
    assert!(!be.has_metal4());
    let names: Vec<String> = be.kernels().into_iter().map(|k| k.name).collect();
    assert!(!names.iter().any(|n| n == MATMUL_BF16_NAX || n == MATMUL_BF16_NAX_RELAXED), "{names:?}");
    assert!(be.pso("gemm_nax_t128x64w4").is_none());
    assert!(names.iter().any(|n| n == MATMUL_BF16));
    // the existing prefill path still runs and is correct, even when the NAX priority would have won
    assert_eq!(select_and_run(&be, true), MATMUL_BF16);
    // launching a NAX kernel by name is an error, not a crash
    let (xb, wb, yb) = (upload(&be, &[0u16; 16 * 16]), upload(&be, &[0u16; 16 * 16]), upload(&be, &[0u16; 16 * 16]));
    let call = matmul_call(tref(&xb, 0, &[16, 16]), tref(&wb, 0, &[16, 16]), tref(&yb, 0, &[16, 16]));
    assert!(be.launch(MATMUL_BF16_NAX, &call, 0).is_err());

    // the explicit constructor does the same
    let be = MetalBackend::new_without_metal4().unwrap();
    assert!(!be.has_metal4());
    assert_eq!(select_and_run(&be, true), MATMUL_BF16);
}

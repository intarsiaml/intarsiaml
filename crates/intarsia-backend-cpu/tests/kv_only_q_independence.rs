//! The cache half of the fused gated prologue (`Op::GateSplitQkNormRopeKv`) does not read the q+gate projection.
//!
//! `Model::attn_block_kv_only` (the MTP head's prompt catch-up, `mtp_layer_kv_only`) skips the q projection (`kv_only_skip_q`)
//! and runs the fused prologue anyway, so its q+gate input is whatever the scratch last held. The K and V cache rows it writes
//! must not depend on that input (review of PR #185, 2026-10-26: the first run of this path under tp > 1 had an uninitialised
//! input). Checked here on the reference kernel over the admission sweep's cases (bf16 and f32, partial rotary, both norm
//! offsets, positions 0 and 37), with the q+gate input replaced by other values, large ones and non-finite ones. The device
//! kernels are held to the reference by the backends' fusion tests; this pins the contract the catch-up relies on.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{sweep, SweepConfig};
use intarsia_core::exec::run_host;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{HostTensor, Op};

#[test]
fn kv_cache_rows_do_not_depend_on_the_q_gate_input() {
    let b = CpuBackend::new();
    let cfg = SweepConfig { quick: true, seed: 7, ..SweepConfig::default() };
    let mut rng = SplitMix64::new(4242);
    let mut cases = 0;
    for case in sweep(Op::GateSplitQkNormRopeKv, &cfg) {
        let want = run_host(&b, "ref_gate_split_qk_norm_rope_kv", &case.call).unwrap_or_else(|e| panic!("{}: {e}", case.label));
        let (dt, shape) = (case.call.inputs[0].dtype, case.call.inputs[0].shape.clone());
        let n: usize = shape.iter().product();
        let other = |vals: Vec<f32>| HostTensor::from_f32_as(dt, &shape, &vals);
        let variants = [
            ("zeros", other(vec![0.0; n])),
            ("other values", other(rng.normal_vec(n, 1.0))),
            ("large values", other(rng.normal_vec(n, 3.0e4))),
            ("nan", other(vec![f32::NAN; n])),
            ("inf", other(vec![f32::INFINITY; n])),
        ];
        for (what, qg) in variants {
            let mut hc = case.call.clone();
            hc.inputs[0] = qg;
            let got = run_host(&b, "ref_gate_split_qk_norm_rope_kv", &hc).unwrap_or_else(|e| panic!("{} / {what}: {e}", case.label));
            for (o, name) in [(2, "k cache"), (3, "v cache")] {
                assert!(got[o].data == want[o].data, "{} with a q+gate input of {what}: {name} differs", case.label);
            }
        }
        cases += 1;
    }
    assert!(cases > 0, "no cases swept");
}

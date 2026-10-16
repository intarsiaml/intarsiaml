//! Unit tests of every cpu op against naive formulas written independently of
//! the implementations (small hand-checkable shapes), for both the oracle
//! (`ref_*`) and the fast (`cpu_f32_*`) kernels.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::{DType, HostTensor, Op, OpAttrs};

fn run(kernel_prefix: &str, hc: &HostCall) -> Vec<HostTensor> {
    let b = CpuBackend::new();
    let name = format!("{kernel_prefix}{}", hc.op.as_str());
    run_host(&b, &name, hc).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn both(hc: &HostCall) -> Vec<Vec<HostTensor>> {
    vec![run("ref_", hc), run("cpu_f32_", hc)]
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}[{i}]: got {g}, want {w}");
    }
}

fn f32t(shape: &[usize], v: &[f32]) -> HostTensor {
    HostTensor::from_f32(shape, v)
}

fn call(op: Op, inputs: Vec<HostTensor>, outs: Vec<OutSpec>, attrs: OpAttrs) -> HostCall {
    HostCall { op, inputs, outputs: outs, attrs }
}

#[test]
fn embedding_gathers_rows() {
    let table = f32t(&[4, 3], &[0., 1., 2., 10., 11., 12., 20., 21., 22., 30., 31., 32.]);
    let ids = HostTensor::from_i32(&[3], &[2, 0, 3]);
    let hc = call(Op::Embedding, vec![ids, table], vec![OutSpec::new(DType::F32, &[3, 3])], OpAttrs::default());
    for r in both(&hc) {
        assert_eq!(r[0].to_f32(), vec![20., 21., 22., 0., 1., 2., 30., 31., 32.]);
    }
}

#[test]
fn rmsnorm_matches_formula() {
    let x = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 0.0, 2.0];
    let w = [1.0f32, 0.5, 2.0, 1.5];
    let eps = 1e-6f32;
    let mut want = vec![];
    for row in x.chunks(4) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let r = 1.0 / (ms + eps).sqrt();
        want.extend(row.iter().zip(&w).map(|(v, w)| v * r * w));
    }
    let hc = call(
        Op::RmsNorm,
        vec![f32t(&[2, 4], &x), f32t(&[4], &w)],
        vec![OutSpec::new(DType::F32, &[2, 4])],
        OpAttrs { eps, ..Default::default() },
    );
    for r in both(&hc) {
        assert_close(&r[0].to_f32(), &want, 1e-5, "rmsnorm");
    }
}

#[test]
fn rmsnorm_add_returns_sum_and_norm() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let res = [0.5f32, -1.0, 1.0, 0.0];
    let w = [1.0f32; 4];
    let sum: Vec<f32> = x.iter().zip(&res).map(|(a, b)| a + b).collect();
    let ms: f32 = sum.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let r = 1.0 / (ms + 1e-6).sqrt();
    let want_n: Vec<f32> = sum.iter().map(|v| v * r).collect();
    let hc = call(
        Op::RmsNormAdd,
        vec![f32t(&[1, 4], &x), f32t(&[1, 4], &res), f32t(&[4], &w)],
        vec![OutSpec::new(DType::F32, &[1, 4]), OutSpec::new(DType::F32, &[1, 4])],
        OpAttrs { eps: 1e-6, ..Default::default() },
    );
    for r in both(&hc) {
        assert_close(&r[0].to_f32(), &want_n, 1e-5, "normed");
        assert_close(&r[1].to_f32(), &sum, 1e-6, "sum");
    }
}

#[test]
fn add_works() {
    let hc = call(
        Op::Add,
        vec![f32t(&[2, 2], &[1., 2., 3., 4.]), f32t(&[2, 2], &[10., 20., 30., 40.])],
        vec![OutSpec::new(DType::F32, &[2, 2])],
        OpAttrs::default(),
    );
    for r in both(&hc) {
        assert_eq!(r[0].to_f32(), vec![11., 22., 33., 44.]);
    }
}

#[test]
fn qk_norm_is_per_head_rmsnorm() {
    // [T=1, heads=2, D=4]: each head normalised independently
    let x = [1.0f32, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0];
    let w = [1.0f32; 4];
    let hc = call(
        Op::QkNorm,
        vec![f32t(&[1, 2, 4], &x), f32t(&[4], &w)],
        vec![OutSpec::new(DType::F32, &[1, 2, 4])],
        OpAttrs { eps: 0.0, ..Default::default() },
    );
    for r in both(&hc) {
        // head0: rms 1 -> unchanged; head1: rms = 1 -> [2,0,0,0]
        assert_close(&r[0].to_f32(), &[1., 1., 1., 1., 2., 0., 0., 0.], 1e-5, "qk_norm");
    }
}

#[test]
fn rope_neox_known_angles() {
    // D=4, theta=100: inv_freq = [1, 100^-0.5 = 0.1]; position 2 -> angles [2, 0.2]
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let (c0, s0) = (2f32.cos(), 2f32.sin());
    let (c1, s1) = (0.2f32.cos(), 0.2f32.sin());
    let want = [x[0] * c0 - x[2] * s0, x[1] * c1 - x[3] * s1, x[2] * c0 + x[0] * s0, x[3] * c1 + x[1] * s1];
    let hc = call(
        Op::Rope,
        vec![f32t(&[1, 1, 4], &x)],
        vec![OutSpec::new(DType::F32, &[1, 1, 4])],
        OpAttrs { pos0: 2, theta: 100.0, ..Default::default() },
    );
    for r in both(&hc) {
        assert_close(&r[0].to_f32(), &want, 1e-5, "rope");
    }
    // position 0 is the identity
    let hc0 = call(
        Op::Rope,
        vec![f32t(&[1, 1, 4], &x)],
        vec![OutSpec::new(DType::F32, &[1, 1, 4])],
        OpAttrs { pos0: 0, theta: 100.0, ..Default::default() },
    );
    for r in both(&hc0) {
        assert_close(&r[0].to_f32(), &x, 1e-6, "rope pos0");
    }
}

#[test]
fn matmul_is_x_times_w_transposed() {
    let x = [1.0f32, 2.0, 3.0, 0.5, -1.0, 2.0];
    let w = [1.0f32, 0.0, 1.0, 2.0, 1.0, -1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0];
    let (m, k, n) = (2, 3, 4);
    let mut want = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            for ki in 0..k {
                want[mi * n + ni] += x[mi * k + ki] * w[ni * k + ki];
            }
        }
    }
    // bf16 weights + activations (values are exactly representable)
    let hc = call(
        Op::Matmul,
        vec![HostTensor::from_f32_as(DType::Bf16, &[m, k], &x), HostTensor::from_f32_as(DType::Bf16, &[n, k], &w)],
        vec![OutSpec::new(DType::Bf16, &[m, n])],
        OpAttrs::default(),
    );
    for r in both(&hc) {
        assert_close(&r[0].to_f32(), &want, 1e-2, "matmul bf16");
    }
    let hl = call(
        Op::Logits,
        vec![f32t(&[m, k], &x), f32t(&[n, k], &w)],
        vec![OutSpec::new(DType::F32, &[m, n])],
        OpAttrs::default(),
    );
    for r in both(&hl) {
        assert_close(&r[0].to_f32(), &want, 1e-6, "logits f32");
    }
}

#[test]
fn kv_append_writes_only_new_rows() {
    // cache [nkv=1, cap=4, D=2], append T=2 rows at pos0=1
    let cache = f32t(&[1, 4, 2], &[9.; 8]);
    let new = f32t(&[2, 1, 2], &[1., 2., 3., 4.]);
    let hc = call(
        Op::KvAppend,
        vec![new],
        vec![OutSpec { dtype: DType::F32, shape: vec![1, 4, 2], init: Some(cache) }],
        OpAttrs { pos0: 1, kv_cap: 4, ..Default::default() },
    );
    for r in both(&hc) {
        assert_eq!(r[0].to_f32(), vec![9., 9., 1., 2., 3., 4., 9., 9.]);
    }
}

#[test]
fn attention_causal_gqa_matches_naive() {
    // heads=2, kv heads=1 (group 2), D=2, T=2 queries at pos0=1 -> keys 0..=1 and 0..=2
    let d = 2usize;
    let q = [0.3f32, -0.2, 0.1, 0.4, 0.5, 0.5, -0.3, 0.2]; // [T=2, N=2, D=2]
    let k = [0.1f32, 0.2, 0.3, -0.1, 0.5, 0.4, 0.0, 0.0]; // [1, cap=4, 2]
    let v = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0];
    let scale = 1.0 / (d as f32).sqrt();
    let mut want = vec![0f32; 2 * 2 * d];
    for t in 0..2 {
        for h in 0..2 {
            let vis = 1 + t + 1;
            let qr = &q[(t * 2 + h) * d..(t * 2 + h + 1) * d];
            let sc: Vec<f32> = (0..vis).map(|j| scale * (qr[0] * k[j * d] + qr[1] * k[j * d + 1])).collect();
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let ex: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
            let den: f32 = ex.iter().sum();
            for j in 0..vis {
                for e in 0..d {
                    want[(t * 2 + h) * d + e] += ex[j] / den * v[j * d + e];
                }
            }
        }
    }
    let attrs = OpAttrs { pos0: 1, scale, n_heads: 2, n_kv_heads: 1, head_dim: d, kv_cap: 4, ..Default::default() };
    let hc = call(
        Op::AttnPrefill,
        vec![f32t(&[2, 2, 2], &q), f32t(&[1, 4, 2], &k), f32t(&[1, 4, 2], &v)],
        vec![OutSpec::new(DType::F32, &[2, 2, 2])],
        attrs.clone(),
    );
    for r in both(&hc) {
        assert_close(&r[0].to_f32(), &want, 1e-5, "attn prefill");
    }
    // decode of the last query alone (pos0 = 2) equals the last row of prefill
    let hd = call(
        Op::AttnDecode,
        vec![f32t(&[1, 2, 2], &q[8 - 4..]), f32t(&[1, 4, 2], &k), f32t(&[1, 4, 2], &v)],
        vec![OutSpec::new(DType::F32, &[1, 2, 2])],
        OpAttrs { pos0: 2, ..attrs },
    );
    for r in both(&hd) {
        assert_close(&r[0].to_f32(), &want[4..], 1e-5, "attn decode");
    }
}

#[test]
fn silu_glu_matches_formula() {
    let g = [0.0f32, 1.0, -2.0, 3.0];
    let u = [1.0f32, 2.0, 3.0, -1.0];
    let want: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
    let hc = call(
        Op::SiluGlu,
        vec![f32t(&[2, 2], &g), f32t(&[2, 2], &u)],
        vec![OutSpec::new(DType::F32, &[2, 2])],
        OpAttrs::default(),
    );
    for r in both(&hc) {
        assert_close(&r[0].to_f32(), &want, 1e-5, "silu_glu");
    }
}

#[test]
fn argmax_lowest_index_on_ties() {
    let l = f32t(&[2, 4], &[0.1, 0.9, 0.9, 0.2, -1.0, -3.0, -0.5, -0.5]);
    let hc = call(Op::Argmax, vec![l], vec![OutSpec::new(DType::I32, &[2])], OpAttrs::default());
    for r in both(&hc) {
        assert_eq!(r[0].to_i32().unwrap(), vec![1, 2]);
    }
}

#[test]
fn sample_greedy_and_seeded_agree_between_impls() {
    let l = f32t(&[1, 5], &[0.1, 2.0, 0.3, 1.9, 0.0]);
    for (t, k, p, u) in [(0.0f32, 0usize, 1.0f32, 0.5f32), (1.0, 2, 1.0, 0.99), (0.8, 0, 0.5, 0.2)] {
        let hc = call(
            Op::Sample,
            vec![l.clone(), f32t(&[1], &[u])],
            vec![OutSpec::new(DType::I32, &[1])],
            OpAttrs { temperature: t, top_k: k, top_p: p, ..Default::default() },
        );
        let r = both(&hc);
        assert_eq!(r[0][0].to_i32().unwrap(), r[1][0].to_i32().unwrap());
    }
}

#[test]
fn fast_matches_reference_on_random_bf16() {
    use intarsia_core::admission::{sweep, SweepConfig};
    let cfg = SweepConfig { quick: true, ..Default::default() };
    let b = CpuBackend::new();
    for op in Op::ALL {
        for case in sweep(op, &cfg).into_iter().take(6) {
            let want = run_host(&b, &format!("ref_{}", op.as_str()), &case.call).unwrap();
            let got = run_host(&b, &format!("cpu_f32_{}", op.as_str()), &case.call).unwrap();
            for (g, w) in got.iter().zip(&want) {
                let (a, bb) = (g.to_f32(), w.to_f32());
                let num: f64 = a.iter().zip(&bb).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
                let den: f64 = bb.iter().map(|y| (*y as f64).powi(2)).sum();
                let rel = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };
                assert!(rel < 1e-2, "{op} {}: rel_l2 {rel}", case.label);
            }
        }
    }
}

//! Diagnostics (lane ink-cuda-mixmoe): the error spread of `moe_experts_situ_bf16` against the cpu reference over many cases
//! built exactly as plan admission builds them (`quant::cases::case_from_call` of an Inkling-2 situ call at T rows, E 32,
//! H 2304, I 1152), judged with the admission's own `judge`. Env-gated: `INTARSIA_SITU_SPREAD=<cases>` (no-op otherwise),
//! `INTARSIA_SITU_SPREAD_T` rows (default 1). Prints the rel_l2 distribution and, for the worst case, how many output
//! elements differ and by how many bf16 ulps.

use intarsia_core::backend::{Buffer, TensorRef};
use intarsia_core::quant::cases::{case_from_call, experts_case};
use intarsia_core::{Backend, DType, HostTensor, Op, OpCall};
use intarsia_core::rng::SplitMix64;

struct P(usize);
impl intarsia_core::backend::BufferImpl for P {
    fn bytes(&self) -> usize {
        self.0
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn ink_situ_admission_spread() {
    let Some(n) = std::env::var("INTARSIA_SITU_SPREAD").ok().and_then(|v| v.parse::<usize>().ok()) else { return };
    let t: usize = std::env::var("INTARSIA_SITU_SPREAD_T").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let cuda: std::sync::Arc<dyn Backend> = intarsia_backend_cuda::create(0).expect("cuda");
    let cpu = intarsia_backend_cpu::CpuBackend::arc();
    let ck = cuda.kernels().into_iter().find(|k| k.name == "moe_experts_situ_bf16").expect("cuda situ kernel");
    let rk = cpu.kernels().into_iter().find(|k| k.op == Op::MoeExperts && k.is_reference).expect("cpu reference");
    // a phantom situ call of the real shapes, as the plan records it (128 experts; the case clamps to 32)
    let mut rng0 = SplitMix64::new(3);
    let shape = experts_case(&mut rng0, t, 128, 8, 2304, 1152, DType::Bf16);
    let mk = |t: &HostTensor| TensorRef::new(Buffer::new(P(t.nbytes())), t.dtype, &t.shape);
    let mut a = shape.attrs.clone();
    (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
    let call = OpCall::new(
        Op::MoeExperts,
        shape.inputs.iter().map(mk).collect(),
        shape.outputs.iter().map(|o| mk(&HostTensor::zeros(o.dtype, &o.shape))).collect(),
        a,
    );
    let caps = intarsia_core::plan::CAPS;
    let mut rels: Vec<(f64, u64)> = Vec::new();
    let mut worst: Option<(f64, Vec<f32>, Vec<f32>)> = None;
    for seed in 0..n as u64 {
        let mut rng = SplitMix64::new(0x5170 + seed);
        let (hc, _label) = case_from_call(&call, &caps, &mut rng).expect("case");
        let want = intarsia_core::exec::run_host(&*cpu, &rk.name, &hc).unwrap();
        let got = intarsia_core::exec::run_host(&*cuda, &ck.name, &hc).unwrap();
        let (pass, max_abs, rel) = intarsia_core::admission::judge(&ck, &got, &want);
        rels.push((rel, seed));
        if !pass {
            println!("seed {seed}: FAIL rel_l2 {rel:.3e} max_abs {max_abs:.3e}");
        }
        if worst.as_ref().map_or(true, |w| rel > w.0) {
            worst = Some((rel, got[0].to_f32(), want[0].to_f32()));
        }
    }
    rels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let q = |p: f64| rels[((rels.len() - 1) as f64 * p) as usize].0;
    let fails = rels.iter().filter(|r| r.0 > ck.tol_for(DType::Bf16).rel_l2 as f64).count();
    println!(
        "T={t} cases {n}: rel_l2 median {:.2e} p90 {:.2e} p99 {:.2e} max {:.2e} (seed {}); over tol {fails}",
        q(0.5),
        q(0.9),
        q(0.99),
        rels.last().unwrap().0,
        rels.last().unwrap().1
    );
    let (rel, g, w) = worst.unwrap();
    let norm = w.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let mut d: Vec<(f64, usize)> = g.iter().zip(&w).enumerate().map(|(i, (a, b))| (((*a - *b) as f64).abs(), i)).collect();
    d.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let ulp = |x: f32| 2f64.powi((x.abs().max(1e-30).log2().floor() as i32) - 7);
    let nz = d.iter().filter(|x| x.0 > 0.0).count();
    let share = |k: usize| d[..k].iter().map(|x| x.0 * x.0).sum::<f64>().sqrt() / norm;
    println!("worst rel_l2 {rel:.3e}: {nz} / {} elements differ; top-1 / top-5 / top-20 carry rel {:.2e} / {:.2e} / {:.2e}", g.len(), share(1), share(5), share(20));
    for &(dv, i) in d.iter().take(6) {
        println!("  out[{i}]: cuda {:.6} ref {:.6} |d| {dv:.3e} = {:.1} ulp of ref", g[i], w[i], dv / ulp(w[i]));
    }
}

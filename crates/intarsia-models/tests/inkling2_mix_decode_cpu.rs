#![cfg(target_os = "macos")]
//! Inkling-2 1-row decode of the mix kernels (`ConvMix`, `AttnResMix`, `LoopBlend`) against the cpu reference, state carried (lane ink-metal, after the
//! decode-mode G2 regression of the m-mix / ink-batch merges, docs/results/2026-11-21-inkling2-metal.md): 20 one-row
//! `ConvMix` steps at the real width D 2304 (the m-mix NE 4 row geometry) on Metal and on the cpu backend, each step's conv
//! ring and output fed to the next (output -> next input), and every tensor the kernel writes (output, ring, sum, mix / blend
//! streams) compared per step within a bf16 budget. The rows-exact tests (tests/inkmix.rs) compare Metal with Metal; this one
//! compares with the reference, so a decode-path error common to the 1-row and the rows-exact kernels shows here.
//! `ne3` runs the same steps with `INTARSIA_INK_MIX_NE=3` (in a child process: the override is read once per process).

use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const D: usize = 2304;
const RING: usize = 19; // K - 1 + 16, the model's conv ring
const K: usize = 4;
const G: usize = 64;
const STEPS: usize = 20;
/// Per-step relative L2 budget of a bf16 tensor against the reference (a few bf16 ulps; the tiny Metal tf_check budgets
/// are 5e-3 to 2e-2 per tensor).
const BUDGET: f64 = 1e-2;

#[derive(Clone)]
struct Ht {
    dt: DType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Ht {
    fn rand(dt: DType, shape: &[usize], scale: f32, rng: &mut Rng) -> Ht {
        let n: usize = shape.iter().product();
        let data = match dt {
            DType::Bf16 => (0..n).flat_map(|_| f32_to_bf16(rng.unit() * scale).to_le_bytes()).collect(),
            _ => (0..n).flat_map(|_| (rng.unit() * scale).to_le_bytes()).collect(),
        };
        Ht { dt, shape: shape.to_vec(), data }
    }
    fn pos(shape: &[usize], rng: &mut Rng) -> Ht {
        let n: usize = shape.iter().product();
        Ht { dt: DType::Bf16, shape: shape.to_vec(), data: (0..n).flat_map(|_| f32_to_bf16(1.0 + 0.5 * rng.unit()).to_le_bytes()).collect() }
    }
    fn zeros(dt: DType, shape: &[usize]) -> Ht {
        Ht { dt, shape: shape.to_vec(), data: vec![0; shape.iter().product::<usize>() * dt.size_bytes()] }
    }
    fn none() -> Ht {
        Ht::zeros(DType::Bf16, &[0])
    }
    fn f64s(&self) -> Vec<f64> {
        match self.dt {
            DType::Bf16 => self.data.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64).collect(),
            _ => self.data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64).collect(),
        }
    }
}

/// Run one call of `op` on `be` (its kernel for `op` named `kernel`): host inputs, outputs as initial contents; the outputs back.
fn run(be: &dyn Backend, kernel: &str, op: Op, ins: &[Ht], outs: &[Ht], attrs: &OpAttrs) -> Vec<Ht> {
    let up = |h: &Ht| -> TensorRef {
        let b = be.alloc(h.data.len().max(16), MemKind::Device).unwrap();
        if !h.data.is_empty() {
            be.upload(&b, 0, &h.data, 0).unwrap();
        }
        TensorRef { buf: b, offset: 0, dtype: h.dt, shape: h.shape.clone() }
    };
    let it: Vec<TensorRef> = ins.iter().map(up).collect();
    let ot: Vec<TensorRef> = outs.iter().map(up).collect();
    let mut call = OpCall::new(op, it, ot.clone(), attrs.clone());
    call.check().unwrap();
    let k = be.kernels().into_iter().find(|k| k.name == kernel).unwrap_or_else(|| panic!("{} has no {kernel}", be.name()));
    let n = k.scratch_bytes(&ShapeInfo::from_call(&call));
    if n > 0 {
        call.scratch = Some(TensorRef { buf: be.alloc(n, MemKind::Device).unwrap(), offset: 0, dtype: DType::U8, shape: vec![n] });
    }
    be.launch(kernel, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    ot.iter()
        .zip(outs)
        .map(|(t, h)| {
            let mut v = vec![0u8; h.data.len()];
            if !v.is_empty() {
                be.download(&t.buf, t.offset, &mut v, 0).unwrap();
            }
            Ht { dt: h.dt, shape: h.shape.clone(), data: v }
        })
        .collect()
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        num += (x - y) * (x - y);
        den += y * y;
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// 20 one-row `ConvMix` steps at D 2304, `mix` = (streams, q rows, norm, blend streams), Metal vs cpu; returns the worst
/// (step, tensor, rel l2).
fn decode_steps(mix: (usize, usize, bool, usize), seed: u64) -> (usize, usize, f64) {
    let (metal, cpu) = backends();
    let mut rng = Rng(seed);
    let (n, q, norm, s) = mix;
    let bf = DType::Bf16;
    // the weights (fixed over the steps)
    let taps = Ht::rand(bf, &[K, D], 0.5, &mut rng);
    let qp = Ht::rand(DType::F32, &[q, D], 0.2, &mut rng);
    let nrows = if s > 0 { s } else { q };
    let nw = if norm || s > 0 { Ht::pos(&[nrows, D], &mut rng) } else { Ht::none() };
    let (gw, uw, dw) = if s > 0 {
        (
            Ht::rand(bf, &[s, G, 2 * D], (2.0 * D as f32).powf(-0.5), &mut rng),
            Ht::rand(bf, &[s, D, G], 0.25, &mut rng),
            Ht::rand(bf, &[s, D, G], 0.25, &mut rng),
        )
    } else {
        (Ht::none(), Ht::none(), Ht::none())
    };
    let mixout = |s: usize| {
        if q == 0 {
            Ht::none()
        } else if s > 0 {
            Ht::zeros(bf, &[s, 1, D])
        } else if norm {
            Ht::zeros(bf, &[q, 1, D])
        } else {
            Ht::zeros(DType::F32, &[q, 1, D])
        }
    };
    let ring0 = Ht::rand(bf, &[RING, D], 1.0, &mut rng);
    let (mut ring_m, mut ring_c) = (ring0.clone(), ring0);
    let x = Ht::rand(bf, &[1, D], 1.0, &mut rng);
    let (mut xm, mut xc) = (x.clone(), x);
    let mut worst = (0, 0, 0.0f64);
    for step in 0..STEPS {
        // per-step inputs shared by both backends: the residual, the mix streams and the blend streams
        let res = Ht::rand(bf, &[1, D], 1.0, &mut rng);
        let add = Ht::rand(bf, &[1, D], 1.0, &mut rng);
        let streams = Ht::rand(bf, &[n, 1, D], 1.5, &mut rng);
        let blend = if s > 0 { Ht::rand(DType::F32, &[s, 1, D], 1.0, &mut rng) } else { Ht::none() };
        let attrs = OpAttrs { pos0: 29 + step, eps: 1e-6, ..OpAttrs::default() };
        let ins = |x: &Ht| vec![x.clone(), taps.clone(), res.clone(), add.clone(), streams.clone(), qp.clone(), nw.clone(), blend.clone(), gw.clone(), uw.clone(), dw.clone()];
        let outs = |ring: &Ht| vec![Ht::zeros(bf, &[1, D]), ring.clone(), Ht::zeros(bf, &[1, D]), mixout(s)];
        let om = run(&metal, intarsia_backend_metal::inkmix::CONV_MIX_BF16, Op::ConvMix, &ins(&xm), &outs(&ring_m), &attrs);
        let oc = run(&cpu, "ref_conv_mix", Op::ConvMix, &ins(&xc), &outs(&ring_c), &attrs);
        let e = check(&format!("ConvMix mix {mix:?} step {step}"), &om, &oc);
        if e > worst.2 {
            worst = (step, 0, e);
        }
        // the state: each backend carries its own ring and feeds its own output to the next step (the decode recurrence)
        ring_m = om[1].clone();
        ring_c = oc[1].clone();
        xm = Ht { shape: vec![1, D], ..om[0].clone() };
        xc = Ht { shape: vec![1, D], ..oc[0].clone() };
    }
    worst
}

fn backends() -> (MetalBackend, intarsia_backend_cpu::CpuBackend) {
    (MetalBackend::new().expect("Metal backend"), intarsia_backend_cpu::CpuBackend::new())
}

fn check(label: &str, a: &[Ht], b: &[Ht]) -> f64 {
    let mut worst = 0.0f64;
    for (ti, (x, y)) in a.iter().zip(b).enumerate() {
        let e = rel_l2(&x.f64s(), &y.f64s());
        worst = worst.max(e);
        assert!(e <= BUDGET, "{label}: output {ti} rel l2 {e:.3e} > {BUDGET:.0e} (Metal vs cpu reference)");
    }
    worst
}

/// `AttnResMix` and `LoopBlend` at D 2304, 1 row (decode) and a few row counts, Metal vs cpu.
fn res_mix_and_blend(label: &str) {
    let (metal, cpu) = backends();
    let mut rng = Rng(0xab5d);
    let bf = DType::Bf16;
    let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
    for &r in &[1usize, 2, 5, 16] {
        for &(n, q, add, norm) in &[(4usize, 1usize, true, true), (7, 3, false, true), (0, 4, true, false), (3, 6, false, false)] {
            let ins = vec![
                Ht::rand(bf, &[n, r, D], 1.5, &mut rng),
                Ht::rand(bf, &[r, D], 1.0, &mut rng),
                if add { Ht::rand(bf, &[r, D], 1.0, &mut rng) } else { Ht::none() },
                Ht::rand(DType::F32, &[q, D], 0.2, &mut rng),
                if norm { Ht::pos(&[q, D], &mut rng) } else { Ht::none() },
            ];
            let out = if norm { Ht::zeros(bf, &[q, r, D]) } else { Ht::zeros(DType::F32, &[q, r, D]) };
            let om = run(&metal, intarsia_backend_metal::inkmix::ATTN_RES_MIX_BF16, Op::AttnResMix, &ins, &[out.clone()], &a);
            let oc = run(&cpu, "ref_attn_res_mix", Op::AttnResMix, &ins, &[out], &a);
            let e = check(&format!("{label} AttnResMix R={r} N={n} Q={q} add={add} norm={norm}"), &om, &oc);
            println!("{label} AttnResMix R={r} N={n} Q={q}: rel l2 {e:.3e}");
        }
        for &(s, sd) in &[(3usize, 1usize), (3, 3), (1, 1)] {
            let ins = vec![
                Ht::rand(DType::F32, &[sd, r, D], 1.0, &mut rng),
                Ht::rand(DType::F32, &[s, r, D], 1.0, &mut rng),
                Ht::rand(bf, &[s, G, 2 * D], (2.0 * D as f32).powf(-0.5), &mut rng),
                Ht::rand(bf, &[s, D, G], 0.25, &mut rng),
                Ht::rand(bf, &[s, D, G], 0.25, &mut rng),
                Ht::pos(&[s, D], &mut rng),
            ];
            let out = Ht::zeros(bf, &[s, r, D]);
            let om = run(&metal, intarsia_backend_metal::inkmix::LOOP_BLEND_BF16, Op::LoopBlend, &ins, &[out.clone()], &a);
            let oc = run(&cpu, "ref_loop_blend", Op::LoopBlend, &ins, &[out], &a);
            let e = check(&format!("{label} LoopBlend R={r} S={s} Sd={sd}"), &om, &oc);
            println!("{label} LoopBlend R={r} S={s} Sd={sd}: rel l2 {e:.3e}");
        }
    }
}

fn all_mixes(label: &str) {
    res_mix_and_blend(label);
    // the model's ConvMix forms: a plain message mix (normed, one q row), and the blended one (3 streams)
    for (i, mix) in [(4usize, 1usize, true, 0usize), (7, 3, true, 0), (3, 1, false, 3), (5, 3, false, 3)].into_iter().enumerate() {
        let (step, _, e) = decode_steps(mix, 0x1de5 + i as u64);
        println!("{label} ConvMix decode mix {mix:?}: worst rel l2 {e:.3e} (step {step})");
    }
}

#[test]
fn mix_one_row_decode_matches_cpu_at_d2304() {
    all_mixes(&format!("NE {}", std::env::var("INTARSIA_INK_MIX_NE").unwrap_or_else(|_| "default".into())));
}

/// The same steps with the old row geometry (NE 3 at D 2304), in a child process with `INTARSIA_INK_MIX_NE=3`.
#[test]
fn mix_one_row_decode_matches_cpu_at_d2304_ne3() {
    if std::env::var("INTARSIA_INK_MIX_NE").is_ok() {
        return; // the child (or an explicit override): the test above covers it
    }
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args(["--exact", "mix_one_row_decode_matches_cpu_at_d2304", "--nocapture", "--test-threads", "1"])
        .env("INTARSIA_INK_MIX_NE", "3")
        .output()
        .unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    println!("{text}");
    assert!(out.status.success() && text.contains("NE 3 ConvMix"), "NE 3 child failed:\n{text}");
}

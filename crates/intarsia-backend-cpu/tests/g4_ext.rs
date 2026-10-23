//! Gemma-4 extended attributes on the cpu kernels (`ref_*` and `cpu_f32_*`), each against an independent naive
//! implementation written here: `GeluGlu`, `Rope` with `rope_prop`, attention `window`, `Add` / `Embedding` with
//! `scale != 1.0`, `MoeExperts` with `gelu`.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::cases::{experts_case, random_experts};
use intarsia_core::quant::reference::nvfp4_row;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, Op, OpAttrs, OpCall, Registry, TensorRef};

const KERNELS: [&str; 2] = ["ref_", "cpu_f32_"];

fn run(prefix: &str, hc: &HostCall) -> Vec<HostTensor> {
    let b = CpuBackend::new();
    let name = format!("{prefix}{}", hc.op.as_str());
    run_host(&b, &name, hc).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn rnd(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale).collect()
}

/// What a tensor of `dt` holds after storing `v`.
fn round_to(dt: DType, v: &[f32]) -> Vec<f32> {
    HostTensor::from_f32_as(dt, &[v.len()], v).to_f32()
}

fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((*g as f64 - w).abs() <= tol * (1.0 + w.abs()), "{what}[{i}]: got {g}, want {w}");
    }
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

// ---------------------------------------------------------------- GeluGlu

fn naive_gelu(x: f64) -> f64 {
    let k = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x.powi(3))).tanh())
}

#[test]
fn gelu_glu_matches_formula() {
    for dt in [DType::F32, DType::Bf16, DType::F16] {
        let mut rng = SplitMix64::new(11);
        let g = round_to(dt, &rnd(&mut rng, 3 * 70, 4.0));
        let u = round_to(dt, &rnd(&mut rng, 3 * 70, 2.0));
        let want: Vec<f64> = g.iter().zip(&u).map(|(g, u)| naive_gelu(*g as f64) * *u as f64).collect();
        let hc = HostCall {
            op: Op::GeluGlu,
            inputs: vec![HostTensor::from_f32_as(dt, &[3, 70], &g), HostTensor::from_f32_as(dt, &[3, 70], &u)],
            outputs: vec![OutSpec::new(dt, &[3, 70])],
            attrs: OpAttrs::default(),
        };
        // one rounding of the exact value; the fast kernel is f32 inside, so allow a 1-ulp flip of the storage type
        let tol = match dt {
            DType::F32 => 1e-5,
            DType::F16 => 2e-3,
            _ => 1e-2,
        };
        for p in KERNELS {
            let r = run(p, &hc);
            assert_close(&r[0].to_f32(), &want, tol, &format!("{p}gelu_glu {dt:?}"));
        }
        // the reference rounds the exact f64 value once
        let r = run("ref_", &hc);
        assert_eq!(r[0].to_f32(), round_to(dt, &f32s(&want)), "ref gelu_glu {dt:?} single rounding");
    }
}

// ---------------------------------------------------------------- Rope

/// HF `_compute_proportional_rope_parameters` + `rotate_half` application (cos/sin over the whole head), in f64.
fn naive_rope_prop(x: &[f32], t: usize, n: usize, d: usize, pos0: usize, theta: f64, rotary_dim: usize) -> Vec<f64> {
    let angles = rotary_dim / 2; // int(partial_rotary_factor * head_dim // 2)
    let mut inv_freq = vec![0f64; d / 2]; // zeros for the unrotated tail
    for j in 0..angles {
        inv_freq[j] = 1.0 / theta.powf((2 * j) as f64 / d as f64);
    }
    let mut out = vec![0f64; x.len()];
    for ti in 0..t {
        let pos = (pos0 + ti) as f64;
        // emb = cat(freqs, freqs)
        let cos: Vec<f64> = (0..d).map(|e| (pos * inv_freq[e % (d / 2)]).cos()).collect();
        let sin: Vec<f64> = (0..d).map(|e| (pos * inv_freq[e % (d / 2)]).sin()).collect();
        for h in 0..n {
            let row = &x[(ti * n + h) * d..(ti * n + h + 1) * d];
            for e in 0..d {
                // rotate_half(x) = cat(-x2, x1)
                let rh = if e < d / 2 { -(row[e + d / 2] as f64) } else { row[e - d / 2] as f64 };
                out[(ti * n + h) * d + e] = row[e] as f64 * cos[e] + rh * sin[e];
            }
        }
    }
    out
}

#[test]
fn rope_proportional_matches_hf_formula() {
    let (t, n, d, rot, pos0, theta) = (3usize, 2usize, 512usize, 128usize, 100usize, 1.0e6f64);
    let mut rng = SplitMix64::new(5);
    let x = rnd(&mut rng, t * n * d, 1.5);
    let want = naive_rope_prop(&x, t, n, d, pos0, theta, rot);
    let attrs = OpAttrs { pos0, theta: theta as f32, rotary_dim: rot, rope_prop: true, ..OpAttrs::default() };
    let hc = HostCall {
        op: Op::Rope,
        inputs: vec![HostTensor::from_f32(&[t, n, d], &x)],
        outputs: vec![OutSpec::new(DType::F32, &[t, n, d])],
        attrs: attrs.clone(),
    };
    for p in KERNELS {
        let got = run(p, &hc)[0].to_f32();
        assert_close(&got, &want, 2e-5, &format!("{p}rope_prop f32"));
        // values past the rotated pairs are copied bit-exactly
        for r in 0..t * n {
            for e in 0..d {
                if e % (d / 2) >= rot / 2 {
                    assert_eq!(got[r * d + e], x[r * d + e], "{p} tail copy r={r} e={e}");
                }
            }
        }
    }
    // bf16: the stored result is the f32 formula rounded once (cos/sin narrowed to f32 like every kernel's table)
    let xb = round_to(DType::Bf16, &x);
    let want_b = naive_rope_prop(&xb, t, n, d, pos0, theta, rot);
    let hcb = HostCall {
        op: Op::Rope,
        inputs: vec![HostTensor::from_f32_as(DType::Bf16, &[t, n, d], &xb)],
        outputs: vec![OutSpec::new(DType::Bf16, &[t, n, d])],
        attrs,
    };
    for p in KERNELS {
        assert_close(&run(p, &hcb)[0].to_f32(), &want_b, 1.2e-2, &format!("{p}rope_prop bf16"));
    }
    // and without the flag the same call is the ordinary partial rope (denominator rotary_dim, pairs (j, j + rot/2)): different
    let plain = HostCall { attrs: OpAttrs { rope_prop: false, ..hc.attrs.clone() }, ..hc.clone() };
    let a = run("ref_", &hc)[0].to_f32();
    let b = run("ref_", &plain)[0].to_f32();
    assert_ne!(a, b);
}

// ---------------------------------------------------------------- attention

#[allow(clippy::too_many_arguments)]
fn naive_attn(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    (t, n, nkv, d, cap): (usize, usize, usize, usize, usize),
    pos0: usize,
    scale: f64,
    window: usize,
) -> Vec<f64> {
    let group = n / nkv;
    let mut out = vec![0f64; t * n * d];
    for ti in 0..t {
        let p = pos0 + ti;
        for h in 0..n {
            let kvh = h / group;
            // masked softmax over every key 0..=p; masked keys get -inf
            let mut s = vec![f64::NEG_INFINITY; p + 1];
            for (j, sj) in s.iter_mut().enumerate() {
                if window > 0 && j + window <= p {
                    continue;
                }
                *sj = scale * (0..d).map(|e| q[(ti * n + h) * d + e] as f64 * kc[(kvh * cap + j) * d + e] as f64).sum::<f64>();
            }
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
            let z: f64 = w.iter().sum();
            for (j, wj) in w.iter().enumerate() {
                for e in 0..d {
                    out[(ti * n + h) * d + e] += wj / z * vc[(kvh * cap + j) * d + e] as f64;
                }
            }
        }
    }
    out
}

fn attn_call(op: Op, q: &[f32], kc: &[f32], vc: &[f32], dims: (usize, usize, usize, usize, usize), attrs: OpAttrs) -> HostCall {
    let (t, n, nkv, d, cap) = dims;
    HostCall {
        op,
        inputs: vec![
            HostTensor::from_f32(&[t, n, d], q),
            HostTensor::from_f32(&[nkv, cap, d], kc),
            HostTensor::from_f32(&[nkv, cap, d], vc),
        ],
        outputs: vec![OutSpec::new(DType::F32, &[t, n, d])],
        attrs: OpAttrs { n_heads: n, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..attrs },
    }
}

#[test]
fn attention_window_gqa_matches_naive_masked_softmax() {
    // (n heads, kv heads, head_dim): Gemma global (group 8, D=512) and sliding (group 2, D=256) layers
    for (n, nkv, d) in [(16usize, 2usize, 512usize), (16, 8, 256)] {
        let cap = 64;
        let dims1 = (1, n, nkv, d, cap);
        let mut rng = SplitMix64::new(21 + d as u64);
        let kc = rnd(&mut rng, nkv * cap * d, 1.0);
        let vc = rnd(&mut rng, nkv * cap * d, 1.0);
        // decode at cache length 40 (position 39) and prefill of 6 rows starting at 30
        let q1 = rnd(&mut rng, n * d, 1.0);
        let dimsp = (6, n, nkv, d, cap);
        let qp = rnd(&mut rng, 6 * n * d, 1.0);
        for window in [0usize, 1024, 5, 1, 17] {
            for scale in [1.0f32, 0.0625] {
                let at = |pos0| OpAttrs { pos0, scale, window, ..OpAttrs::default() };
                let want = naive_attn(&q1, &kc, &vc, dims1, 39, scale as f64, window);
                let hc = attn_call(Op::AttnDecode, &q1, &kc, &vc, dims1, at(39));
                let wantp = naive_attn(&qp, &kc, &vc, dimsp, 30, scale as f64, window);
                let hcp = attn_call(Op::AttnPrefill, &qp, &kc, &vc, dimsp, at(30));
                for p in KERNELS {
                    let tag = format!("{p} n={n} nkv={nkv} d={d} window={window} scale={scale}");
                    assert_close(&run(p, &hc)[0].to_f32(), &want, 1e-4, &format!("decode {tag}"));
                    assert_close(&run(p, &hcp)[0].to_f32(), &wantp, 1e-4, &format!("prefill {tag}"));
                }
            }
        }
        // a window larger than the cache is the unwindowed result, bit for bit, on both kernels
        for p in KERNELS {
            let at = |window| OpAttrs { pos0: 39, scale: 1.0, window, ..OpAttrs::default() };
            let full = run(p, &attn_call(Op::AttnDecode, &q1, &kc, &vc, dims1, at(0)));
            let big = run(p, &attn_call(Op::AttnDecode, &q1, &kc, &vc, dims1, at(1024)));
            assert_eq!(full[0].data, big[0].data, "{p} window 1024 == no window");
            let small = run(p, &attn_call(Op::AttnDecode, &q1, &kc, &vc, dims1, at(5)));
            assert_ne!(full[0].data, small[0].data, "{p} window 5 must change the result");
        }
    }
}

#[test]
fn chunked_windowed_prefill_equals_unchunked() {
    let (n, nkv, d, cap, total) = (16usize, 2usize, 512usize, 32usize, 12usize);
    let mut rng = SplitMix64::new(77);
    let kc = rnd(&mut rng, nkv * cap * d, 1.0);
    let vc = rnd(&mut rng, nkv * cap * d, 1.0);
    let q = rnd(&mut rng, total * n * d, 1.0);
    for window in [0usize, 5, 1024] {
        let at = |pos0| OpAttrs { pos0, scale: 1.0, window, ..OpAttrs::default() };
        for p in KERNELS {
            let whole = run(p, &attn_call(Op::AttnPrefill, &q, &kc, &vc, (total, n, nkv, d, cap), at(0)))[0].to_f32();
            let split = 7;
            let a = run(p, &attn_call(Op::AttnPrefill, &q[..split * n * d], &kc, &vc, (split, n, nkv, d, cap), at(0)))[0].to_f32();
            let b = run(p, &attn_call(Op::AttnPrefill, &q[split * n * d..], &kc, &vc, (total - split, n, nkv, d, cap), at(split)))[0].to_f32();
            let chunked: Vec<f32> = a.into_iter().chain(b).collect();
            assert_eq!(whole.len(), chunked.len());
            for (i, (x, y)) in whole.iter().zip(&chunked).enumerate() {
                assert!((x - y).abs() <= 1e-6 * (1.0 + x.abs()), "{p} window={window} [{i}]: {x} vs {y}");
            }
            // each row of the second chunk equals the same position decoded alone
            let last = total - 1;
            let one = run(p, &attn_call(Op::AttnDecode, &q[last * n * d..], &kc, &vc, (1, n, nkv, d, cap), at(last)))[0].to_f32();
            let tail = &whole[last * n * d..];
            for (i, (x, y)) in tail.iter().zip(&one).enumerate() {
                assert!((x - y).abs() <= 1e-6 * (1.0 + x.abs()), "{p} window={window} decode [{i}]: {x} vs {y}");
            }
        }
    }
}

// ---------------------------------------------------------------- Add / Embedding scale

#[test]
fn add_and_embedding_scale_round_once() {
    let scale = 1.732_050_8f32; // not a power of two, so a double rounding would show
    for dt in [DType::Bf16, DType::F32, DType::F16] {
        let mut rng = SplitMix64::new(3);
        let a = round_to(dt, &rnd(&mut rng, 2 * 50, 3.0));
        let b = round_to(dt, &rnd(&mut rng, 2 * 50, 3.0));
        let want = round_to(dt, &a.iter().zip(&b).map(|(x, y)| (x + y) * scale).collect::<Vec<_>>());
        let hc = HostCall {
            op: Op::Add,
            inputs: vec![HostTensor::from_f32_as(dt, &[2, 50], &a), HostTensor::from_f32_as(dt, &[2, 50], &b)],
            outputs: vec![OutSpec::new(dt, &[2, 50])],
            attrs: OpAttrs { scale, ..OpAttrs::default() },
        };
        for p in KERNELS {
            assert_eq!(run(p, &hc)[0].to_f32(), want, "{p}add scale {dt:?}");
        }
        // unscaled Add is unchanged (plain rounded sum)
        let plain = HostCall { attrs: OpAttrs::default(), ..hc.clone() };
        let want_plain = round_to(dt, &a.iter().zip(&b).map(|(x, y)| x + y).collect::<Vec<_>>());
        for p in KERNELS {
            assert_eq!(run(p, &plain)[0].to_f32(), want_plain, "{p}add unscaled {dt:?}");
        }

        let table = round_to(dt, &rnd(&mut rng, 6 * 9, 2.0));
        let ids = [4i32, 0, 5, 4];
        let mut want_e = Vec::new();
        for id in ids {
            want_e.extend(table[id as usize * 9..(id as usize + 1) * 9].iter().map(|v| v * scale));
        }
        let want_e = round_to(dt, &want_e);
        let he = HostCall {
            op: Op::Embedding,
            inputs: vec![HostTensor::from_i32(&[4], &ids), HostTensor::from_f32_as(dt, &[6, 9], &table)],
            outputs: vec![OutSpec::new(dt, &[4, 9])],
            attrs: OpAttrs { scale, ..OpAttrs::default() },
        };
        for p in KERNELS {
            assert_eq!(run(p, &he)[0].to_f32(), want_e, "{p}embedding scale {dt:?}");
        }
        // table in a different dtype than the output (f32 table -> bf16 activations) also scales before the single rounding
        if dt == DType::Bf16 {
            let t32 = rnd(&mut rng, 6 * 9, 2.0);
            let mut w = Vec::new();
            for id in ids {
                w.extend(t32[id as usize * 9..(id as usize + 1) * 9].iter().map(|v| v * scale));
            }
            let w = round_to(DType::Bf16, &w);
            let hm = HostCall {
                inputs: vec![HostTensor::from_i32(&[4], &ids), HostTensor::from_f32(&[6, 9], &t32)],
                ..he.clone()
            };
            for p in KERNELS {
                assert_eq!(run(p, &hm)[0].to_f32(), w, "{p}embedding scale f32 table -> bf16");
            }
        }
    }
}

// ---------------------------------------------------------------- MoeExperts gelu

/// Naive expert FFN on the dequantised weights in f64: sum_k w_k * down(act(gate x) * (up x)).
fn naive_moe(hc: &HostCall, t: usize, e: usize, k: usize, h: usize, inter: usize, gelu: bool) -> Vec<f64> {
    let x = hc.inputs[0].to_f32();
    let ids = hc.inputs[1].to_i32().unwrap();
    let wts = hc.inputs[2].to_f32();
    let (guw, gus, dw, ds) = (&hc.inputs[3].data, &hc.inputs[4].data, &hc.inputs[5].data, &hc.inputs[6].data);
    let gs = hc.inputs[7].to_f32();
    let mut out = vec![0f64; t * h];
    let mut row = vec![0f64; h.max(inter)];
    for ti in 0..t {
        for kk in 0..k {
            let id = ids[ti * k + kk];
            if id < 0 || id as usize >= e {
                continue;
            }
            let ex = id as usize;
            let g3 = &gs[ex * 3..ex * 3 + 3];
            let gw = &guw[ex * 2 * inter * (h / 2)..(ex + 1) * 2 * inter * (h / 2)];
            let gsb = &gus[ex * 2 * inter * (h / 16)..(ex + 1) * 2 * inter * (h / 16)];
            let mut act = vec![0f64; inter];
            for r in 0..inter {
                nvfp4_row(gw, gsb, g3[0], r, h, &mut row);
                let g: f64 = (0..h).map(|c| x[ti * h + c] as f64 * row[c]).sum();
                nvfp4_row(gw, gsb, g3[1], inter + r, h, &mut row);
                let u: f64 = (0..h).map(|c| x[ti * h + c] as f64 * row[c]).sum();
                act[r] = if gelu { naive_gelu(g) } else { g / (1.0 + (-g).exp()) } * u;
            }
            let dww = &dw[ex * h * (inter / 2)..(ex + 1) * h * (inter / 2)];
            let dss = &ds[ex * h * (inter / 16)..(ex + 1) * h * (inter / 16)];
            for c in 0..h {
                nvfp4_row(dww, dss, g3[2], c, inter, &mut row);
                let y: f64 = (0..inter).map(|r| act[r] * row[r]).sum();
                out[ti * h + c] += wts[ti * k + kk] as f64 * y;
            }
        }
    }
    out
}

#[test]
fn moe_experts_gelu_matches_naive() {
    let (t, e, k, h, inter) = (3usize, 4usize, 2usize, 64usize, 64usize);
    let mut rng = SplitMix64::new(9);
    let _ = random_experts; // layout helper used by experts_case
    let mut hc = experts_case(&mut rng, t, e, k, h, inter, DType::F32);
    for gelu in [false, true] {
        hc.attrs.gelu = gelu;
        let want = naive_moe(&hc, t, e, k, h, inter, gelu);
        for p in KERNELS {
            let got = run(p, &hc)[0].to_f32();
            assert_close(&got, &want, 2e-4, &format!("{p}moe_experts gelu={gelu}"));
        }
    }
    // the two activations really differ on this data
    let a = naive_moe(&hc, t, e, k, h, inter, false);
    let b = naive_moe(&hc, t, e, k, h, inter, true);
    assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3), "silu and gelu experts must differ");
}

// ---------------------------------------------------------------- registration

#[test]
fn only_kernels_that_implement_the_extension_are_ext() {
    let b = CpuBackend::new();
    let ks = Backend::kernels(&b);
    let ext_ops = [Op::Embedding, Op::Add, Op::Rope, Op::AttnPrefill, Op::AttnDecode, Op::GeluGlu, Op::MoeExperts];
    for k in &ks {
        assert_eq!(k.ext, ext_ops.contains(&k.op), "{}", k.name);
    }
    // selection: an extended call only reaches an ext kernel; fused MoE (does not forward gelu) is never ext
    let reg = Registry::new("cpu", &Backend::info(&b), ks);
    let t = |dt, shape: &[usize]| TensorRef::new(b.alloc(shape.iter().product::<usize>() * 4, intarsia_core::backend::MemKind::Device).unwrap(), dt, shape);
    let call = |op, attrs| {
        OpCall::new(op, vec![t(DType::F32, &[1, 2, 8]), t(DType::F32, &[1, 4, 8]), t(DType::F32, &[1, 4, 8])], vec![t(DType::F32, &[1, 2, 8])], attrs)
    };
    let attn = call(Op::AttnDecode, OpAttrs { window: 3, n_heads: 2, n_kv_heads: 1, head_dim: 8, kv_cap: 4, ..OpAttrs::default() });
    assert!(reg.select(&attn).unwrap().ext);
    let plain = call(Op::AttnDecode, OpAttrs { n_heads: 2, n_kv_heads: 1, head_dim: 8, kv_cap: 4, ..OpAttrs::default() });
    assert!(reg.select(&plain).is_ok());
}

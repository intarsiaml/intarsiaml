//! P12 (lossless 12-bit bf16) GEMVs on HIP (lane ink-hip-p12, kernels/hip/ink_p12.hip): each P12 kernel against its bf16 HIP twin,
//! bit for bit, at the real Inkling-2-Mini shapes and every row count 1..=16 (wo [2304, 3072], the unembed [200058, 2304], the
//! in-projection stack [6552 + 1056 + 1056, 2304] at S = 1 and 3), every launch shape (`inkp12::set_p12_shape`) and the unembed
//! split on and off (`set_dec3`); plus small weight-like and adversarial shapes. And the microbench of the lane's step 1:
//!   cargo test --release -p intarsia-backend-hip --test ink_p12 -- --test-threads=1 --nocapture            (bitwise)
//!   P12_BENCH=1 cargo test --release -p intarsia-backend-hip --test ink_p12 p12_microbench -- --nocapture  (timing)
#![cfg(ik_hip)]

mod common;

use intarsia_backend_hip::inkp12::set_p12_shape;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::p12;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::Arc;

/// the process-wide launch-shape / dec3 switches are held while a test flips them
static SHAPE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn backend(test: &str) -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip(test, &format!("no usable HIP device ({e})"));
            None
        }
    }
}

fn put(b: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = b.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    b.zero(&buf, 0).unwrap();
    if t.nbytes() > 0 {
        b.upload(&buf, 0, &t.data, 0).unwrap();
    }
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn out_t(b: &dyn Backend, dt: DType, shape: &[usize]) -> TensorRef {
    let mut t = HostTensor::zeros(dt, shape);
    t.data.fill(0xA5); // NaN in bf16 / f32: an unwritten output cannot pass
    put(b, &t)
}

fn get(b: &dyn Backend, o: &TensorRef) -> Vec<u8> {
    let mut data = vec![0u8; o.nbytes()];
    b.download(&o.buf, o.offset, &mut data, 0).unwrap();
    data
}

/// Launch `kernel` on device inputs (`exact_rows` for the `_rows` names) and return the outputs' bytes.
fn run(b: &dyn Backend, kernel: &str, op: Op, ins: &[TensorRef], outs: &[(DType, Vec<usize>)]) -> Vec<Vec<u8>> {
    let outputs: Vec<TensorRef> = outs.iter().map(|(dt, s)| out_t(b, *dt, s)).collect();
    let mut call = OpCall::new(op, ins.to_vec(), outputs.clone(), OpAttrs::default());
    call.exact_rows = kernel.ends_with("_rows");
    call.check().unwrap_or_else(|e| panic!("{kernel}: {e}"));
    let k = b.kernels().into_iter().find(|k| k.name == kernel && k.op == op).unwrap_or_else(|| panic!("no {kernel}"));
    assert!(k.accepts(&ShapeInfo::from_call(&call)), "{kernel} refuses the call");
    b.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    b.synchronize(0).unwrap();
    outputs.iter().map(|o| get(b, o)).collect()
}

/// A fast weight-like bf16 [n, k]: sign and mantissa random, exponents 121 - (0..14) with geometric weights (values over ~14
/// binades below 2^-6), and about 8e-5 of the elements far below the window (an escape), some rows with an escape in the first and
/// the last chunk. (`p12::random_weight` at 461 M elements spends seconds of a GPU window on Box-Muller.)
fn fast_weight(seed: u64, n: usize, k: usize) -> Vec<u16> {
    let mut w = vec![0u16; n * k];
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get()).min(32);
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (t, ch) in w.chunks_mut(per * k).enumerate() {
            sc.spawn(move || {
                let mut s = seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(t as u64 + 1));
                for (i, v) in ch.iter_mut().enumerate() {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let r = s;
                    let row = t * per + i / k;
                    let col = i % k;
                    let esc = (r >> 40) % 12000 == 0 || (row % 97 == 5 && (col == 0 || col == k - 1));
                    let e: u16 = if esc { [0u16, 40, 60, 200][(r >> 20) as usize & 3] } else { 121 - ((r.trailing_zeros() as u16) % 14) };
                    *v = ((r >> 8) as u16 & 0x8000) | e << 7 | ((r >> 24) as u16 & 0x7F);
                }
            });
        }
    });
    w
}

fn bf16_tensor(shape: &[usize], bits: &[u16]) -> HostTensor {
    HostTensor::new(DType::Bf16, shape.to_vec(), bits.iter().flat_map(|v| v.to_le_bytes()).collect()).unwrap()
}

fn act(rng: &mut SplitMix64, shape: &[usize]) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), 1.0))
}

/// A weight on the device as bf16 and as P12 (codes, aux).
struct W {
    n: usize,
    k: usize,
    bf: TensorRef,
    codes: TensorRef,
    aux: TensorRef,
    escapes: usize,
}

fn weight(b: &dyn Backend, bits: &[u16], n: usize, k: usize) -> W {
    let p = p12::pack(bits, n, k, 0).unwrap();
    let parts = p.host_parts();
    W { n, k, bf: put(b, &bf16_tensor(&[n, k], bits)), codes: put(b, &parts[0]), aux: put(b, &parts[1]), escapes: p.escapes() }
}

/// MatmulP12 against the bf16 twin at rows `ms`: (bf16 kernel, P12 kernel) pairs per output dtype, ordinary (<= 4 rows) and rows.
fn check_matmul(b: &dyn Backend, rng: &mut SplitMix64, w: &W, dt: DType, ms: &[usize], what: &str) -> usize {
    let (bk, pk, bop) = if dt == DType::F32 { ("logits_bf16_gemv", "matmul_p12_logits_gemv", Op::Logits) } else { ("matmul_bf16_gemv", "matmul_p12_gemv", Op::Matmul) };
    let mut n = 0;
    let mut one_row: Option<(HostTensor, Vec<u8>)> = None;
    for &m in ms {
        let xh = act(rng, &[m, w.k]);
        let x = put(b, &xh);
        let outs = [(dt, vec![m, w.n])];
        for sfx in ["", "_rows"] {
            if sfx.is_empty() && m > 4 {
                continue;
            }
            let want = run(b, &format!("{bk}{sfx}"), bop, &[x.clone(), w.bf.clone()], &outs);
            let got = run(b, &format!("{pk}{sfx}"), Op::MatmulP12, &[x.clone(), w.codes.clone(), w.aux.clone()], &outs);
            assert_eq!(got[0], want[0], "{what}: {pk}{sfx} M={m} differs from {bk}{sfx}");
            n += 1;
            if m == 1 && sfx.is_empty() {
                one_row = Some((xh.clone(), want[0].clone()));
            }
        }
    }
    // rows-exact: row m of a 16-row call (when run) is the one-row call on that row
    if let (Some((x1, o1)), true) = (one_row, ms.contains(&16)) {
        let mut xs = x1.data.clone();
        let x16 = act(rng, &[15, w.k]);
        xs.extend_from_slice(&x16.data);
        let x = put(b, &HostTensor::new(DType::Bf16, vec![16, w.k], xs).unwrap());
        let got = run(b, &format!("{pk}_rows"), Op::MatmulP12, &[x, w.codes.clone(), w.aux.clone()], &[(dt, vec![16, w.n])]);
        assert_eq!(&got[0][..o1.len()], &o1[..], "{what}: row 0 of a 16-row call differs from the one-row call");
        n += 1;
    }
    n
}

fn check_in_proj(b: &dyn Backend, rng: &mut SplitMix64, w: &W, ns: [usize; 3], s: usize, ms: &[usize], what: &str) -> usize {
    let mut n = 0;
    for &t in ms {
        let x = put(b, &act(rng, &[s, t, w.k]));
        let outs: Vec<(DType, Vec<usize>)> = ns.iter().map(|&c| (DType::Bf16, vec![t, c])).collect();
        for sfx in ["", "_rows"] {
            let want = run(b, &format!("ink_in_proj_bf16_gemv{sfx}"), Op::InkInProj, &[x.clone(), w.bf.clone()], &outs);
            let got = run(b, &format!("ink_in_proj_p12_gemv{sfx}"), Op::InkInProj, &[x.clone(), w.codes.clone(), w.aux.clone()], &outs);
            for j in 0..3 {
                assert_eq!(got[j], want[j], "{what}: ink_in_proj_p12_gemv{sfx} T={t} S={s} segment {j} differs from bf16");
            }
            n += 1;
        }
    }
    n
}

const GEMV_SHAPES_ONE: [(i32, i32); 7] = [(1, 8), (1, 4), (1, 9), (1, 12), (1, 16), (2, 4), (2, 8)];
const GEMV_SHAPES_MANY: [(i32, i32); 4] = [(2, 4), (2, 8), (1, 4), (1, 8)];

/// Gate 1: the real shapes, rows 1..=16, every launch shape and the split on / off.
#[test]
fn p12_bitwise_real_shapes() {
    let Some(be) = backend("p12_bitwise_real_shapes") else { return };
    let b = &*be;
    let _g = SHAPE.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = SplitMix64::new(0x12_12);
    let all: Vec<usize> = (1..=16).collect();
    let mut n = 0;
    // wo [2304, 3072] (bf16 out)
    let wo = weight(b, &p12::random_weight(&mut rng, 2304, 3072, (3072f32).powf(-0.5), false), 2304, 3072);
    n += check_matmul(b, &mut rng, &wo, DType::Bf16, &all, "wo");
    // the unembed [200058, 2304] (f32 out), dec3 default (split) and dec3 = 3 (one launch)
    let ue = weight(b, &fast_weight(0xe1, 200_058, 2304), 200_058, 2304);
    eprintln!("unembed: {} escapes", ue.escapes);
    n += check_matmul(b, &mut rng, &ue, DType::F32, &all, "unembed");
    intarsia_backend_hip::set_dec3(3);
    n += check_matmul(b, &mut rng, &ue, DType::F32, &[1, 2, 4, 9, 16], "unembed one launch");
    intarsia_backend_hip::set_dec3(-1);
    // the in-projection stack at S = 1 and 3
    let ns = [6552usize, 1056, 1056];
    let ip = weight(b, &p12::random_weight(&mut rng, 8664, 2304, (2304f32).powf(-0.5), false), 8664, 2304);
    for s in [1usize, 3] {
        n += check_in_proj(b, &mut rng, &ip, ns, s, &all, "in_proj");
    }
    // every launch shape (the same chain, other schedules)
    for &(r, u) in &GEMV_SHAPES_ONE {
        set_p12_shape(Some((r, u)), -1);
        n += check_matmul(b, &mut rng, &wo, DType::Bf16, &[1], &format!("wo shape {r},{u}"));
        n += check_matmul(b, &mut rng, &ue, DType::F32, &[1], &format!("unembed shape {r},{u}"));
    }
    for &(r, u) in &GEMV_SHAPES_MANY {
        set_p12_shape(Some((r, u)), -1);
        n += check_matmul(b, &mut rng, &wo, DType::Bf16, &[2, 4, 8], &format!("wo shape {r},{u}"));
        n += check_matmul(b, &mut rng, &ue, DType::F32, &[4, 8], &format!("unembed shape {r},{u}"));
    }
    for u in [4, 8, 9] {
        set_p12_shape(None, u);
        n += check_in_proj(b, &mut rng, &ip, ns, 3, &[1, 4, 16], &format!("in_proj u={u}"));
    }
    set_p12_shape(None, -1);
    eprintln!("p12_bitwise_real_shapes: {n} cases bitwise");
}

/// Small shapes with K tails and the adversarial rows (`p12::random_weight`: +-0, subnormals, all-escape rows, escapes in the first
/// and last chunk, values far above the window).
#[test]
fn p12_bitwise_small_and_adversarial() {
    let Some(be) = backend("p12_bitwise_small_and_adversarial") else { return };
    let b = &*be;
    let _g = SHAPE.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = SplitMix64::new(0xad);
    let mut n = 0;
    for &(nr, k) in &[(64usize, 64usize), (37, 96), (130, 2304), (61, 3072), (5000, 2304)] {
        for adv in [false, true] {
            let w = weight(b, &p12::random_weight(&mut rng, nr, k, (k as f32).powf(-0.5), adv), nr, k);
            for dt in [DType::Bf16, DType::F32] {
                n += check_matmul(b, &mut rng, &w, dt, &[1, 2, 3, 4, 5, 8, 9, 16], &format!("N={nr} K={k} adv={adv}"));
            }
        }
    }
    for &(d, ns) in &[(64usize, [246usize, 40, 40]), (2304, [130, 17, 17])] {
        for adv in [false, true] {
            let w = weight(b, &p12::random_weight(&mut rng, ns.iter().sum(), d, (d as f32).powf(-0.5), adv), ns.iter().sum(), d);
            for s in [1usize, 3] {
                n += check_in_proj(b, &mut rng, &w, ns, s, &[1, 2, 7, 16], &format!("in_proj D={d} adv={adv}"));
            }
        }
    }
    eprintln!("p12_bitwise_small_and_adversarial: {n} cases bitwise");
}

// ---- step 1 microbench -------------------------------------------------------------------------------------------------------

/// Mean microseconds per call of `kernel` over `calls` (rotated: each call's weight is another copy, so the 64 MB Infinity Cache
/// does not hold it, as in a decode step where every other weight streams through between two visits of one layer's).
fn time_calls(b: &dyn Backend, kernel: &str, calls: &[OpCall], iters: usize) -> f64 {
    for c in calls.iter().cycle().take(calls.len().max(8)) {
        b.launch(kernel, c, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    let t0 = std::time::Instant::now();
    for c in calls.iter().cycle().take(iters) {
        b.launch(kernel, c, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

struct Copies {
    bf: Vec<TensorRef>,
    p12: Vec<(TensorRef, TensorRef)>,
    bf_bytes: usize,
    p12_bytes: usize,
}

fn copies(b: &dyn Backend, bits: &[u16], n: usize, k: usize, min_bytes: usize) -> Copies {
    let p = p12::pack(bits, n, k, 0).unwrap();
    let parts = p.host_parts();
    let bft = bf16_tensor(&[n, k], bits);
    let cnt = min_bytes.div_ceil(bft.nbytes()).max(1);
    Copies {
        bf: (0..cnt).map(|_| put(b, &bft)).collect(),
        p12: (0..cnt).map(|_| (put(b, &parts[0]), put(b, &parts[1]))).collect(),
        bf_bytes: bft.nbytes(),
        p12_bytes: p.bytes(),
    }
}

fn mk_calls(op: Op, x: &TensorRef, ws: Vec<Vec<TensorRef>>, outs: &[TensorRef], exact: bool) -> Vec<OpCall> {
    ws.into_iter()
        .map(|w| {
            let mut ins = vec![x.clone()];
            ins.extend(w);
            let mut c = OpCall::new(op, ins, outs.to_vec(), OpAttrs::default());
            c.exact_rows = exact;
            c.check().unwrap();
            c
        })
        .collect()
}

#[test]
fn p12_microbench() {
    if std::env::var_os("P12_BENCH").is_none() {
        eprintln!("skipping: P12_BENCH unset");
        return;
    }
    let Some(be) = backend("p12_microbench") else { return };
    let b = &*be;
    let _g = SHAPE.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = SplitMix64::new(0xbe12);
    let ms: Vec<usize> = std::env::var("P12_MS").ok().map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 4]);
    let iters = 200usize;
    println!("BENCH shape | M | kernel / variant | us per call | GB/s of weight bytes | P12 speedup over bf16");
    // (name, N, K, out dtype, bf16 kernel, P12 kernel, op)
    let wo_bits = p12::random_weight(&mut rng, 2304, 3072, (3072f32).powf(-0.5), false);
    let ue_bits = fast_weight(0xe2, 200_058, 2304);
    for (name, n, k, bits, dt) in [("wo", 2304usize, 3072usize, &wo_bits, DType::Bf16), ("unembed", 200_058, 2304, &ue_bits, DType::F32)] {
        let cp = copies(b, bits, n, k, 400 << 20);
        let (bk, pk, bop) = if dt == DType::F32 { ("logits_bf16_gemv", "matmul_p12_logits_gemv", Op::Logits) } else { ("matmul_bf16_gemv", "matmul_p12_gemv", Op::Matmul) };
        for &m in &ms {
            let x = put(b, &act(&mut rng, &[m, k]));
            let out = out_t(b, dt, &[m, n]);
            let sfx = if m > 4 { "_rows" } else { "" };
            let it = if name == "unembed" { 40 } else { iters };
            let bc = mk_calls(bop, &x, cp.bf.iter().map(|w| vec![w.clone()]).collect(), &[out.clone()], m > 4);
            let tb = time_calls(b, &format!("{bk}{sfx}"), &bc, it);
            println!("BENCH {name} | {m} | {bk}{sfx} | {tb:.2} | {:.1} | 1.000", cp.bf_bytes as f64 / tb / 1e3);
            let pc = mk_calls(Op::MatmulP12, &x, cp.p12.iter().map(|(c, a)| vec![c.clone(), a.clone()]).collect(), &[out.clone()], m > 4);
            let shapes: Vec<Option<(i32, i32)>> = match m {
                1 => std::iter::once(None).chain(GEMV_SHAPES_ONE[1..].iter().map(|s| Some(*s))).collect(),
                2..=8 => std::iter::once(None).chain(GEMV_SHAPES_MANY[1..].iter().map(|s| Some(*s))).collect(),
                _ => vec![None],
            };
            for sh in shapes {
                set_p12_shape(sh, -1);
                let tp = time_calls(b, &format!("{pk}{sfx}"), &pc, it);
                let lbl = sh.map_or("default".to_string(), |(r, u)| format!("R={r},U={u}"));
                println!("BENCH {name} | {m} | {pk}{sfx} {lbl} | {tp:.2} | {:.1} | {:.3}", cp.p12_bytes as f64 / tp / 1e3, tb / tp);
            }
            set_p12_shape(None, -1);
            if name == "unembed" && m == 1 {
                // one launch (dec3 bit 4 off) for both, to see the stall's share
                intarsia_backend_hip::set_dec3(3);
                let tb1 = time_calls(b, bk, &bc, it);
                let tp1 = time_calls(b, pk, &pc, it);
                intarsia_backend_hip::set_dec3(-1);
                println!("BENCH {name} | {m} | one launch each (dec3=3): bf16 {tb1:.2} / P12 {tp1:.2} | | {:.3}", tb1 / tp1);
            }
        }
    }
    // the in-projection stack at S = 1 (decode) and 3 (the T = 1 verify / draft form takes S = 1 too)
    let ns = [6552usize, 1056, 1056];
    let ip_bits = p12::random_weight(&mut rng, 8664, 2304, (2304f32).powf(-0.5), false);
    let cp = copies(b, &ip_bits, 8664, 2304, 400 << 20);
    for &m in &ms {
        let x = put(b, &act(&mut rng, &[1, m, 2304]));
        let outs: Vec<TensorRef> = ns.iter().map(|&c| out_t(b, DType::Bf16, &[m, c])).collect();
        let bc = mk_calls(Op::InkInProj, &x, cp.bf.iter().map(|w| vec![w.clone()]).collect(), &outs, false);
        let tb = time_calls(b, "ink_in_proj_bf16_gemv", &bc, iters);
        println!("BENCH in_proj | {m} | ink_in_proj_bf16_gemv | {tb:.2} | {:.1} | 1.000", cp.bf_bytes as f64 / tb / 1e3);
        let pc = mk_calls(Op::InkInProj, &x, cp.p12.iter().map(|(c, a)| vec![c.clone(), a.clone()]).collect(), &outs, false);
        for u in [0, 4, 8, 9] {
            set_p12_shape(None, u);
            let tp = time_calls(b, "ink_in_proj_p12_gemv", &pc, iters);
            println!("BENCH in_proj | {m} | ink_in_proj_p12_gemv u={u} | {tp:.2} | {:.1} | {:.3}", cp.p12_bytes as f64 / tp / 1e3, tb / tp);
        }
        set_p12_shape(None, -1);
    }
}

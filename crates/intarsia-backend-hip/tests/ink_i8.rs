//! Int8 per-row GEMVs on HIP (lane ink-hip-i8, kernels/hip/ink_i8.hip): each kernel against the f64 cpu reference contract
//! (`rA(scale[n] * sum_k x[m,k] * q[n,k])`) at the real Inkling-2-Mini shapes (wo [2304, 3072], the unembed [200058, 2304], the
//! in-projection stack [6552 + 1056 + 1056, 2304] at S = 1 and 3), every row count 1..=16 and every instantiated launch shape
//! (`inki8::set_i8_shape`); rows-exactness (row m of an M-row call equals the one-row call); and the lane's step-2 microbench against
//! the P12 kernels of main:
//!   cargo test --release -p intarsia-backend-hip --test ink_i8 -- --test-threads=1 --nocapture               (correctness)
//!   I8_BENCH=1 cargo test --release -p intarsia-backend-hip --test ink_i8 i8_microbench -- --nocapture       (timing)
#![cfg(ik_hip)]
mod common;

use intarsia_backend_hip::inki8::set_i8_shape;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::p12;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::Arc;

/// the process-wide launch-shape switch is held while a test flips it
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

fn get_f32(b: &dyn Backend, o: &TensorRef) -> Vec<f32> {
    let mut data = vec![0u8; o.nbytes()];
    b.download(&o.buf, o.offset, &mut data, 0).unwrap();
    let mut v = vec![0f32; o.numel()];
    intarsia_core::dtype::decode_to_f32(o.dtype, &data, &mut v);
    v
}

/// A fast random i8 weight [n, k] (codes uniform in -127..=127; a few rows with the extreme codes in the first and the last chunk) and
/// per-row scales in [2e-4, 1e-3]: the kernel does not care how the codes were chosen.
fn fast_i8(seed: u64, n: usize, k: usize) -> (HostTensor, HostTensor) {
    let mut q = vec![0u8; n * k];
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get()).min(32);
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (t, ch) in q.chunks_mut(per * k).enumerate() {
            sc.spawn(move || {
                let mut s = seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(t as u64 + 1));
                for (i, v) in ch.iter_mut().enumerate() {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let row = t * per + i / k;
                    let col = i % k;
                    let c = ((s >> 20) % 255) as i32 - 127;
                    *v = if row % 53 == 7 && col == 0 {
                        127
                    } else if row % 53 == 7 && col == k - 1 {
                        (-127i8) as u8
                    } else {
                        c as i8 as u8
                    };
                }
            });
        }
    });
    let mut r = SplitMix64::new(seed ^ 0x5ca1e);
    let sc: Vec<f32> = (0..n).map(|_| 2e-4 + 8e-4 * (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32).collect();
    (HostTensor::new(DType::I8, vec![n, k], q).unwrap(), HostTensor::from_f32(&[n, 1], &sc))
}

fn act(rng: &mut SplitMix64, shape: &[usize]) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), 1.0))
}

/// f64 reference rows `rows` of `x [m, k] . q^T * scale`, rounded to the output dtype.
fn reference(x: &[f32], m: usize, k: usize, q: &HostTensor, sc: &[f32], rows: &[usize], odt: DType) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|&n| {
            (0..m)
                .map(|mi| {
                    let s: f64 = (0..k).map(|ki| x[mi * k + ki] as f64 * (q.data[n * k + ki] as i8) as f64).sum();
                    intarsia_core::quant::reference::round(odt, s * sc[n] as f64)
                })
                .collect()
        })
        .collect()
}

/// tolerance of the f32-accumulating kernel against the f64 reference after the output rounding
fn close(g: f32, r: f64, odt: DType, scale: f64) -> bool {
    let (rel, abs) = if odt == DType::F32 { (2e-5, 2e-6) } else { (9e-3, 1e-3) };
    ((g as f64) - r).abs() <= rel * r.abs() + abs * scale
}

/// rows to check of an n-row weight: all when small, else the first / last 40 and a stride through the middle
fn pick_rows(n: usize) -> Vec<usize> {
    if n <= 400 {
        return (0..n).collect();
    }
    let mut v: Vec<usize> = (0..40).chain(n - 40..n).collect();
    v.extend((40..n - 40).step_by(n / 300 + 1));
    v
}

fn launch(b: &dyn Backend, kernel: &str, op: Op, ins: &[TensorRef], outs: &[TensorRef]) {
    let mut call = OpCall::new(op, ins.to_vec(), outs.to_vec(), OpAttrs::default());
    call.exact_rows = kernel.ends_with("_rows");
    call.check().unwrap_or_else(|e| panic!("{kernel}: {e}"));
    let k = b.kernels().into_iter().find(|k| k.name == kernel && k.op == op).unwrap_or_else(|| panic!("no {kernel}"));
    assert!(k.accepts(&ShapeInfo::from_call(&call)), "{kernel}: registration does not accept the call");
    b.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    b.synchronize(0).unwrap();
}

/// `MatmulI8` at rows `ms` against the f64 reference, and rows-exactness (bitwise up to a signed zero) against the one-row call.
fn check_matmul(b: &dyn Backend, rng: &mut SplitMix64, n: usize, k: usize, odt: DType, ms: &[usize], what: &str) -> usize {
    let (q, sc) = fast_i8(0x1234 + n as u64 + k as u64, n, k);
    let (qd, sd) = (put(b, &q), put(b, &sc));
    let scv = sc.to_f32();
    let rows = pick_rows(n);
    let mut checked = 0;
    let (ok, okr) = if odt == DType::F32 { ("matmul_i8_logits_gemv", "matmul_i8_logits_gemv_rows") } else { ("matmul_i8_gemv", "matmul_i8_gemv_rows") };
    for &m in ms {
        let xh = act(rng, &[m, k]);
        let xv = xh.to_f32();
        let x = put(b, &xh);
        let out = out_t(b, odt, &[m, n]);
        let kernel = if m <= 4 { ok } else { okr };
        launch(b, kernel, Op::MatmulI8, &[x.clone(), qd.clone(), sd.clone()], &[out.clone()]);
        let got = get_f32(b, &out);
        let want = reference(&xv, m, k, &q, &scv, &rows, odt);
        let maxabs = want.iter().flatten().fold(0f64, |a, v| a.max(v.abs()));
        for (ri, &r) in rows.iter().enumerate() {
            for mi in 0..m {
                let (g, w) = (got[mi * n + r], want[ri][mi]);
                assert!(close(g, w, odt, maxabs), "{what} M={m} row {r} m {mi}: got {g} want {w}");
                checked += 1;
            }
        }
        // rows-exact: every row of the M-row call equals the one-row call of its activation row
        if m > 1 {
            for mi in 0..m.min(3) {
                let x1 = put(b, &HostTensor::from_f32_as(DType::Bf16, &[1, k], &xv[mi * k..(mi + 1) * k]));
                let o1 = out_t(b, odt, &[1, n]);
                launch(b, ok, Op::MatmulI8, &[x1, qd.clone(), sd.clone()], &[o1.clone()]);
                let g1 = get_f32(b, &o1);
                for r in 0..n {
                    assert!(got[mi * n + r] == g1[r], "{what} M={m}: row {mi} of the call differs from the one-row call at n={r}: {} vs {}", got[mi * n + r], g1[r]);
                }
            }
        }
    }
    checked
}

fn check_in_proj(b: &dyn Backend, rng: &mut SplitMix64, ns: [usize; 3], d: usize, ms: &[usize], what: &str) -> usize {
    let n: usize = ns.iter().sum();
    let (q, sc) = fast_i8(0xabc0 + n as u64, n, d);
    let (qd, sd) = (put(b, &q), put(b, &sc));
    let scv = sc.to_f32();
    let rows = pick_rows(n);
    let mut checked = 0;
    for s in [1usize, 3] {
        for &m in ms {
            let xh = act(rng, &[s, m, d]);
            let xv = xh.to_f32();
            let x = put(b, &xh);
            let outs: Vec<TensorRef> = ns.iter().map(|&c| out_t(b, DType::Bf16, &[m, c])).collect();
            let kernel = if m <= 4 { "ink_in_proj_i8_gemv" } else { "ink_in_proj_i8_gemv_rows" };
            launch(b, kernel, Op::InkInProj, &[x.clone(), qd.clone(), sd.clone()], &outs);
            let gots: Vec<Vec<f32>> = outs.iter().map(|o| get_f32(b, o)).collect();
            for &r in &rows {
                let (j, nn) = if r < ns[0] { (0, r) } else if r < ns[0] + ns[1] { (1, r - ns[0]) } else { (2, r - ns[0] - ns[1]) };
                let xs = &xv[j.min(s - 1) * m * d..(j.min(s - 1) + 1) * m * d];
                let w = reference(xs, m, d, &q, &scv, &[r], DType::Bf16);
                let maxabs = w[0].iter().fold(0f64, |a, v| a.max(v.abs()));
                for mi in 0..m {
                    let g = gots[j][mi * ns[j] + nn];
                    assert!(close(g, w[0][mi], DType::Bf16, maxabs.max(1.0)), "{what} S={s} M={m} row {r} m {mi}: got {g} want {}", w[0][mi]);
                    checked += 1;
                }
            }
        }
    }
    checked
}

const GEMV_SHAPES_FEW: [(i32, i32); 6] = [(1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (2, 8)];
const GEMV_SHAPES_MANY: [(i32, i32); 4] = [(1, 1), (1, 2), (2, 1), (2, 2)];

#[test]
fn i8_real_shapes_match_the_reference() {
    let Some(be) = backend("i8_real_shapes_match_the_reference") else { return };
    let b = &*be;
    let _g = SHAPE.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = SplitMix64::new(0x18);
    let all: Vec<usize> = (1..=16).collect();
    let mut checked = 0;
    checked += check_matmul(b, &mut rng, 2304, 3072, DType::Bf16, &all, "wo");
    checked += check_matmul(b, &mut rng, 200_058, 2304, DType::F32, &[1, 2, 4, 5, 8, 9, 16], "unembed");
    checked += check_in_proj(b, &mut rng, [6552, 1056, 1056], 2304, &all, "in_proj");
    println!("checked {checked} outputs");
}

#[test]
fn i8_small_shapes_tails_and_every_launch_shape() {
    let Some(be) = backend("i8_small_shapes_tails_and_every_launch_shape") else { return };
    let b = &*be;
    let _g = SHAPE.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = SplitMix64::new(0x19);
    let all: Vec<usize> = (1..=16).collect();
    let mut checked = 0;
    for &(n, k) in &[(64usize, 64usize), (37, 96), (130, 2304), (61, 3072), (1, 16), (9, 48)] {
        for odt in [DType::Bf16, DType::F32] {
            checked += check_matmul(b, &mut rng, n, k, odt, &all, &format!("small N={n} K={k} {odt}"));
        }
    }
    // every instantiated (R, U) at real wo and a tail-heavy shape
    for sh in GEMV_SHAPES_FEW.iter().chain(GEMV_SHAPES_MANY.iter()) {
        set_i8_shape(Some(*sh), -1);
        let ms: &[usize] = if GEMV_SHAPES_FEW.contains(sh) { &[1, 2, 4] } else { &[5, 8, 16] };
        checked += check_matmul(b, &mut rng, 2304, 3072, DType::Bf16, ms, &format!("wo R,U={sh:?}"));
        checked += check_matmul(b, &mut rng, 130, 2304, DType::F32, ms, &format!("tail R,U={sh:?}"));
    }
    set_i8_shape(None, -1);
    for u in 1..=5 {
        set_i8_shape(None, u);
        checked += check_in_proj(b, &mut rng, [24, 8, 8], 2304, &[1, 3, 4, 16], &format!("in_proj u={u}"));
    }
    set_i8_shape(None, -1);
    checked += check_in_proj(b, &mut rng, [24, 8, 8], 96, &all, "in_proj small");
    println!("checked {checked} outputs");
}

/// Mean microseconds per call of `kernel` over `calls` (rotated: each call's weight is another copy, so the 64 MB Infinity Cache does
/// not hold it, as in a decode step where every other weight streams through between two visits of one layer's).
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

/// A fast weight-like bf16 [n, k] for the P12 baseline (the lane ink-hip-p12 bench's generator: exponents 121 - (0..14), a few escapes).
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

#[test]
fn i8_microbench() {
    if std::env::var_os("I8_BENCH").is_none() {
        eprintln!("skipping: I8_BENCH unset");
        return;
    }
    let Some(be) = backend("i8_microbench") else { return };
    let b = &*be;
    let _g = SHAPE.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = SplitMix64::new(0xbe18);
    let ms: Vec<usize> = std::env::var("I8_MS").ok().map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 4]);
    let iters = 200usize;
    println!("BENCH shape | M | kernel / variant | us per call | GB/s of weight bytes | speedup over P12 (main's kernel)");
    // (name, N, K, out dtype)
    for (name, n, k, dt) in [("wo", 2304usize, 3072usize, DType::Bf16), ("unembed", 200_058, 2304, DType::F32), ("in_proj", 8664, 2304, DType::Bf16)] {
        let bits = if name == "wo" { p12::random_weight(&mut rng, n, k, (k as f32).powf(-0.5), false) } else { fast_weight(0xe2, n, k) };
        let p = p12::pack(&bits, n, k, 0).unwrap();
        let parts = p.host_parts();
        let (q, sc) = fast_i8(0x77 + n as u64, n, k);
        let min_bytes = 400usize << 20;
        let cnt_p = min_bytes.div_ceil(p.bytes()).max(2);
        let cnt_i = min_bytes.div_ceil(q.nbytes()).max(2);
        let pw: Vec<(TensorRef, TensorRef)> = (0..cnt_p).map(|_| (put(b, &parts[0]), put(b, &parts[1]))).collect();
        let iw: Vec<(TensorRef, TensorRef)> = (0..cnt_i).map(|_| (put(b, &q), put(b, &sc))).collect();
        let (p12_bytes, i8_bytes) = (p.bytes(), q.nbytes() + sc.nbytes());
        for &m in &ms {
            let (x, outs): (TensorRef, Vec<TensorRef>) = if name == "in_proj" {
                (put(b, &act(&mut rng, &[1, m, k])), [6552usize, 1056, 1056].iter().map(|&c| out_t(b, DType::Bf16, &[m, c])).collect())
            } else {
                (put(b, &act(&mut rng, &[m, k])), vec![out_t(b, dt, &[m, n])])
            };
            let exact = m > 4;
            let sfx = if exact { "_rows" } else { "" };
            let (pop, pk, ik) = match name {
                "wo" => (Op::MatmulP12, "matmul_p12_gemv", "matmul_i8_gemv"),
                "unembed" => (Op::MatmulP12, "matmul_p12_logits_gemv", "matmul_i8_logits_gemv"),
                _ => (Op::InkInProj, "ink_in_proj_p12_gemv", "ink_in_proj_i8_gemv"),
            };
            let iop = if name == "in_proj" { Op::InkInProj } else { Op::MatmulI8 };
            // P12 as main runs it: the kernel name the registration picks (the `_rows` name at > 4 rows; the unembed's rows kernel up to 8)
            let pname = format!("{pk}{sfx}");
            let it = if name == "unembed" { 40 } else { iters };
            let pc = mk_calls(pop, &x, pw.iter().map(|(c, a)| vec![c.clone(), a.clone()]).collect(), &outs, exact);
            let tp = time_calls(b, &pname, &pc, it);
            println!("BENCH {name} | {m} | {pname} (P12, main default) | {tp:.2} | {:.1} | 1.000", p12_bytes as f64 / tp / 1e3);
            let ic = mk_calls(iop, &x, iw.iter().map(|(c, a)| vec![c.clone(), a.clone()]).collect(), &outs, exact);
            let shapes: Vec<Option<(i32, i32)>> = if name == "in_proj" {
                vec![None]
            } else if m <= 4 {
                std::iter::once(None).chain(GEMV_SHAPES_FEW.iter().map(|s| Some(*s))).collect()
            } else {
                std::iter::once(None).chain(GEMV_SHAPES_MANY.iter().map(|s| Some(*s))).collect()
            };
            for sh in shapes {
                set_i8_shape(sh, -1);
                let ti = time_calls(b, &format!("{ik}{sfx}"), &ic, it);
                let lbl = sh.map_or("default".to_string(), |(r, u)| format!("R={r},U={u}"));
                println!("BENCH {name} | {m} | {ik}{sfx} (i8) {lbl} | {ti:.2} | {:.1} | {:.3}", i8_bytes as f64 / ti / 1e3, tp / ti);
            }
            set_i8_shape(None, -1);
            if name == "in_proj" {
                for u in 1..=5 {
                    set_i8_shape(None, u);
                    let ti = time_calls(b, &format!("{ik}{sfx}"), &ic, it);
                    println!("BENCH {name} | {m} | {ik}{sfx} (i8) u={u} | {ti:.2} | {:.1} | {:.3}", i8_bytes as f64 / ti / 1e3, tp / ti);
                }
                set_i8_shape(None, -1);
            }
        }
    }
}

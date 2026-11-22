#![cfg(target_os = "macos")]
//! P12 (lane ink-pack12, p12.rs): every P12 kernel is BITWISE its bf16 twin on the same weights, rows 1..=16, at the real
//! Inkling-2 shapes (wo [2304, 3072], the unembed [200058, 2304], the in-projection stacks [6552 + 1056 + 1056, 2304] and
//! [6552 + 1024 + 1024, 2304]), with weight-like rows (escapes where the checkpoint has them) and adversarial rows (zeros,
//! subnormals, a row of only escapes, escapes in the first / last chunk). A row of an R-row P12 call is also bitwise its 1-row
//! call (rows-exact). Numerics against the f64 reference are the admission sweep (`intarsia kernels test --backend metal`).

mod common;
use common::*;
use intarsia_backend_metal::p12::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::p12;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, Op, OpAttrs, OpCall};

fn up(be: &MetalBackend, dt: DType, shape: &[usize], data: &[u8]) -> TensorRef {
    let b = be.alloc(data.len().max(16), MemKind::Device).unwrap();
    if !data.is_empty() {
        be.upload(&b, 0, data, 0).unwrap();
    }
    TensorRef { buf: b, offset: 0, dtype: dt, shape: shape.to_vec() }
}

fn zeros(be: &MetalBackend, dt: DType, shape: &[usize]) -> TensorRef {
    up(be, dt, shape, &vec![0u8; shape.iter().product::<usize>() * dt.size_bytes()])
}

fn down(be: &MetalBackend, t: &TensorRef) -> Vec<u8> {
    let mut v = vec![0u8; t.numel() * t.dtype.size_bytes()];
    be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    v
}

fn bytes16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A weight `[n, k]` (bf16 bits) and its P12 parts on the device.
struct W {
    bf: TensorRef,
    codes: TensorRef,
    aux: TensorRef,
    escapes: usize,
}

fn weight(be: &MetalBackend, rng: &mut SplitMix64, n: usize, k: usize, adversarial: bool, seg_scales: &[(usize, f32)]) -> W {
    let mut w = Vec::with_capacity(n * k);
    for &(rows, sc) in seg_scales {
        w.extend(p12::random_weight(rng, rows, k, sc * (k as f32).powf(-0.5), adversarial));
    }
    assert_eq!(w.len(), n * k);
    let p = p12::pack(&w, n, k, 0).unwrap();
    assert_eq!(p12::decode(&p.codes, &p.aux, n, k).unwrap(), w, "host round trip");
    let aux: Vec<u8> = p.aux.iter().flat_map(|v| v.to_le_bytes()).collect();
    W {
        bf: up(be, DType::Bf16, &[n, k], &bytes16(&w)),
        codes: up(be, DType::U8, &[n, p12::row_bytes(k)], &p.codes),
        aux: up(be, DType::I32, &[p.aux.len()], &aux),
        escapes: p.escapes(),
    }
}

fn x_rows(be: &MetalBackend, rng: &mut SplitMix64, shape: &[usize]) -> (TensorRef, Vec<u16>) {
    let n: usize = shape.iter().product();
    let v: Vec<u16> = rng.normal_vec(n, 1.0).iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
    (up(be, DType::Bf16, shape, &bytes16(&v)), v)
}

fn launch(be: &MetalBackend, kernel: &str, op: Op, ins: Vec<TensorRef>, outs: Vec<TensorRef>, exact: bool) -> Vec<Vec<u8>> {
    let mut call = OpCall::new(op, ins, outs.clone(), OpAttrs::default());
    call.exact_rows = exact;
    call.check().unwrap_or_else(|e| panic!("{kernel}: {e}"));
    be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    be.synchronize(0).unwrap();
    outs.iter().map(|t| down(be, t)).collect()
}

/// `MatmulP12` (bf16 out = wo, f32 out = the unembed) vs `Matmul` / `Logits` on the bf16 weight, rows `rows`.
fn check_gemv(be: &MetalBackend, rng: &mut SplitMix64, w: &W, n: usize, k: usize, f32o: bool, rows: &[usize], what: &str) {
    let odt = if f32o { DType::F32 } else { DType::Bf16 };
    let (bf_one, bf_rows, bf_op) = if f32o { ("logits_bf16_gemv", "logits_bf16_gemv_rows", Op::Logits) } else { ("matmul_bf16_gemv", "matmul_bf16_gemv_rows", Op::Matmul) };
    let (pk_one, pk_rows) = if f32o { (MATMUL_P12_LOGITS_GEMV, MATMUL_P12_LOGITS_GEMV_ROWS) } else { (MATMUL_P12_GEMV, MATMUL_P12_GEMV_ROWS) };
    let es = odt.size_bytes();
    for &m in rows {
        let (x, xv) = x_rows(be, rng, &[m, k]);
        let bf = if m == 1 { bf_one } else { bf_rows };
        let want = launch(be, bf, bf_op, vec![x.clone(), w.bf.clone()], vec![zeros(be, odt, &[m, n])], m > 1).remove(0);
        for (kern, exact) in [(pk_one, false), (pk_rows, true)] {
            let got = launch(be, kern, Op::MatmulP12, vec![x.clone(), w.codes.clone(), w.aux.clone()], vec![zeros(be, odt, &[m, n])], exact).remove(0);
            if got != want {
                let first = got.chunks(es).zip(want.chunks(es)).position(|(a, b)| a != b).unwrap();
                panic!("{what} M={m} {kern}: not bitwise {bf}; first diff at output element {first} (row {}, col {})", first / n, first % n);
            }
        }
        // rows-exact: row t of the m-row call is the 1-row call on row t
        if m > 1 {
            let multi = launch(be, pk_rows, Op::MatmulP12, vec![x.clone(), w.codes.clone(), w.aux.clone()], vec![zeros(be, odt, &[m, n])], true).remove(0);
            for t in [0, m / 2, m - 1] {
                let x1 = up(be, DType::Bf16, &[1, k], &bytes16(&xv[t * k..(t + 1) * k]));
                let one = launch(be, pk_one, Op::MatmulP12, vec![x1, w.codes.clone(), w.aux.clone()], vec![zeros(be, odt, &[1, n])], false).remove(0);
                assert_eq!(&multi[t * n * es..(t + 1) * n * es], &one[..], "{what} M={m} row {t}: rows-exact");
            }
        }
    }
}

#[test]
fn matmul_p12_bitwise_bf16_wo_and_small() {
    let be = backend();
    let mut rng = SplitMix64::new(0x9e12);
    let all: Vec<usize> = (1..=16).collect();
    for &(n, k, adv) in &[(2304usize, 3072usize, false), (2304, 3072, true), (61, 3072, true), (37, 2304, true), (130, 64, true)] {
        let w = weight(&be, &mut rng, n, k, adv, &[(n, 1.0)]);
        assert!(w.escapes > 0, "the weight has escapes to patch");
        check_gemv(&be, &mut rng, &w, n, k, false, &all, &format!("Matmul N={n} K={k} adversarial={adv} escapes={}", w.escapes));
        check_gemv(&be, &mut rng, &w, n, k, true, &[1, 2, 5, 16], &format!("Logits N={n} K={k} adversarial={adv}"));
    }
}

#[test]
fn matmul_p12_bitwise_bf16_unembed() {
    let be = backend();
    let mut rng = SplitMix64::new(0x9e13);
    // a cut of the vocab at every row count, then the full unpadded vocab at 1 / 4 / 16 rows
    let (n, k) = (20011usize, 2304usize);
    let w = weight(&be, &mut rng, n, k, true, &[(n, 1.0)]);
    check_gemv(&be, &mut rng, &w, n, k, true, &(1..=16).collect::<Vec<_>>(), "Logits N=20011 K=2304 adversarial");
    let n = 200058usize;
    let w = weight(&be, &mut rng, n, k, false, &[(n, 1.0)]);
    check_gemv(&be, &mut rng, &w, n, k, true, &[1, 4, 16], &format!("Logits unembed N={n} K={k} escapes={}", w.escapes));
}

#[test]
fn in_proj_p12_bitwise_bf16() {
    let be = backend();
    let mut rng = SplitMix64::new(0x9e14);
    let d = 2304usize;
    for (ns, adv) in [([6552usize, 1056, 1056], false), ([6552, 1024, 1024], true), ([40, 24, 24], true)] {
        let n: usize = ns.iter().sum();
        let w = weight(&be, &mut rng, n, d, adv, &[(ns[0], 1.0), (ns[1], 0.25), (ns[2], 3.7)]);
        for s in [1usize, 3] {
            for r in 1..=16usize {
                let (x, xv) = x_rows(&be, &mut rng, &[s, r, d]);
                let outs = |be: &MetalBackend, r: usize| ns.iter().map(|&k| zeros(be, DType::Bf16, &[r, k])).collect::<Vec<_>>();
                let bf = if r == 1 { "ink_in_proj_bf16_gemv" } else { "ink_in_proj_bf16_gemv_rows" };
                let want = launch(&be, bf, Op::InkInProj, vec![x.clone(), w.bf.clone()], outs(&be, r), r > 1);
                for (kern, exact) in [(INK_IN_PROJ_P12_GEMV, false), (INK_IN_PROJ_P12_GEMV_ROWS, true)] {
                    let got = launch(&be, kern, Op::InkInProj, vec![x.clone(), w.codes.clone(), w.aux.clone()], outs(&be, r), exact);
                    for j in 0..3 {
                        assert!(got[j] == want[j], "N={ns:?} adversarial={adv} S={s} R={r} {kern} segment {j}: not bitwise {bf}");
                    }
                }
                if r > 1 {
                    // rows-exact: row t is the 1-row call on row t of every plane
                    let multi = launch(&be, INK_IN_PROJ_P12_GEMV_ROWS, Op::InkInProj, vec![x.clone(), w.codes.clone(), w.aux.clone()], outs(&be, r), true);
                    let t = r - 1;
                    let row: Vec<u16> = (0..s).flat_map(|a| xv[(a * r + t) * d..(a * r + t + 1) * d].to_vec()).collect();
                    let x1 = up(&be, DType::Bf16, &[s, 1, d], &bytes16(&row));
                    let one = launch(&be, INK_IN_PROJ_P12_GEMV, Op::InkInProj, vec![x1, w.codes.clone(), w.aux.clone()], outs(&be, 1), false);
                    for j in 0..3 {
                        assert_eq!(&multi[j][t * ns[j] * 2..(t + 1) * ns[j] * 2], &one[j][..], "N={ns:?} S={s} R={r} row {t} segment {j}: rows-exact");
                    }
                }
            }
        }
    }
}

/// The registry picks the P12 kernels exactly where the bf16 call runs the twin GEMV: ordinary calls at 1 row (Matmul /
/// Logits) or 1..=8 rows (InkInProj), `exact_rows` calls at 1..=16 rows; nothing else.
#[test]
fn p12_selection() {
    let be = backend();
    let reg = intarsia_core::Registry::new("metal", &be.info(), be.kernels());
    let (n, k) = (64usize, 2304usize);
    let codes = zeros(&be, DType::U8, &[n, p12::row_bytes(k)]);
    let aux = zeros(&be, DType::I32, &[n + 1]);
    for m in [1usize, 2, 8, 9, 16, 17] {
        for (odt, exact) in [(DType::Bf16, false), (DType::Bf16, true), (DType::F32, false), (DType::F32, true)] {
            let mut c = OpCall::new(Op::MatmulP12, vec![zeros(&be, DType::Bf16, &[m, k]), codes.clone(), aux.clone()], vec![zeros(&be, odt, &[m, n])], OpAttrs::default());
            c.exact_rows = exact;
            let got = reg.select(&c).map(|k| k.name.clone()).ok();
            let want = match (odt, exact) {
                (DType::Bf16, false) if m == 1 => Some(MATMUL_P12_GEMV),
                (DType::Bf16, true) if m <= ROWS_MAX => Some(MATMUL_P12_GEMV_ROWS),
                (DType::F32, false) if m == 1 => Some(MATMUL_P12_LOGITS_GEMV),
                (DType::F32, true) if m <= ROWS_MAX => Some(MATMUL_P12_LOGITS_GEMV_ROWS),
                _ => None,
            };
            assert_eq!(got.as_deref(), want, "MatmulP12 M={m} out={odt} exact={exact}");
        }
    }
    let ns = [40usize, 24, 24];
    let pcodes = zeros(&be, DType::U8, &[ns.iter().sum(), p12::row_bytes(k)]);
    let paux = zeros(&be, DType::I32, &[ns.iter().sum::<usize>() + 1]);
    for t in [1usize, 8, 9, 16, 17] {
        for exact in [false, true] {
            let outs = ns.iter().map(|&j| zeros(&be, DType::Bf16, &[t, j])).collect();
            let mut c = OpCall::new(Op::InkInProj, vec![zeros(&be, DType::Bf16, &[1, t, k]), pcodes.clone(), paux.clone()], outs, OpAttrs::default());
            c.exact_rows = exact;
            let got = reg.select(&c).map(|k| k.name.clone()).ok();
            let want = match exact {
                false if t <= intarsia_backend_metal::inkmix::PROJ_GEMV_ORDINARY_MAX => Some(INK_IN_PROJ_P12_GEMV),
                true if t <= ROWS_MAX => Some(INK_IN_PROJ_P12_GEMV_ROWS),
                _ => None,
            };
            assert_eq!(got.as_deref(), want, "InkInProj p12 T={t} exact={exact}");
        }
    }
}

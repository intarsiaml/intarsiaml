//! `matmul_mxfp4_naxr` / `matmul_mxfp4_naxr_rows` (kernels/metal/mxfp4_naxr.metal, metal4.0 library; lane mnr, opt-in
//! `INTARSIA_METAL_NAX_ROWS=1`): MXFP4 rows on the neural accelerators with the weights decoded per tile in the kernel.
//! * row invariance: a row's bits depend on neither the call's M, the row's index, the other rows, the form nor buffer offsets
//!   (bf16 and f32 out, K 32 .. 17408, N tails);
//! * a planted fault (the last row's tile left zero) is caught by that comparison;
//! * against an f64 reference and against the GEMV (accumulation order only);
//! * every scale byte the integer decode special-cases decodes to the bf16 of `mxfp4_w_to_bf16`;
//! * registration: never with the knob unset; with it, the M ranges.
//! The kernels are launched by name, so the tests run whatever the knob says.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16, Rng};
use intarsia_backend_metal::mxfp4::{self, NAXR_FORMS};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const NAXR: &str = "matmul_mxfp4_naxr";
const ROWS: &str = "matmul_mxfp4_naxr_rows";
const GEMV: &str = "matmul_mxfp4_gemv";
const FP4: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

fn up(be: &MetalBackend, data: &[u8]) -> Buffer {
    let b = be.alloc(data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; data.len() + 16], 0).unwrap();
    be.upload(&b, 0, data, 0).unwrap();
    b
}

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn e8m0(b: u8) -> f32 {
    match b {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        b => f32::from_bits((b as u32) << 23),
    }
}

/// Decoded weights `[n, k]` as f32 (e2m1 * 2^(scale - 127), exact).
fn decode(w: &[u8], sc: &[u8], n: usize, k: usize) -> Vec<f32> {
    let mut dec = vec![0f32; n * k];
    for row in 0..n {
        for e in 0..k {
            let byte = w[row * (k / 2) + e / 2];
            let code = if e % 2 == 0 { byte & 0xF } else { byte >> 4 };
            dec[row * k + e] = FP4[code as usize] * e8m0(sc[row * (k / 32) + e / 32]);
        }
    }
    dec
}

struct Run<'a> {
    be: &'a MetalBackend,
    n: usize,
    k: usize,
    x: Vec<u16>,
    w: Vec<u8>,
    sc: Vec<u8>,
    wb: Buffer,
    sb: Buffer,
}

#[derive(Clone, Copy)]
struct Opt {
    kernel: &'static str,
    act: f32,
    /// leading bf16 elements before x in its buffer (1: x sits at an odd element, a 2-byte offset)
    x_lead: usize,
    y_off: usize,
    f32_out: bool,
    form: Option<usize>,
    fault: bool,
}

const O: Opt = Opt { kernel: ROWS, act: 0.0, x_lead: 0, y_off: 0, f32_out: false, form: None, fault: false };

impl<'a> Run<'a> {
    /// `m_total` random activation rows, random codes, scales 2^-8 .. 2^3.
    fn new(be: &'a MetalBackend, m_total: usize, n: usize, k: usize, seed: u64) -> Self {
        let x = rand_bf16(m_total * k, seed);
        let mut r = Rng((seed + 1) | 1);
        let w: Vec<u8> = (0..n * k / 2).map(|_| (r.next_u64() >> 40) as u8).collect();
        let sc: Vec<u8> = (0..n * k / 32).map(|_| 119 + ((r.next_u64() >> 40) % 12) as u8).collect();
        Self::with(be, x, w, sc, n, k)
    }

    fn with(be: &'a MetalBackend, x: Vec<u16>, w: Vec<u8>, sc: Vec<u8>, n: usize, k: usize) -> Self {
        let (wb, sb) = (up(be, &w), up(be, &sc));
        Run { be, n, k, x, w, sc, wb, sb }
    }

    /// Output bytes of a call whose activation rows are `rows` of `self.x`, in that order.
    fn go(&self, rows: &[usize], o: Opt) -> Vec<u8> {
        let (n, k, m) = (self.n, self.k, rows.len());
        let mut xs = vec![0x3F80u16; o.x_lead];
        for &r in rows {
            xs.extend_from_slice(&self.x[r * k..(r + 1) * k]);
        }
        let xb = up(self.be, as_bytes(&xs));
        let esz = if o.f32_out { 4 } else { 2 };
        let ysz = o.y_off + m * n * esz + 16;
        let yb = self.be.alloc(ysz, MemKind::Device).unwrap();
        self.be.upload(&yb, 0, &vec![0xEEu8; ysz], 0).unwrap();
        let ab = up(self.be, &o.act.to_le_bytes());
        let odt = if o.f32_out { DType::F32 } else { DType::Bf16 };
        let mut cl = OpCall::new(
            Op::MatmulMxfp4,
            vec![
                t(&xb, o.x_lead * 2, DType::Bf16, &[m, k]),
                t(&self.wb, 0, DType::U8, &[n, k / 2]),
                t(&self.sb, 0, DType::U8, &[n, k / 32]),
                t(&ab, 0, DType::F32, &[1]),
            ],
            vec![t(&yb, o.y_off, odt, &[m, n])],
            OpAttrs { mxfp4_w4a16: o.act == 0.0, ..OpAttrs::default() },
        );
        let s = ShapeInfo::from_call(&cl);
        let need = mxfp4::scratch(&s);
        let b = self.be.alloc(need + 8, MemKind::Device).unwrap();
        self.be.upload(&b, 0, &vec![0xEEu8; need + 8], 0).unwrap();
        cl.scratch = Some(TensorRef { buf: b, offset: 4, dtype: DType::U8, shape: vec![need] });
        let launch = || self.be.launch(o.kernel, &cl, 0).unwrap();
        let launch = || if o.fault { mxfp4::with_naxr_fault(launch) } else { launch() };
        match o.form {
            Some(i) => mxfp4::with_naxr_form(i, launch),
            None => launch(),
        }
        self.be.synchronize(0).unwrap();
        let mut raw = vec![0u8; m * n * esz];
        self.be.download(&yb, o.y_off, &mut raw, 0).unwrap();
        raw
    }

    /// Every row alone (M = 1): the bits each row must have in every other call.
    fn solo(&self, m_total: usize, o: Opt) -> Vec<Vec<u8>> {
        (0..m_total).map(|r| self.go(&[r], o)).collect()
    }
}

fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn bf(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// Rows of `got` (a call on `rows`) that differ from the solo bits.
fn bad_rows(got: &[u8], rows: &[usize], solo: &[Vec<u8>]) -> Vec<usize> {
    let per = solo[0].len();
    rows.iter().enumerate().filter(|&(i, &r)| got[i * per..(i + 1) * per] != solo[r][..]).map(|(i, _)| i).collect()
}

/// (N, K): K 32 .. 17408 (the 27B's FFN), N tails 37 / 130 / 515, the 27B projection shapes.
const SHAPES: &[(usize, usize)] =
    &[(37, 32), (130, 96), (515, 160), (64, 128), (100, 384), (515, 1024), (37, 17408), (1024, 5120), (130, 6144)];

#[test]
fn naxr_rows_do_not_depend_on_the_call() {
    let be = backend();
    const MT: usize = 24;
    for (si, &(n, k)) in SHAPES.iter().enumerate() {
        let r = Run::new(&be, MT, n, k, 900 + si as u64);
        for f32_out in [false, true] {
            let o = Opt { f32_out, ..O };
            let solo = r.solo(MT, o);
            let mut rng = Rng(77 + si as u64);
            // every M 1..=16 and a few above, the rows drawn at random (so each lands at many indices beside many co-rows)
            for m in (1..=16).chain([17, 23, 33, 40]) {
                for rep in 0..3 {
                    let rows: Vec<usize> = (0..m).map(|_| (rng.next_u64() >> 33) as usize % MT).collect();
                    let o = Opt {
                        x_lead: rep % 2,
                        y_off: [0, 2, 4][rep % 3] * if f32_out { 2 } else { 1 },
                        kernel: if rep == 1 && m <= 8 { NAXR } else { ROWS },
                        ..o
                    };
                    let got = r.go(&rows, o);
                    let bad = bad_rows(&got, &rows, &solo);
                    assert!(bad.is_empty(), "n={n} k={k} f32={f32_out} m={m} rows {rows:?}: indices {bad:?} differ from M = 1");
                }
            }
            // every index 0..16 for one row, the co-rows fixed and then replaced
            for i in 0..16 {
                for co in [1usize, 5] {
                    let mut rows = vec![co; 16];
                    rows[i] = 0;
                    let got = r.go(&rows, o);
                    assert!(bad_rows(&got, &rows, &solo).is_empty(), "n={n} k={k} f32={f32_out}: row at index {i}, co-row {co}");
                }
            }
        }
    }
}

/// The forms (weight rows per threadgroup, K step) give the same bits: the accelerator's K order does not depend on how K is cut
/// into runs of 128 or 256.
#[test]
fn naxr_forms_give_the_same_bits() {
    let be = backend();
    for (si, &(n, k)) in SHAPES.iter().enumerate() {
        let r = Run::new(&be, 16, n, k, 300 + si as u64);
        let rows: Vec<usize> = (0..16).collect();
        for f32_out in [false, true] {
            let want = r.go(&rows, Opt { f32_out, form: Some(0), ..O });
            // the accelerator forms and the ALU twin (index NAXR_FORMS.len())
            for form in 1..=NAXR_FORMS.len() {
                for m in [1usize, 2, 3, 5, 16] {
                    let got = r.go(&rows[..m], Opt { f32_out, form: Some(form), ..O });
                    assert!(
                        got[..] == want[..got.len()],
                        "n={n} k={k} f32={f32_out} m={m}: form {} differs from form 0",
                        NAXR_FORMS.get(form).map_or("the ALU twin", |f| f.shaders[0])
                    );
                }
            }
        }
    }
}

#[test]
fn naxr_planted_fault_is_caught() {
    let be = backend();
    let (n, k) = (130, 384);
    let r = Run::new(&be, 16, n, k, 5);
    let solo = r.solo(16, O);
    for m in [2usize, 7, 16] {
        let rows: Vec<usize> = (0..m).collect();
        assert!(bad_rows(&r.go(&rows, O), &rows, &solo).is_empty());
        let bad = bad_rows(&r.go(&rows, Opt { fault: true, ..O }), &rows, &solo);
        assert_eq!(bad, vec![m - 1], "m={m}: the zeroed last row must be the one row that differs");
    }
}

#[test]
fn naxr_matches_an_f64_reference() {
    let be = backend();
    for (si, &(n, k)) in SHAPES.iter().enumerate() {
        let m = 16;
        let r = Run::new(&be, m, n, k, 40 + si as u64);
        let rows: Vec<usize> = (0..m).collect();
        let dec = decode(&r.w, &r.sc, n, k);
        let got = f32s(&r.go(&rows, Opt { f32_out: true, ..O }));
        let got_bf = bf(&r.go(&rows, O));
        let (mut num, mut den, mut max_abs) = (0f64, 0f64, 0f64);
        for i in 0..m {
            for j in 0..n {
                let (mut s, mut mag) = (0f64, 0f64);
                for e in 0..k {
                    let p = bf16_to_f32(r.x[i * k + e]) as f64 * dec[j * k + e] as f64;
                    s += p;
                    mag += p.abs();
                }
                let d = got[i * n + j] as f64 - s;
                // per element: f32 accumulation of K exact products
                assert!(d.abs() <= 1e-6 * mag + 1e-30, "n={n} k={k} [{i},{j}]: {} vs {s} (sum of |products| {mag})", got[i * n + j]);
                num += d * d;
                den += s * s;
                max_abs = max_abs.max(d.abs());
                // the bf16 form is the same float rounded once (nearest even)
                assert_eq!(got_bf[i * n + j], f32_to_bf16(got[i * n + j]), "n={n} k={k} [{i},{j}]: bf16 is not the rounded f32");
            }
        }
        let rel = (num / den).sqrt();
        eprintln!("f64 reference n={n} k={k}: rel_l2 {rel:.3e} max_abs {max_abs:.3e}");
        assert!(rel < 5e-5, "n={n} k={k}: rel_l2 {rel:e}");
    }
}

/// The accelerator's K order, found by the absorption probe (examples/naxr_order.rs `tree`): one f32 accumulator over quads in K
/// order, a quad being (p[a] + p[a+1]) + (p[a+8] + p[a+9]) for a = 16 t + 2 j (j = 0..4), every add rounded to nearest even and
/// every product exact. The kernel's f32 output is this sum bit for bit, on operands with a wide exponent spread (where the order
/// matters in most outputs).
#[test]
fn naxr_is_the_quad_order_sum_bit_for_bit() {
    let be = backend();
    for (si, &(n, k)) in [(64usize, 32usize), (130, 96), (37, 160), (64, 1024), (100, 5120), (16, 17408)].iter().enumerate() {
        let m = 5;
        let mut r = Rng(4242 + si as u64);
        let mut next = |md: u64| (r.next_u64() >> 40) % md;
        let w: Vec<u8> = (0..n * k / 2).map(|_| next(256) as u8).collect();
        let sc: Vec<u8> = (0..n * k / 32).map(|_| 110 + next(12) as u8).collect();
        let x: Vec<u16> = (0..m * k).map(|_| ((next(2) as u16) << 15) | ((120 + next(12) as u16) << 7) | next(128) as u16).collect();
        let dec = decode(&w, &sc, n, k);
        let run = Run::with(&be, x.clone(), w, sc, n, k);
        let rows: Vec<usize> = (0..m).collect();
        let got = f32s(&run.go(&rows, Opt { f32_out: true, ..O }));
        let (mut bad, mut differs_from_seq) = (0, 0);
        for i in 0..m {
            for j in 0..n {
                let p = |e: usize| bf16_to_f32(x[i * k + e]) as f64 * dec[j * k + e] as f64; // exact
                let (mut acc, mut seq) = (0f32, 0f32);
                for a in (0..k).step_by(16) {
                    for q in 0..4 {
                        let a = a + 2 * q;
                        acc += (p(a) + p(a + 1)) as f32 + (p(a + 8) + p(a + 9)) as f32;
                    }
                }
                for e in 0..k {
                    seq = (seq as f64 + p(e)) as f32;
                }
                bad += (acc.to_bits() != got[i * n + j].to_bits()) as usize;
                differs_from_seq += (acc.to_bits() != seq.to_bits()) as usize;
            }
        }
        eprintln!("quad order n={n} k={k}: {bad} of {} differ; the plain K-ascending chain would differ in {differs_from_seq}", m * n);
        assert_eq!(bad, 0, "n={n} k={k}: outputs that are not the quad-order sum");
        assert!(k < 64 || differs_from_seq > m * n / 10, "the operands must make the order matter");
    }
}

/// Against the GEMV (same activations, fp32 FMA chain): only the K accumulation order differs. W4A16 and W4A4.
#[test]
fn naxr_matches_the_gemv_within_accumulation_order() {
    let be = backend();
    for act in [0.0f32, 1.0] {
        for &(m, n, k) in &[(1usize, 512usize, 1024usize), (8, 1000, 512), (16, 2048, 5120), (5, 5120, 17408)] {
            let r = Run::new(&be, m, n, k, 77);
            let rows: Vec<usize> = (0..m).collect();
            let a = f32s(&r.go(&rows, Opt { act, f32_out: true, ..O }));
            let b = f32s(&r.go(&rows, Opt { act, f32_out: true, kernel: GEMV, ..O }));
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..m * n {
                num += ((a[i] - b[i]) as f64).powi(2);
                den += (b[i] as f64).powi(2);
            }
            let rel = (num / den).sqrt();
            eprintln!("vs gemv act {act} m={m} n={n} k={k}: relative RMS {rel:.3e}");
            assert!(rel < 1e-5, "act {act} m={m} n={n} k={k}: relative RMS difference {rel:e}");
        }
    }
}

/// W4A4 rows are invariant too (the QDQ is per row and group).
#[test]
fn naxr_w4a4_rows_do_not_depend_on_the_call() {
    let be = backend();
    let (n, k) = (130, 1024);
    let r = Run::new(&be, 16, n, k, 11);
    let o = Opt { act: 1.0, ..O };
    let solo = r.solo(16, o);
    for m in [2usize, 5, 8, 16] {
        let rows: Vec<usize> = (0..m).map(|i| (i * 7 + m) % 16).collect();
        assert!(bad_rows(&r.go(&rows, o), &rows, &solo).is_empty(), "m={m}");
    }
}

/// Scale bytes at the edges of the integer decode (the first and last normal exponents; from 253 up the largest codes overflow
/// f32 in every decode) and every
/// code: K = 32, one-hot activations of a power of two, so each output is one decoded weight times that power exactly.
/// Scale bytes 0 and 1 are left out: they give weights below 2^-126, subnormal as bf16, and the accelerator reads a subnormal
/// operand as zero (measured; `matmul_mxfp4_nax` feeds it the same bf16). No checkpoint has such scales.
#[test]
fn naxr_decodes_every_code_and_edge_scale() {
    let be = backend();
    let scs: [u8; 12] = [2, 3, 4, 5, 6, 7, 100, 127, 200, 250, 251, 252];
    let (n, k) = (scs.len() * 2, 32usize);
    // row j: scale scs[j / 2]; codes: element e holds code (e + 8 * (j % 2)) % 16 -> every code at even and odd elements
    let mut w = vec![0u8; n * k / 2];
    for j in 0..n {
        for e in 0..k {
            let code = ((e + 8 * (j % 2) + e / 16) % 16) as u8;
            w[j * (k / 2) + e / 2] |= code << (4 * (e % 2));
        }
    }
    let sc: Vec<u8> = (0..n).map(|j| scs[j / 2]).collect();
    let dec = decode(&w, &sc, n, k);
    for (pow, lo, hi) in [(100i32, 2u8, 127u8), (-100, 127, 252)] {
        let xv = 2f32.powi(pow);
        let mut x = vec![0u16; 32 * k];
        for e in 0..32 {
            x[e * k + e] = f32_to_bf16(xv);
        }
        let r = Run::with(&be, x, w.clone(), sc.clone(), n, k);
        for (r0, m) in [(0usize, 16usize), (16, 16)] {
            let rows: Vec<usize> = (r0..r0 + m).collect();
            let got = f32s(&r.go(&rows, Opt { f32_out: true, ..O }));
            for (i, &e) in rows.iter().enumerate() {
                for j in 0..n {
                    if sc[j] < lo || sc[j] > hi {
                        continue;
                    }
                    let want = dec[j * k + e] * xv;
                    let g = got[i * n + j];
                    // a zero weight gives a zero of either sign (the sum of signed zeros)
                    assert!(g.to_bits() == want.to_bits() || (g == 0.0 && want == 0.0), "scale byte {} element {e}: {g} vs {want}", sc[j]);
                }
            }
        }
    }
}

#[test]
fn naxr_registration_follows_the_knob() {
    let be = backend();
    let shape = |m: usize, n: usize, k: usize| ShapeInfo {
        m,
        n,
        k,
        act: Some(DType::Bf16),
        weight: Some(DType::U8),
        out: Some(DType::Bf16),
        ..Default::default()
    };
    // the predicates as functions of the knob
    for m in [1usize, 2, 8, 9, 16, 17, 300] {
        for (n, k) in [(17408usize, 5120usize), (5120, 17408), (5120, 6144)] {
            let s = shape(m, n, k);
            assert!(!mxfp4::naxr_accepts(false, &s) && !mxfp4::naxr_rows_accepts(false, &s), "knob unset: never");
            assert_eq!(mxfp4::naxr_accepts(true, &s), m <= 8, "ordinary calls: M 1..=8 (m={m})");
            assert!(mxfp4::naxr_rows_accepts(true, &s), "exact_rows calls: every M (m={m})");
        }
    }
    for s in [ShapeInfo { mxfp4_a8: true, ..shape(4, 5120, 5120) }, shape(4, 64, 48), shape(4, 64, 16)] {
        assert!(!mxfp4::naxr_accepts(true, &s) && !mxfp4::naxr_rows_accepts(true, &s), "W4A8 / K not a multiple of 32");
    }
    // this process: registered exactly when the knob is set, and then above the GEMV with `rows_exact` on the rows entry only
    let ks = be.kernels();
    let find = |name: &str| ks.iter().find(|k| k.name == name);
    let knob = mxfp4::naxr_knob();
    assert_eq!(find(NAXR).is_some(), knob && be.has_metal4());
    assert_eq!(find(ROWS).is_some(), knob && be.has_metal4());
    let all = intarsia_backend_metal::kernels::all();
    let (a, b) = (all.iter().find(|k| k.name == NAXR).unwrap(), all.iter().find(|k| k.name == ROWS).unwrap());
    let gemv = all.iter().find(|k| k.name == GEMV).unwrap();
    assert!(a.op == Op::MatmulMxfp4 && b.op == Op::MatmulMxfp4 && !a.rows_exact && b.rows_exact);
    assert!(a.priority > gemv.priority);
    for m in [1usize, 8, 9, 16] {
        let s = shape(m, 17408, 5120);
        assert_eq!(a.accepts(&s), knob && m <= 8);
        assert_eq!(b.accepts(&s), knob);
    }
    eprintln!("knob {knob}: registered {}", find(NAXR).is_some());
}

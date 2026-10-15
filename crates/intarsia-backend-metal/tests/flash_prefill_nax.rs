//! `attn_prefill_bf16_flash_nax` (kernels/metal/attention_flash_nax.metal, Metal 4 / NAX) against an f64 reference: GQA group
//! sizes 1 / 2 / 4 / 8, partial query blocks and key tiles, chunk offsets (pos0 > 0, keys = cache prefix + current chunk),
//! caches whose tail beyond the visible keys holds NaN (a masked P.V must not multiply it in), the end of the buffer,
//! bitwise chunk invariance and bitwise agreement across threadgroup shapes of the same tile, and the fallback to the P2
//! flash kernel for operands the kernel cannot take. Sibling of flash_prefill.rs. `intarsia kernels test --backend metal`
//! compares the kernel with the cpu backend's own implementation.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::bf16_to_f32;
use intarsia_backend_metal::flash_nax::{fallbacks, select_variant, NAX_FLASH_VARIANTS};
use intarsia_backend_metal::kernels::{ATTN_PREFILL_BF16, ATTN_PREFILL_BF16_FLASH, ATTN_PREFILL_BF16_FLASH_NAX};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::registry::Registry;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const NAN_BF16: u16 = 0x7FC0;

/// The variant selection and the fallback counter are process-wide: tests that use them run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Clone, Copy, Debug)]
struct Case {
    t: usize,
    n: usize,
    nkv: usize,
    d: usize,
    pos0: usize,
    cap: usize,
}

fn c(t: usize, n: usize, nkv: usize, d: usize, pos0: usize, cap: usize) -> Case {
    Case { t, n, nkv, d, pos0, cap }
}

struct Data {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
}

/// Random q / k / v; every cache row at or beyond `pos0 + t` is NaN.
fn data(cs: &Case, seed: u64) -> Data {
    let q = rand_bf16(cs.t * cs.n * cs.d, seed);
    let (mut k, mut v) = (rand_bf16(cs.nkv * cs.cap * cs.d, seed + 1), rand_bf16(cs.nkv * cs.cap * cs.d, seed + 2));
    for h in 0..cs.nkv {
        for j in cs.pos0 + cs.t..cs.cap {
            for x in 0..cs.d {
                k[(h * cs.cap + j) * cs.d + x] = NAN_BF16;
                v[(h * cs.cap + j) * cs.d + x] = NAN_BF16;
            }
        }
    }
    Data { q, k, v }
}

fn read_u16(be: &MetalBackend, b: &Buffer, off: usize, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, off, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// Upload `data` at byte offset `off` of a fresh buffer (0xEE guards around it).
fn upload_at(be: &MetalBackend, off: usize, data: &[u16]) -> Buffer {
    let b = be.alloc(off + data.len() * 2 + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() * 2 + 16], 0).unwrap();
    be.upload(&b, off, as_bytes(data), 0).unwrap();
    b
}

/// f64 causal GQA attention (row `ti` sees keys `0 ..= pos0 + ti`): the output and, per element, the root of the sum of
/// squares of its terms `p_j * v_j` (what rounding every probability independently perturbs it by, to first order).
fn reference_attn(cs: &Case, scale: f32, dt: &Data) -> (Vec<f64>, Vec<f64>) {
    let (t, n, d) = (cs.t, cs.n, cs.d);
    let group = n / cs.nkv;
    let mut out = vec![0f64; t * n * d];
    let mut noise = vec![0f64; t * n * d];
    let threads = std::thread::available_parallelism().map_or(4, |p| p.get());
    let rows = (t * n).div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, (oc, nc)) in out.chunks_mut(rows * d).zip(noise.chunks_mut(rows * d)).enumerate() {
            s.spawn(move || {
                for (r, (o, nz)) in oc.chunks_mut(d).zip(nc.chunks_mut(d)).enumerate() {
                    let row = ci * rows + r;
                    let (ti, h) = (row / n, row % n);
                    let kvh = h / group;
                    let vis = cs.pos0 + ti + 1;
                    let qr = &dt.q[row * d..][..d];
                    let sc: Vec<f64> = (0..vis)
                        .map(|j| {
                            let kr = &dt.k[(kvh * cs.cap + j) * d..][..d];
                            qr.iter().zip(kr).map(|(a, b)| bf16_to_f32(*a) as f64 * bf16_to_f32(*b) as f64).sum::<f64>()
                                * scale as f64
                        })
                        .collect();
                    let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let ex: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let den: f64 = ex.iter().sum();
                    for (j, e) in ex.iter().enumerate() {
                        let vr = &dt.v[(kvh * cs.cap + j) * d..][..d];
                        for x in 0..d {
                            let term = e / den * bf16_to_f32(vr[x]) as f64;
                            o[x] += term;
                            nz[x] += term * term;
                        }
                    }
                    nz.iter_mut().for_each(|x| *x = x.sqrt());
                }
            });
        }
    });
    (out, noise)
}

struct Run {
    be: MetalBackend,
    qb: Buffer,
    kb: Buffer,
    vb: Buffer,
    off: [usize; 3],
}

fn setup(cs: &Case, dt: &Data, off: [usize; 3]) -> Run {
    let be = backend();
    let (qb, kb, vb) = (upload_at(&be, off[0], &dt.q), upload_at(&be, off[1], &dt.k), upload_at(&be, off[2], &dt.v));
    let _ = cs;
    Run { be, qb, kb, vb, off }
}

/// Rows `t0 .. t0 + nt` of the call `cs` as one launch (q / out tensors offset by whole rows, pos0 advanced).
fn call_rows(r: &Run, cs: &Case, out: &Buffer, out_off: usize, t0: usize, nt: usize, scale: f32) -> OpCall {
    let t = |b: &Buffer, off: usize, shape: &[usize]| TensorRef {
        buf: b.clone(),
        offset: off,
        dtype: DType::Bf16,
        shape: shape.to_vec(),
    };
    let row = cs.n * cs.d * 2;
    OpCall::new(
        Op::AttnPrefill,
        vec![
            t(&r.qb, r.off[0] + t0 * row, &[nt, cs.n, cs.d]),
            t(&r.kb, r.off[1], &[cs.nkv, cs.cap, cs.d]),
            t(&r.vb, r.off[2], &[cs.nkv, cs.cap, cs.d]),
        ],
        vec![t(out, out_off + t0 * row, &[nt, cs.n, cs.d])],
        OpAttrs {
            pos0: cs.pos0 + t0,
            scale,
            n_heads: cs.n,
            n_kv_heads: cs.nkv,
            head_dim: cs.d,
            kv_cap: cs.cap,
            ..Default::default()
        },
    )
}

fn out_buf(be: &MetalBackend, cs: &Case, off: usize) -> Buffer {
    let len = cs.t * cs.n * cs.d;
    let out = be.alloc(off + len * 2 + 16, MemKind::Device).unwrap();
    be.upload(&out, 0, &vec![0xEEu8; off + len * 2 + 16], 0).unwrap();
    out
}

/// Run the whole case as one launch of `kernel` and return the output.
fn run_once(r: &Run, cs: &Case, kernel: &str, scale: f32, out_off: usize) -> Vec<u16> {
    let out = out_buf(&r.be, cs, out_off);
    r.be.launch(kernel, &call_rows(r, cs, &out, out_off, 0, cs.t, scale), 0).unwrap();
    r.be.synchronize(0).unwrap();
    let len = cs.t * cs.n * cs.d;
    let raw = read_u16(&r.be, &out, 0, (out_off + len * 2 + 16) / 2);
    assert!(
        raw[..out_off / 2].iter().all(|v| *v == 0xEEEE) && raw[out_off / 2 + len..].iter().all(|v| *v == 0xEEEE),
        "{kernel} {cs:?}: guard bytes overwritten"
    );
    raw[out_off / 2..out_off / 2 + len].to_vec()
}

/// Relative error bound of a probability rounded to bf16 (8 bits): the kernel's P.V operand.
const P_EPS: f64 = 1.0 / 256.0;

/// Compare with the f64 reference; returns the worst |err| in units of the allowed error. Allowed: half a bf16 ulp of the
/// result (2% slack) + 5 sigma of the probability rounding noise (independent errors of relative size eps / sqrt(3))
/// + f32 accumulation noise.
fn check(cs: &Case, got: &[u16], want: &(Vec<f64>, Vec<f64>), what: &str) -> f64 {
    let mut worst = 0.0f64;
    for (i, g) in got.iter().enumerate() {
        let (w, nz) = (want.0[i], want.1[i]);
        let g = bf16_to_f32(*g) as f64;
        let tol = w.abs() * 2f64.powi(-8) * 1.02 + 5.0 * P_EPS / 3f64.sqrt() * nz + 2e-6;
        assert!(g.is_finite() && (g - w).abs() <= tol, "{what} {cs:?}: element {i}: gpu {g} ref {w} tol {tol}");
        worst = worst.max((g - w).abs() / tol);
    }
    worst
}

fn all_variants() -> impl Iterator<Item = &'static str> {
    NAX_FLASH_VARIANTS.iter().map(|v| v.name)
}

#[test]
fn flash_matches_f64_reference() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cases = [
        c(1, 4, 2, 128, 0, 16),         // T = 1, one key
        c(5, 4, 2, 128, 0, 16),         // partial block, partial tile
        c(8, 8, 8, 128, 0, 8),          // MHA, exactly one 8-row slab
        c(40, 8, 2, 128, 0, 64),        // group 4, tile boundary at 32
        c(33, 4, 2, 128, 30, 64),       // continuation chunk, tile boundaries mid-block
        c(64, 16, 2, 128, 0, 80),       // group 8
        c(64, 16, 8, 128, 191, 255),    // group 2, cap = pos0 + T (end of the cache), odd key count
        c(205, 32, 8, 128, 3, 300),     // Qwen3-8B heads, ragged block, NaN tail
        c(256, 32, 8, 128, 256, 512),   // exactly full blocks after a prefix, cap = pos0 + T
        c(257, 32, 8, 128, 3839, 4200), // one row past a block, 4k context
        c(1024, 8, 2, 128, 0, 1030),    // long chunk
        c(256, 32, 8, 128, 3840, 4096), // the 4k chunk the benchmark measures
        c(3, 4, 4, 128, 61, 64),        // MHA, cache filled to the end
        c(17, 4, 2, 128, 15, 40),       // block and tile boundaries straddle the first row
        c(16, 4, 1, 128, 0, 16),        // group 4 with kv head 1, exactly one 16-row slab
        c(129, 16, 2, 128, 63, 400),    // group 8, key tile boundary at 64 inside the block
        c(300, 8, 8, 128, 1000, 1300),  // MHA, continuation, cap = pos0 + T
    ];
    for name in all_variants() {
        select_variant(Some(name));
        let mut worst = 0.0f64;
        for (i, cs) in cases.iter().enumerate() {
            let dt = data(cs, 500 + i as u64);
            let r = setup(cs, &dt, [0; 3]);
            let scale = 1.0 / (cs.d as f32).sqrt();
            let got = run_once(&r, cs, ATTN_PREFILL_BF16_FLASH_NAX, scale, 0);
            let want = reference_attn(cs, scale, &dt);
            worst = worst.max(check(cs, &got, &want, name));
        }
        eprintln!("{name}: worst error {worst:.3} of tolerance over {} cases", cases.len());
    }
    select_variant(None);
}

/// A row's result does not depend on how the prompt is split into calls: one call of T rows and several calls of fewer rows
/// give the same bits (the arithmetic of a row is a function of the row and the key tiles only, which start at key 0).
#[test]
fn flash_chunk_invariant() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cs = c(256, 32, 8, 128, 300, 556);
    let dt = data(&cs, 900);
    let scale = 1.0 / (cs.d as f32).sqrt();
    let mut first: Option<Vec<u16>> = None;
    for name in all_variants() {
        select_variant(Some(name));
        let r = setup(&cs, &dt, [0; 3]);
        let whole = run_once(&r, &cs, ATTN_PREFILL_BF16_FLASH_NAX, scale, 0);
        // the threadgroup shape (heads x slabs per threadgroup) does not change a row's bits either
        assert!(first.get_or_insert_with(|| whole.clone()) == &whole, "{name}: differs from the first variant");
        for split in [vec![64, 64, 64, 64], vec![100, 156], vec![1, 255], vec![7, 13, 97, 139], vec![128, 128]] {
            let out = out_buf(&r.be, &cs, 0);
            let mut t0 = 0;
            for &nt in &split {
                r.be.launch(ATTN_PREFILL_BF16_FLASH_NAX, &call_rows(&r, &cs, &out, 0, t0, nt, scale), 0).unwrap();
                t0 += nt;
            }
            r.be.synchronize(0).unwrap();
            let got = read_u16(&r.be, &out, 0, cs.t * cs.n * cs.d);
            assert!(got == whole, "{name}: split {split:?} differs from the single call");
        }
    }
    select_variant(None);
}

/// Operands with a leading 2-byte skip go to `attn_prefill_bf16` (counted), and the result is still right.
#[test]
fn flash_falls_back_on_misaligned_operands() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cs = c(40, 8, 2, 128, 20, 64);
    let dt = data(&cs, 700);
    let scale = 1.0 / (cs.d as f32).sqrt();
    let want = reference_attn(&cs, scale, &dt);
    for off in [[2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 2, 0], [0, 0, 8], [4, 4, 4], [16, 16, 16], [32, 0, 48]] {
        let r = setup(&cs, &dt, off);
        let before = fallbacks();
        let got = run_once(&r, &cs, ATTN_PREFILL_BF16_FLASH_NAX, scale, 0);
        let expect_fallback = off.iter().any(|o| o % 16 != 0);
        assert_eq!(fallbacks() - before, expect_fallback as u64, "offsets {off:?}");
        check(&cs, &got, &want, "fallback");
    }
    // an output at a 2-byte offset
    let r = setup(&cs, &dt, [0; 3]);
    let before = fallbacks();
    let got = run_once(&r, &cs, ATTN_PREFILL_BF16_FLASH_NAX, scale, 2);
    assert_eq!(fallbacks() - before, 1);
    check(&cs, &got, &want, "fallback out");
}

#[test]
fn flash_predicate() {
    let be = backend();
    let ks = be.kernels();
    let b = be.alloc(1 << 16, MemKind::Device).unwrap();
    let mk = |t: usize, n: usize, nkv: usize, d: usize, kdt: DType| {
        let tr = |dt: DType, shape: &[usize]| TensorRef { buf: b.clone(), offset: 0, dtype: dt, shape: shape.to_vec() };
        OpCall::new(
            Op::AttnPrefill,
            vec![tr(DType::Bf16, &[t, n, d]), tr(kdt, &[nkv, 8, d]), tr(kdt, &[nkv, 8, d])],
            vec![tr(DType::Bf16, &[t, n, d])],
            OpAttrs { n_heads: n, n_kv_heads: nkv, head_dim: d, kv_cap: 8, ..Default::default() },
        )
    };
    let accepts = |name: &str, c: &OpCall| ks.iter().any(|k| k.name == name && k.accepts(&ShapeInfo::from_call(c)));
    let f = ATTN_PREFILL_BF16_FLASH_NAX;
    assert!(be.has_metal4(), "this test needs the metal4.0 library");
    for g in [1, 2, 4, 8] {
        assert!(accepts(f, &mk(256, 8 * g, 8, 128, DType::Bf16)), "group {g}");
    }
    assert!(accepts(f, &mk(2, 32, 8, 128, DType::Bf16)), "the smallest chunk");
    assert!(accepts(f, &mk(47, 32, 8, 128, DType::Bf16)), "a short tail chunk takes the NAX kernel too");
    assert!(!accepts(f, &mk(256, 24, 8, 128, DType::Bf16)), "group 3");
    assert!(!accepts(f, &mk(256, 16, 1, 128, DType::Bf16)), "group 16");
    assert!(!accepts(f, &mk(256, 4, 2, 64, DType::Bf16)), "head_dim 64 stays on the P2 kernel");
    assert!(!accepts(f, &mk(256, 4, 2, 256, DType::Bf16)), "head_dim 256 stays on the P2 kernel");
    assert!(!accepts(f, &mk(256, 4, 2, 128, DType::F32)), "f32 cache");
    assert!(!accepts(f, &mk(256, 4, 2, 128, DType::Fp8E4m3)), "fp8 cache");
    assert!(accepts(ATTN_PREFILL_BF16, &mk(4, 4, 2, 128, DType::Bf16)), "the old kernel stays registered");
    assert!(accepts(ATTN_PREFILL_BF16_FLASH, &mk(4, 4, 2, 64, DType::Bf16)), "the P2 kernel stays registered");
}

/// Chunks through the registry, the way the engine launches them: every chunk, short tails included, runs the NAX kernel,
/// so every split gives the same bits as the single call (which is within the f64-reference tolerance).
#[test]
fn registry_splits_are_bitwise_the_single_call() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    select_variant(None);
    let cs = c(300, 32, 8, 128, 100, 400);
    let dt = data(&cs, 1100);
    let scale = 1.0 / (cs.d as f32).sqrt();
    let want = reference_attn(&cs, scale, &dt);
    let r = setup(&cs, &dt, [0; 3]);
    let reg = Registry::new("metal", &r.be.info(), r.be.kernels());
    let whole = run_once(&r, &cs, ATTN_PREFILL_BF16_FLASH_NAX, scale, 0);
    check(&cs, &whole, &want, "single call");
    for split in [vec![256, 44], vec![300], vec![150, 150], vec![100, 200], vec![253, 47], vec![288, 12], vec![298, 2], vec![7, 293]] {
        let out = out_buf(&r.be, &cs, 0);
        let mut t0 = 0;
        for &nt in &split {
            let call = call_rows(&r, &cs, &out, 0, t0, nt, scale);
            let name = reg.select(&call).unwrap().name.clone();
            assert_eq!(name, ATTN_PREFILL_BF16_FLASH_NAX, "split {split:?}");
            r.be.launch(&name, &call, 0).unwrap();
            t0 += nt;
        }
        r.be.synchronize(0).unwrap();
        let got = read_u16(&r.be, &out, 0, cs.t * cs.n * cs.d);
        assert!(got == whole, "split {split:?} differs from the single call");
    }
}

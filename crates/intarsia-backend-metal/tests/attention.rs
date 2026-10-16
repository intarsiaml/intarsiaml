//! `AttnPrefill` / `AttnDecode` against an f64 reference (causal GQA, scale, KV cache `[Nkv, cap, D]`): GQA ratios,
//! head dims that are not a multiple of 32, ragged query blocks, odd (2 mod 4) element offsets of every tensor, the
//! device-side position, and the predicates that keep non-bf16 caches off the kernels. The comparison of the kernels
//! with the cpu backend's own implementation is `intarsia kernels test --backend metal`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::bf16_to_f32;
use intarsia_backend_metal::kernels::{ATTN_DECODE_BF16, ATTN_DECODE_BF16_SPLITKV, ATTN_PREFILL_BF16};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

/// Upload `data` at byte offset `off` of a fresh buffer (0xEE guards around it).
fn upload_at(be: &MetalBackend, off: usize, data: &[u16]) -> Buffer {
    let b = be.alloc(off + data.len() * 2 + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() * 2 + 16], 0).unwrap();
    be.upload(&b, off, as_bytes(data), 0).unwrap();
    b
}

fn read_u16(be: &MetalBackend, b: &Buffer, off: usize, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, off, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

#[derive(Clone, Copy, Debug)]
struct Case {
    t: usize,
    n: usize,
    nkv: usize,
    d: usize,
    pos0: usize,
    cap: usize,
    /// byte offsets of q, k cache, v cache, out
    off: [usize; 4],
}

fn c(t: usize, n: usize, nkv: usize, d: usize, pos0: usize, cap: usize, off: [usize; 4]) -> Case {
    Case { t, n, nkv, d, pos0, cap, off }
}

/// f64 attention for one case; `visible(ti)` is the number of keys query `ti` sees.
fn reference_attn(
    cs: &Case,
    scale: f32,
    q: &[u16],
    k: &[u16],
    v: &[u16],
    visible: impl Fn(usize) -> usize,
) -> Vec<f64> {
    let (t, n, d) = (cs.t, cs.n, cs.d);
    let group = n / cs.nkv;
    let mut out = vec![0f64; t * n * d];
    for ti in 0..t {
        let vis = visible(ti);
        for h in 0..n {
            let kvh = h / group;
            let qr = &q[(ti * n + h) * d..][..d];
            let sc: Vec<f64> = (0..vis)
                .map(|j| {
                    let kr = &k[(kvh * cs.cap + j) * d..][..d];
                    qr.iter().zip(kr).map(|(a, b)| bf16_to_f32(*a) as f64 * bf16_to_f32(*b) as f64).sum::<f64>()
                        * scale as f64
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
            let den: f64 = ex.iter().sum();
            for (j, e) in ex.iter().enumerate() {
                let vr = &v[(kvh * cs.cap + j) * d..][..d];
                for x in 0..d {
                    out[(ti * n + h) * d + x] += e / den * bf16_to_f32(vr[x]) as f64;
                }
            }
        }
    }
    out
}

struct Run {
    be: MetalBackend,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    qb: Buffer,
    kb: Buffer,
    vb: Buffer,
}

fn setup(cs: &Case, seed: u64) -> Run {
    let be = backend();
    let q = rand_bf16(cs.t * cs.n * cs.d, seed);
    let k = rand_bf16(cs.nkv * cs.cap * cs.d, seed + 1);
    let v = rand_bf16(cs.nkv * cs.cap * cs.d, seed + 2);
    let (qb, kb, vb) = (upload_at(&be, cs.off[0], &q), upload_at(&be, cs.off[1], &k), upload_at(&be, cs.off[2], &v));
    Run { be, q, k, v, qb, kb, vb }
}

fn call(r: &Run, cs: &Case, op: Op, out: &Buffer, scale: f32) -> OpCall {
    let t = |b: &Buffer, off: usize, shape: &[usize]| TensorRef {
        buf: b.clone(),
        offset: off,
        dtype: DType::Bf16,
        shape: shape.to_vec(),
    };
    OpCall::new(
        op,
        vec![
            t(&r.qb, cs.off[0], &[cs.t, cs.n, cs.d]),
            t(&r.kb, cs.off[1], &[cs.nkv, cs.cap, cs.d]),
            t(&r.vb, cs.off[2], &[cs.nkv, cs.cap, cs.d]),
        ],
        vec![t(out, cs.off[3], &[cs.t, cs.n, cs.d])],
        OpAttrs {
            pos0: cs.pos0,
            scale,
            n_heads: cs.n,
            n_kv_heads: cs.nkv,
            head_dim: cs.d,
            kv_cap: cs.cap,
            ..Default::default()
        },
    )
}

/// The scratch workspace the registry would hand a kernel that declares one (plan-time upper bound, 0xEE-filled so a
/// kernel reading what it did not write shows up).
fn attach_scratch(be: &MetalBackend, kernel: &str, cl: &mut OpCall) {
    let need = be.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(cl)));
    if need > 0 {
        let b = be.alloc(need, MemKind::Device).unwrap();
        be.upload(&b, 0, &vec![0xEEu8; need], 0).unwrap();
        cl.scratch = Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] });
    }
}

fn run_case(cs: &Case, op: Op, kernel: &str, seed: u64) {
    let r = setup(cs, seed);
    let scale = 1.0 / (cs.d as f32).sqrt();
    let len = cs.t * cs.n * cs.d;
    let out = r.be.alloc(cs.off[3] + len * 2 + 16, MemKind::Device).unwrap();
    r.be.upload(&out, 0, &vec![0xEEu8; cs.off[3] + len * 2 + 16], 0).unwrap();
    let mut cl = call(&r, cs, op, &out, scale);
    attach_scratch(&r.be, kernel, &mut cl);
    r.be.launch(kernel, &cl, 0).unwrap();
    r.be.synchronize(0).unwrap();
    let got = read_u16(&r.be, &out, cs.off[3], len);
    let want = reference_attn(cs, scale, &r.q, &r.k, &r.v, |ti| cs.pos0 + ti + 1);
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let g = bf16_to_f32(*g) as f64;
        // half a bf16 ulp of the result (2% slack) + f32 accumulation noise
        let tol = w.abs() * 2f64.powi(-8) * 1.02 + 1e-5;
        assert!((g - w).abs() <= tol, "{kernel} {cs:?}: element {i}: gpu {g} ref {w}");
        worst = worst.max((g - w).abs() / tol);
    }
    eprintln!("{kernel} {cs:?}: worst error {worst:.3} of tolerance");
    let raw = read_u16(&r.be, &out, 0, (cs.off[3] + len * 2 + 16) / 2);
    assert!(
        raw[..cs.off[3] / 2].iter().all(|v| *v == 0xEEEE) && raw[cs.off[3] / 2 + len..].iter().all(|v| *v == 0xEEEE),
        "{kernel} {cs:?}: guard bytes overwritten"
    );
}

#[test]
fn decode_matches_f64_reference() {
    let z = [0; 4];
    let cases = [
        c(1, 32, 8, 128, 0, 64, z),              // first token: one key
        c(1, 32, 8, 128, 4095, 4096, z),         // Qwen3-8B geometry at 4k context
        c(1, 4, 2, 64, 17, 40, z),               // more keys than the 8 simdgroups
        c(1, 4, 2, 64, 5, 40, z),                // fewer keys than simdgroups: empty simdgroups
        c(1, 6, 3, 96, 100, 128, [2, 2, 2, 2]),  // D = 96, odd element offsets everywhere
        c(1, 4, 4, 30, 33, 64, [0, 2, 0, 2]),    // D not a multiple of 32, MHA
        c(1, 2, 1, 2, 3, 8, z),                  // D = 2
        c(1, 8, 2, 256, 300, 512, [2, 0, 2, 0]), // D = 256, group 4
        c(1, 16, 1, 128, 63, 64, [0, 0, 0, 0]),  // MQA, context fills the cache
        // 16-byte-load kernel (power-of-two head_dim, 16-byte aligned caches): every lanes-per-key / keys-per-simdgroup split
        c(1, 4, 2, 256, 100, 128, z),
        c(1, 4, 2, 128, 100, 128, [16, 32, 48, 16]),
        c(1, 8, 2, 64, 200, 256, z),
        c(1, 4, 1, 32, 50, 64, z),
        c(1, 2, 1, 16, 9, 16, z),
        c(1, 2, 2, 8, 1, 4, z),
        c(1, 4, 2, 128, 3, 8, z),                // fewer keys than the 16 simdgroups
        c(1, 4, 2, 128, 100, 128, [2, 2, 2, 2]), // same shape, odd offsets: the 2-byte-load kernel
    ];
    for (i, cs) in cases.iter().enumerate() {
        run_case(cs, Op::AttnDecode, ATTN_DECODE_BF16, 100 + i as u64);
    }
}

#[test]
fn prefill_matches_f64_reference() {
    let z = [0; 4];
    let cases = [
        c(1, 4, 2, 64, 0, 16, z),              // T = 1
        c(5, 4, 2, 64, 0, 16, z),              // ragged block (5 of 8 rows)
        c(8, 4, 2, 64, 0, 8, z),               // exactly one block
        c(40, 8, 2, 128, 0, 64, z),            // several blocks, a K tile boundary at 32
        c(33, 4, 2, 128, 30, 64, z),           // continuation chunk (pos0 > 0), tile boundaries mid-block
        c(17, 2, 1, 256, 5, 32, [2, 2, 2, 2]), // D = 256, odd offsets
        c(9, 4, 4, 80, 0, 16, [0, 2, 2, 0]),   // D = 80 (not a multiple of 32)
        c(70, 32, 8, 128, 100, 192, z),        // Qwen3-8B heads, long chunk
        c(3, 6, 3, 30, 61, 64, [2, 0, 0, 2]),  // D = 30, cache filled to the end
    ];
    for (i, cs) in cases.iter().enumerate() {
        run_case(cs, Op::AttnPrefill, ATTN_PREFILL_BF16, 200 + i as u64);
    }
}

/// The decode kernel with `OpCall::pos`: bit-identical to the host `pos0` launch, and a position past the cache is
/// clamped to its end instead of reading out of bounds.
#[test]
fn decode_device_position() {
    // 16-byte aligned caches take the 16-byte-load kernel, odd element offsets the 2-byte-load one
    for off in [[0; 4], [2; 4]] {
        device_position(c(1, 8, 2, 128, 77, 128, off));
    }
}

fn device_position(cs: Case) {
    let r = setup(&cs, 300);
    let mk_pos = |v: i32| {
        let pb = r.be.alloc(4, MemKind::Device).unwrap();
        r.be.upload(&pb, 0, &v.to_le_bytes(), 0).unwrap();
        TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![1] }
    };
    let launch = |pos0_attr: usize, dev: Option<i32>| {
        let mut cs2 = cs;
        cs2.pos0 = pos0_attr;
        let out = r.be.alloc(cs.off[3] + cs.n * cs.d * 2, MemKind::Device).unwrap();
        let mut cl = call(&r, &cs2, Op::AttnDecode, &out, 0.125);
        if let Some(v) = dev {
            cl.pos = Some(mk_pos(v));
        }
        r.be.launch(ATTN_DECODE_BF16, &cl, 0).unwrap();
        read_u16(&r.be, &out, cs.off[3], cs.n * cs.d)
    };
    let host = launch(77, None);
    assert_eq!(host, launch(77, Some(77)));
    // the host attribute only bounds the call; the tensor decides (here: a different position)
    assert_eq!(launch(10, Some(77)), host);
    assert_ne!(launch(10, None), host);
    assert_eq!(launch(0, Some(5000)), launch(127, None), "position past the cache is clamped to cap");
    assert_eq!(launch(0, Some(-3)), launch(0, None), "a negative position sees one key");
}

#[test]
fn predicates_accept_bf16_caches_only() {
    let be = backend();
    let ks = be.kernels();
    let b = be.alloc(1 << 16, MemKind::Device).unwrap();
    let mk = |op: Op, t: usize, d: usize, kdt: DType| {
        let tr = |dt: DType, shape: &[usize]| TensorRef { buf: b.clone(), offset: 0, dtype: dt, shape: shape.to_vec() };
        OpCall::new(
            op,
            vec![tr(DType::Bf16, &[t, 4, d]), tr(kdt, &[2, 8, d]), tr(kdt, &[2, 8, d])],
            vec![tr(DType::Bf16, &[t, 4, d])],
            OpAttrs { n_heads: 4, n_kv_heads: 2, head_dim: d, kv_cap: 8, ..Default::default() },
        )
    };
    let accepts = |name: &str, c: &OpCall| ks.iter().any(|k| k.name == name && k.accepts(&ShapeInfo::from_call(c)));
    assert!(accepts(ATTN_DECODE_BF16, &mk(Op::AttnDecode, 1, 64, DType::Bf16)));
    assert!(accepts(ATTN_PREFILL_BF16, &mk(Op::AttnPrefill, 4, 128, DType::Bf16)));
    assert!(!accepts(ATTN_DECODE_BF16, &mk(Op::AttnDecode, 1, 64, DType::F32)), "f32 cache");
    assert!(!accepts(ATTN_PREFILL_BF16, &mk(Op::AttnPrefill, 4, 64, DType::F16)), "f16 cache");
    assert!(!accepts(ATTN_PREFILL_BF16, &mk(Op::AttnPrefill, 4, 288, DType::Bf16)), "head_dim > 256");
    assert!(!accepts(ATTN_PREFILL_BF16, &mk(Op::AttnPrefill, 4, 63, DType::Bf16)), "odd head_dim");
    let op_of = |name: &str| ks.iter().find(|k| k.name == name).unwrap().op;
    assert_eq!((op_of(ATTN_PREFILL_BF16), op_of(ATTN_DECODE_BF16)), (Op::AttnPrefill, Op::AttnDecode));
}

#[test]
fn fingerprint_names_device_and_metallib() {
    let be = backend();
    let f = be.fingerprint();
    assert!(f.starts_with("metal|"), "{f}");
    assert!(f.contains(&format!("metallib:{:016x}", intarsia_backend_metal::metallib_hash())), "{f}");
    assert_ne!(intarsia_backend_metal::metallib_hash(), 0);
    // the Rust launcher sources (build.rs LAUNCHER_SOURCES) are part of the identity too
    assert!(f.ends_with(&format!("|launch:{}", intarsia_backend_metal::LAUNCHER_HASH)), "{f}");
    assert_eq!(intarsia_backend_metal::LAUNCHER_HASH.len(), 16);
    assert_eq!(f, backend().fingerprint(), "stable across instances");
}

// ---- GQA-grouped split-KV decode (attn_decode_bf16_splitkv) -------------------------------------------------------------

/// Every group size (1 = MHA .. 8 incl. the padded 3 / 5 / 6), the head dims of the 16-byte kernels, and context lengths
/// from one key to ~16k (1 / 2 / 4+ splits, empty trailing splits, a context that fills the cache).
#[test]
fn splitkv_matches_f64_reference() {
    let z = [0; 4];
    let cases = [
        c(1, 32, 8, 128, 0, 64, z),                     // one key (nsplit 1, a single slot active)
        c(1, 32, 8, 128, 16, 64, z),                    // 17 keys
        c(1, 32, 8, 128, 127, 128, z),                  // 128 keys: the split boundary of the default policy
        c(1, 32, 8, 128, 128, 256, z),                  // 129 keys: two splits, the second holds one key
        c(1, 32, 8, 128, 999, 1024, z),                 // 1k
        c(1, 32, 8, 128, 4095, 4096, z),                // Qwen3-8B at 4k
        c(1, 32, 8, 128, 16100, 16384, z),              // ~16k
        c(1, 8, 8, 128, 700, 1024, z),                  // group 1 (MHA)
        c(1, 4, 2, 64, 300, 512, z),                    // group 2
        c(1, 6, 2, 64, 300, 512, z),                    // group 3 (padded to 4)
        c(1, 16, 2, 128, 2000, 2048, z),                // group 8
        c(1, 10, 2, 32, 1500, 2048, z),                 // group 5 (padded to 8), D = 32: 8 keys per simdgroup step
        c(1, 12, 2, 256, 1200, 2048, z),                // group 6, D = 256: one key per step, fewer simdgroups
        c(1, 16, 2, 256, 900, 1024, z),                 // group 8, D = 256
        c(1, 16, 1, 16, 600, 1024, z),                  // MQA, D = 16
        c(1, 2, 1, 8, 400, 512, z),                     // D = 8: one lane per key
        c(1, 32, 8, 128, 4000, 4096, [16, 32, 48, 16]), // 16-byte aligned, non-zero offsets
        c(1, 32, 8, 128, 4000, 4096, [2, 0, 0, 2]),     // q / out at odd element offsets: q not 16-byte aligned, falls back
        c(1, 32, 8, 128, 4000, 4096, [4, 0, 0, 0]),     // q 4 mod 16: falls back to the single-pass kernel (vector q loads)
        c(1, 32, 8, 128, 4000, 4096, [16, 0, 0, 2]),    // q 16-byte aligned, out at an odd element offset (skip)
        c(1, 16, 2, 128, 20000, 20480, z),              // 2 kv heads, 60 splits: the combine's second and later batches
        c(1, 8, 1, 128, 30000, 30720, z),               // 1 kv head: 64 splits (the cap)
        c(1, 32, 8, 128, 4000, 4096, [0, 2, 2, 0]),     // caches 2 mod 4: falls back to the single-pass kernel
        c(1, 4, 2, 96, 500, 512, z),                    // D = 96: not a power of two, falls back
        c(1, 4, 4, 30, 33, 64, [0, 2, 0, 2]),           // D = 30, MHA: falls back
        c(1, 18, 2, 128, 500, 512, z),                  // group 9: falls back
    ];
    for (i, cs) in cases.iter().enumerate() {
        run_case(cs, Op::AttnDecode, ATTN_DECODE_BF16_SPLITKV, 400 + i as u64);
    }
}

/// The split-KV kernel against the single-pass one on the same data (both are within half a bf16 ulp of the f64
/// reference, so they differ by at most a ulp), and a repeat launch is bit-identical.
#[test]
fn splitkv_is_deterministic_and_close_to_single_pass() {
    let cs = c(1, 32, 8, 128, 6000, 8192, [0; 4]);
    let r = setup(&cs, 500);
    let len = cs.n * cs.d;
    let go = |kernel: &str| {
        let out = r.be.alloc(len * 2, MemKind::Device).unwrap();
        let mut cl = call(&r, &cs, Op::AttnDecode, &out, 0.0883883);
        attach_scratch(&r.be, kernel, &mut cl);
        r.be.launch(kernel, &cl, 0).unwrap();
        read_u16(&r.be, &out, 0, len)
    };
    let (a, b) = (go(ATTN_DECODE_BF16_SPLITKV), go(ATTN_DECODE_BF16_SPLITKV));
    assert_eq!(a, b, "run-to-run");
    let single = go(ATTN_DECODE_BF16);
    let worst = a
        .iter()
        .zip(&single)
        .map(|(x, y)| (bf16_to_f32(*x) - bf16_to_f32(*y)).abs() / bf16_to_f32(*y).abs().max(1e-3))
        .fold(0f32, f32::max);
    assert!(worst < 0.02, "split-KV vs single-pass: worst relative difference {worst}");
}

/// Elementwise within one bf16 ulp (the host and device launches may pick different threadgroup shapes, so their
/// rounding order differs).
fn assert_close(a: &[u16], b: &[u16], what: &str) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let (x, y) = (bf16_to_f32(*x), bf16_to_f32(*y));
        assert!((x - y).abs() <= 2f32.powi(-7) * x.abs().max(y.abs()) + 1e-6, "{what}: element {i}: {x} vs {y}");
    }
}

/// The device position tensor decides: host `pos0` and tensor agree up to rounding order (same split count, the
/// threadgroup shape may differ), the attribute only bounds the call (results with the same tensor are bit-identical),
/// a position past the cache is clamped.
#[test]
fn splitkv_device_position() {
    for off in [[0; 4], [2; 4]] {
        // cap 4096 so that the position can pick 1, 3 and 15 splits; D = 128 and 64
        for (n, nkv, d) in [(32, 8, 128), (8, 2, 64)] {
            let cs = c(1, n, nkv, d, 0, 4096, off);
            let r = setup(&cs, 600);
            let mk_pos = |v: i32| {
                let pb = r.be.alloc(4, MemKind::Device).unwrap();
                r.be.upload(&pb, 0, &v.to_le_bytes(), 0).unwrap();
                TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![1] }
            };
            let launch = |pos0_attr: usize, dev: Option<i32>| {
                let mut cs2 = cs;
                cs2.pos0 = pos0_attr;
                let out = r.be.alloc(cs.off[3] + cs.n * cs.d * 2, MemKind::Device).unwrap();
                let mut cl = call(&r, &cs2, Op::AttnDecode, &out, 0.125);
                attach_scratch(&r.be, ATTN_DECODE_BF16_SPLITKV, &mut cl);
                if let Some(v) = dev {
                    cl.pos = Some(mk_pos(v));
                }
                r.be.launch(ATTN_DECODE_BF16_SPLITKV, &cl, 0).unwrap();
                read_u16(&r.be, &out, cs.off[3], cs.n * cs.d)
            };
            for pos in [0usize, 100, 127, 128, 129, 700, 4000, 4095] {
                let host = launch(pos, None);
                let dev = launch(pos, Some(pos as i32));
                assert_close(&host, &dev, &format!("pos {pos}: device position vs host position"));
                // the tensor decides, the attribute (here: a different position) only bounds the call
                assert_eq!(launch(10, Some(pos as i32)), dev, "pos {pos}: attribute 10, tensor decides");
                assert_eq!(launch(pos, Some(pos as i32)), dev, "pos {pos}: run-to-run");
            }
            assert_ne!(launch(10, None), launch(4000, None));
            assert_eq!(launch(0, Some(5000)), launch(0, Some(4095)), "position past the cache is clamped to cap");
            assert_close(&launch(0, Some(5000)), &launch(4095, None), "clamped position vs host cap");
            assert_close(&launch(0, Some(-3)), &launch(0, None), "a negative position sees one key");
        }
    }
}

#[test]
fn splitkv_scratch_is_required_and_checked() {
    let cs = c(1, 32, 8, 128, 4000, 4096, [0; 4]);
    let r = setup(&cs, 700);
    let out = r.be.alloc(cs.n * cs.d * 2, MemKind::Device).unwrap();
    let cl = call(&r, &cs, Op::AttnDecode, &out, 0.125);
    assert!(r.be.launch(ATTN_DECODE_BF16_SPLITKV, &cl, 0).is_err(), "no scratch");
    let mut small = call(&r, &cs, Op::AttnDecode, &out, 0.125);
    let b = r.be.alloc(1024, MemKind::Device).unwrap();
    small.scratch = Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![1024] });
    assert!(r.be.launch(ATTN_DECODE_BF16_SPLITKV, &small, 0).is_err(), "scratch too small");
    // the single-pass kernel needs none
    r.be.launch(ATTN_DECODE_BF16, &cl, 0).unwrap();
    r.be.synchronize(0).unwrap();
}

#[test]
fn splitkv_predicate_and_scratch_declaration() {
    let be = backend();
    let ks = be.kernels();
    let k = ks.iter().find(|k| k.name == ATTN_DECODE_BF16_SPLITKV).unwrap();
    let single = ks.iter().find(|k| k.name == ATTN_DECODE_BF16).unwrap();
    assert_eq!((k.op, k.priority > single.priority), (Op::AttnDecode, true));
    let shape = |m: usize, n: usize, nkv: usize, d: usize| ShapeInfo {
        m,
        n_heads: n,
        n_kv_heads: nkv,
        head_dim: d,
        act: Some(DType::Bf16),
        weight: Some(DType::Bf16),
        ..Default::default()
    };
    assert!(k.accepts(&shape(1, 32, 8, 128)));
    assert!(k.accepts(&shape(1, 8, 8, 64)), "group 1");
    assert!(k.accepts(&shape(1, 16, 2, 256)), "group 8");
    for (s, why) in [
        (shape(1, 4, 2, 96), "D = 96"),
        (shape(1, 4, 2, 30), "D = 30"),
        (shape(1, 18, 2, 128), "group 9"),
        (shape(1, 5, 2, 64), "heads not a multiple"),
        (shape(2, 32, 8, 128), "two rows"),
    ] {
        assert!(!k.accepts(&s), "{why}");
    }
    assert!(single.accepts(&shape(1, 4, 2, 96)), "the single-pass kernel still takes the shapes split-KV rejects");
    let mut f32kv = shape(1, 32, 8, 128);
    f32kv.weight = Some(DType::F32);
    assert!(!k.accepts(&f32kv), "f32 cache");
    // scratch bound: does not depend on the context length, covers the largest split count
    let mut s = shape(1, 32, 8, 128);
    let base = k.scratch_bytes(&s);
    s.kv_len = 1 << 20;
    assert_eq!(k.scratch_bytes(&s), base);
    assert!(base >= 32 * 64 * 130 * 4);
    assert_eq!(single.scratch_bytes(&s), 0);
}

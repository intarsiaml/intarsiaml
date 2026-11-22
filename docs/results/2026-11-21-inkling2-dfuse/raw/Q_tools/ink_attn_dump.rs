//! Lane ink-dfuse attention probe (measurement only, not shipped): one `ink_attn_decode_bf16` visit per (kind, ctx) at the
//! real decode shapes (ink_attn_bench's `visit`, rows 1), its operands and results written as raw little-endian files, so
//! the Swift probe harness can check that its copy of the launch (InkP, grid, threadgroup memory) is bitwise the backend's.
//!
//!   cargo run --release -p intarsia-backend-metal --example ink_attn_dump -- OUT_DIR
//!
//! Files per case `<kind>_<ctx>_`: pq pk pv kt vt qs rs rp kc vc kr vr (before) and out kc1 vc1 kr1 vr1 (after); meta.txt.

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::inkling2::ops::nq;
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};
    use std::io::Write;

    const H: usize = 24;
    const HK: usize = 8;
    const DH: usize = 128;
    const DR: usize = 16;
    const E: usize = 1024;
    const K: usize = 4;
    const RR: usize = K - 1 + 16;

    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("OUT_DIR"));
    std::fs::create_dir_all(&dir).unwrap();
    let be = MetalBackend::new().expect("metal backend");
    let t = |b: &Buffer, shape: &[usize]| TensorRef { buf: b.clone(), offset: 0, dtype: DType::Bf16, shape: shape.to_vec() };
    let mut meta = std::fs::File::create(dir.join("meta.txt")).unwrap();
    for &(full, ctx) in &[(false, 150usize), (true, 150), (false, 512), (true, 512), (false, 4096), (true, 4096), (false, 5180), (true, 5180)] {
        let mut rng = Rng(0x9e3779b97f4a7c15 ^ (ctx as u64 * 2 + full as u64));
        let mut bufs: Vec<(&str, Buffer, usize)> = Vec::new();
        let mk = |name: &'static str, n: usize, rng: &mut Rng, bufs: &mut Vec<(&str, Buffer, usize)>| -> Buffer {
            let v: Vec<u16> = (0..n).map(|_| f32_to_bf16((rng.unit() * 2.0 - 1.0) * 1.0)).collect();
            let b = be.alloc(n * 2 + 64, MemKind::Device).unwrap();
            // SAFETY: u16 -> u8 view of the same memory.
            be.upload(&b, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }, 0).unwrap();
            bufs.push((name, b.clone(), n * 2));
            b
        };
        let ckv = HK * DH;
        let pw = if full { ckv + HK * K } else { ckv };
        let nqw = nq(H, DH, DR);
        let (win, ring, cr) = if full { (0, false, ctx + 1 + 16) } else { (1024, true, 1040) };
        let mut a = OpAttrs { pos0: ctx, n_heads: H, n_kv_heads: HK, head_dim: DH, window: win, scale: (DH as f32).powf(-0.5), eps: 1e-6, ..OpAttrs::default() };
        a.ink.qk_clip = 22.6;
        a.ink.rel_clip = 2.0;
        a.ink.d_rel = DR;
        a.ink.kv_ring = ring;
        a.ink.dyn_conv = full;
        if full {
            (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
        }
        let pq = mk("pq", nqw, &mut rng, &mut bufs);
        let pk = mk("pk", pw, &mut rng, &mut bufs);
        let pv = mk("pv", pw, &mut rng, &mut bufs);
        let kt = mk("kt", K * ckv, &mut rng, &mut bufs);
        let vt = mk("vt", K * ckv, &mut rng, &mut bufs);
        let qs = mk("qs", H * DH, &mut rng, &mut bufs);
        let rs = mk("rs", H * DR, &mut rng, &mut bufs);
        let rp = mk("rp", DR * E, &mut rng, &mut bufs);
        let kc = mk("kc", HK * cr * DH, &mut rng, &mut bufs);
        let vc = mk("vc", HK * cr * DH, &mut rng, &mut bufs);
        let kr = mk("kr", RR * ckv, &mut rng, &mut bufs);
        let vr = mk("vr", RR * ckv, &mut rng, &mut bufs);
        let o = mk("out0", H * DH, &mut rng, &mut bufs);
        for (name, b, n) in &bufs {
            let mut v = vec![0u8; *n];
            be.download(b, 0, &mut v, 0).unwrap();
            std::fs::write(dir.join(format!("{}_{ctx}_{name}", if full { "full" } else { "swa" })), &v).unwrap();
        }
        let mut f = OpCall::new(
            Op::InkAttnDecode,
            vec![t(&pq, &[1, nqw]), t(&pk, &[1, pw]), t(&pv, &[1, pw]), t(&kt, &[K, ckv]), t(&vt, &[K, ckv]), t(&qs, &[H, DH]), t(&rs, &[H, DR]), t(&rp, &[DR, E]), t(&kc, &[HK, cr, DH]), t(&vc, &[HK, cr, DH])],
            vec![t(&o, &[1, H * DH]), t(&kc, &[HK, cr, DH]), t(&vc, &[HK, cr, DH]), t(&kr, &[RR, ckv]), t(&vr, &[RR, ckv])],
            a,
        );
        let need = be.kernels().iter().find(|k| k.name == "ink_attn_decode_bf16").unwrap().scratch_bytes(&ShapeInfo::from_call(&f));
        let sb = be.alloc(need.max(16), MemKind::Device).unwrap();
        f.scratch = Some(TensorRef { buf: sb, offset: 0, dtype: DType::U8, shape: vec![need.max(16)] });
        be.launch("ink_attn_decode_bf16", &f, 0).unwrap();
        be.synchronize(0).unwrap();
        for (name, b, n) in [("out", &o, H * DH * 2), ("kc1", &kc, HK * cr * DH * 2), ("vc1", &vc, HK * cr * DH * 2), ("kr1", &kr, RR * ckv * 2), ("vr1", &vr, RR * ckv * 2)] {
            let mut v = vec![0u8; n];
            be.download(b, 0, &mut v, 0).unwrap();
            std::fs::write(dir.join(format!("{}_{ctx}_{name}", if full { "full" } else { "swa" })), &v).unwrap();
        }
        writeln!(meta, "{} {ctx} cr {cr} scratch {need}", if full { "full" } else { "swa" }).unwrap();
    }
    println!("dumped to {}", dir.display());
}

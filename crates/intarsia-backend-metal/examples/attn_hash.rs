//! Bitwise fingerprint of the split-KV decode attention (phase B5): fixed random q / K / V per case, one launch, FNV-1a of
//! the output bytes. Run it on two builds and diff the output: identical lines mean bitwise identical results.
//! The cases cover one split, the 15-split Qwen3-8B steady state and shapes whose split count passes the combine's
//! batch of 16 (2 kv heads: up to 60 splits; 1 kv head: 64), group sizes 1 to 8, head_dim 64 / 128 / 256.
//!
//!   cargo run --release -p intarsia-backend-metal --example attn_hash

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("attn_hash needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
    use intarsia_backend_metal::kernels::ATTN_DECODE_BF16_SPLITKV;
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

    let be = MetalBackend::new().expect("metal backend");
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut rand_buf = |n: usize, scale: f32| -> Buffer {
        let b = be.alloc(n * 2, MemKind::Device).unwrap();
        let v: Vec<u16> = (0..n).map(|_| f32_to_bf16(rng.unit() * scale)).collect();
        // SAFETY: u16 -> u8 view of the same memory.
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 2) };
        be.upload(&b, 0, bytes, 0).unwrap();
        b
    };
    // (heads, kv heads, head_dim, visible keys)
    let cases: &[(usize, usize, usize, usize)] = &[
        (32, 8, 128, 1),
        (32, 8, 128, 100),
        (32, 8, 128, 129),
        (32, 8, 128, 1000),
        (32, 8, 128, 4090),
        (32, 8, 128, 4096),
        (32, 8, 128, 16384),
        (16, 2, 128, 100000),
        (8, 1, 128, 70000),
        (16, 2, 64, 33333),
        (12, 2, 256, 20000),
        (6, 2, 32, 9000),
        (8, 8, 128, 5000),
        (40, 8, 128, 8000),
        (64, 8, 128, 12000),
    ];
    for &(heads, kv, dim, vis) in cases {
        let cap = vis.next_multiple_of(64);
        let q = rand_buf(heads * dim, 1.0);
        let k = rand_buf(kv * cap * dim, 1.0);
        let v = rand_buf(kv * cap * dim, 1.0);
        let out = be.alloc(heads * dim * 2, MemKind::Device).unwrap();
        let t = |b: &Buffer, shape: &[usize]| TensorRef {
            buf: b.clone(),
            offset: 0,
            dtype: DType::Bf16,
            shape: shape.to_vec(),
        };
        let attn = OpAttrs {
            pos0: vis - 1,
            scale: 1.0 / (dim as f32).sqrt(),
            n_heads: heads,
            n_kv_heads: kv,
            head_dim: dim,
            kv_cap: cap,
            ..Default::default()
        };
        let mut call = OpCall::new(
            Op::AttnDecode,
            vec![t(&q, &[1, heads, dim]), t(&k, &[kv, cap, dim]), t(&v, &[kv, cap, dim])],
            vec![t(&out, &[1, heads, dim])],
            attn,
        );
        let need = be
            .kernels()
            .iter()
            .find(|k| k.name == ATTN_DECODE_BF16_SPLITKV)
            .map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(&call)));
        let sb = be.alloc(need, MemKind::Device).unwrap();
        call.scratch = Some(TensorRef { buf: sb, offset: 0, dtype: DType::U8, shape: vec![need] });
        be.launch(ATTN_DECODE_BF16_SPLITKV, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let mut bytes = vec![0u8; heads * dim * 2];
        be.download(&out, 0, &mut bytes, 0).unwrap();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in &bytes {
            h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
        }
        println!("heads {heads:>2} kv {kv} dim {dim:>3} visible {vis:>6}: {h:016x}");
    }
}

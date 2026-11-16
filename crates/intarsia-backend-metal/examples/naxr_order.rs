//! Lane mnr probe: in which order does the neural accelerator (`mpp::tensor_ops::matmul2d`, multiply-accumulate into a float
//! cooperative tensor, bf16 operands) sum the K products of an output element? Runs `matmul_mxfp4_naxr_rows` (f32 out) on
//! operands with a wide exponent spread and counts, per candidate order computed on the host, the outputs whose bits differ.
//! A candidate with 0 mismatches at every K is an order an ALU kernel could reproduce bit for bit.
//!
//!   cargo run --release -p intarsia-backend-metal --example naxr_order      (inside a fleetq slot, `--quiet gpu`)

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("naxr_order needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::bench::{bf16_to_f32, Rng};
    use intarsia_backend_metal::MetalBackend;
    use intarsia_core::backend::{Backend, MemKind, TensorRef};
    use intarsia_core::{DType, Op, OpAttrs, OpCall};

    const FP4: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    let be = MetalBackend::new().expect("metal backend");
    let t = |b: &intarsia_core::backend::Buffer, dt: DType, shape: &[usize]| TensorRef {
        buf: b.clone(),
        offset: 0,
        dtype: dt,
        shape: shape.to_vec(),
    };
    let up = |v: &[u8]| {
        let b = be.alloc(v.len() + 16, MemKind::Device).unwrap();
        be.upload(&b, 0, v, 0).unwrap();
        b
    };
    // `tree`: absorption probe. One weight row of ones; activation row (a, b) holds +B at a, -B at b and 1 elsewhere, B so large
    // that any partial sum of ones merged into a node holding one unbalanced B is absorbed. The result is then the number of ones
    // outside the smallest summation subtree that holds both a and b: K - |subtree(lca(a, b))|, which gives the tree.
    if std::env::args().nth(1).as_deref() == Some("tree") {
        for &(k, bexp) in &[(32usize, 30i32), (32, 60), (64, 30), (128, 30), (256, 30), (384, 30)] {
            let pairs: Vec<(usize, usize)> = (0..k).flat_map(|a| (a + 1..k).map(move |b| (a, b))).collect();
            // above 64 only pairs with a at the start of a 16-block (enough to see the block structure)
            let pairs: Vec<(usize, usize)> = if k > 64 { pairs.into_iter().filter(|&(a, _)| a % 16 == 0 || a % 16 == 5).collect() } else { pairs };
            let m = pairs.len();
            let big = (((127 + bexp) as u16) << 7, 0x8000u16 | (((127 + bexp) as u16) << 7));
            let mut x = vec![0x3F80u16; m * k];
            for (i, &(a, b)) in pairs.iter().enumerate() {
                x[i * k + a] = big.0;
                x[i * k + b] = big.1;
            }
            let w = vec![0x22u8; k / 2]; // code 2 = 1.0 in both nibbles
            let sc = vec![127u8; k / 32];
            let xbytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let (xb, wb, sb) = (up(&xbytes), up(&w), up(&sc));
            let ab = up(&0f32.to_le_bytes());
            let yb = be.alloc(m * 4 + 16, MemKind::Device).unwrap();
            let cl = OpCall::new(
                Op::MatmulMxfp4,
                vec![t(&xb, DType::Bf16, &[m, k]), t(&wb, DType::U8, &[1, k / 2]), t(&sb, DType::U8, &[1, k / 32]), t(&ab, DType::F32, &[1])],
                vec![t(&yb, DType::F32, &[m, 1])],
                OpAttrs { mxfp4_w4a16: true, ..OpAttrs::default() },
            );
            be.launch("matmul_mxfp4_naxr_rows", &cl, 0).unwrap();
            be.synchronize(0).unwrap();
            let mut raw = vec![0u8; m * 4];
            be.download(&yb, 0, &mut raw, 0).unwrap();
            let got: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            println!("TREE K = {k} B = 2^{bexp}: rows a, columns b > a, entry = K - result (the ones absorbed with a and b, plus 2)");
            let mut i = 0;
            let mut last_a = usize::MAX;
            let mut line = String::new();
            for &(a, _b) in &pairs {
                if a != last_a {
                    if !line.is_empty() {
                        println!("{line}");
                    }
                    line = format!("a={a:3}:");
                    last_a = a;
                }
                let v = k as f32 - got[i];
                line.push_str(&format!(" {}", if v == v.round() && v.abs() < 1e6 { format!("{}", v as i64) } else { format!("{v:e}") }));
                i += 1;
            }
            println!("{line}");
        }
        return;
    }
    // spread: 0 = narrow exponents (like the model), 1 = wide (order-sensitive)
    for spread in [0u32, 1] {
        for &k in &[8usize, 10, 12, 16, 20, 24, 32, 48, 64, 128, 160, 256, 1024, 5120, 17408] {
            let kk = k.next_multiple_of(32);
            let (n, m) = (256usize, 4usize);
            let mut r = Rng(1234 + k as u64 + spread as u64 * 77);
            // a K below the padded width: the tail codes are zero
            let mut w = vec![0u8; n * kk / 2];
            for row in 0..n {
                for e in 0..k {
                    w[row * (kk / 2) + e / 2] |= (((r.next_u64() >> 40) & 0xF) as u8) << (4 * (e % 2));
                }
            }
            let srange = if spread == 1 { 24 } else { 6 };
            let sc: Vec<u8> = (0..n * kk / 32).map(|_| 110 + ((r.next_u64() >> 40) % srange) as u8).collect();
            let x: Vec<u16> = (0..m * kk)
                .map(|_| {
                    let mant = (r.next_u64() >> 40) as u16 & 0x7F;
                    let exp = 120 + ((r.next_u64() >> 40) % if spread == 1 { 16 } else { 4 }) as u16;
                    let sign = ((r.next_u64() >> 40) & 1) as u16;
                    (sign << 15) | (exp << 7) | mant
                })
                .collect();
            let xbytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let (xb, wb, sb) = (up(&xbytes), up(&w), up(&sc));
            let ab = up(&0f32.to_le_bytes());
            let yb = be.alloc(m * n * 4 + 16, MemKind::Device).unwrap();
            let cl = OpCall::new(
                Op::MatmulMxfp4,
                vec![
                    t(&xb, DType::Bf16, &[m, kk]),
                    t(&wb, DType::U8, &[n, kk / 2]),
                    t(&sb, DType::U8, &[n, kk / 32]),
                    t(&ab, DType::F32, &[1]),
                ],
                vec![t(&yb, DType::F32, &[m, n])],
                OpAttrs { mxfp4_w4a16: true, ..OpAttrs::default() },
            );
            be.launch("matmul_mxfp4_naxr_rows", &cl, 0).unwrap();
            be.synchronize(0).unwrap();
            let mut raw = vec![0u8; m * n * 4];
            be.download(&yb, 0, &mut raw, 0).unwrap();
            let got: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

            // exact products as f64 (bf16 x bf16: 16 significant bits)
            let prod = |i: usize, j: usize| -> Vec<f64> {
                (0..kk)
                    .map(|e| {
                        let byte = w[j * (kk / 2) + e / 2];
                        let code = if e % 2 == 0 { byte & 0xF } else { byte >> 4 };
                        let wv = FP4[code as usize] * f32::from_bits((sc[j * (kk / 32) + e / 32] as u32) << 23);
                        bf16_to_f32(x[i * kk + e]) as f64 * wv as f64
                    })
                    .collect()
            };
            let add = |a: f32, p: f64| (a as f64 + p) as f32; // one rounded add of an exact product
            type Cand = (&'static str, Box<dyn Fn(&[f64]) -> f32>);
            let chunked = |c: usize, inner: fn(&[f64]) -> f64, fuse: bool| -> Box<dyn Fn(&[f64]) -> f32> {
                Box::new(move |p: &[f64]| {
                    let mut acc = 0f32;
                    for ch in p.chunks(c) {
                        let s = inner(ch);
                        acc = if fuse { (acc as f64 + s) as f32 } else { acc + s as f32 };
                    }
                    acc
                })
            };
            fn seq32(ch: &[f64]) -> f64 {
                let mut a = 0f32;
                for &p in ch {
                    a = (a as f64 + p) as f32;
                }
                a as f64
            }
            fn exact(ch: &[f64]) -> f64 {
                ch.iter().sum()
            }
            fn tree32(ch: &[f64]) -> f64 {
                let mut v: Vec<f32> = ch.iter().map(|&p| p as f32).collect();
                while v.len() > 1 {
                    v = v.chunks(2).map(|c| if c.len() == 2 { c[0] + c[1] } else { c[0] }).collect();
                }
                v[0] as f64
            }
            let strided = |s: usize| -> Box<dyn Fn(&[f64]) -> f32> {
                Box::new(move |p: &[f64]| {
                    let mut a = vec![0f32; s];
                    for (e, &v) in p.iter().enumerate() {
                        a[e % s] = (a[e % s] as f64 + v) as f32;
                    }
                    let mut t = a.clone();
                    while t.len() > 1 {
                        t = t.chunks(2).map(|c| c[0] + c[1]).collect();
                    }
                    t[0]
                })
            };
            // the order the absorption probe (`tree`) shows: one accumulator over quads in K order, a quad being
            // (p[a] + p[a+1]) + (p[a+8] + p[a+9]) for a = 16 t + 2 j, j = 0..3
            fn quad_order(p: &[f64]) -> f32 {
                let mut acc = 0f32;
                for blk in p.chunks(16) {
                    let g = |i: usize| blk.get(i).copied().unwrap_or(0.0);
                    for j in 0..4 {
                        let lo = (g(2 * j) + g(2 * j + 1)) as f32;
                        let hi = (g(2 * j + 8) + g(2 * j + 9)) as f32;
                        acc += lo + hi;
                    }
                }
                acc
            }
            let mut cands: Vec<Cand> = vec![
                ("QUAD ORDER: acc += (p[a]+p[a+1]) + (p[a+8]+p[a+9])", Box::new(quad_order)),
                ("seq asc (rounded add per product)", Box::new(move |p: &[f64]| p.iter().fold(0f32, |a, &v| add(a, v)))),
                ("seq desc", Box::new(move |p: &[f64]| p.iter().rev().fold(0f32, |a, &v| add(a, v)))),
                ("exact sum, one rounding", Box::new(|p: &[f64]| p.iter().sum::<f64>() as f32)),
            ];
            for c in [2usize, 4, 8, 16, 32, 64, 128] {
                cands.push((Box::leak(format!("chunk {c}: seq f32, then add").into_boxed_str()), chunked(c, seq32, false)));
                cands.push((Box::leak(format!("chunk {c}: exact, rounded, then add").into_boxed_str()), chunked(c, exact, false)));
                cands.push((Box::leak(format!("chunk {c}: exact, fused into acc").into_boxed_str()), chunked(c, exact, true)));
                cands.push((Box::leak(format!("chunk {c}: pair tree f32, then add").into_boxed_str()), chunked(c, tree32, false)));
                cands.push((Box::leak(format!("{c} strided accumulators, tree").into_boxed_str()), strided(c)));
            }
            // rounding statistics against the exact sum (f64; exact for the short K)
            if spread == 1 && k <= 32 {
                let (mut rn, mut rz, mut rd, mut ru, mut other, mut shown) = (0, 0, 0, 0, 0, 0);
                for i in 0..m {
                    for j in 0..n {
                        let p = prod(i, j);
                        let e: f64 = p.iter().sum();
                        let g = got[i * n + j];
                        let near = e as f32;
                        let (lo, hi) = if (near as f64) <= e {
                            (near, if near as f64 == e { near } else { f32::from_bits(if near >= 0.0 { near.to_bits() + 1 } else { near.to_bits() - 1 }) })
                        } else {
                            (f32::from_bits(if near > 0.0 { near.to_bits() - 1 } else { near.to_bits() + 1 }), near)
                        };
                        let tz = if e >= 0.0 { lo } else { hi };
                        if g == near { rn += 1 }
                        if g == tz { rz += 1 }
                        if g == lo { rd += 1 }
                        if g == hi { ru += 1 }
                        if g != lo && g != hi {
                            other += 1;
                            if shown < 6 {
                                shown += 1;
                                println!("      off by more than a rounding: got {g:e} exact {e:e} ({} ulp) products {:?}", ((g as f64 - e) / (near.abs() as f64 * 1.19e-7)).round(), &p[..k]);
                            }
                        }
                    }
                }
                println!("    K = {k}: of {} = nearest {rn}, toward zero {rz}, down {rd}, up {ru}, neither neighbour {other}", m * n);
            }
            let mut miss = vec![0usize; cands.len()];
            for i in 0..m {
                for j in 0..n {
                    let p = prod(i, j);
                    for (ci, (_, f)) in cands.iter().enumerate() {
                        if f(&p[..]).to_bits() != got[i * n + j].to_bits() {
                            miss[ci] += 1;
                        }
                    }
                }
            }
            let mut order: Vec<usize> = (0..cands.len()).collect();
            order.sort_by_key(|&c| miss[c]);
            println!("spread {spread} K = {k}: {} outputs", m * n);
            for &c in order.iter().take(5) {
                println!("    {:5} mismatches  {}", miss[c], cands[c].0);
            }
        }
    }
}

//! `Backend::topk_rows` on Metal (lane ink-mtp, kernels/metal/topk.metal) against the sampler's host order
//! (`sampler::topk_row`: value descending, ties by ascending index, a NaN row flagged): random rows, heavy ties, -inf, a NaN
//! row, short rows (chunks with fewer than k elements), f32 and bf16 logits, k 1 / 8 / 64, the Inkling-2 vocabulary.

mod common;
use common::*;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::DType;

fn bits_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn run(be: &intarsia_backend_metal::MetalBackend, rows: &[Vec<f32>], k: usize, bf16: bool) {
    let r = rows.len();
    let v = rows[0].len();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let (bytes, dt): (Vec<u8>, DType) = if bf16 {
        (flat.iter().flat_map(|&x| bits_bf16(x).to_le_bytes()).collect(), DType::Bf16)
    } else {
        (flat.iter().flat_map(|x| x.to_le_bytes()).collect(), DType::F32)
    };
    let lb = be.alloc(bytes.len() + 64, MemKind::Device).unwrap();
    be.upload(&lb, 0, &bytes, 0).unwrap();
    let mk = |n: usize| be.alloc(n.max(16), MemKind::Device).unwrap();
    let (ib, vb, nb, sb) = (mk(r * k * 4), mk(r * k * 4), mk(r * 4), mk(r * 32 * k * 8));
    let logits = TensorRef::new(lb, dt, &[r, v]);
    let ids = TensorRef::new(ib.clone(), DType::I32, &[r, k]);
    let vals = TensorRef::new(vb.clone(), DType::F32, &[r, k]);
    let nan = TensorRef::new(nb.clone(), DType::I32, &[r]);
    let scratch = TensorRef::new(sb, DType::F32, &[r * 32 * k * 2]);
    be.topk_rows(&logits, k, &ids, &vals, &nan, &scratch, 0).unwrap();
    be.synchronize(0).unwrap();
    let rd = |b: &intarsia_core::backend::Buffer, n: usize| {
        let mut raw = vec![0u8; n * 4];
        be.download(b, 0, &mut raw, 0).unwrap();
        raw.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect::<Vec<_>>()
    };
    let gi: Vec<i32> = rd(&ib, r * k).into_iter().map(i32::from_le_bytes).collect();
    let gv: Vec<f32> = rd(&vb, r * k).into_iter().map(f32::from_le_bytes).collect();
    let gn: Vec<i32> = rd(&nb, r).into_iter().map(i32::from_le_bytes).collect();
    for (ri, row) in rows.iter().enumerate() {
        // the reference sees what the kernel reads (bf16-rounded values for bf16 logits)
        let seen: Vec<f32> = if bf16 { row.iter().map(|&x| f32::from_bits((bits_bf16(x) as u32) << 16)).collect() } else { row.clone() };
        match intarsia_core::sampler::topk_row(&seen, k) {
            None => assert_eq!(gn[ri], 1, "row {ri}: NaN row not flagged"),
            Some((i, x)) => {
                assert_eq!(gn[ri], 0, "row {ri}: flagged without a NaN");
                let got_i: Vec<u32> = gi[ri * k..(ri + 1) * k].iter().map(|&q| q as u32).collect();
                assert_eq!(got_i, i, "row {ri} (V {v}, k {k}, bf16 {bf16}): ids");
                assert_eq!(&gv[ri * k..(ri + 1) * k], &x[..], "row {ri}: values");
            }
        }
    }
}

#[test]
fn topk_rows_matches_the_sampler_order() {
    let be = backend();
    let mut seed = 1u64;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
        seed += 1;
        rand_bf16(n, seed).iter().map(|&b| f32::from_bits((b as u32) << 16) * scale).collect()
    };
    for &v in &[200058usize, 4096, 333, 70] {
        for &k in &[1usize, 8, 64] {
            if k > v {
                continue;
            }
            let mut rows = vec![rnd(v, 10.0), rnd(v, 10.0)];
            // heavy ties: few distinct values
            rows.push(rnd(v, 1.0).iter().map(|x| (x * 3.0).round()).collect());
            // -inf everywhere but a few
            let mut r = vec![f32::NEG_INFINITY; v];
            for j in (0..v).step_by(v / 5 + 1) {
                r[j] = j as f32;
            }
            rows.push(r);
            // a NaN row
            let mut r = rnd(v, 1.0);
            r[v / 2] = f32::NAN;
            rows.push(r);
            for bf16 in [false, true] {
                run(&be, &rows, k, bf16);
            }
        }
    }
}

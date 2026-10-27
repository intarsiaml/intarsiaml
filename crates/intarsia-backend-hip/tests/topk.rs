//! G-B for the device top-k of sampling on HIP (lane htk, `Backend::topk_rows`, kernels/hip/topk.hip, the port of kernels/cuda/topk.cu): the
//! k largest values of each f32 row, in the order `intarsia_core::sampler` walks them (value descending, ties by ascending index, -0.0 ==
//! +0.0), with their indices, BITWISE equal to a host sort; and the token `sample_from_topk` draws from them is the token `sample_row`
//! draws from the whole row. `ik_topk_f32` is not an `Op` (the model calls it for `HeadMode::LastTopK`), so it is not in the
//! `every_hip_kernel_*` registry sweep; this file is its admission: random bf16-grid rows with many ties, signed zeros, infinities, planted
//! maxima on every slice edge (the grid is 32 slices of ceil(V / 32)), k at both ends (1, 64, k == V), NaN rows (flagged, the rows around
//! them untouched) and the launcher's rejections. scripts/hip-decode/mutate.py has the `htk-*` mutants that fail it.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_hip)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::rng::SplitMix64;
use intarsia_core::sampler::{sample_from_topk, sample_row, SamplingParams};
use intarsia_core::{DType, HostTensor};
use std::sync::Arc;

const SLICES: usize = 32;

struct Rig {
    be: Arc<dyn Backend>,
}

impl Rig {
    fn new(test: &str) -> Option<Rig> {
        let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        match intarsia_backend_hip::create(ord) {
            Ok(be) => Some(Rig { be }),
            Err(e) => {
                common::skip(test, &format!("no usable HIP device ({e})"));
                None
            }
        }
    }

    fn put_bytes(&self, data: &[u8], dtype: DType, shape: &[usize]) -> TensorRef {
        let buf = self.be.alloc(data.len().max(16), MemKind::Device).unwrap();
        self.be.zero(&buf, 0).unwrap();
        if !data.is_empty() {
            self.be.upload(&buf, 0, data, 0).unwrap();
        }
        TensorRef::new(buf, dtype, shape)
    }

    fn f32s(&self, vals: &[f32], shape: &[usize]) -> TensorRef {
        self.put_bytes(&HostTensor::from_f32_as(DType::F32, shape, vals).data, DType::F32, shape)
    }

    /// An output buffer of `bytes` filled with 0xA5.
    fn poison(&self, bytes: usize, dtype: DType, shape: &[usize]) -> TensorRef {
        self.put_bytes(&vec![0xA5u8; bytes], dtype, shape)
    }

    fn down(&self, t: &TensorRef) -> Vec<u8> {
        let mut d = vec![0u8; t.nbytes()];
        self.be.download(&t.buf, t.offset, &mut d, 0).unwrap();
        d
    }

    /// Launch `topk_rows` on `data` ([rows, v]) with scratch filled with `fill`; (ids, vals, nan flags).
    fn topk(&self, data: &[f32], rows: usize, v: usize, k: usize, fill: u8) -> Result<(Vec<i32>, Vec<f32>, Vec<i32>), String> {
        let logits = self.f32s(data, &[rows, v]);
        let ids = self.poison(rows * k * 4, DType::I32, &[rows, k]);
        let vals = self.poison(rows * k * 4, DType::F32, &[rows, k]);
        let nan = self.poison(rows * 4, DType::I32, &[rows]);
        let sb = rows * SLICES * k * 8;
        let scratch = self.put_bytes(&vec![fill; sb], DType::U8, &[sb]);
        self.be.topk_rows(&logits, k, &ids, &vals, &nan, &scratch, 0).map_err(|e| e.to_string())?;
        self.be.synchronize(0).unwrap();
        let i32s = |b: Vec<u8>| b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
        let f32s = |b: Vec<u8>| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
        Ok((i32s(self.down(&ids)), f32s(self.down(&vals)), i32s(self.down(&nan))))
    }
}

/// The sampler's candidate order: value descending, ties by ascending index (-0.0 == +0.0).
fn host_topk(row: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| row[*b as usize].partial_cmp(&row[*a as usize]).unwrap().then(a.cmp(b));
    if k < row.len() {
        order.select_nth_unstable_by(k - 1, cmp);
        order.truncate(k);
    }
    order.sort_unstable_by(cmp);
    order
}

/// Check a launch on NaN-free `data` against the host order, bit for bit, and the draw of the sampler on top of it. Returns the pairs checked.
fn check(rig: &Rig, what: &str, data: &[f32], rows: usize, v: usize, k: usize) -> usize {
    let mut n = 0;
    // zeroed and non-zero scratch: the kernel must not depend on what the scratch held
    let mut first: Option<(Vec<i32>, Vec<f32>)> = None;
    for fill in [0u8, 0x01] {
        let (gi, gv, gn) = rig.topk(data, rows, v, k, fill).unwrap_or_else(|e| panic!("{what}: topk_rows: {e}"));
        assert!(gn.iter().all(|&f| f == 0), "{what}: no NaN in the rows, flags {gn:?}");
        for r in 0..rows {
            let row = &data[r * v..(r + 1) * v];
            let order = host_topk(row, k);
            for j in 0..k {
                let (id, val) = (gi[r * k + j], gv[r * k + j]);
                assert_eq!(id as u32, order[j], "{what}: v {v} rows {rows} k {k} row {r} rank {j}");
                assert_eq!(val.to_bits(), row[order[j] as usize].to_bits(), "{what}: v {v} rows {rows} k {k} row {r} rank {j} value");
                n += 1;
            }
            // the token the sampler draws from the candidates is the token it draws from the whole row (once per launch shape)
            if fill != 0 {
                continue;
            }
            let ids: Vec<u32> = gi[r * k..(r + 1) * k].iter().map(|&i| i as u32).collect();
            let vals = &gv[r * k..(r + 1) * k];
            for (temperature, top_p) in [(0.7f32, 1.0f32), (1.0, 0.9), (1.3, 0.5)] {
                let p = SamplingParams { temperature, top_k: k, top_p };
                for u in [0.0f32, 0.25, 0.5, 0.75, 0.999] {
                    assert_eq!(sample_from_topk(&ids, vals, &p, u), sample_row(row, &p, u), "{what}: row {r} k {k} t {temperature} tp {top_p} u {u}");
                }
            }
        }
        match &first {
            None => first = Some((gi, gv)),
            Some((i0, v0)) => {
                assert_eq!(&gi, i0, "{what}: ids depend on the scratch contents");
                assert!(gv.iter().zip(v0).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: values depend on the scratch contents");
            }
        }
    }
    n
}

/// Random bf16-grid rows (many exact ties), some with signed zeros and -inf, one all tied; the CUDA test's shapes (V = 248320 is the 35B
/// vocabulary), k from 1 to 64.
#[test]
fn device_topk_is_the_host_candidate_order() {
    let Some(rig) = Rig::new("topk") else { return };
    let mut rng = SplitMix64::new(16);
    let mut checked = 0usize;
    for v in [248320usize, 151936, 1000, 70] {
        for (rows, k) in [(1usize, 1usize), (3, 20), (16, 40), (16, 64), (5, 64.min(v - 1)), (2, 2)] {
            let mut data = Vec::new();
            for r in 0..rows {
                // bf16-grid logits, skewed like a real row; some special values
                let mut row: Vec<f32> = (0..v)
                    .map(|i| {
                        let x = rng.normal_vec(1, 3.0)[0] + if i % 97 == 0 { 9.0 } else { 0.0 };
                        f32::from_bits(x.to_bits() & 0xffff_0000)
                    })
                    .collect();
                if r % 3 == 1 {
                    for (i, x) in row.iter_mut().enumerate() {
                        match i % 11 {
                            1 => *x = 0.0,
                            2 => *x = -0.0,
                            3 => *x = f32::NEG_INFINITY,
                            _ => {}
                        }
                    }
                }
                if r % 5 == 4 {
                    row.iter_mut().for_each(|x| *x = 2.0); // all tied: the k lowest indices
                }
                data.extend(row);
            }
            checked += check(&rig, "random", &data, rows, v, k);
        }
    }
    println!("G-B topk_f32 (hip): {checked} (id, value) pairs equal to the host order (V 248320 / 151936 / 1000 / 70, R <= 16, k <= 64), sampler draws equal");
}

/// Maxima planted on the slice edges of V = 248320 (slice width 7760), at both ends of the row and in the last slice alone, equal values
/// among them so the tie order across slices counts. A kernel that drops the end of a slice, the last slice, the last element or a slice's
/// tie loses one of them.
#[test]
fn device_topk_finds_planted_values_on_every_slice_edge() {
    let Some(rig) = Rig::new("topk_edges") else { return };
    let v = 248_320usize;
    let per = v.div_ceil(SLICES);
    let mut rng = SplitMix64::new(33);
    let mut checked = 0usize;
    let base = |rng: &mut SplitMix64| -> Vec<f32> { rng.normal_vec(v, 1.0).into_iter().map(|x| f32::from_bits(x.to_bits() & 0xffff_0000)).collect() };
    // edges: first and last element of the row, first and last of slices 1, 15 and 31, the middle of a slice
    let edges = [0, per - 1, per, 2 * per - 1, 15 * per, 16 * per - 1, 31 * per - 1, 31 * per, v - 2, v - 1, 5 * per + per / 2];
    for k in [1usize, 2, 8, 11, 64] {
        // distinct planted values, descending in the order of `edges` shuffled by k so every edge takes the top rank once
        let mut data = Vec::new();
        for shift in 0..3usize {
            let mut row = base(&mut rng);
            for (n, &e) in edges.iter().enumerate() {
                row[e] = 100.0 + ((n + shift * 4) % edges.len()) as f32;
            }
            data.extend(row);
        }
        // the same value at every edge: the k lowest edge indices win the ties
        let mut tied = base(&mut rng);
        for &e in &edges {
            tied[e] = 50.0;
        }
        data.extend(tied);
        // everything the row holds that matters is in the last slice (and the last element), or in the first
        let mut last = base(&mut rng);
        for j in 0..64 {
            last[31 * per + 3 * j] = 60.0 + (j % 5) as f32;
        }
        last[v - 1] = 70.0;
        data.extend(last);
        let mut first = base(&mut rng);
        for j in 0..64 {
            first[2 * j] = 60.0 + (j % 7) as f32;
        }
        data.extend(first);
        checked += check(&rig, "planted", &data, 6, v, k);
    }
    println!("G-B topk_f32 (hip) planted: {checked} pairs equal to the host order (maxima on every slice edge, last slice, last element)");
}

/// k at the edges: 1, the maximum (64), k == V, and the ties, signed zeros, infinities and subnormals the sampler orders by index.
#[test]
fn device_topk_edge_k_and_special_values() {
    let Some(rig) = Rig::new("topk_special") else { return };
    let mut checked = 0usize;
    // k == V and V below the slice count: slices are empty or hold a single element
    for (v, k) in [(1usize, 1usize), (2, 2), (8, 8), (31, 31), (64, 64), (65, 64), (70, 64)] {
        let mut rng = SplitMix64::new(v as u64);
        let data: Vec<f32> = (0..3 * v).map(|i| if i / v == 1 { 1.5 } else { f32::from_bits(rng.normal_vec(1, 2.0)[0].to_bits() & 0xffff_0000) }).collect();
        checked += check(&rig, "k==V", &data, 3, v, k);
    }
    // ties and special values in one row (V = 4096): the order is by index inside each tie class
    let v = 4096usize;
    let specials = [f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, f32::MAX, f32::MIN, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1.0e-45, -1.0e-45, 1.0, -1.0];
    let mut rng = SplitMix64::new(9);
    let mut data = Vec::new();
    for r in 0..8usize {
        let mut row: Vec<f32> = (0..v).map(|i| specials[(i * 7 + r * 5 + (rng.next_u64() % 3) as usize) % specials.len()]).collect();
        if r == 7 {
            row.iter_mut().for_each(|x| *x = f32::NEG_INFINITY); // all tied at -inf
        }
        data.extend(row);
    }
    for k in [1usize, 3, 17, 64] {
        checked += check(&rig, "specials", &data, 8, v, k);
    }
    // +0.0 and -0.0 tie: the lower index wins whichever sign it has, and the value keeps its sign bit
    let mut zeros = vec![-1.0f32; 300];
    for (i, z) in [(5usize, -0.0f32), (6, 0.0), (7, -0.0), (200, 0.0), (299, -0.0)] {
        zeros[i] = z;
    }
    let (gi, gv, gn) = rig.topk(&zeros, 1, 300, 4, 0).unwrap();
    assert_eq!((gi, gn), (vec![5, 6, 7, 200], vec![0]));
    assert_eq!(gv.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), vec![(-0.0f32).to_bits(), 0.0f32.to_bits(), (-0.0f32).to_bits(), 0.0f32.to_bits()]);
    println!("G-B topk_f32 (hip) edges: {checked} pairs equal to the host order (k = 1 / 64 / V, ties, +-0, +-inf, subnormals, extremes)");
}

/// A row that holds a NaN anywhere is flagged (its entries are not meaningful); the rows around it are untouched.
#[test]
fn device_topk_flags_nan_rows_and_only_those() {
    let Some(rig) = Rig::new("topk_nan") else { return };
    let v = 248_320usize;
    let per = v.div_ceil(SLICES);
    let mut rng = SplitMix64::new(5);
    let at = [0usize, per - 1, per, 17 * per + 3, 31 * per, v - 1];
    for k in [1usize, 8, 64] {
        // [clean, NaN at a position, clean, ...]: every NaN position gets a row of its own between two clean rows
        let mut data = Vec::new();
        let mut want = Vec::new();
        for &p in &at {
            let clean: Vec<f32> = rng.normal_vec(v, 1.0).into_iter().map(|x| f32::from_bits(x.to_bits() & 0xffff_0000)).collect();
            let mut dirty = clean.clone();
            dirty[p] = f32::NAN;
            data.extend(clean);
            data.extend(dirty);
            want.extend([0, 1]);
        }
        let mut all_nan = vec![f32::NAN; v];
        all_nan[3] = f32::NAN;
        data.extend(all_nan);
        want.push(1);
        let rows = want.len();
        let (gi, gv, gn) = rig.topk(&data, rows, v, k, 0).unwrap();
        assert_eq!(gn, want, "k {k}: NaN flags");
        for r in (0..rows - 1).step_by(2) {
            let row = &data[r * v..(r + 1) * v];
            let order = host_topk(row, k);
            for j in 0..k {
                assert_eq!(gi[r * k + j] as u32, order[j], "k {k} clean row {r} rank {j}");
                assert_eq!(gv[r * k + j].to_bits(), row[order[j] as usize].to_bits(), "k {k} clean row {r} rank {j} value");
            }
        }
    }
    // a NaN row does not leave a stale flag: the same buffers, a clean launch afterwards
    let clean: Vec<f32> = (0..500).map(|i| (i % 13) as f32).collect();
    check(&rig, "after-nan", &clean, 1, 500, 8);
    println!("G-B topk_f32 (hip) NaN: rows flagged at every slice edge and the last element, neighbours exact, all-NaN row flagged");
}

/// The launcher's rejections: k out of 1..=64 or above V, shapes that do not match, a scratch too small. A rejected launch leaves the
/// stream usable.
#[test]
fn device_topk_rejects_bad_arguments() {
    let Some(rig) = Rig::new("topk_reject") else { return };
    let (rows, v) = (2usize, 100usize);
    let data: Vec<f32> = (0..rows * v).map(|i| (i % 17) as f32).collect();
    let logits = rig.f32s(&data, &[rows, v]);
    let launch = |k: usize, out_k: usize, scratch_bytes: usize| {
        let ids = rig.poison(rows * out_k * 4, DType::I32, &[rows, out_k]);
        let vals = rig.poison(rows * out_k * 4, DType::F32, &[rows, out_k]);
        let nan = rig.poison(rows * 4, DType::I32, &[rows]);
        let scratch = rig.put_bytes(&vec![0u8; scratch_bytes.max(16)], DType::U8, &[scratch_bytes.max(16)]);
        let r = rig.be.topk_rows(&logits, k, &ids, &vals, &nan, &scratch, 0);
        rig.be.synchronize(0).unwrap();
        r
    };
    let need = |k: usize| rows * SLICES * k * 8;
    assert!(launch(0, 1, need(1)).is_err(), "k = 0");
    assert!(launch(65, 65, need(65)).is_err(), "k = 65 (above the maximum)");
    assert!(launch(101, 101, need(101)).is_err(), "k above V");
    // k above V with k within 1..=64: a row of 10 logits, k = 11
    {
        let small = rig.f32s(&(0..rows * 10).map(|i| i as f32).collect::<Vec<_>>(), &[rows, 10]);
        let ids = rig.poison(rows * 11 * 4, DType::I32, &[rows, 11]);
        let vals = rig.poison(rows * 11 * 4, DType::F32, &[rows, 11]);
        let nan = rig.poison(rows * 4, DType::I32, &[rows]);
        let scratch = rig.put_bytes(&vec![0u8; need(11)], DType::U8, &[need(11)]);
        assert!(rig.be.topk_rows(&small, 11, &ids, &vals, &nan, &scratch, 0).is_err(), "k = 11 above V = 10");
        rig.be.synchronize(0).unwrap();
    }
    assert!(launch(8, 9, need(8)).is_err(), "ids / vals shaped [R, 9] for k = 8");
    assert!(launch(8, 8, need(8) - 1).is_err(), "scratch one byte short");
    assert!(launch(64, 64, need(64)).is_ok(), "k = 64 with the exact scratch");
    assert!(launch(1, 1, need(1)).is_ok(), "k = 1 with the exact scratch");
    // a rejected launch left nothing behind
    check(&rig, "after-reject", &data, rows, v, 5);
    println!("G-B topk_f32 (hip) rejections: k 0 / 65 / > V, wrong output shape, short scratch");
}

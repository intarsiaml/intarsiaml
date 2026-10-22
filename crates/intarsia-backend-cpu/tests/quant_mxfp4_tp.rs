//! Tensor-parallel slices of MXFP4 weights (Qwen3.8-27B at tp=2, lane T2): what each rank loads through
//! `quant::load_linear` with a `Shard`, run through the cpu reference `MatmulMxfp4`, must give the unsharded result.
//!
//! * column-parallel (`Rows`, q / k / v / gate / up / in_z / in_b / in_a): the ranks' outputs concatenated are the full
//!   output, bit for bit (each output row is computed from the same bytes);
//! * the GatedDeltaNet `[q | k | v]` row groups (`RowGroups`): each group split on its own, concatenated in group order;
//! * row-parallel (`Cols`, o_proj / out_proj / down_proj): the ranks' f32 partials summed are the full output up to the
//!   order of the f32 sum. With W4A4 the activation QDQ groups of a rank's K slice are exactly the full matrix's
//!   (the slice holds whole 32-element groups), checked bitwise;
//! * a column split that would cut a 32-element group is refused.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::quant::{self, mxfp4, HostQTensor, Mxfp4Act, QuantLoadOpts};
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::{MemWeights, Shard, ShardAxis};
use intarsia_core::{DType, HostTensor, Op, OpAttrs};

fn put(w: &mut MemWeights, prefix: &str, q: &HostQTensor) {
    w.insert(&format!("{prefix}.weight"), q.parts[0].clone());
    w.insert(&format!("{prefix}.weight_scale"), q.parts[1].clone());
}

/// The rank's (or the whole, `shard = None`) weight as loaded onto the cpu backend, back on the host.
fn load(w: &MemWeights, prefix: &str, act: Mxfp4Act, shard: Option<Shard>) -> intarsia_core::error::Result<HostQTensor> {
    let b = CpuBackend::new();
    let q = quant::load_linear(w, prefix, &b, &QuantLoadOpts { mxfp4_act: act, shard, ..Default::default() })?;
    assert_eq!(q.format, quant::MXFP4);
    q.download(&b)
}

/// `x [m, k] @ W^T` through the cpu reference `MatmulMxfp4`, output in `odt`, returned as f32.
fn matmul(x: &HostTensor, q: &HostQTensor, odt: DType) -> Vec<f32> {
    let b = CpuBackend::new();
    let m = x.shape[0];
    let hc = HostCall {
        op: Op::MatmulMxfp4,
        inputs: vec![x.clone(), q.parts[0].clone(), q.parts[1].clone(), q.parts[2].clone()],
        outputs: vec![OutSpec::new(odt, &[m, q.n])],
        attrs: OpAttrs::default(),
    };
    run_host(&b, &format!("ref_{}", Op::MatmulMxfp4.as_str()), &hc).unwrap().remove(0).to_f32()
}

fn x_rows(rng: &mut SplitMix64, m: usize, k: usize, dt: DType) -> HostTensor {
    HostTensor::from_f32_as(dt, &[m, k], &rng.normal_vec(m * k, 1.0))
}

const ACTS: [Mxfp4Act; 2] = [Mxfp4Act::Dynamic, Mxfp4Act::None];

#[test]
fn row_shards_concatenate_to_the_full_output_bitwise() {
    let mut rng = SplitMix64::new(0x72);
    let (n, k, m) = (96, 160, 5);
    let mut w = MemWeights::new();
    put(&mut w, "p", &mxfp4::random_rows(&mut rng, n, k, Mxfp4Act::Dynamic));
    for act in ACTS {
        let full = load(&w, "p", act, None).unwrap();
        for (xdt, odt) in [(DType::Bf16, DType::Bf16), (DType::Bf16, DType::F32), (DType::F32, DType::F32)] {
            let x = x_rows(&mut rng, m, k, xdt);
            let want = matmul(&x, &full, odt);
            for world in [2, 4] {
                let mut got = vec![Vec::new(); m];
                for rank in 0..world {
                    let q = load(&w, "p", act, Some(Shard { axis: ShardAxis::Rows, rank, world })).unwrap();
                    assert_eq!((q.n, q.k), (n / world, k));
                    let y = matmul(&x, &q, odt);
                    for (r, row) in got.iter_mut().enumerate() {
                        row.extend_from_slice(&y[r * q.n..(r + 1) * q.n]);
                    }
                }
                let got: Vec<f32> = got.concat();
                assert!(
                    got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "rows world {world} act {} x {xdt} out {odt}",
                    act.name()
                );
            }
        }
    }
}

#[test]
fn row_group_shards_concatenate_in_group_order_bitwise() {
    // q | k | v of a GatedDeltaNet in_proj_qkv: groups of 64, 64, 128 rows (the 27B: 2048, 2048, 6144)
    let mut rng = SplitMix64::new(0x73);
    let g = [64usize, 64, 128];
    let (n, k, m) = (g.iter().sum::<usize>(), 128, 3);
    let mut w = MemWeights::new();
    put(&mut w, "qkv", &mxfp4::random_rows(&mut rng, n, k, Mxfp4Act::Dynamic));
    for act in ACTS {
        let full = load(&w, "qkv", act, None).unwrap();
        let x = x_rows(&mut rng, m, k, DType::Bf16);
        let want = matmul(&x, &full, DType::Bf16);
        let world = 2;
        for rank in 0..world {
            let sh = Shard { axis: ShardAxis::RowGroups(g), rank, world };
            let q = load(&w, "qkv", act, Some(sh)).unwrap();
            assert_eq!((q.n, q.k), (n / world, k));
            let y = matmul(&x, &q, DType::Bf16);
            // this rank's rows of the full output, group by group
            let rows: Vec<usize> = sh.row_ranges(n).unwrap().into_iter().flat_map(|(r0, rn)| r0..r0 + rn).collect();
            assert_eq!(rows.len(), q.n);
            for mi in 0..m {
                for (j, &r) in rows.iter().enumerate() {
                    assert_eq!(y[mi * q.n + j].to_bits(), want[mi * n + r].to_bits(), "rank {rank} act {} row {mi} col {r}", act.name());
                }
            }
        }
    }
}

/// QDQ of a K slice of whole groups is the slice of the full QDQ: what makes the W4A4 row-parallel partials exact.
#[test]
fn w4a4_activation_qdq_of_a_column_slice_is_the_slice_of_the_full_qdq() {
    let mut rng = SplitMix64::new(0x74);
    let (m, k) = (4, 256);
    let x = rng.normal_vec(m * k, 2.0);
    let full = mxfp4::qdq_rows(&x, m, k);
    for world in [2, 4, 8] {
        let lk = k / world;
        for rank in 0..world {
            let part: Vec<f32> = (0..m).flat_map(|r| x[r * k + rank * lk..r * k + (rank + 1) * lk].to_vec()).collect();
            let q = mxfp4::qdq_rows(&part, m, lk);
            for r in 0..m {
                for c in 0..lk {
                    assert_eq!(q[r * lk + c].to_bits(), full[r * k + rank * lk + c].to_bits(), "world {world} rank {rank}");
                }
            }
        }
    }
}

#[test]
fn column_shards_sum_to_the_full_output() {
    let mut rng = SplitMix64::new(0x75);
    // (n, k): a small one and the 27B's FFN down / o_proj input widths at tp=2 scaled down (K / world a multiple of 32)
    for (n, k) in [(48usize, 128usize), (64, 384), (40, 1088)] {
        let mut w = MemWeights::new();
        put(&mut w, "o", &mxfp4::random_rows(&mut rng, n, k, Mxfp4Act::Dynamic));
        for act in ACTS {
            let full = load(&w, "o", act, None).unwrap();
            for m in [1usize, 9] {
                let x = x_rows(&mut rng, m, k, DType::Bf16);
                let want = matmul(&x, &full, DType::F32);
                for world in [2, 4] {
                    if k % (world * 32) != 0 {
                        continue;
                    }
                    let lk = k / world;
                    let mut sum = vec![0f32; m * n];
                    for rank in 0..world {
                        let q = load(&w, "o", act, Some(Shard { axis: ShardAxis::Cols, rank, world })).unwrap();
                        assert_eq!((q.n, q.k), (n, lk));
                        // the rank's activation columns, as the rank's workspace holds them
                        let xs: Vec<f32> = {
                            let xf = x.to_f32();
                            (0..m).flat_map(|r| xf[r * k + rank * lk..r * k + (rank + 1) * lk].to_vec()).collect()
                        };
                        let xs = HostTensor::from_f32_as(DType::Bf16, &[m, lk], &xs);
                        for (s, p) in sum.iter_mut().zip(matmul(&xs, &q, DType::F32)) {
                            *s += p;
                        }
                    }
                    let num: f64 = sum.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
                    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
                    let rel = (num / den.max(1e-30)).sqrt();
                    assert!(rel < 1e-6, "cols n {n} k {k} m {m} world {world} act {}: rel l2 {rel:.3e}", act.name());
                }
            }
        }
    }
}

#[test]
fn a_column_split_through_a_32_group_is_refused() {
    let mut rng = SplitMix64::new(0x76);
    let mut w = MemWeights::new();
    put(&mut w, "d", &mxfp4::random_rows(&mut rng, 16, 96, Mxfp4Act::Dynamic)); // K/2 = 48: not whole groups
    put(&mut w, "e", &mxfp4::random_rows(&mut rng, 16, 64, Mxfp4Act::Dynamic)); // K/4 = 16
    for (p, world) in [("d", 2usize), ("e", 4)] {
        let e = load(&w, p, Mxfp4Act::Dynamic, Some(Shard { axis: ShardAxis::Cols, rank: 0, world })).unwrap_err();
        assert!(e.to_string().contains("whole 32-element groups"), "{p}: {e}");
    }
    // rows of the same weights split fine, and 64 columns in two halves too
    load(&w, "d", Mxfp4Act::Dynamic, Some(Shard { axis: ShardAxis::Rows, rank: 1, world: 2 })).unwrap();
    load(&w, "e", Mxfp4Act::Dynamic, Some(Shard { axis: ShardAxis::Cols, rank: 1, world: 2 })).unwrap();
}

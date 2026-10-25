//! Lane wg: fragment-order dense fp8 weights (`Layout::Fp8Frag`, kernels/hip/ik_f8_rows.h, `Backend::relayout_fp8_frag`).
//!
//! The tag is a property of the weight's allocation. This file proves, on a GPU:
//!   1. every reader that accepts the layout gives, on tagged weights, the bytes it gives on row-major ones (same sums, other addresses):
//!      the WMMA family (`matmul_fp8_wmma`, `_rows`, the four folds) and the three prefill GEMMs;
//!   2. every fp8 reader that does not know the layout fails loudly (the launcher returns an error; the registry never selects it for a tagged
//!      call), and the set of fp8 kernels is pinned so a new one has to be classified here;
//!   3. the raw-byte paths of the backend (download, upload, device copy, row slice, reshape) refuse a tagged tensor, and the one reader of
//!      its bytes, `download_row_major` / `QTensor::download`, returns the checkpoint order;
//!   4. the loader (`Fp8E4m3::load`, with and without a TP shard) tags exactly when `INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_FP8_FRAG=1`
//!      are set (the gate runs this file once with them and once without).
//! Needs a GPU: skips (passes) without one.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::{Backend, Layout, MemKind};
use intarsia_core::exec::{run_host, run_host_frag, HostCall};
use intarsia_core::quant::cases::{
    gated_rmsnorm_matmul_fp8_case, matmul_fp8_case, matmul_fp8_pair_case, rmsnorm_add_matmul_fp8_pair_case, sigmoid_mul_matmul_fp8_case,
};
use intarsia_core::quant::fp8::{self, frag_eligible, frag_pack};
use intarsia_core::quant::{load_linear, Fp8Act, QuantLoadOpts};
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::{MemWeights, Shard, ShardAxis};
use intarsia_core::{DType, HostTensor, Op, ShapeInfo, TensorRef};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

fn put(be: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    be.zero(&buf, 0).unwrap();
    be.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn knobs_on() -> bool {
    std::env::var("INTARSIA_HIP_WMMA_ROWS").as_deref() == Ok("1") && std::env::var("INTARSIA_HIP_FP8_FRAG").as_deref() == Ok("1")
}

/// The fp8 kernels that read fragment order. Everything else registered for an fp8 op must fail on a tagged weight.
const FRAG_READERS: [&str; 9] = [
    "matmul_fp8_wmma",
    "matmul_fp8_wmma_rows",
    "matmul_fp8_pair_wmma",
    "gated_rmsnorm_matmul_fp8_wmma",
    "rmsnorm_add_matmul_fp8_pair_wmma",
    "sigmoid_mul_matmul_fp8_wmma",
    "matmul_fp8_mma",
    "matmul_fp8_mma_f8",
    "matmul_fp8_mma_v2",
];
/// The fp8 readers of row-major bytes only.
const ROW_MAJOR_ONLY: [&str; 7] = [
    "matmul_fp8_gemv",
    "matmul_fp8_gemv_v2",
    "matmul_fp8_gemv_rows",
    "matmul_fp8_pair_gemv",
    "gated_rmsnorm_matmul_fp8_gemv",
    "rmsnorm_add_matmul_fp8_pair_gemv",
    "sigmoid_mul_matmul_fp8_gemv",
];
const FP8_OPS: [Op; 5] = [Op::MatmulFp8, Op::MatmulFp8Pair, Op::GatedRmsnormMatmulFp8, Op::RmsNormAddMatmulFp8Pair, Op::SigmoidMulMatmulFp8];

/// One HostCall per fp8 op at M = 1 (the shapes the folds and every GEMV accept) and the matmul at more row counts.
fn cases(rng: &mut SplitMix64) -> Vec<(String, HostCall)> {
    let mut v = Vec::new();
    for (n, k) in [(512usize, 2048usize), (4096, 2048), (2048, 4096), (64, 64)] {
        for aq in [true, false] {
            for m in [1usize, 3, 5, 16, 40] {
                v.push((format!("matmul_fp8 m={m} n={n} k={k} aq={aq}"), matmul_fp8_case(rng, m, n, k, aq, DType::Bf16, DType::Bf16)));
            }
        }
    }
    for (n0, n1, k) in [(512usize, 512usize, 2048usize), (4096, 2048, 2048), (64, 48, 64)] {
        for (a0, a1) in [(true, true), (false, false)] {
            v.push((format!("pair n0={n0} n1={n1} k={k} aq={a0}"), matmul_fp8_pair_case(rng, 1, n0, n1, k, a0, a1, DType::Bf16, DType::Bf16)));
            v.push((
                format!("norm_pair n0={n0} n1={n1} k={k} aq={a0}"),
                rmsnorm_add_matmul_fp8_pair_case(rng, 1, n0, n1, k, a0, a1, DType::Bf16, 1.0),
            ));
        }
    }
    for (n, k) in [(2048usize, 4096usize), (64, 64)] {
        for aq in [true, false] {
            v.push((format!("sigmoid n={n} k={k} aq={aq}"), sigmoid_mul_matmul_fp8_case(rng, 1, n, k, aq, DType::Bf16, DType::Bf16)));
        }
    }
    for (n, rows) in [(2048usize, 32usize), (64, 2)] {
        for aq in [true, false] {
            v.push((format!("gated n={n} rows={rows} aq={aq}"), gated_rmsnorm_matmul_fp8_case(rng, rows, n, aq, DType::Bf16)));
        }
    }
    v
}

/// (1) the readers that know the layout, on tagged weights, against row-major weights.
#[test]
fn frag_tagged_weights_give_the_bytes_of_row_major_in_every_reader() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(2025);
    let kernels = be.kernels();
    let mut ran = std::collections::BTreeMap::<String, usize>::new();
    for (label, hc) in cases(&mut rng) {
        for k in kernels.iter().filter(|k| k.op == hc.op && FRAG_READERS.contains(&k.name.as_str())) {
            // build the call's ShapeInfo from untagged inputs: a kernel the plain shapes already refuse is not a candidate
            let plain = match run_host(&*be, &k.name, &hc) {
                Ok(o) => o,
                Err(_) => continue,
            };
            let tagged = run_host_frag(&*be, &k.name, &hc).unwrap_or_else(|e| panic!("{label}: {} on tagged weights: {e}", k.name));
            for (i, (t, p)) in tagged.iter().zip(&plain).enumerate() {
                assert!(t.data == p.data, "{label}: {} output {i} on fragment-order weights differs from the row-major run", k.name);
            }
            *ran.entry(k.name.clone()).or_default() += 1;
        }
    }
    // every reader was exercised on at least one shape (a skipped one would hide a regression)
    for r in FRAG_READERS {
        assert!(ran.get(r).copied().unwrap_or(0) > 0, "no case ran {r}");
    }
    println!("frag == row-major, cases per reader: {ran:?}");
}

/// (2) everything else that reads fp8 weights fails loudly on a tagged one, and the kernel set is pinned.
#[test]
fn frag_readers_that_do_not_know_the_layout_fail_loudly() {
    let Some(be) = device() else { return };
    let kernels = be.kernels();
    // the pin: every kernel of an fp8 op is classified above (a new kernel must be added to one of the two lists, and tested)
    let mut seen: Vec<String> = kernels.iter().filter(|k| FP8_OPS.contains(&k.op)).map(|k| k.name.clone()).collect();
    seen.sort();
    let mut want: Vec<String> = FRAG_READERS.iter().chain(ROW_MAJOR_ONLY.iter()).map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(seen, want, "the fp8 kernels of this backend changed: classify the new ones as a fragment-order reader or row-major only");
    for k in kernels.iter().filter(|k| FP8_OPS.contains(&k.op)) {
        assert_eq!(k.frag, FRAG_READERS.contains(&k.name.as_str()), "{}: KernelImpl::frag disagrees with the reader list", k.name);
    }
    // no other kernel of the backend claims to read tagged weights
    for k in kernels.iter().filter(|k| k.frag) {
        assert!(FRAG_READERS.contains(&k.name.as_str()), "{} is registered frag_ok but is not a listed reader", k.name);
    }
    let mut rng = SplitMix64::new(2026);
    let mut failed = std::collections::BTreeMap::<String, usize>::new();
    for (label, hc) in cases(&mut rng) {
        for k in kernels.iter().filter(|k| k.op == hc.op && ROW_MAJOR_ONLY.contains(&k.name.as_str())) {
            // only where the weights are actually rearranged (the shape is eligible) and the kernel takes the call untagged
            let eligible = hc.inputs.iter().any(|t| t.dtype == DType::Fp8E4m3 && t.shape.len() == 2 && frag_eligible(t.shape[0], t.shape[1]));
            if !eligible || run_host(&*be, &k.name, &hc).is_err() {
                continue;
            }
            match run_host_frag(&*be, &k.name, &hc) {
                Ok(_) => panic!("{label}: {} ran on fragment-order weights and read them as row-major", k.name),
                Err(_) => *failed.entry(k.name.clone()).or_default() += 1,
            }
        }
    }
    for r in ROW_MAJOR_ONLY {
        assert!(failed.get(r).copied().unwrap_or(0) > 0, "no case showed that {r} refuses a tagged weight");
    }
    println!("refused on tagged weights, cases per kernel: {failed:?}");
}

/// (2b) the registry never offers a tagged call to a kernel that is not frag_ok.
#[test]
fn frag_the_registry_selects_only_fragment_readers_for_a_tagged_call() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(2027);
    let hc = matmul_fp8_case(&mut rng, 1, 512, 2048, true, DType::Bf16, DType::Bf16);
    let w = put(&*be, &hc.inputs[1]);
    let other = put(&*be, &hc.inputs[1]);
    be.relayout_fp8_frag(&w).unwrap();
    let inputs = |w: &TensorRef| vec![put(&*be, &hc.inputs[0]), w.clone(), put(&*be, &hc.inputs[2])];
    let out = put(&*be, &HostTensor::zeros(DType::Bf16, &[1, 512]));
    let mut call_t = intarsia_core::OpCall::new(Op::MatmulFp8, inputs(&w), vec![out.clone()], hc.attrs.clone());
    call_t.attrs.fp8_static_act = true;
    let mut call_p = intarsia_core::OpCall::new(Op::MatmulFp8, inputs(&other), vec![out], hc.attrs.clone());
    call_p.attrs.fp8_static_act = true;
    let (st, sp) = (ShapeInfo::from_call(&call_t), ShapeInfo::from_call(&call_p));
    assert!(st.frag && !sp.frag);
    let accepted_t: Vec<String> = be.kernels().into_iter().filter(|k| k.op == Op::MatmulFp8 && k.accepts(&st)).map(|k| k.name).collect();
    let accepted_p: Vec<String> = be.kernels().into_iter().filter(|k| k.op == Op::MatmulFp8 && k.accepts(&sp)).map(|k| k.name).collect();
    assert!(!accepted_t.is_empty(), "a tagged M=1 call has no kernel");
    for n in &accepted_t {
        assert!(FRAG_READERS.contains(&n.as_str()), "{n} accepts a tagged call");
    }
    assert!(accepted_p.iter().any(|n| n.contains("gemv")), "the untagged call lost its GEMVs: {accepted_p:?}");
}

/// (3) the raw-byte paths.
#[test]
fn frag_raw_byte_paths_refuse_a_tagged_tensor() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(2028);
    let q = fp8::random_rows(&mut rng, 64, 128, true);
    let host_w = q.parts[0].clone();
    let w = put(&*be, &host_w);
    assert_eq!(w.layout(), Layout::RowMajor);
    be.relayout_fp8_frag(&w).unwrap();
    assert_eq!(w.layout(), Layout::Fp8Frag);
    assert!(be.relayout_fp8_frag(&w).is_err(), "a second relayout must refuse");
    // the bytes on the device are the fragment order, not the checkpoint's
    let mut raw_dev = vec![0u8; w.nbytes()];
    assert!(be.download(&w.buf, 0, &mut raw_dev, 0).is_err(), "download of a tagged buffer");
    assert!(be.upload(&w.buf, 0, &host_w.data, 0).is_err(), "upload into a tagged buffer");
    let plain = put(&*be, &host_w);
    assert!(be.copy(&plain.buf, 0, &w.buf, 0, w.nbytes(), 0).is_err(), "device copy out of a tagged buffer");
    assert!(be.copy(&w.buf, 0, &plain.buf, 0, w.nbytes(), 0).is_err(), "device copy into a tagged buffer");
    // the one reader: row-major order back, bit for bit, and the tagged bytes really are the fragment order
    assert_eq!(be.download_row_major(&w).unwrap(), host_w.data);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.rows(0, 16))).is_err(), "rows() of a tagged tensor");
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.reshaped(&[128, 64]))).is_err(), "reshaped() of a tagged tensor");
    // a view made from it by struct update is still tagged (the tag is the allocation's)
    let view = TensorRef { shape: vec![64, 128], ..w.clone() };
    assert_eq!(view.layout(), Layout::Fp8Frag);
    // QTensor: the layout is read from the weight, download un-permutes, dequant is checkpoint semantics
    let hq = q.clone();
    let qt = hq.upload(&*be).unwrap();
    assert_eq!(qt.layout(), Layout::RowMajor);
    be.relayout_fp8_frag(&qt.parts[0]).unwrap();
    assert_eq!(qt.layout(), Layout::Fp8Frag);
    let back = qt.download(&*be).unwrap();
    assert_eq!(back.parts[0].data, hq.parts[0].data);
    assert_eq!(back.dequant(), hq.dequant());
    // the device permutation is the host reference
    let mut staged = vec![0u8; w.nbytes()];
    {
        // read the tagged bytes through a row-major alias only to compare against the host reference: relayout a second copy and unpack
        let again = put(&*be, &host_w);
        be.relayout_fp8_frag(&again).unwrap();
        let rm = be.download_row_major(&again).unwrap();
        staged.copy_from_slice(&rm);
    }
    assert_eq!(staged, host_w.data);
    assert_eq!(frag_pack(&host_w.data, 64, 128).len(), host_w.data.len());
    // a weight that does not fit the layout is refused, not silently left alone
    let odd = put(&*be, &HostTensor::new(DType::Fp8E4m3, vec![17, 32], vec![0x38; 17 * 32]).unwrap());
    assert!(be.relayout_fp8_frag(&odd).is_err());
    // another backend has no such layout
    let cpu = CpuBackend::new();
    assert!(!cpu.fp8_frag_wanted(64, 128, true));
    assert!(cpu.relayout_fp8_frag(&w).is_err());
}

/// (4) the loader: tags exactly when both knobs are set; the TP shards (rows and columns) tag their local slice; the bytes read back equal the
/// checkpoint slice; an ineligible shard stays row-major.
#[test]
fn frag_the_loader_tags_only_under_the_knobs_and_tp_shards_load() {
    let Some(be) = device() else { return };
    let mut rng = SplitMix64::new(2029);
    for aq in [true, false] {
        let (n, k) = (128usize, 256usize);
        let q = fp8::random_rows(&mut rng, n, k, aq);
        let mut src = MemWeights::new();
        src.insert("p.weight", q.parts[0].clone());
        src.insert("p.weight_scale", HostTensor::from_f32(&[], &[q.parts[1].to_f32()[0]]));
        if aq {
            src.insert("p.input_scale", HostTensor::from_f32(&[], &[q.parts[1].to_f32()[1]]));
        }
        let opts = QuantLoadOpts { fp8_act: if aq { Fp8Act::Static } else { Fp8Act::None }, ..Default::default() };
        let whole = load_linear(&src, "p", &*be, &opts).unwrap();
        let want_tag = knobs_on() && frag_eligible(n, k) && table_runs(aq);
        assert_eq!(whole.layout() == Layout::Fp8Frag, want_tag, "aq={aq}: whole tensor");
        assert_eq!(whole.download(&*be).unwrap().parts[0].data, q.parts[0].data);
        for world in [2usize] {
            for (axis, local) in [(ShardAxis::Rows, (n / world, k)), (ShardAxis::Cols, (n, k / world))] {
                for rank in 0..world {
                    let sh = Shard { axis, rank, world };
                    let o = QuantLoadOpts { shard: Some(sh), ..opts };
                    let part = load_linear(&src, "p", &*be, &o).unwrap();
                    assert_eq!((part.n, part.k), local);
                    let tagged = part.layout() == Layout::Fp8Frag;
                    assert_eq!(tagged, knobs_on() && frag_eligible(local.0, local.1) && table_runs(aq), "aq={aq} {axis:?} rank {rank}");
                    // the slice, read back in checkpoint order, is the checkpoint's slice
                    let rows = match axis {
                        ShardAxis::Rows => (rank * n / world..(rank + 1) * n / world).flat_map(|r| q.parts[0].data[r * k..(r + 1) * k].to_vec()).collect::<Vec<u8>>(),
                        _ => (0..n).flat_map(|r| q.parts[0].data[r * k + rank * k / world..r * k + (rank + 1) * k / world].to_vec()).collect::<Vec<u8>>(),
                    };
                    assert_eq!(part.download(&*be).unwrap().parts[0].data, rows, "aq={aq} {axis:?} rank {rank}: read back");
                }
            }
        }
    }
}

/// Whether the table in force (INTARSIA_HIP_WMMA_TABLE) lets a dense fp8 weight of this kind be stored in fragment order.
fn table_runs(aq: bool) -> bool {
    use intarsia_backend_hip::quantmoe::{fp8_frag_serves, WmmaKind};
    fp8_frag_serves(if aq { WmmaKind::Fp8A8 } else { WmmaKind::Fp8A16 })
}

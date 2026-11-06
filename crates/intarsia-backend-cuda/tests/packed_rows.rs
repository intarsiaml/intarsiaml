//! Lane pf (docs/results/2026-11-05-cuda-pack-flip.md): the CUDA twin of the HIP `packed_rows` test. A packed prefill pass
//! (`Model::forward_packed`) runs every op the backend does not split per prompt (`Backend::pack_split_gemms`: only the cuBLAS
//! matmul family is split) ONCE over the concatenated rows of several prompts, so the registry picks the kernel by the packed
//! total, while the prompt's solo prefill picks it by its own row count. A segment keeps its solo bits only if both picks give
//! its rows the same bits.
//!
//! For the 35B's row-wise prefill ops that pick by row count (the grouped MoE experts, the dense NVFP4 GEMM of the shared expert,
//! the MoE top-k), each case runs the call the registry selects for the packed total and, per segment, the call the registry
//! selects for that segment alone, and compares the segment's rows byte for byte. Segments of 15..40 rows and around the other
//! kernel switches (255 / 256, 767 / 768), packed beside partners that move the total into each regime (< 256, 256..767, >= 768).
//! Every case is reported; the assertion is that every segment of at least `Backend::pack_min_rows` rows is identical.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::exec::{HostCall, OutSpec};
use intarsia_core::quant::cases::{experts_case, matmul_nvfp4_case, topk_case};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, OpCall, Registry, ShapeInfo};
use std::sync::Arc;

fn device(test: &str) -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip(test, &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn put(be: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    be.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

/// The inputs of one case on the device, and which of them are row-wise (leading dimension = the rows).
struct Case {
    hc: HostCall,
    dev: Vec<TensorRef>,
    row_inputs: Vec<usize>,
}

impl Case {
    fn new(be: &dyn Backend, hc: HostCall, row_inputs: &[usize]) -> Case {
        let dev = hc.inputs.iter().map(|t| put(be, t)).collect();
        Case { hc, dev, row_inputs: row_inputs.to_vec() }
    }

    /// Run the call over rows `lo..lo + n` with the kernel the registry selects for `n` rows; returns that kernel and the outputs.
    fn run(&self, be: &dyn Backend, reg: &Registry, lo: usize, n: usize) -> (String, Vec<Vec<u8>>) {
        let inputs: Vec<TensorRef> =
            self.dev.iter().enumerate().map(|(i, t)| if self.row_inputs.contains(&i) { t.rows(lo, n) } else { t.clone() }).collect();
        let outputs: Vec<TensorRef> = self
            .hc
            .outputs
            .iter()
            .map(|o: &OutSpec| {
                let mut shape = o.shape.clone();
                shape[0] = n;
                let t = HostTensor::zeros(o.dtype, &shape);
                put(be, &t)
            })
            .collect();
        let mut call = OpCall::new(self.hc.op, inputs, outputs.clone(), self.hc.attrs.clone());
        call.check().unwrap();
        let k = reg.select(&call).unwrap_or_else(|e| panic!("{:?} at {n} rows: no kernel ({e})", self.hc.op)).clone();
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            let buf = be.alloc(need.max(16), MemKind::Device).unwrap();
            call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
        }
        be.launch(&k.name, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let outs = outputs
            .iter()
            .map(|o| {
                let mut data = vec![0u8; o.nbytes()];
                be.download(&o.buf, o.offset, &mut data, 0).unwrap();
                data
            })
            .collect();
        (k.name.clone(), outs)
    }
}

/// Segment row counts under test: around the CUDA kernel switches (MoE mma at 32, NVFP4 GEMM at 33, the register kernels at 256,
/// MoE w4a16 at 768) and HIP's 16.
const SEGS: [usize; 16] = [15, 16, 17, 24, 31, 32, 33, 34, 40, 64, 255, 256, 257, 767, 768, 1000];
/// Partners packed after the segment: the packed total lands below 256, in 256..767 and at 768 or more.
const PARTNERS: [usize; 3] = [16, 300, 1100];

/// Every (segment, partner) packing of `case`: the segment's rows of the packed call against the segment alone. Returns the
/// report lines and the segments (>= `min` rows) that differ.
fn sweep(be: &dyn Backend, reg: &Registry, case: &Case, label: &str, min: usize) -> (Vec<String>, Vec<String>) {
    let total_rows = case.hc.inputs[0].shape[0];
    let (mut lines, mut bad) = (Vec::new(), Vec::new());
    for &seg in &SEGS {
        for &p in &PARTNERS {
            let total = seg + p;
            assert!(total <= total_rows);
            // the packed call over rows 0..total; the segment first (rows 0..seg) or after its partner (rows p..total)
            let (k_pack, packed) = case.run(be, reg, 0, total);
            for lo in [0, p] {
                let (k_solo, solo) = case.run(be, reg, lo, seg);
                let diff_rows = (0..seg)
                    .filter(|&i| {
                        packed.iter().zip(&solo).any(|(pk, so)| {
                            let r = pk.len() / total;
                            pk[(lo + i) * r..(lo + i + 1) * r] != so[i * r..(i + 1) * r]
                        })
                    })
                    .count();
                let line = format!(
                    "{label}: segment {seg:>4} at row {lo:>4} of {total:>4}: solo {k_solo} / packed {k_pack}: {}",
                    if diff_rows == 0 { "identical".to_string() } else { format!("DIFFERS ({diff_rows}/{seg} rows)") }
                );
                eprintln!("PACKROWS {line}");
                if diff_rows > 0 && seg >= min {
                    bad.push(line.clone());
                }
                lines.push(line);
            }
        }
    }
    (lines, bad)
}

fn registry(be: &dyn Backend) -> Registry {
    Registry::new("cuda", &be.info(), be.kernels())
}

const MAX_ROWS: usize = 1000 + 1100;

#[test]
fn moe_experts_segments_keep_their_solo_bits_from_pack_min_rows() {
    let Some(be) = device("moe_experts_segments_keep_their_solo_bits_from_pack_min_rows") else { return };
    let reg = registry(&*be);
    let min = be.pack_min_rows();
    let mut rng = SplitMix64::new(91);
    // the 35B: 256 experts, top-8, hidden 2048, expert width 512 (inputs 0..=2: activations, ids, weights are row-wise)
    let case = Case::new(&*be, experts_case(&mut rng, MAX_ROWS, 256, 8, 2048, 512, DType::Bf16), &[0, 1, 2]);
    let (_, bad) = sweep(&*be, &reg, &case, "moe_experts 35B", min);
    assert!(bad.is_empty(), "segments of >= {min} rows that lose their solo bits packed:\n{}", bad.join("\n"));
}

#[test]
fn nvfp4_matmul_segments_keep_their_solo_bits_from_pack_min_rows() {
    let Some(be) = device("nvfp4_matmul_segments_keep_their_solo_bits_from_pack_min_rows") else { return };
    let reg = registry(&*be);
    let min = be.pack_min_rows();
    let mut rng = SplitMix64::new(92);
    let mut all_bad = Vec::new();
    // the shared expert: gate / up (N 512, K 2048) and down (N 2048, K 512)
    for (n, k) in [(512usize, 2048usize), (2048, 512)] {
        let case = Case::new(&*be, matmul_nvfp4_case(&mut rng, MAX_ROWS, n, k, DType::Bf16, DType::Bf16), &[0]);
        let (_, bad) = sweep(&*be, &reg, &case, &format!("matmul_nvfp4 N{n} K{k}"), min);
        all_bad.extend(bad);
    }
    assert!(all_bad.is_empty(), "segments of >= {min} rows that lose their solo bits packed:\n{}", all_bad.join("\n"));
}

#[test]
fn moe_topk_segments_keep_their_solo_bits_from_pack_min_rows() {
    let Some(be) = device("moe_topk_segments_keep_their_solo_bits_from_pack_min_rows") else { return };
    let reg = registry(&*be);
    let min = be.pack_min_rows();
    let mut rng = SplitMix64::new(93);
    let case = Case::new(&*be, topk_case(&mut rng, MAX_ROWS, 256, 8, false, DType::Bf16, true), &[0]);
    let (_, bad) = sweep(&*be, &reg, &case, "moe_topk", min);
    assert!(bad.is_empty(), "segments of >= {min} rows that lose their solo bits packed:\n{}", bad.join("\n"));
}

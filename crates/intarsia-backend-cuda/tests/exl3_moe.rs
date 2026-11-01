//! cuda `moe_experts_exl3` (kernels/cuda/exl3_moe*.cu, `Op::MoeExperts` in its EXL3 encoding): the scratch formula against
//! the device layout, accuracy against the cpu reference over the op's EXL3 sweep, what the two fp16 roundings account
//! for, a row's bits against its batch and path (decode / prompt), run-to-run bits; and, with the Mia DeepSeek-V4.1 EXL3
//! checkpoint (`INTARSIA_MIA_DIR=<snapshot dir>`, `--ignored`), real experts against the cpu reference whole and at TP=2,
//! and the decode / prefill microbenchmark (`bench_dsv41_moe`) against separate `matmul_exl3_gemv` launches.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_backend_cuda::exl3::{launch::moe_scratch_bytes_device, moe_scratch_bytes};
use intarsia_core::admission::{judge, sweep, SweepConfig};
use intarsia_core::backend::{upload_new, Backend, MemKind, TensorRef};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::ops::OpAttrs;
use intarsia_core::quant::exl3::{self, Codebook, HAD};
use intarsia_core::quant::exl3_moe::{self, Mat, View};
use intarsia_core::rng::SplitMix64;
use intarsia_core::weights::MemWeights;
use intarsia_core::{DType, HostTensor, Op, OpCall, ShapeInfo};
use std::sync::Arc;
use std::time::Instant;

const KERNEL: &str = "moe_experts_exl3";
const REF: &str = "ref_moe_experts";

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("exl3_moe", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

/// (rel_l2, max_abs scaled by max(1, max|ref|)), the admission harness's measures.
fn measure(got: &[f32], want: &[f64]) -> (f64, f64) {
    let (mut num, mut den, mut ma, mut rmax) = (0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - w).abs();
        num += d * d;
        den += w * w;
        ma = ma.max(d);
        rmax = rmax.max(w.abs());
    }
    ((num / den.max(1e-30)).sqrt(), ma / rmax.max(1.0))
}

fn exl3_cases(quick: bool) -> Vec<intarsia_core::admission::Case> {
    sweep(Op::MoeExperts, &SweepConfig { quick, ..SweepConfig::default() }).into_iter().filter(|c| c.label.starts_with("exl3")).collect()
}

#[test]
fn scratch_formula_mirrors_the_device_layout() {
    for (t, h, i, e, k) in [(1, 5120, 1152, 385, 7), (7, 256, 128, 8, 2), (63, 5120, 1152, 385, 7), (64, 5120, 1152, 385, 7), (600, 384, 256, 12, 6), (4096, 5120, 1152, 385, 7)] {
        let s = ShapeInfo { m: t, k: h, n: i, experts: e, top_k: k, ..ShapeInfo::default() };
        assert_eq!(moe_scratch_bytes(&s), moe_scratch_bytes_device(t, h, i, e, k), "T={t} H={h} I={i} E={e} K={k}");
    }
}

/// Every EXL3 case of the `moe_experts` sweep (quick and full) against `ref_moe_experts`, judged by the kernel's own
/// tolerance. Prints the worst rel_l2 / max_abs per output dtype (the numbers the tolerance constants are set from).
#[test]
fn matches_the_cpu_reference_over_the_sweep() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let kern = cuda.kernels().into_iter().find(|k| k.name == KERNEL).unwrap();
    let mut worst: std::collections::BTreeMap<String, (f64, f64, String)> = Default::default();
    let (mut n, mut failed) = (0, Vec::new());
    for quick in [true, false] {
        for case in exl3_cases(quick) {
            let call_shape = shape_of(&case.call);
            assert!(kern.accepts(&call_shape), "{} not accepted", case.label);
            let want = run_host(&cpu, REF, &case.call).unwrap();
            let got = run_host(&*cuda, KERNEL, &case.call).unwrap();
            let (ok, ma, rl) = judge(&kern, &got, &want);
            let w = worst.entry(format!("{}", got[0].dtype)).or_insert((0.0, 0.0, String::new()));
            if rl > w.0 {
                *w = (rl, w.1, case.label.clone());
            }
            w.1 = w.1.max(ma);
            println!("{}: max_abs {ma:.3e} rel_l2 {rl:.3e}{}", case.label, if ok { "" } else { "  OUTSIDE" });
            if !ok {
                failed.push(format!("{}: max_abs {ma:.3e} rel_l2 {rl:.3e}", case.label));
            }
            n += 1;
        }
    }
    for (dt, (rl, ma, label)) in &worst {
        println!("worst {dt} output: rel_l2 {rl:.3e} (at {label}), max_abs {ma:.3e}");
    }
    println!("{n} cases, {} outside the kernel's tolerance", failed.len());
    assert!(n >= 7 && failed.is_empty(), "outside the kernel's tolerance:\n{}", failed.join("\n"));
}

fn shape_of(hc: &HostCall) -> ShapeInfo {
    struct P(usize);
    impl intarsia_core::backend::BufferImpl for P {
        fn bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    let mk = |dt: DType, shape: &[usize]| TensorRef::new(intarsia_core::backend::Buffer::new(P(1)), dt, shape);
    let call = OpCall::new(
        hc.op,
        hc.inputs.iter().map(|t| mk(t.dtype, &t.shape)).collect(),
        hc.outputs.iter().map(|o| mk(o.dtype, &o.shape)).collect(),
        hc.attrs.clone(),
    );
    ShapeInfo::from_call(&call)
}

/// The reference with the kernel's two fp16 roundings (the rotated input of gate / up, the rotated activation of down)
/// and nothing else: f64 math otherwise. `[T, H]`.
fn reference_with_fp16_roundings(hc: &HostCall) -> Vec<f64> {
    let inp = &hc.inputs;
    let (t, h) = (inp[0].shape[0], inp[0].shape[1]);
    let k = inp[1].shape[1];
    let e = inp[5].shape[0];
    let i = (inp[4].shape[1] - 3 * h) / 3;
    let v = View::new(e, h, i, &inp[3].data, &inp[4].data, &inp[5].data, &inp[6].data, &inp[7].data).unwrap();
    let x = inp[0].to_f32();
    let (ids, ws) = (inp[1].to_i32().unwrap(), inp[2].to_f32());
    let r16 = |a: Vec<f64>| -> Vec<f64> { a.into_iter().map(|v| half::f16::from_f32(v as f32).to_f64()).collect() };
    let proj = |xh: &[f64], q: &exl3_moe::MatView| -> Vec<f64> {
        (0..q.n / HAD).flat_map(|nb| exl3::forward_block(xh, 1, q.k, q.n, q.trellis, &q.svh, nb, q.k2, v.cb)).collect()
    };
    let l = v.limit as f64;
    let mut out = vec![0f64; t * h];
    for r in 0..t {
        for j in 0..k {
            let ex = ids[r * k + j];
            if ex < 0 || ex as usize >= e {
                continue;
            }
            let (g, u, d) = (v.mat(ex as usize, Mat::Gate).unwrap(), v.mat(ex as usize, Mat::Up).unwrap(), v.mat(ex as usize, Mat::Down).unwrap());
            let xr = &x[r * h..(r + 1) * h];
            let gy = proj(&r16(exl3::rotate_input(xr, 1, h, &g.suh)), &g);
            let uy = proj(&r16(exl3::rotate_input(xr, 1, h, &u.suh)), &u);
            let a: Vec<f32> = gy
                .iter()
                .zip(&uy)
                .map(|(g, u)| {
                    let g = g.min(l);
                    (g / (1.0 + (-g).exp()) * u.max(-l).min(l)) as f32
                })
                .collect();
            let y = proj(&r16(exl3::rotate_input(&a, 1, i, &d.suh)), &d);
            for (o, yv) in out[r * h..(r + 1) * h].iter_mut().zip(y) {
                *o += ws[r * k + j] as f64 * yv;
            }
        }
    }
    out
}

/// Against a reference that rounds where the kernel rounds (and keeps the activation f32), the f32 outputs agree to
/// f32 accumulation noise: the tolerance is the two fp16 roundings, not the kernel's arithmetic.
#[test]
fn only_the_fp16_roundings_separate_it_from_the_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut n = 0;
    for case in exl3_cases(true).into_iter().filter(|c| c.call.outputs[0].dtype == DType::F32) {
        let got = run_host(&*cuda, KERNEL, &case.call).unwrap().remove(0).to_f32();
        let want = run_host(&cpu, REF, &case.call).unwrap().remove(0).to_f32();
        let want: Vec<f64> = want.iter().map(|v| *v as f64).collect();
        let emu = reference_with_fp16_roundings(&case.call);
        let (rl, ma) = measure(&got, &emu);
        let (rl0, ma0) = measure(&got, &want);
        println!("{}: vs fp16-rounding reference rel_l2 {rl:.2e} max_abs {ma:.2e}; vs f64 reference rel_l2 {rl0:.2e} max_abs {ma0:.2e}", case.label);
        assert!(rl < 2e-5 && ma < 2e-5, "{}: {rl:e} {ma:e}", case.label);
        n += 1;
    }
    assert!(n >= 2);
}

fn rows(t: &HostTensor, r0: usize, r1: usize) -> HostTensor {
    let w = t.data.len() / t.shape[0];
    let mut shape = t.shape.clone();
    shape[0] = r1 - r0;
    HostTensor { dtype: t.dtype, shape, data: t.data[r0 * w..r1 * w].to_vec() }
}

fn sub_call(hc: &HostCall, r0: usize, r1: usize) -> HostCall {
    let mut inputs: Vec<HostTensor> = hc.inputs[..3].iter().map(|t| rows(t, r0, r1)).collect();
    inputs.extend(hc.inputs[3..].iter().cloned());
    let mut out = hc.outputs[0].clone();
    out.shape[0] = r1 - r0;
    HostCall { op: hc.op, inputs, outputs: vec![out], attrs: hc.attrs.clone() }
}

/// A row's output bits do not depend on the batch it runs in, nor on the path: T = 600 (two prompt slices) and T = 70
/// (one prompt slice) against T = 40 / T = 1 (the decode kernel), bf16 and f32 outputs, mixed widths with the shared
/// expert's slot. And two runs of the same call give the same bits.
#[test]
fn rows_do_not_depend_on_the_batch_or_the_path() {
    let Some(cuda) = device() else { return };
    let mut rng = SplitMix64::new(5);
    for odt in [DType::Bf16, DType::F32] {
        let (e, k, h, i) = (24usize, 6usize, 384usize, 256usize);
        let hc = exl3_moe::case(&mut rng, 600, e, k, h, i, &|x, m| if x == e - 1 { 10 } else { [6, 4, 6][m as usize] }, Codebook::Mul1, true, DType::Bf16, odt, 10.0);
        let full = run_host(&*cuda, KERNEL, &hc).unwrap().remove(0);
        let again = run_host(&*cuda, KERNEL, &hc).unwrap().remove(0);
        assert_eq!(full.data, again.data, "run to run ({odt})");
        let rb = full.data.len() / 600;
        for (r0, r1) in [(0, 70), (530, 600), (100, 140), (0, 1), (511, 512), (512, 513), (599, 600)] {
            let part = run_host(&*cuda, KERNEL, &sub_call(&hc, r0, r1)).unwrap().remove(0);
            assert_eq!(part.data, full.data[r0 * rb..r1 * rb], "rows {r0}..{r1} ({odt})");
        }
    }
}

// ---- the Mia DeepSeek-V4.1 checkpoint ---------------------------------------------------------------------------------

/// Tensors of a safetensors checkpoint read one at a time (offset reads, never a whole shard).
struct Ckpt {
    dir: std::path::PathBuf,
    index: serde_json::Map<String, serde_json::Value>,
    headers: std::cell::RefCell<std::collections::HashMap<String, (usize, serde_json::Value)>>,
}

impl Ckpt {
    fn open() -> Option<Ckpt> {
        let dir = std::path::PathBuf::from(std::env::var("INTARSIA_MIA_DIR").ok()?);
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).ok()?).ok()?;
        Some(Ckpt { dir, index: idx["weight_map"].as_object()?.clone(), headers: Default::default() })
    }

    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn read(&self, name: &str) -> HostTensor {
        use std::io::{Read, Seek, SeekFrom};
        let file = self.index[name].as_str().unwrap().to_string();
        let path = self.dir.join(&file);
        let mut hs = self.headers.borrow_mut();
        let (base, hdr) = hs.entry(file).or_insert_with(|| {
            let mut f = std::fs::File::open(&path).unwrap();
            let mut n = [0u8; 8];
            f.read_exact(&mut n).unwrap();
            let n = u64::from_le_bytes(n) as usize;
            let mut b = vec![0u8; n];
            f.read_exact(&mut b).unwrap();
            (8 + n, serde_json::from_slice(&b).unwrap())
        });
        let v = &hdr[name];
        let dt = DType::from_safetensors(v["dtype"].as_str().unwrap()).unwrap();
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let (s, e) = (v["data_offsets"][0].as_u64().unwrap() as usize, v["data_offsets"][1].as_u64().unwrap() as usize);
        let mut f = std::fs::File::open(&path).unwrap();
        f.seek(SeekFrom::Start((*base + s) as u64)).unwrap();
        let mut data = vec![0u8; e - s];
        f.read_exact(&mut data).unwrap();
        HostTensor { dtype: dt, shape, data }
    }

    /// The EXL3 tensors of the given expert matrices (prefixes) in a `MemWeights`.
    fn experts(&self, names: &[[String; 3]]) -> MemWeights {
        let mut w = MemWeights::new();
        for p in names.iter().flatten() {
            for part in ["trellis", "suh", "svh", "su", "sv", "mcg", "mul1"] {
                let n = format!("{p}.{part}");
                if self.has(&n) {
                    w.insert(&n, self.read(&n));
                }
            }
        }
        w
    }
}

fn expert_names(layer: usize, ids: &[usize], shared: bool) -> Vec<[String; 3]> {
    let mut v: Vec<[String; 3]> = ids
        .iter()
        .map(|x| {
            let p = format!("layers.{layer}.ffn.experts.{x}");
            [format!("{p}.w1"), format!("{p}.w3"), format!("{p}.w2")]
        })
        .collect();
    if shared {
        let p = format!("layers.{layer}.ffn.shared_experts");
        v.push([format!("{p}.w1"), format!("{p}.w3"), format!("{p}.w2")]);
    }
    v
}

fn host_parts(b: &dyn Backend, ex: &intarsia_core::quant::moe::Experts) -> Vec<HostTensor> {
    [&ex.gate_up_w, &ex.gate_up_s, &ex.down_w, &ex.down_s, &ex.gscales]
        .map(|t| HostTensor::new(t.dtype, t.shape.clone(), b.download_row_major(t).unwrap()).unwrap())
        .to_vec()
}

const H: usize = 5120;
const I: usize = 2304;
const LIMIT: f32 = 10.0;

/// Real experts (layer 3: six routed experts of mixed 2 / 3-bit widths and the 4 / 5-bit shared expert as the seventh slot)
/// through the cuda kernel against the cpu reference, whole (I = 2304) and as TP=2 ranks (I = 1152 each), decode (T = 1,
/// 4) and a prompt slice (T = 96); and rank 0 + rank 1 against the whole layer, on both backends.
#[test]
#[ignore]
fn real_mia_experts_match_the_cpu_reference_and_shard() {
    let Some(cuda) = device() else { return };
    let Some(ck) = Ckpt::open() else {
        println!("SKIPPED: set INTARSIA_MIA_DIR to the Mia DeepSeek-V4.1 EXL3 snapshot");
        return;
    };
    let cpu = CpuBackend::new();
    let kern = cuda.kernels().into_iter().find(|k| k.name == KERNEL).unwrap();
    // routed experts with both widths in layer 3 (2-bit ones exist in some layers: pick from the checkpoint)
    let layer = 3;
    let picks = [0usize, 17, 101, 222, 300, 383];
    let names = expert_names(layer, &picks, true);
    let w = ck.experts(&names);
    let e = names.len();
    for (name, m) in names.iter().zip(0..) {
        let k2s: Vec<usize> = name.iter().map(|p| intarsia_core::weights::WeightSource::info(&w, &format!("{p}.trellis")).unwrap().shape[2] / 8).collect();
        println!("expert slot {m} ({}): k2 gate/up/down {k2s:?}", name[0]);
    }
    let mut rng = SplitMix64::new(17);
    for t in [1usize, 4, 96] {
        let k = 7;
        let x = HostTensor::from_f32_as(DType::Bf16, &[t, H], &rng.normal_vec(t * H, 1.0));
        // every row: six routed slots in a row-dependent order, weights summing to 1.5 (routed_scaling_factor), then shared
        let mut ids = Vec::new();
        let mut wts = Vec::new();
        for r in 0..t {
            let mut order: Vec<i32> = (0..6).collect();
            order.rotate_left(r % 6);
            let raw: Vec<f32> = (0..6).map(|_| 0.1 + rng.next_f32()).collect();
            let s: f32 = raw.iter().sum();
            ids.extend(order);
            ids.push(6);
            wts.extend(raw.iter().map(|v| 1.5 * v / s));
            wts.push(1.0);
        }
        let mut outs: Vec<(String, Vec<f32>, Vec<f32>)> = Vec::new();
        for (rank, world) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let mut parts = Vec::new();
            for b in [&cpu as &dyn Backend, &*cuda] {
                parts.push(host_parts(b, &exl3_moe::load(b, &w, &names, H, I, rank, world, LIMIT).unwrap()));
            }
            assert_eq!(parts[0].iter().map(|p| p.data.clone()).collect::<Vec<_>>(), parts[1].iter().map(|p| p.data.clone()).collect::<Vec<_>>(), "loads differ by backend");
            let mut inputs = vec![x.clone(), HostTensor::from_i32(&[t, k], &ids), HostTensor::from_f32(&[t, k], &wts)];
            inputs.extend(parts.remove(0));
            let hc = HostCall { op: Op::MoeExperts, inputs, outputs: vec![OutSpec::new(DType::F32, &[t, H])], attrs: OpAttrs { top_k: k, ..OpAttrs::default() } };
            let mut want = run_host(&cpu, REF, &hc).unwrap();
            let mut got = run_host(&*cuda, KERNEL, &hc).unwrap();
            let (ok, ma, rl) = judge(&kern, &got, &want);
            println!("T={t} rank {rank}/{world} (E={e}, I={}): cuda vs cpu max_abs {ma:.3e} rel_l2 {rl:.3e}", I / world);
            assert!(ok, "T={t} rank {rank}/{world}: outside tolerance");
            outs.push((format!("{rank}/{world}"), want.remove(0).to_f32(), got.remove(0).to_f32()));
        }
        for (which, idx) in [("cpu", 1usize), ("cuda", 2)] {
            let pick = |o: &(String, Vec<f32>, Vec<f32>)| if idx == 1 { o.1.clone() } else { o.2.clone() };
            let whole: Vec<f64> = pick(&outs[0]).iter().map(|v| *v as f64).collect();
            let sum: Vec<f32> = pick(&outs[1]).iter().zip(pick(&outs[2])).map(|(a, b)| a + b).collect();
            let (rl, ma) = measure(&sum, &whole);
            println!("T={t} {which}: rank 0 + rank 1 vs whole rel_l2 {rl:.3e} max_abs {ma:.3e}");
            assert!(rl < if idx == 1 { 1e-6 } else { 2e-3 }, "{which} shard sum");
        }
    }
}

/// Decode and prefill timings per TP=2 rank on real experts (layer 3; a pool of 64 routed experts and the shared expert,
/// ~0.43 GB per rank, so picks rotate through far more than the L2): (a) 18 / 21 separate `matmul_exl3_gemv` launches
/// (6 routed experts x gate, up, down, + the shared expert's three; no activation) against (b) one `moe_experts_exl3`
/// call (routed only, K = 6; with the shared expert, K = 7), and prompt slices of 512 / 2048 rows. GB/s counts the
/// trellis and scale bytes of every matrix read once.
#[test]
#[ignore]
fn bench_dsv41_moe() {
    let Some(cuda) = device() else { return };
    let Some(ck) = Ckpt::open() else {
        println!("SKIPPED: set INTARSIA_MIA_DIR to the Mia DeepSeek-V4.1 EXL3 snapshot");
        return;
    };
    let pool: usize = std::env::var("BENCH_POOL").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let layer = 3;
    let routed: Vec<usize> = (0..pool).map(|j| j * 384 / pool).collect();
    let names = expert_names(layer, &routed, true);
    let t0 = Instant::now();
    let w = ck.experts(&names);
    println!("read {} matrices of layer {layer} in {:.1} s", names.len() * 3, t0.elapsed().as_secs_f64());
    let (rank, world) = (0usize, 2usize);
    let il = I / world;
    let e = names.len();
    let ex = exl3_moe::load(&*cuda, &w, &names, H, I, rank, world, LIMIT).unwrap();
    let shared = e - 1;
    // the same matrices as separate EXL3 linears (this rank's slices), for the launch-per-matrix baseline
    let opts = |axis| intarsia_core::quant::QuantLoadOpts { shard: Some(intarsia_core::weights::Shard { axis, rank, world }), ..Default::default() };
    let lin: Vec<[intarsia_core::quant::QTensor; 3]> = names
        .iter()
        .map(|n| {
            use intarsia_core::weights::ShardAxis::{Cols, Rows};
            [
                intarsia_core::quant::load_linear(&w, &n[0], &*cuda, &opts(Rows)).unwrap(),
                intarsia_core::quant::load_linear(&w, &n[1], &*cuda, &opts(Rows)).unwrap(),
                intarsia_core::quant::load_linear(&w, &n[2], &*cuda, &opts(Cols)).unwrap(),
            ]
        })
        .collect();
    let mbytes = |x: usize| -> usize { lin[x].iter().map(|q| q.parts.iter().map(|p| p.nbytes()).sum::<usize>()).sum() };
    let routed_bytes: f64 = (0..pool).map(mbytes).sum::<usize>() as f64 / pool as f64; // mean per expert
    let shared_bytes = mbytes(shared) as f64;
    println!("rank {rank}/{world}: H={H} I={il}; mean routed expert {:.2} MB, shared {:.2} MB (trellis + scales)", routed_bytes / 1e6, shared_bytes / 1e6);
    let dev = |t: &HostTensor| TensorRef::new(upload_new(&*cuda, &t.data).unwrap(), t.dtype, &t.shape);
    let sync = || cuda.synchronize(0).unwrap();
    let mut rng = SplitMix64::new(23);
    let iters = 200;

    // (a) separate GEMV launches, T = 1
    let x = dev(&HostTensor::from_f32_as(DType::Bf16, &[1, H], &rng.normal_vec(H, 1.0)));
    let gu_out = dev(&HostTensor::zeros(DType::Bf16, &[1, il]));
    let d_in = dev(&HostTensor::from_f32_as(DType::Bf16, &[1, il], &rng.normal_vec(il, 0.1)));
    let d_out = dev(&HostTensor::zeros(DType::Bf16, &[1, H]));
    let gemv = cuda.kernels().into_iter().find(|k| k.name == "matmul_exl3_gemv").unwrap();
    let mk_call = |xin: &TensorRef, q: &intarsia_core::quant::QTensor, out: &TensorRef| -> OpCall {
        let mut inputs = vec![xin.clone()];
        inputs.extend(q.parts.iter().cloned());
        let mut c = OpCall::new(Op::MatmulExl3, inputs, vec![out.clone()], OpAttrs::default());
        let s = ShapeInfo::from_call(&c);
        let sb = gemv.scratch_bytes(&s);
        c.scratch = Some(TensorRef::new(cuda.alloc(sb.max(256), MemKind::Device).unwrap(), DType::U8, &[sb.max(256)]));
        c
    };
    let picks: Vec<Vec<usize>> = (0..iters)
        .map(|_| {
            let mut p: Vec<usize> = (0..pool).collect();
            for j in 0..6 {
                let r = j + (rng.next_u64() as usize) % (pool - j);
                p.swap(j, r);
            }
            p[..6].to_vec()
        })
        .collect();
    let baseline_calls: Vec<Vec<OpCall>> = picks
        .iter()
        .map(|p| {
            let mut v = Vec::new();
            for &x_ in p.iter().chain(std::iter::once(&shared)) {
                v.push(mk_call(&x, &lin[x_][0], &gu_out));
                v.push(mk_call(&x, &lin[x_][1], &gu_out));
                v.push(mk_call(&d_in, &lin[x_][2], &d_out));
            }
            v
        })
        .collect();
    let time = |f: &dyn Fn(usize)| -> f64 {
        for it in 0..10 {
            f(it);
        }
        sync();
        let t0 = Instant::now();
        for it in 0..iters {
            f(it);
        }
        sync();
        t0.elapsed().as_secs_f64() / iters as f64
    };
    for with_shared in [false, true] {
        let n = if with_shared { 21 } else { 18 };
        let tb = time(&|it| {
            for c in &baseline_calls[it][..n] {
                cuda.launch("matmul_exl3_gemv", c, 0).unwrap();
            }
        });
        let bytes = 6.0 * routed_bytes + if with_shared { shared_bytes } else { 0.0 };
        println!("decode T=1 {n} separate matmul_exl3_gemv: {:.1} us/token, {:.0} GB/s", tb * 1e6, bytes / tb / 1e9);
    }

    // (b) the grouped kernel
    let moe_call = |t: usize, k: usize, ids: &[i32], wts: &[f32], xin: &TensorRef, out: &TensorRef| -> OpCall {
        let mut inputs = vec![xin.clone(), dev(&HostTensor::from_i32(&[t, k], ids)), dev(&HostTensor::from_f32(&[t, k], wts))];
        inputs.extend([&ex.gate_up_w, &ex.gate_up_s, &ex.down_w, &ex.down_s, &ex.gscales].map(|t| t.clone()));
        let mut c = OpCall::new(Op::MoeExperts, inputs, vec![out.clone()], OpAttrs { top_k: k, ..OpAttrs::default() });
        let sb = moe_scratch_bytes(&ShapeInfo::from_call(&c));
        c.scratch = Some(TensorRef::new(cuda.alloc(sb.max(256), MemKind::Device).unwrap(), DType::U8, &[sb.max(256)]));
        c
    };
    let out1 = dev(&HostTensor::zeros(DType::F32, &[1, H]));
    for with_shared in [false, true] {
        let k = if with_shared { 7 } else { 6 };
        let calls: Vec<OpCall> = picks
            .iter()
            .map(|p| {
                let mut ids: Vec<i32> = p.iter().map(|v| *v as i32).collect();
                let mut wts = vec![0.25f32; 6];
                if with_shared {
                    ids.push(shared as i32);
                    wts.push(1.0);
                }
                moe_call(1, k, &ids, &wts, &x, &out1)
            })
            .collect();
        let tg = time(&|it| cuda.launch(KERNEL, &calls[it], 0).unwrap());
        let bytes = 6.0 * routed_bytes + if with_shared { shared_bytes } else { 0.0 };
        println!("decode T=1 moe_experts_exl3 K={k}: {:.1} us/token, {:.0} GB/s", tg * 1e6, bytes / tg / 1e9);
    }
    // a verify window: T = 4, 6 routed + shared
    for t in [4usize, 8] {
        let xs = dev(&HostTensor::from_f32_as(DType::Bf16, &[t, H], &rng.normal_vec(t * H, 1.0)));
        let outs = dev(&HostTensor::zeros(DType::F32, &[t, H]));
        let calls: Vec<OpCall> = (0..20)
            .map(|c| {
                let mut ids = Vec::new();
                for r in 0..t {
                    ids.extend(picks[(c * t + r) % iters].iter().map(|v| *v as i32));
                    ids.push(shared as i32);
                }
                moe_call(t, 7, &ids, &vec![0.25f32; t * 7], &xs, &outs)
            })
            .collect();
        let tg = time(&|it| cuda.launch(KERNEL, &calls[it % 20], 0).unwrap());
        println!("decode T={t} moe_experts_exl3 K=7: {:.1} us/call, {:.1} us/token", tg * 1e6, tg * 1e6 / t as f64);
    }
    // prompt slices: uniform routing over the pool's routed experts, + shared
    for t in [512usize, 2048] {
        let xs = dev(&HostTensor::from_f32_as(DType::Bf16, &[t, H], &rng.normal_vec(t * H, 1.0)));
        let outs = dev(&HostTensor::zeros(DType::Bf16, &[t, H]));
        let mut ids = Vec::new();
        for _ in 0..t {
            let mut p: Vec<i32> = (0..pool as i32).collect();
            for j in 0..6 {
                let r = j + (rng.next_u64() as usize) % (pool - j);
                p.swap(j, r);
            }
            ids.extend_from_slice(&p[..6]);
            ids.push(shared as i32);
        }
        let c = moe_call(t, 7, &ids, &vec![0.25f32; t * 7], &xs, &outs);
        let reps = 5;
        cuda.launch(KERNEL, &c, 0).unwrap();
        sync();
        let t0 = Instant::now();
        for _ in 0..reps {
            cuda.launch(KERNEL, &c, 0).unwrap();
        }
        sync();
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let rows_per_expert = t as f64 * 6.0 / pool as f64;
        let bytes = (pool as f64 * routed_bytes + shared_bytes) * (t as f64 / 512.0).ceil();
        println!(
            "prefill T={t} (pool {pool}: {rows_per_expert:.0} rows per routed expert) moe_experts_exl3 K=7: {:.2} ms, {:.1} us/token, weights {:.0} GB/s",
            dt * 1e3,
            dt * 1e6 / t as f64,
            bytes / dt / 1e9
        );
    }
}

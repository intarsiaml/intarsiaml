//! The DeepSeek-V4.1 runtime (`intarsia_core::dsv41::runtime`) end to end on a few-layer rank cache against the official
//! code's layer outputs (`golden/dsv41` generator `layers_bf16.safetensors`: 16 prefill rows, then two decode steps, layers
//! 0-2 of the Mia EXL3 checkpoint). Unlike `dsv41_layers.rs` nothing is taken from the fixture except the token ids and the
//! Engram rows (served by a tier keyed by the fixture's hash ids, so the runtime's own hashing must ask for exactly those
//! rows): the routed and shared experts run through the fused `MoeExperts` kernel, every layer reads the previous one's
//! output, the state carries from the prefill into the decode steps.
//!
//! Gated on files outside the repository: `INTARSIA_DSV41_CACHE` = a `--layers 3` rank cache of world 1
//! (`intarsia dsv41-cache --layers 3 --rank 0 --world 1`), `INTARSIA_DSV41_FIXTURES` = directory with `layers_bf16.safetensors`.
//! `INTARSIA_DSV41_BACKEND=cuda` runs it on the CUDA backend (a Spark) instead of the cpu reference.
#[path = "../../intarsia-backend-cpu/tests/dsv41_common/mod.rs"]
mod dsv41_common;

use dsv41_common::*;
use intarsia_core::dsv41::engram::{record_bytes, EngramLayout, EngramShard, EngramTicket, EngramTier};
use intarsia_core::dsv41::graph::Buf;
use intarsia_core::dsv41::runtime::{Dsv41Runtime, EngramInputs};
use intarsia_core::{Backend, DType, Error, Result};
use intarsia_models::dsv41_engine::{load_runtime_with, Dsv41Opts};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Re-encode decoded Engram rows (exact in bf16) as packed records: per 32-block the smallest-exponent E8M0 scale whose
/// quotients are all exact e4m3 values (any such scale decodes to the same row).
fn encode_row(vals: &[f32]) -> Vec<u8> {
    let dh = vals.len();
    let mut table: HashMap<u32, u8> = HashMap::new();
    for b in 0u8..=255 {
        let f = intarsia_core::dtype::fp8_e4m3_to_f32(b);
        if f.is_finite() {
            table.entry(f.to_bits()).or_insert(b);
        }
    }
    let mut payload = vec![0u8; dh];
    let mut scales = vec![0u8; dh / 32];
    for blk in 0..dh / 32 {
        let v = &vals[blk * 32..(blk + 1) * 32];
        let mut done = false;
        for s in 0u8..=254 {
            let m = (2f64).powi(s as i32 - 127);
            let codes: Option<Vec<u8>> = v.iter().map(|&x| table.get(&((x as f64 / m) as f32).to_bits()).copied().filter(|_| ((x as f64 / m) as f32) as f64 * m == x as f64)).collect();
            if let Some(c) = codes {
                payload[blk * 32..(blk + 1) * 32].copy_from_slice(&c);
                scales[blk] = s;
                done = true;
                break;
            }
        }
        assert!(done, "engram row block {blk} has no exact e4m3 + E8M0 encoding");
    }
    payload.extend(scales);
    payload
}

/// Serves the fixture's Engram rows by row id; a row it does not know is an error (the runtime hashed wrong).
struct FixtureTier {
    layout: EngramLayout,
    rows: HashMap<u32, Vec<u8>>,
    /// rows of layer 1 this rank owns (others come back as zero payload with unit scales)
    owned: std::ops::Range<u64>,
    rank: usize,
    world: usize,
    next: AtomicU64,
    pending: Mutex<HashMap<u64, Vec<u32>>>,
}

impl EngramTier for FixtureTier {
    fn layout(&self) -> &EngramLayout {
        &self.layout
    }
    fn shard(&self) -> EngramShard {
        EngramShard { rank: self.rank, world: self.world }
    }
    fn owned_rows(&self, _layer: usize) -> std::ops::Range<u64> {
        self.owned.clone()
    }
    fn issue(&self, layer: usize, rows: &[u32]) -> Result<EngramTicket> {
        if layer != 0 {
            return Err(Error::invalid("fixture tier holds layer 1 only"));
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.pending.lock().unwrap().insert(id, rows.to_vec());
        Ok(EngramTicket(id))
    }
    fn wait(&self, t: EngramTicket, out: &mut [u8]) -> Result<()> {
        let rows = self.pending.lock().unwrap().remove(&t.0).unwrap();
        let rb = record_bytes(256);
        for (i, r) in rows.iter().enumerate() {
            if !self.owned.contains(&(*r as u64)) {
                out[i * rb..i * rb + 256].fill(0);
                out[i * rb + 256..(i + 1) * rb].fill(127);
                continue;
            }
            let rec = self.rows.get(r).ok_or_else(|| Error::invalid(format!("engram row {r} is not one the official hash asks for")))?;
            out[i * rb..(i + 1) * rb].copy_from_slice(rec);
        }
        Ok(())
    }
}

fn err(got: &[f32], want: &[f32]) -> (f64, f64) {
    let scale = want.iter().fold(0f64, |m, &x| m.max(x.abs() as f64));
    let max_abs = got.iter().zip(want).fold(0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
    let rms = (got.iter().zip(want).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    let rrms = (want.iter().map(|&b| (b as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    (max_abs / scale.max(1e-30), rms / rrms.max(1e-30))
}

/// The residual the head reads (the last layer's output), `[t, 4, dim]` as f32.
fn residual(rt: &Dsv41Runtime, t: usize, pos0: usize) -> Vec<f32> {
    let g = rt.graph(t, pos0).unwrap();
    let head = g.stages.last().unwrap();
    let Buf::Ws(name) = &head.steps[0].inputs[0].buf else { panic!("head input is not a workspace") };
    let n = t * 4 * rt.cfg.dim;
    if rt.opts.act == DType::F32 {
        let b = rt.read_ws(name, DType::F32, n).unwrap();
        return b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    }
    let b = rt.read_ws(name, DType::Bf16, n).unwrap();
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

fn fixture_engram(fx: &St, rank: usize, world: usize) -> EngramInputs {
    let hash = intarsia_engram::HashLayout::dsv41();
    let mut rows = HashMap::new();
    for s in ["s0", "s1", "s2"] {
        let ids = fx.get(&format!("bf16/L1/{s}/engram.hash_ids")).i32();
        let vals = fx.get(&format!("bf16/L1/{s}/engram.rows")).f32();
        for (i, &id) in ids.iter().enumerate() {
            rows.insert(id as u32, encode_row(&vals[i * 256..(i + 1) * 256]));
        }
    }
    let tm: Vec<u32> = fx.get("token_map").i32().into_iter().map(|v| v as u32).collect();
    let owned = hash.rank_rows(0, rank, world);
    let tier = FixtureTier { layout: hash.layout().clone(), rows, owned, rank, world, next: AtomicU64::new(1), pending: Mutex::new(HashMap::new()) };
    EngramInputs { tier: Arc::new(tier), hasher: hash.hasher.clone(), token_map: tm }
}

/// Prefill 16 then decode 2, per truncation depth `layers`; returns (layer, step, max_abs/max|ref|, rel_rms).
fn run_steps(rt: &mut Dsv41Runtime, fx: &St, read: bool) -> Vec<(usize, &'static str, f64, f64)> {
    run_steps_flow(rt, fx, fx, "bf16", read)
}

/// `fxf`: the fixture of the compared flow (`flow`/L../out.h); `fx`: the bf16 fixture (ids, routes).
fn run_steps_flow(rt: &mut Dsv41Runtime, fx: &St, fxf: &St, flow: &str, read: bool) -> Vec<(usize, &'static str, f64, f64)> {
    let ids: Vec<u32> = fx.get("ids").i32().into_iter().map(|v| v as u32).collect();
    let mut out = vec![];
    for l in 0..3 {
        rt.opts.tp.layers = Some(l + 1);
        let mut st = rt.new_state().unwrap();
        for (step, (a, b)) in [("s0", (0, 16)), ("s1", (16, 17)), ("s2", (17, 18))] {
            let pos0 = st.len;
            rt.forward(&mut st, &ids[a..b], true).unwrap();
            if read && b - a == 1 && std::env::var_os("INTARSIA_DSV41_ROUTES").is_some() {
                // the last layer's routing of this decode step against the official one (a near-tie flip explains a jump)
                let r = rt.read_ws("route_ids7", DType::I32, 7).unwrap();
                let mut got: Vec<i32> = r.chunks_exact(4).take(6).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
                got.sort();
                let k = format!("bf16/L{l}/{step}/ffn.gate_idx");
                let mut want = if fx.has(&k) { fx.get(&k).i32() } else { vec![] };
                want.sort();
                println!("ROUTE L{l} {step}: ours {got:?} official {want:?}");
            }
            if read {
                let got = residual(rt, b - a, pos0);
                let want = fxf.get(&format!("{flow}/L{l}/{step}/out.h")).f32();
                let (mx, rr) = err(&got, &want);
                out.push((l, step, mx, rr));
            }
        }
    }
    out
}

/// Two in-process ranks (host communicator) on the two `--layers 3` rank caches of world 2: the sharded weights, the
/// collectives and the column-split Engram rows must reproduce the official layer outputs like one rank does.
/// `INTARSIA_DSV41_CACHE_R0` / `_R1` = the rank caches (`intarsia dsv41-cache --layers 3 --rank {0,1} --world 2`).
#[test]
fn few_layer_runtime_two_ranks_match_the_official_layer_outputs() {
    let (Some(c0), Some(c1), Some(fxd)) = (
        std::env::var_os("INTARSIA_DSV41_CACHE_R0").map(PathBuf::from),
        std::env::var_os("INTARSIA_DSV41_CACHE_R1").map(PathBuf::from),
        std::env::var_os("INTARSIA_DSV41_FIXTURES").map(PathBuf::from),
    ) else {
        println!("skipped: set INTARSIA_DSV41_CACHE_R0, INTARSIA_DSV41_CACHE_R1 and INTARSIA_DSV41_FIXTURES");
        return;
    };
    // bf16 (bound as for one rank) and fp32 (the sharding and the collectives must not change the result beyond rounding)
    for (flow, act, bound) in [("fp32", DType::F32, 1e-4), ("bf16", DType::Bf16, 3e-2)] {
        let backends: Vec<Arc<dyn Backend>> = vec![intarsia_backend_cpu::CpuBackend::arc(), intarsia_backend_cpu::CpuBackend::arc()];
        let comms = intarsia_core::comm::HostCommGroup::new(backends.clone());
        let hs: Vec<_> = [c0.clone(), c1.clone()]
            .into_iter()
            .enumerate()
            .map(|(r, cache)| {
                let (b, c, fxd) = (backends[r].clone(), comms[r].clone(), fxd.clone());
                std::thread::spawn(move || {
                    let fx = St::open(&fxd.join("layers_bf16.safetensors"));
                    let fxf = St::open(&fxd.join(format!("layers_{flow}.safetensors")));
                    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act };
                    let (mut rt, m) = load_runtime_with(&cache, b, Some(c), &o, Some(fixture_engram(&fx, r, 2))).unwrap();
                    assert_eq!((m.rank, m.world, m.layers), (r, 2, Some(3)));
                    run_steps_flow(&mut rt, &fx, &fxf, flow, r == 0)
                })
            })
            .collect();
        let res: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        let mut worst = 0f64;
        for (l, step, mx, rr) in &res[0] {
            println!("RUNTIME TP2 {flow} L{l} {step}: max_abs/max|ref| {mx:.3e}  rel_rms {rr:.3e}");
            worst = worst.max(*mx);
        }
        assert!(worst < bound, "{flow}: worst layer output error {worst:.3e}");
    }
}

#[test]
fn few_layer_runtime_fp32_matches_the_official_fp32_layer_outputs() {
    let (Some(cache), Some(fxd)) = (std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from), std::env::var_os("INTARSIA_DSV41_FIXTURES").map(PathBuf::from)) else {
        println!("skipped: set INTARSIA_DSV41_CACHE and INTARSIA_DSV41_FIXTURES");
        return;
    };
    let fx = St::open(&fxd.join("layers_bf16.safetensors"));
    let fx32 = St::open(&fxd.join("layers_fp32.safetensors"));
    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act: DType::F32 };
    // cpu only: the CUDA backend's shared kernels (embedding, norms) take bf16 activations only
    if std::env::var("INTARSIA_DSV41_BACKEND").as_deref() == Ok("cuda") {
        println!("skipped: the fp32 flow runs on the cpu reference only (CUDA embedding / norms are bf16-only)");
        return;
    }
    let (backend, bound): (Arc<dyn Backend>, f64) = (intarsia_backend_cpu::CpuBackend::arc(), 1e-4);
    let (mut rt, _) = load_runtime_with(&cache, backend, None, &o, Some(fixture_engram(&fx, 0, 1))).unwrap();
    let mut worst = 0f64;
    for (l, step, mx, rr) in run_steps_flow(&mut rt, &fx, &fx32, "fp32", true) {
        println!("RUNTIME FP32 L{l} {step}: max_abs/max|ref| {mx:.3e}  rel_rms {rr:.3e}");
        worst = worst.max(mx);
    }
    assert!(worst < bound, "worst fp32 layer output error {worst:.3e}");
}

#[test]
fn few_layer_runtime_matches_the_official_layer_outputs() {
    let (Some(cache), Some(fx)) = (std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from), std::env::var_os("INTARSIA_DSV41_FIXTURES").map(PathBuf::from)) else {
        println!("skipped: set INTARSIA_DSV41_CACHE and INTARSIA_DSV41_FIXTURES (see the file header)");
        return;
    };
    let fx = St::open(&fx.join("layers_bf16.safetensors"));
    let backend: Arc<dyn Backend> = match std::env::var("INTARSIA_DSV41_BACKEND").as_deref() {
        Ok("cuda") => cuda_backend(),
        _ => intarsia_backend_cpu::CpuBackend::arc(),
    };
    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act: DType::Bf16 };
    let (mut rt, m) = load_runtime_with(&cache, backend, None, &o, Some(fixture_engram(&fx, 0, 1))).unwrap();
    assert_eq!((m.world, m.layers), (1, Some(3)), "a --layers 3 cache of world 1");
    let mut worst = 0f64;
    for (l, step, mx, rr) in run_steps(&mut rt, &fx, true) {
        println!("RUNTIME L{l} {step}: max_abs/max|ref| {mx:.3e}  rel_rms {rr:.3e}");
        worst = worst.max(mx);
    }
    // bf16 flow: the per-layer parity of the ops is 1.4e-3 .. 5.5e-3 (dsv41_layers.rs); here the routed experts are
    // computed too and errors carry through three layers and the decode steps
    assert!(worst < 3e-2, "worst layer output error {worst:.3e}");
}

fn cuda_backend() -> Arc<dyn Backend> {
    intarsia_backend_cuda::create(0).expect("cuda backend")
}

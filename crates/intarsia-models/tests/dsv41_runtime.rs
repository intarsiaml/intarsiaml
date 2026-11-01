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
                    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act, spec: false };
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
    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act: DType::F32, spec: false };
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
    let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act: DType::Bf16, spec: false };
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

fn to_f32s(b: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        _ => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
    }
}

/// DSpark drafting rounds against the official drafter (`tools/dsv41/fixtures.py dspark`: 16 seed rows absorbed, then 3
/// rounds of one absorbed row + a 5-row block): the head input, the head's logits at the official top-64 ids, the drafts
/// (Markov loop, greedy) and the confidences. `INTARSIA_DSV41_CACHE` = a world-1 rank cache with its DSpark sub-cache
/// (`intarsia dsv41-cache --mtp-only`), `INTARSIA_DSV41_FIXTURES` = the dir with `layers_bf16.safetensors` (for the Engram
/// tier the loader wants) and `dspark_{fp32,bf16}.safetensors` (else the CI cuts in `golden/dsv41` are used).
#[test]
fn dspark_rounds_match_the_official_drafter() {
    let (Some(cache), Some(fxd)) = (std::env::var_os("INTARSIA_DSV41_CACHE").map(PathBuf::from), std::env::var_os("INTARSIA_DSV41_FIXTURES").map(PathBuf::from)) else {
        println!("skipped: set INTARSIA_DSV41_CACHE (with its mtp sub-cache) and INTARSIA_DSV41_FIXTURES");
        return;
    };
    let cuda = std::env::var("INTARSIA_DSV41_BACKEND").as_deref() == Ok("cuda");
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/dsv41");
    let fx = St::open(&fxd.join("layers_bf16.safetensors"));
    for (flow, act) in [("fp32", DType::F32), ("bf16", DType::Bf16)] {
        if cuda && act == DType::F32 {
            println!("DSPARK {flow}: skipped on cuda (bf16 activations only)");
            continue;
        }
        let full = fxd.join(format!("dspark_{flow}.safetensors"));
        let ds = St::open(&if full.exists() { full } else { golden.join(format!("dspark_{flow}_ci.safetensors")) });
        let backend: Arc<dyn Backend> = if cuda { cuda_backend() } else { intarsia_backend_cpu::CpuBackend::arc() };
        let o = Dsv41Opts { cap: 512, chunk: 16, engram_dir: PathBuf::new(), hot_mb: 0, act, spec: true };
        let (rt, _) = load_runtime_with(&cache, backend, None, &o, Some(fixture_engram(&fx, 0, 1))).unwrap();
        assert_eq!(rt.dspark, 3, "the sub-cache's three stages");
        let (d, v) = (rt.cfg.dim, rt.cfg.vocab);
        let mh = ds.get("in/main_hidden").f32();
        let w = mh.len() / 19;
        let ids: Vec<u32> = ds.get("in/input_ids").i32().into_iter().map(|x| x as u32).collect();
        let mut st = rt.new_state().unwrap();
        rt.absorb(&mut st, &mh[..16 * w]).unwrap();
        let (mut same_ids, mut worst_hx, mut worst_lg, mut worst_conf) = (0, 0f64, 0f64, 0f64);
        for (j, &first) in ids.iter().enumerate() {
            rt.absorb(&mut st, &mh[(16 + j) * w..(17 + j) * w]).unwrap();
            let dr = rt.draft(&st, first).unwrap().expect("rank 0 drafts");
            let r = format!("r{j}");
            let want_ids: Vec<u32> = ds.get(&format!("{r}/head.output_ids")).i32().into_iter().map(|x| x as u32).collect();
            let hx = to_f32s(&rt.read_ws("dspark.hx", act, 5 * d).unwrap(), act);
            let (_, hx_rr) = err(&hx, &ds.get(&format!("{r}/head.x")).f32());
            // the head's raw logits (before the Markov bias) at the official top-64 ids of every row
            let lg = to_f32s(&rt.read_ws("dspark.logits", DType::F32, 5 * v).unwrap(), DType::F32);
            // CI cut: the official top-64 per row; full fixture: the whole `head.logits_raw [block, V]`, top-64 taken here
            let (ti, tv) = if ds.has(&format!("{r}/head.logits_raw_top64.i")) {
                (ds.get(&format!("{r}/head.logits_raw_top64.i")).i32(), ds.get(&format!("{r}/head.logits_raw_top64.v")).f32())
            } else {
                let raw = ds.get(&format!("{r}/head.logits_raw")).f32();
                let (mut ti, mut tv) = (Vec::new(), Vec::new());
                for row in raw.chunks_exact(v) {
                    let mut ix: Vec<usize> = (0..v).collect();
                    ix.select_nth_unstable_by(63, |&a, &b| row[b].total_cmp(&row[a]));
                    ix.truncate(64);
                    ix.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                    ti.extend(ix.iter().map(|&i| i as i32));
                    tv.extend(ix.iter().map(|&i| row[i]));
                }
                (ti, tv)
            };
            let got: Vec<f32> = ti.iter().enumerate().map(|(k, &id)| lg[(k / 64) * v + id as usize]).collect();
            let (_, lg_rr) = err(&got, &tv);
            let want_conf = ds.get(&format!("{r}/head.confidence")).f32();
            let conf = dr.confidence.iter().zip(&want_conf).fold(0f64, |m, (a, b)| m.max((a - b).abs() as f64));
            println!("DSPARK {flow} round {j}: ids {:?} official {want_ids:?}; head.x rel_rms {hx_rr:.3e}, top-64 raw logits rel_rms {lg_rr:.3e}, confidence max |diff| {conf:.3e} ({:?} vs {want_conf:?})", dr.ids, dr.confidence);
            same_ids += dr.ids.iter().zip(&want_ids).skip(1).take_while(|(a, b)| a == b).count();
            (worst_hx, worst_lg, worst_conf) = (worst_hx.max(hx_rr), worst_lg.max(lg_rr), worst_conf.max(conf));
        }
        println!("DSPARK {flow}: leading drafts equal {same_ids}/15, worst head.x {worst_hx:.3e}, logits {worst_lg:.3e}, confidence {worst_conf:.3e}");
        if act == DType::F32 {
            assert!(worst_hx < 1e-3 && worst_lg < 1e-3, "fp32 drafter parity: head.x {worst_hx:.3e}, logits {worst_lg:.3e}");
            assert_eq!(same_ids, 15, "fp32: every draft equals the official one");
            assert!(worst_conf < 1e-2, "fp32 confidence {worst_conf:.3e}");
        } else {
            assert!(worst_hx < 5e-2 && worst_lg < 5e-2, "bf16 drafter parity: head.x {worst_hx:.3e}, logits {worst_lg:.3e}");
        }
    }
}

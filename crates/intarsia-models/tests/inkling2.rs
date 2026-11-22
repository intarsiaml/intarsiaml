//! Inkling-2 on the tiny fixture (`golden/inkling2-tiny`, `tools/inkling2_ref.py tiny`): gate G1 of the port spec
//! (docs/plans/2026-11-21-inkling2-port.md section i) on the cpu backend, plus the state contract (chunked prefill across the
//! window and the 6-layer blocks, prefill then decode, snapshot restore, failed-pass reset, speculative rollback by length,
//! batched == solo) and NoKernel refusal on a backend without the Inkling-2 kernels.
//!
//! `INTARSIA_INKLING2_BACKEND=metal` runs the bf16 tests (G1(b), the state contract, the failed-pass reset) on the Metal
//! backend instead (G1 on Metal, lane ink-metal; `cuda`: the same on CUDA, lane ink-cuda-mixmoe); G1(a) (f32 activations) and
//! the NoKernel refusal stay on cpu.
//!
//! Skipped (loudly) when the fixture is missing; fails under `CI=1`.

use intarsia_core::model::{HeadMode, InkTensors, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::Telemetry;
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn tiny_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("INTARSIA_INKLING2_TINY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/inkling2-tiny"));
    dir.join("config.json").exists().then_some(dir)
}

fn skip(why: &str) {
    println!("SKIPPED: inkling2: {why}");
    eprintln!("SKIPPED: inkling2: {why}");
    if std::env::var("CI").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("inkling2 was SKIPPED ({why}) but CI is set");
    }
}

fn load_with(dir: &Path, chunk: usize, backend: Arc<dyn intarsia_core::Backend>) -> intarsia_core::Result<Model> {
    load_act(dir, chunk, backend, None)
}

fn load_act(dir: &Path, chunk: usize, backend: Arc<dyn intarsia_core::Backend>, act: Option<intarsia_core::DType>) -> intarsia_core::Result<Model> {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    assert!(dims.ink.is_some());
    let src = SafeTensors::open_dir(dir).unwrap();
    Model::load(backend, &recipe, &dims, &src, &LoadOptions { max_seq: 128, prefill_chunk: chunk, graphs: false, act_dtype: act, kv_dtype: act, ..Default::default() }, Arc::new(Telemetry::new()))
}

/// The backend of the bf16 tests: cpu, or Metal / CUDA with `INTARSIA_INKLING2_BACKEND=metal` / `cuda`.
fn backend() -> Arc<dyn intarsia_core::Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        #[cfg(target_os = "macos")]
        Ok("metal") => intarsia_backend_metal::MetalBackend::arc().unwrap(),
        #[cfg(not(target_os = "macos"))]
        Ok("metal") => panic!("INTARSIA_INKLING2_BACKEND=metal: the Metal backend is macOS only"),
        Ok("cuda") => intarsia_backend_cuda::create(0).unwrap(),
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu, metal or cuda"),
    }
}

/// A GPU backend (Metal or CUDA): passes of 2..=16 rows take the rows-exact kernels, larger ones the prefill kernels.
fn on_metal() -> bool {
    matches!(std::env::var("INTARSIA_INKLING2_BACKEND").as_deref(), Ok("metal") | Ok("cuda"))
}

/// The prefill chunk of the state tests: one pass for the 40-token prompt on cpu (its kernels are rows-exact at any row count);
/// 16 on Metal, where passes of 2..=16 rows take the row-exact kernels (`ink_rows_exact`) and larger ones the fast prefill
/// kernels, which are not bitwise a decode step's rows.
fn chunk() -> usize {
    if on_metal() {
        16
    } else {
        64
    }
}

/// Load with `INTARSIA_INK_HEAD_SCREEN` set to `screen` for this load only. The variable is read at load time; a process-wide
/// lock keeps two such loads (the tests run in parallel) from seeing each other's value, and it is removed again before the
/// lock is released, so other tests load with it unset.
fn load_screen(dir: &Path, chunk: usize, screen: &str) -> Model {
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("INTARSIA_INK_HEAD_SCREEN", screen);
    let m = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load(dir, chunk)));
    std::env::remove_var("INTARSIA_INK_HEAD_SCREEN");
    m.unwrap_or_else(|e| std::panic::resume_unwind(e))
}

fn load(dir: &Path, chunk: usize) -> Model {
    load_with(dir, chunk, backend()).unwrap()
}

/// Every tensor of a safetensors file as f32 (F32 / BF16 / I64 / I32), by name.
fn read_all(path: &Path) -> InkTensors {
    let b = std::fs::read(path).unwrap();
    let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hl]).unwrap();
    let mut out = InkTensors::new();
    for (name, t) in h.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let (s, e) = (t["data_offsets"][0].as_u64().unwrap() as usize, t["data_offsets"][1].as_u64().unwrap() as usize);
        let raw = &b[8 + hl + s..8 + hl + e];
        let data: Vec<f32> = match t["dtype"].as_str().unwrap() {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "I64" => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            "I32" => raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            other => panic!("{name}: dtype {other}"),
        };
        out.insert(name.clone(), (shape, data));
    }
    out
}

fn prompt(dir: &Path) -> Vec<u32> {
    read_all(&dir.join("ref_logits.safetensors"))["prompt_ids"].1.iter().map(|&v| v as u32).collect()
}

fn all_logits(m: &Model, st: &mut intarsia_core::state::SequenceState, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<f32>> = Vec::new();
    m.forward(st, ids, HeadMode::AllLogits, Some(&mut |_, row: &[f32]| {
        rows.push(row.to_vec());
        Ok(())
    }))
    .unwrap();
    rows
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
fn inkling2_tiny_loads() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir, 64);
    let ids = prompt(&dir);
    let mut st = m.new_state().unwrap();
    let rows = all_logits(&m, &mut st, &ids);
    assert_eq!(rows.len(), ids.len());
}

/// G1(a): the f32-activation path vs the f64 oracle: top-1 at all 40 positions, max abs logit error <= 1e-2.
#[test]
fn inkling2_tiny_g1a_f32_vs_f64() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load_act(&dir, 64, intarsia_backend_cpu::CpuBackend::arc(), Some(intarsia_core::DType::F32)).unwrap();
    let ids = prompt(&dir);
    let r = read_all(&dir.join("ref_logits.safetensors"));
    let mut st = m.new_state().unwrap();
    let rows = all_logits(&m, &mut st, &ids);
    let ours: Vec<f32> = rows.concat();
    for k in ["logits_bf16", "logits_f32"] {
        println!("oracle {k} vs f64: {:.3e}", rel_l2(&r[k].1, &r["logits_f64"].1));
    }
    println!("ours vs f64: {:.3e}", rel_l2(&ours, &r["logits_f64"].1));
    let v = 300;
    let agree = (0..ids.len()).filter(|&t| argmax(&ours[t * v..(t + 1) * v]) == argmax(&r["logits_f64"].1[t * v..(t + 1) * v])).count();
    println!("top1 {agree}/{}", ids.len());
    let mx = ours.iter().zip(&r["logits_f64"].1).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("max abs {mx:.3e}");
    assert_eq!(agree, ids.len());
    assert!(mx <= 1e-2, "max abs {mx}");
}

const TF_SKIP: [&str; 6] = ["expert_ids", "blocks", "bank", "partial_in", "partial", "route_margin"];
const ROUTED: [&str; 3] = ["expert_w", "moe_out", "partial_out"];

/// Rows of an `[R, ...]` tensor selected by `keep`.
fn select(v: &[f32], shape: &[usize], keep: &[bool]) -> Vec<f64> {
    let w: usize = shape[1..].iter().product();
    keep.iter().enumerate().filter(|(_, k)| **k).flat_map(|(r, _)| v[r * w..(r + 1) * w].iter().map(|&x| x as f64)).collect()
}

/// `expert_w` reordered by expert id per row (slot order is not part of the contract).
fn by_expert(d: &InkTensors, g: &str) -> Vec<f32> {
    let (sh, ids) = &d[&format!("{g}.expert_ids")];
    let w = &d[&format!("{g}.expert_w")].1;
    let k = sh[1];
    (0..sh[0])
        .flat_map(|r| {
            let mut ix: Vec<usize> = (0..k).collect();
            ix.sort_by(|&a, &b| ids[r * k + a].partial_cmp(&ids[r * k + b]).unwrap());
            ix.into_iter().map(move |i| w[r * k + i]).collect::<Vec<_>>()
        })
        .collect()
}

/// G1(b), the oracle's `tf_check` (tools/inkling2_ref.py) with budget `factor` x the oracle's own teacher-forced error.
fn tf_check(test: &InkTensors, f64d: &InkTensors, oracle: &InkTensors, factor: f64, verbose: bool) -> Vec<String> {
    let mut fails = Vec::new();
    let grp = |k: &str| if k.contains('.') { k.split('.').next().unwrap().to_string() } else { "head".into() };
    let mut groups: Vec<String> = f64d.keys().map(|k| grp(k)).collect();
    groups.sort();
    groups.dedup();
    for g in groups {
        let mut rows: Option<Vec<bool>> = None;
        let mut conv_rows: Option<Vec<bool>> = None;
        if let Some((_, margin)) = f64d.get(&format!("{g}.route_margin")) {
            let key = format!("{g}.router_sel");
            let err = oracle[&key].1.iter().zip(&f64d[&key].1).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let r: Vec<bool> = margin.iter().map(|m| *m > 5.0 * err).collect();
            let mut c = r.clone();
            for lag in 1..4 {
                for i in (lag..c.len()).rev() {
                    c[i] = c[i] && r[i - lag];
                }
            }
            let (sh, want) = &f64d[&format!("{g}.expert_ids")];
            let k = sh[1];
            let sorted = |v: &[f32], i: usize| {
                let mut x = v[i * k..(i + 1) * k].to_vec();
                x.sort_by(|a, b| a.partial_cmp(b).unwrap());
                x
            };
            for (name, d) in [("oracle", oracle), ("test", test)] {
                let got = &d[&format!("{g}.expert_ids")].1;
                if (0..sh[0]).any(|i| r[i] && sorted(got, i) != sorted(want, i)) {
                    fails.push(format!("{g}: {name} expert set differs on routable rows"));
                }
            }
            rows = Some(r);
            conv_rows = Some(c);
        }
        let mut line = Vec::new();
        for k in f64d.keys().filter(|k| grp(k) == g) {
            let name = k.splitn(2, '.').last().unwrap();
            if TF_SKIP.contains(&name) {
                continue;
            }
            let Some(tv) = test.get(k) else {
                fails.push(format!("{k}: missing from the test dump"));
                continue;
            };
            let (sh, rv) = &f64d[k];
            let (rv, tv, ov) = if name == "expert_w" { (by_expert(f64d, &g), by_expert(test, &g), by_expert(oracle, &g)) } else { (rv.clone(), tv.1.clone(), oracle[k].1.clone()) };
            if tv.len() != rv.len() {
                fails.push(format!("{k}: {} values, expected {}", tv.len(), rv.len()));
                continue;
            }
            let all = vec![true; sh[0]];
            let keep = match (&rows, ROUTED.contains(&name)) {
                (Some(r), true) => if name == "partial_out" { conv_rows.as_ref().unwrap() } else { r },
                _ => &all,
            };
            let (r, t, o) = (select(&rv, sh, keep), select(&tv, sh, keep), select(&ov, sh, keep));
            let nrm = r.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
            let e = |a: &[f64]| a.iter().zip(&r).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt() / nrm;
            let (e_o, e_t) = (e(&o), e(&t));
            let budget = (factor * e_o).max(1e-3);
            line.push(format!("{name}={e_t:.1e}/{budget:.1e}"));
            if e_t > budget {
                fails.push(format!("{k}: rel err {e_t:.3e} > budget {budget:.3e}"));
            }
        }
        if verbose {
            let nr = rows.as_ref().map_or(String::new(), |r| format!(" routable {}/{}", r.iter().filter(|x| **x).count(), r.len()));
            println!("{g}{nr}: {}", line.join(" "));
        }
    }
    fails
}

/// G1(b): the bf16 path teacher-forced per visit (one 40-token prefill fed the f64 run's partials, blocks and bank).
#[test]
fn inkling2_tiny_g1b_teacher_forced_bf16() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let m = load(&dir, 64);
    let ids = prompt(&dir);
    let f64d = read_all(&dir.join("ref_dump_f64.safetensors"));
    let oracle = read_all(&dir.join("ref_dump_bf16_tf.safetensors"));
    let mut st = m.new_state().unwrap();
    let test = m.ink_dump(&mut st, &ids, Some(&f64d)).unwrap();
    let fails = tf_check(&test, &f64d, &oracle, 3.0, true);
    assert!(fails.is_empty(), "G1(b) failures:\n{}", fails.join("\n"));
}

fn logits_chunked(dir: &Path, chunk: usize, ids: &[u32]) -> Vec<Vec<f32>> {
    let m = load(dir, chunk);
    let mut st = m.new_state().unwrap();
    all_logits(&m, &mut st, ids)
}

/// Chunked prefill (chunks of 7 and 13 straddle the window of 8, the rel extent of 12 and the attn-res block edges) and
/// token-by-token decode give the one-shot prefill's logits bit for bit (the cpu kernels are rows-exact).
#[test]
fn inkling2_tiny_chunked_and_decode_match_prefill_bitwise() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let whole = logits_chunked(&dir, chunk(), &ids);
    for chunk in [7, 13] {
        assert_eq!(logits_chunked(&dir, chunk, &ids), whole, "chunk {chunk}");
    }
    if on_metal() {
        // one 40-row pass takes the prefill kernels, which are not bitwise a decode step's rows. The tiny random net is chaotic
        // under bf16 when free-running (spec section i), so the two schedules are not comparable here: the prefill kernels are
        // gated teacher-forced (G1(b), one 40-row pass) instead. Printed for the record only.
        let one = logits_chunked(&dir, 64, &ids);
        let e = rel_l2(&one.concat(), &whole.concat());
        let agree = one.iter().zip(&whole).filter(|(a, b)| argmax(a) == argmax(b)).count();
        println!("metal: one 40-row prefill vs 16-row passes (free-running, not gated): rel l2 {e:.3e}, top-1 {agree}/{}", ids.len());
    }
    // prefill 9 (past the window), then decode one token at a time
    let m = load(&dir, chunk());
    let mut st = m.new_state().unwrap();
    let mut rows = all_logits(&m, &mut st, &ids[..9]);
    for &id in &ids[9..] {
        rows.extend(all_logits(&m, &mut st, &[id]));
    }
    let first = (0..rows.len()).find(|&i| rows[i] != whole[i]);
    assert!(first.is_none(), "prefill then decode: first differing row {first:?} of {}", rows.len());
}

/// The captured decode step (CUDA graph: the positioned ops read the device position) gives the eager decode's logits bit for
/// bit, token by token after a 9-token prefill, and replays once per step. On a backend without graphs both loads decode
/// eagerly (the comparison is then trivially equal; the replay count is only checked where graphs are usable).
#[test]
fn inkling2_tiny_graph_decode_is_bitwise_the_eager_decode() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let load_g = |graphs: bool| {
        let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
        let src = SafeTensors::open_dir(&dir).unwrap();
        let o = LoadOptions { max_seq: 128, prefill_chunk: chunk(), graphs, profile_every: 0, ..Default::default() };
        Model::load(backend(), &recipe, &dims, &src, &o, Arc::new(Telemetry::new())).unwrap()
    };
    let run = |m: &Model| {
        let mut st = m.new_state().unwrap();
        let mut rows = all_logits(m, &mut st, &ids[..9]);
        for &id in &ids[9..] {
            let intarsia_core::model::StepOut::Logits(l) = m.forward(&mut st, &[id], HeadMode::LastLogits, None).unwrap() else { panic!() };
            rows.push(l);
        }
        rows
    };
    let eager = load_g(false);
    let graphed = load_g(true);
    let (le, lg) = (run(&eager), run(&graphed));
    assert_eq!(le.len(), lg.len());
    for (i, (a, b)) in le.iter().zip(&lg).enumerate() {
        assert!(a == b, "logits of row {i} differ between eager and graph decode");
    }
    let gs = graphed.graph_stats();
    if gs.enabled {
        assert_eq!(gs.replays as usize, ids.len() - 9, "one replay per decode step");
    }
    println!("graph decode: enabled {}, captures {}, replays {}, {} decode steps", gs.enabled, gs.captures, gs.replays, ids.len() - 9);
}

/// The logits of prefill 9 then decode one token at a time, written as raw f32 to `INTARSIA_INKLING2_DUMP` (two runs with
/// different `INTARSIA_INK_ATTN_FUSED` are compared byte for byte by the caller). No-op without the variable.
#[test]
fn inkling2_tiny_logits_dump() {
    let Some(path) = std::env::var_os("INTARSIA_INKLING2_DUMP") else { return };
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let m = load(&dir, chunk());
    let mut st = m.new_state().unwrap();
    let mut rows = all_logits(&m, &mut st, &ids[..9]);
    for &id in &ids[9..] {
        rows.extend(all_logits(&m, &mut st, &[id]));
    }
    let bytes: Vec<u8> = rows.concat().iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    println!("dump: {} rows, top-1 {:?}", rows.len(), rows.iter().map(|r| argmax(r)).collect::<Vec<_>>());
}

/// Diagnostics (`INTARSIA_INKLING2_CHUNK_SCAN=1`): which prefill chunk sizes give the one-token-at-a-time logits bit for bit,
/// and the first differing row of the others.
#[test]
fn inkling2_tiny_chunk_scan() {
    if std::env::var_os("INTARSIA_INKLING2_CHUNK_SCAN").is_none() {
        return;
    }
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let m = load(&dir, 1);
    let mut st = m.new_state().unwrap();
    let mut dec = Vec::new();
    for &id in &ids {
        dec.extend(all_logits(&m, &mut st, &[id]));
    }
    for chunk in 1..=16 {
        let got = logits_chunked(&dir, chunk, &ids);
        let first = got.iter().zip(&dec).position(|(a, b)| a != b);
        println!("scan chunk {chunk:2}: first row differing from decode {first:?}");
    }
}

/// The speculative verify pass (`Model::forward_verify_logits`, R rows through the rows-exact kernels) gives the 1-row decode
/// steps' logits bit for bit at R = 2, 3, 4 and at several positions (before and past the window and the rel extent), and its
/// greedy picks are the decode argmaxes: greedy spec == no spec needs exactly this (lane ink-cuda-mtp).
///
/// The verify rows come back BEFORE the final logit softcap (`final_logit_softcapping`; a decode step's readback applies it on
/// the host, `Model::softcap_logits`), so the comparison caps them the same way, `cap * tanh(x / cap)`. The cap is monotone:
/// greedy picks are unaffected, a sampled verify would draw from uncapped logits (reported, not changed here).
#[test]
fn inkling2_tiny_verify_rows_equal_decode_bitwise() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let mut m = load(&dir, chunk());
    m.enable_verify(4).unwrap();
    let cap = m.dims.ink.as_ref().map_or(0.0, |i| i.softcap);
    for &p in &[5usize, 9, 17, 26] {
        for r in 2..=4usize {
            if p + r > ids.len() {
                continue;
            }
            let mut a = m.new_state().unwrap();
            all_logits(&m, &mut a, &ids[..p]);
            let (picks, mut rows) = m.forward_verify_logits(&mut a, &ids[p..p + r]).unwrap();
            if cap > 0.0 {
                rows.iter_mut().flatten().for_each(|x| *x = cap * (*x / cap).tanh());
            }
            let mut b = m.new_state().unwrap();
            all_logits(&m, &mut b, &ids[..p]);
            for (j, &id) in ids[p..p + r].iter().enumerate() {
                let intarsia_core::model::StepOut::Logits(one) = m.forward(&mut b, &[id], HeadMode::LastLogits, None).unwrap() else { panic!() };
                let diff = rows[j].iter().zip(&one).position(|(x, y)| x.to_bits() != y.to_bits());
                assert!(
                    diff.is_none(),
                    "{}: verify R={r} at pos {p}, row {j}: logit {diff:?} differs from the 1-row decode (rel l2 {:.3e}, len {} vs {}, \
                     argmax {} vs {}, pick {}, first values {:?} vs {:?})",
                    m.backend.name(),
                    rel_l2(&rows[j], &one),
                    rows[j].len(),
                    one.len(),
                    argmax(&rows[j]),
                    argmax(&one),
                    picks[j],
                    &rows[j][..3],
                    &one[..3]
                );
                assert_eq!(picks[j] as usize, argmax(&one), "verify R={r} at pos {p}, row {j}: greedy pick");
            }
        }
    }
}

/// The captured verify-only pass (`Model::verify_replay`; default on for Inkling-2 on CUDA, `INTARSIA_VERIFY_GRAPH=0` off): ONE state runs R-row verify passes
/// back to back, so one recording per R is replayed at several positions (through the window and the rel extent), and every
/// row is the 1-row decode bit for bit (after the final softcap, as above). A position baked into the recording would show
/// here; a fresh state per pass would not. Asserts that replays happened (a failed capture falls back to the eager pass,
/// which passes the comparison too). Skipped where graphs are unusable (cpu) and under `INTARSIA_VERIFY_GRAPH=0`.
#[test]
fn inkling2_tiny_verify_graph_replays_equal_decode_bitwise() {
    if std::env::var("INTARSIA_VERIFY_GRAPH").ok().as_deref() == Some("0") {
        return skip("INTARSIA_VERIFY_GRAPH=0");
    }
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    // no profiled steps (`LoadOptions::profile_every` defaults to 1, and a profiled verify pass runs eagerly)
    let mut m = {
        let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
        let src = SafeTensors::open_dir(&dir).unwrap();
        let o = LoadOptions { max_seq: 128, prefill_chunk: chunk(), graphs: false, profile_every: 0, ..Default::default() };
        Model::load(backend(), &recipe, &dims, &src, &o, Arc::new(Telemetry::new())).unwrap()
    };
    // graphs on for the verify passes (turning them off drops the recordings, `Model::set_graphs`); the 1-row decode they
    // are compared against runs eagerly on a second load (graphs off, as the other tests)
    m.set_graphs(true);
    if !m.graph_stats().enabled {
        return skip("graphs not usable on this backend");
    }
    m.enable_verify(4).unwrap();
    let e = load(&dir, chunk());
    let cap = m.dims.ink.as_ref().map_or(0.0, |i| i.softcap);
    for r in 2..=4usize {
        let p0 = 5usize;
        let before = m.graph_stats();
        let mut a = m.new_state().unwrap();
        all_logits(&m, &mut a, &ids[..p0]);
        let mut b = e.new_state().unwrap();
        all_logits(&e, &mut b, &ids[..p0]);
        let mut p = p0;
        let mut passes = 0u64;
        while p + r <= ids.len() {
            let (picks, mut rows) = m.forward_verify_logits(&mut a, &ids[p..p + r]).unwrap();
            if cap > 0.0 {
                rows.iter_mut().flatten().for_each(|x| *x = cap * (*x / cap).tanh());
            }
            for (j, &id) in ids[p..p + r].iter().enumerate() {
                let intarsia_core::model::StepOut::Logits(one) = e.forward(&mut b, &[id], HeadMode::LastLogits, None).unwrap() else { panic!() };
                let diff = rows[j].iter().zip(&one).position(|(x, y)| x.to_bits() != y.to_bits());
                assert!(
                    diff.is_none(),
                    "{}: replayed verify R={r} at pos {p}, row {j}: logit {diff:?} differs from the 1-row decode (rel l2 {:.3e})",
                    m.backend.name(),
                    rel_l2(&rows[j], &one)
                );
                assert_eq!(picks[j] as usize, argmax(&one), "replayed verify R={r} at pos {p}, row {j}: greedy pick");
            }
            p += r;
            passes += 1;
        }
        let after = m.graph_stats();
        let (caps, reps) = (after.spec_captures - before.spec_captures, after.spec_replays - before.spec_replays);
        println!("{}: R={r}: {passes} verify passes from pos {p0}, {caps} capture(s), {reps} replays", m.backend.name());
        assert!(passes >= 3, "R={r}: too few passes ({passes}) to replay at several positions");
        assert_eq!(caps, 1, "R={r}: one recording for the state");
        assert_eq!(reps, passes, "R={r}: every pass replayed");
    }
}

/// Rollback by length (a rejected speculative tail, a prompt edit) within the ring margin, and the refusal past it.
#[test]
fn inkling2_tiny_rollback_by_length() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let whole = logits_chunked(&dir, chunk(), &ids);
    let m = load(&dir, chunk());
    let mut st = m.new_state().unwrap();
    all_logits(&m, &mut st, &ids[..30]);
    // a 5-token "verify" pass with wrong tokens, rolled back to keep 2 of them
    let wrong: Vec<u32> = ids[30..35].iter().map(|v| (v + 1) % 300).collect();
    let mut draft = ids[30..32].to_vec();
    draft.extend(&wrong[2..]);
    all_logits(&m, &mut st, &draft);
    st.truncate(32, &*m.backend).unwrap();
    assert_eq!(all_logits(&m, &mut st, &ids[32..]), whole[32..].to_vec(), "after rollback to 32");
    // prompt reuse (`Model::reuse_len`): within the margin the prefix is kept, past it the sequence starts over
    // (CUDA keeps its conservative policy, `prefix_reuse_any_cut` false and no `prefix_reuse_min_pass_rows`: a cut off the
    // cold-prefill ladder starts over, which is safe; the resume below then prefills the whole prompt)
    if m.backend.name() == "cuda" {
        assert!(matches!(m.reuse_len(&st, 40, 30), 0 | 30));
    } else {
        assert_eq!(m.reuse_len(&st, 40, 30), 30);
    }
    assert_eq!(m.reuse_len(&st, 40, 10), 0);
    // a cut beyond the ring margin (16) is refused, and restorable_len says so
    assert_eq!(st.restorable_len(10), 0);
    assert!(st.truncate(10, &*m.backend).is_err());
    // a prompt that shares 30 tokens resumes there (the engine's prefix reuse path: reuse_len, truncate, prefill the rest)
    let keep = m.reuse_len(&st, 40, 30);
    st.truncate(keep, &*m.backend).unwrap();
    assert_eq!(all_logits(&m, &mut st, &ids[keep..]), whole[keep..].to_vec(), "resumed at {keep}");
    // the empty state always works
    st.truncate(0, &*m.backend).unwrap();
    assert_eq!(all_logits(&m, &mut st, &ids), whole, "after a reset to 0");
}

/// Two sequences interleaved on one model keep their own state (the workspace is per pass, the rings per sequence); batched
/// decode itself is refused (one sequence per pass for now).
#[test]
fn inkling2_tiny_interleaved_sequences_equal_solo() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let other: Vec<u32> = ids.iter().rev().copied().collect();
    let (wa, wb) = (logits_chunked(&dir, chunk(), &ids), logits_chunked(&dir, chunk(), &other));
    let m = load(&dir, chunk());
    let (mut a, mut b) = (m.new_state().unwrap(), m.new_state().unwrap());
    let (mut ra, mut rb) = (Vec::new(), Vec::new());
    for (i, c) in [(0, 11), (11, 20), (20, 33), (33, 40)] {
        ra.extend(all_logits(&m, &mut a, &ids[i..c]));
        rb.extend(all_logits(&m, &mut b, &other[i..c]));
    }
    assert_eq!(ra, wa);
    assert_eq!(rb, wb);
}

use intarsia_core::backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId};
use intarsia_core::registry::KernelImpl;
use intarsia_core::{Op, OpCall};
use std::sync::atomic::{AtomicBool, Ordering};

/// The cpu backend with the Inkling-2 kernels of `drop` removed (as a backend without them registers), or whose `ConvMix`
/// launches fail while `fail` is set (a failed pass).
struct Wrapped {
    inner: Arc<dyn Backend>,
    drop: Vec<Op>,
    fail: Arc<AtomicBool>,
}

impl Backend for Wrapped {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn info(&self) -> DeviceInfo {
        self.inner.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> intarsia_core::Result<Buffer> {
        self.inner.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> intarsia_core::Result<()> {
        self.inner.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> intarsia_core::Result<()> {
        self.inner.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> intarsia_core::Result<()> {
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.inner.kernels().into_iter().filter(|k| !self.drop.contains(&k.op)).collect()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> intarsia_core::Result<()> {
        if call.op == Op::ConvMix && self.fail.load(Ordering::SeqCst) {
            return Err(intarsia_core::Error::invalid("injected failure"));
        }
        self.inner.launch(kernel, call, stream)
    }
    fn synchronize(&self, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.synchronize(s)
    }
    // the scheduling hints and stream calls the Metal backend overrides, so a wrapped Metal backend runs the same schedule
    fn avoid_single_row_prefill(&self) -> bool {
        self.inner.avoid_single_row_prefill()
    }
    fn prefix_reuse_any_cut(&self) -> bool {
        self.inner.prefix_reuse_any_cut()
    }
    fn prefix_reuse_min_pass_rows(&self) -> Option<usize> {
        self.inner.prefix_reuse_min_pass_rows()
    }
    fn zero(&self, dst: &Buffer, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.zero(dst, s)
    }
    fn hold(&self, s: StreamId, after: usize) -> intarsia_core::Result<()> {
        self.inner.hold(s, after)
    }
    fn release(&self, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.release(s)
    }
    fn discard(&self, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.discard(s)
    }
    // zero-copy weights: the Metal GEMV variant follows the weight's alignment, so uploaded copies could take another one
    fn wrap_host(&self, mem: Arc<dyn intarsia_core::backend::HostMemory>, offset: usize, len: usize) -> Option<Buffer> {
        self.inner.wrap_host(mem, offset, len)
    }
}

/// Records (op, kernel, the first row of output 0 when its leading dim is the pass's rows) of every launch.
struct Logged {
    inner: Arc<dyn Backend>,
    rows: std::sync::atomic::AtomicUsize,
    log: std::sync::Mutex<Vec<(String, String, Vec<u8>)>>,
}

impl Backend for Logged {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn info(&self) -> DeviceInfo {
        self.inner.info()
    }
    fn alloc(&self, bytes: usize, kind: MemKind) -> intarsia_core::Result<Buffer> {
        self.inner.alloc(bytes, kind)
    }
    fn upload(&self, dst: &Buffer, o: usize, src: &[u8], s: StreamId) -> intarsia_core::Result<()> {
        self.inner.upload(dst, o, src, s)
    }
    fn download(&self, src: &Buffer, o: usize, dst: &mut [u8], s: StreamId) -> intarsia_core::Result<()> {
        self.inner.download(src, o, dst, s)
    }
    fn copy(&self, d: &Buffer, d_o: usize, s: &Buffer, s_o: usize, n: usize, st: StreamId) -> intarsia_core::Result<()> {
        self.inner.copy(d, d_o, s, s_o, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.inner.kernels()
    }
    fn launch(&self, kernel: &str, call: &OpCall, stream: StreamId) -> intarsia_core::Result<()> {
        self.inner.launch(kernel, call, stream)?;
        let t = self.rows.load(Ordering::SeqCst);
        let mut row = Vec::new();
        let shapes: Vec<String> = call.outputs.iter().map(|o| format!("{:?}", o.shape)).collect();
        if let Some(o) = call.outputs.first() {
            if o.shape.first() == Some(&t) && t > 0 {
                self.inner.synchronize(stream)?;
                let mut data = vec![0u8; o.nbytes() / t];
                self.inner.download(&o.buf, o.offset, &mut data, stream)?;
                row = data;
            }
        }
        self.log.lock().unwrap().push((call.op.to_string(), format!("{kernel} out {}", shapes.join(" ")), row));
        Ok(())
    }
    fn synchronize(&self, s: StreamId) -> intarsia_core::Result<()> {
        self.inner.synchronize(s)
    }
}

/// Diagnostics (`INTARSIA_INKLING2_CHUNK_SCAN=1`): the launches of a 1-row pass and of a `INTARSIA_INKLING2_SCAN_T`-row pass at
/// position 0, side by side, with whether output 0's first row is bitwise the same.
#[test]
fn inkling2_tiny_launch_diff() {
    if std::env::var_os("INTARSIA_INKLING2_CHUNK_SCAN").is_none() {
        return;
    }
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let t: usize = std::env::var("INTARSIA_INKLING2_SCAN_T").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut logs = Vec::new();
    for n in [1usize, t] {
        let b = Arc::new(Logged { inner: backend(), rows: std::sync::atomic::AtomicUsize::new(n), log: Default::default() });
        let m = load_with(&dir, 16, b.clone()).unwrap();
        b.log.lock().unwrap().clear();
        let mut st = m.new_state().unwrap();
        all_logits(&m, &mut st, &ids[..n]);
        logs.push(b.log.lock().unwrap().clone());
    }
    let (a, b) = (&logs[0], &logs[1]);
    println!("launches: 1 row {}, {t} rows {}", a.len(), b.len());
    let mut shown = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let same = !x.2.is_empty() && x.2.len() == y.2.len() && x.2 == y.2;
        if !same && shown < 40 {
            println!("diff #{i}: {} {} vs {} {} (row bytes {} / {}, equal {})", x.0, x.1, y.0, y.1, x.2.len(), y.2.len(), x.2 == y.2);
            shown += 1;
        }
    }
}

/// A pass that fails half-way leaves the sequence empty (its rings half-written), and the next prefill is the clean one.
#[test]
fn inkling2_tiny_failed_pass_resets_the_sequence() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let ids = prompt(&dir);
    let whole = logits_chunked(&dir, chunk(), &ids);
    let fail = Arc::new(AtomicBool::new(false));
    let m = load_with(&dir, chunk(), Arc::new(Wrapped { inner: backend(), drop: vec![], fail: fail.clone() })).unwrap();
    {
        let mut c = m.new_state().unwrap();
        let ctl = all_logits(&m, &mut c, &ids);
        let first = (0..ctl.len()).find(|&i| ctl[i] != whole[i]);
        assert!(first.is_none(), "wrapped backend, no failure: first differing row {first:?} (the wrapper must run what the backend runs)");
    }
    let mut st = m.new_state().unwrap();
    all_logits(&m, &mut st, &ids[..20]);
    fail.store(true, Ordering::SeqCst);
    assert!(m.forward(&mut st, &ids[20..], HeadMode::LastLogits, None).is_err());
    fail.store(false, Ordering::SeqCst);
    assert_eq!(st.len, 0, "a failed pass resets the sequence");
    let again = all_logits(&m, &mut st, &ids);
    let first = (0..again.len()).find(|&i| again[i] != whole[i]);
    assert!(first.is_none(), "after the reset: first differing row {first:?} of {}", again.len());
}

/// `INTARSIA_INK_PACK12=1` (lane ink-pack12) on cpu, whose reference registers the P12 kernels, packs at load and gives the
/// bf16 logits bit for bit, over 1-row decode steps and 4- and 5-row passes (both sides of `quant::p12::DENSE_ROWS_MAX`).
/// On a backend without the P12 kernels (`MatmulP12` dropped, as CUDA and HIP register none) the forced `1` keeps the bf16
/// weights instead of packing. The variable is read at load, so this runs only when the test binary has it set to `1`.
#[test]
fn inkling2_tiny_p12_forced_on() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    if !std::env::var(intarsia_core::quant::p12::ENV).is_ok_and(|v| v.trim() == "1") {
        eprintln!("inkling2_tiny_p12_forced_on: skipped (run with INTARSIA_INK_PACK12=1)");
        return;
    }
    let ids = prompt(&dir);
    // the backend under test (INTARSIA_INKLING2_BACKEND: cpu by default, cuda for lane ink-cuda-p12)
    let cpu = backend();
    let packed = load_with(&dir, chunk(), cpu.clone()).unwrap();
    let no_p12 = Arc::new(Wrapped { inner: cpu, drop: vec![Op::MatmulP12], fail: Arc::new(AtomicBool::new(false)) });
    let plain = load_with(&dir, chunk(), no_p12).unwrap();
    assert!(
        packed.weights_bytes > plain.weights_bytes,
        "the cpu load packs ({} B) and the one without P12 kernels keeps bf16 only ({} B)",
        packed.weights_bytes,
        plain.weights_bytes
    );
    // 8 one-row steps, then passes of 4 and 5 rows, then one-row steps to the end
    let mut cuts = vec![0usize];
    for _ in 0..8 {
        cuts.push(cuts.last().unwrap() + 1);
    }
    for n in [4, 5, 4, 5] {
        cuts.push(cuts.last().unwrap() + n);
    }
    while *cuts.last().unwrap() < ids.len() {
        cuts.push(cuts.last().unwrap() + 1);
    }
    let run = |m: &Model| {
        let mut st = m.new_state().unwrap();
        let n = ids.len();
        cuts.windows(2).filter(|w| w[0] < n).flat_map(|w| all_logits(m, &mut st, &ids[w[0]..w[1].min(n)])).collect::<Vec<_>>()
    };
    let (a, b) = (run(&packed), run(&plain));
    assert_eq!(a.len(), ids.len());
    let first = (0..a.len()).find(|&i| a[i] != b[i]);
    assert!(first.is_none(), "P12 vs bf16 logits: first differing row {first:?} of {}", a.len());
}

/// A backend without one of the Inkling-2 kernels refuses the load, naming the op (CUDA and HIP register none of them).
#[test]
fn inkling2_tiny_no_kernel_refusal_names_the_op() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    // SconvRes is in the op set for kernels that split ConvMix; the model graph itself emits ConvMix. InkAttnDecode is optional:
    // without it the graph runs the InkKvPrologue + RelSinkAttn pair
    // InkRoute (76) is optional too: without it the graph runs the router matmul + MoeRoute
    for op in intarsia_core::inkling2::ops::OPS.iter().copied().filter(|&o| o != Op::SconvRes && o != Op::InkAttnDecode && o != Op::InkRoute) {
        let b = Arc::new(Wrapped { inner: intarsia_backend_cpu::CpuBackend::arc(), drop: vec![op], fail: Arc::new(AtomicBool::new(false)) });
        let e = load_with(&dir, chunk(), b).err().unwrap_or_else(|| panic!("load without {op} must fail"));
        let msg = e.to_string();
        assert!(matches!(e, intarsia_core::Error::NoKernel(_)), "{op}: {msg}");
        assert!(msg.contains(op.as_str()), "{op}: {msg}");
    }
    for op in intarsia_core::inkling2::ops::OPS {
        // CUDA: every op has a kernel: the attention ops (lane ink-cuda-attn, inkattn.rs), the in-projection, mix / conv ops and
        // the MXFP8 experts (lane ink-cuda-mixmoe, inkmix.rs / inkmoe.rs)
        let on_cuda = |t: &[(&str, Op)]| t.iter().any(|(_, o)| *o == op);
        // except InkRoute (op 76, lane ink-dfuse): Metal only and optional; CUDA runs the router matmul + MoeRoute
        if op == Op::InkRoute {
            assert!(!intarsia_backend_cuda::kernels::all().iter().any(|k| k.op == op), "cuda {op}");
        } else {
            let lanes = on_cuda(intarsia_backend_cuda::inkattn::TABLE) || on_cuda(intarsia_backend_cuda::inkmix::TABLE) || on_cuda(intarsia_backend_cuda::inkmoe::TABLE);
            assert!(lanes, "cuda {op}: no lane table lists it");
            assert!(intarsia_backend_cuda::kernels::all().iter().any(|k| k.op == op), "cuda {op}");
        }
        assert!(intarsia_backend_hip::kernels::UNIMPLEMENTED.contains(&op), "hip {op}");
        // Metal: every op has a kernel (lanes ink-m-attn, ink-m-moe, ink-m-mix); the table alone, no device
        // (a macOS-only dev-dependency, so the check is too)
        #[cfg(target_os = "macos")]
        assert!(intarsia_backend_metal::kernels::all().iter().any(|k| k.op == op), "metal {op}");
    }
}

/// The screened greedy head on a multi-row verify (lane ink-fp8, brief 12): when any row of a c-row group falls back, the
/// whole group takes the caller's full head (one read of the unembed for all rows, no per-row full row in LogitsGather), and
/// the tokens are bitwise the full head's. Fallbacks forced on 1, 2 and all rows of a 5- and a 16-row verify.
#[test]
fn inkling2_tiny_screened_verify_fallback_reads_the_head_once() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let mut m = load_screen(&dir, chunk(), "1");
    m.enable_verify(16).unwrap();
    assert!(m.head_screen_stats().is_some(), "INTARSIA_INK_HEAD_SCREEN=1 loads the screen on the tiny model");
    let ids = prompt(&dir);
    let run = |m: &Model, rows: usize| -> Vec<u32> {
        let mut st = m.new_state().unwrap();
        all_logits(m, &mut st, &ids[..20]);
        m.forward_verify(&mut st, &ids[20..20 + rows]).unwrap()
    };
    // the screen must engage on cpu; on Metal it turns itself off when the gather refuses this head (the fixture's lm_head is
    // not 16-byte aligned there, so the head runs the scalar GEMV the gather does not repeat): nothing to test then
    run(&m, 5);
    let (g, f) = m.head_screen_stats().unwrap();
    if g + f == 0 {
        assert!(on_metal(), "the screen did not engage on cpu");
        return skip("screened verify: the screen is off for the tiny head on this backend (see the load log)");
    }
    for rows in [5usize, 16] {
        m.head_screen_test_hooks(false, 0);
        let want = run(&m, rows);
        let all = if rows == 64 { u64::MAX } else { (1u64 << rows) - 1 };
        for (what, mask) in [("1 row", 1u64 << (rows - 1)), ("2 rows", 0b101), ("all rows", all)] {
            m.head_screen_test_hooks(true, mask);
            let (g0, f0) = m.head_screen_stats().unwrap();
            assert_eq!(run(&m, rows), want, "{rows}-row verify, fallback forced on {what}");
            let (g1, f1) = m.head_screen_stats().unwrap();
            assert_eq!((g1 - g0, f1 - f0), (0, 1), "{rows}-row verify, {what}: one full-head pass, nothing gathered");
        }
        // unforced: screened (gathered) or, if a row really falls back, the one full-head pass; same tokens either way
        m.head_screen_test_hooks(true, 0);
        let (g0, f0) = m.head_screen_stats().unwrap();
        assert_eq!(run(&m, rows), want, "{rows}-row verify, screened");
        let (g1, f1) = m.head_screen_stats().unwrap();
        assert_eq!((g1 - g0) + (f1 - f0), 1, "{rows}-row verify: exactly one screened call");
    }
}

/// The screened head never changes what batching accepts (brief 12, endor #11315): its plan-probe calls are recorded as
/// `extra`, so the solo decode step the batched step is checked against is the full head's. `enable_batch` + `configure_plan`
/// give the same result with the screen on as off.
#[test]
fn inkling2_tiny_screen_keeps_batching() {
    let Some(dir) = tiny_dir() else { return skip("no fixture") };
    let plan = |screen: &str| -> std::result::Result<(), String> {
        let mut m = load_screen(&dir, chunk(), screen);
        m.enable_batch(4).map_err(|e| format!("enable_batch: {e}"))?;
        m.configure_plan(&Default::default(), None).map(|_| ()).map_err(|e| format!("configure_plan: {e}"))
    };
    let off = plan("0");
    let on = plan("1");
    assert_eq!(on, off, "batching with the screen on vs off");
}

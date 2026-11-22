#![cfg(target_os = "macos")]
//! Inkling-2-Mini (real checkpoint, `INTARSIA_METAL_TEST_MODEL`) 1-row decode on Metal with the mix row kernels at NE 4 against
//! NE 3 (lane ink-metal, the decode-mode G2 regression of m-mix 25ff7864a: NE 4 at D 2304 took decode-mode G2 short from
//! 2.589e-3 to 1.109e-2 while NE 3 is bitwise the pre-merge tree). The synthetic kernel test (inkling2_mix_decode_cpu.rs) passes
//! at NE 4, so the fault needs the real call layout; this test runs the real graph.
//!
//! One child process per geometry (`INTARSIA_INK_MIX_NE`, read once per process: 3, unset = the default, 4): prefill 24 tokens of the golden's `chat_system`
//! prompt (63 ids), then decode the next 38 tokens (rows 24..61: the regression grew past row 50 there) one at a time (teacher-forced: the same ids in both runs), the logits of each decode
//! step written out, then one `ink_dump` decode step whose per-visit tensors are written out. The parent compares with NE 3: per step
//! the softmax KL(NE 3 || default) must stay at reduction-order size (<= 2e-3 every step); NE 4 forced is reported, with the
//! dumped tensors furthest from NE 3, not gated. Skips without the model or the golden (fails under `CI=1`).

use intarsia_core::model::{HeadMode, InkTensors, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::Telemetry;
use intarsia_models::SafeTensors;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const PREFILL: usize = 24;
const DECODE: usize = 38;
/// Per decode step KL(NE 3 || NE 4) of the softmax: two reduction orders of the same math (bf16 activations) stay well below
/// this; the regression reaches 1e-2 .. 3e-1 per row by row 57 of this prompt.
const KL_MAX: f64 = 2e-3;

fn skip(why: &str) {
    assert!(std::env::var("CI").is_err(), "inkling2_real_decode_ne: {why} under CI");
    eprintln!("SKIP inkling2_real_decode_ne: {why}");
}

fn model_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("INTARSIA_METAL_TEST_MODEL").ok()?);
    p.join("config.json").exists().then_some(p)
}

fn golden() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/oracle/inkling2-mini-oracle-w4a16-short-top20.safetensors");
    p.exists().then_some(p)
}

/// The token ids of the golden's prompt `name` (`ids_<i>`, i its index in the `prompts` metadata).
fn golden_ids(path: &Path, name: &str) -> Vec<u32> {
    let b = std::fs::read(path).unwrap();
    let hl = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hl]).unwrap();
    let prompts: serde_json::Value = serde_json::from_str(h["__metadata__"]["prompts"].as_str().unwrap()).unwrap();
    let i = prompts.as_array().unwrap().iter().position(|p| p["name"] == name).unwrap_or_else(|| panic!("golden has no prompt {name}"));
    let t = &h[format!("ids_{i}")];
    let (s, e) = (t["data_offsets"][0].as_u64().unwrap() as usize, t["data_offsets"][1].as_u64().unwrap() as usize);
    let raw = &b[8 + hl + s..8 + hl + e];
    match t["dtype"].as_str().unwrap() {
        "I64" => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
        "I32" => raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
        other => panic!("ids dtype {other}"),
    }
}

fn load(dir: &Path) -> Model {
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(dir).unwrap();
    let backend = intarsia_backend_metal::MetalBackend::arc().unwrap();
    let mut m = Model::load(
        backend,
        &recipe,
        &dims,
        &src,
        &LoadOptions { max_seq: 256, prefill_chunk: 256, graphs: false, ..Default::default() },
        Arc::new(Telemetry::new()),
    )
    .unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

// ---- a flat record file: (name, shape, f32 data) ----
fn put(w: &mut impl Write, name: &str, shape: &[usize], v: &[f32]) {
    let nb = name.as_bytes();
    w.write_all(&(nb.len() as u32).to_le_bytes()).unwrap();
    w.write_all(nb).unwrap();
    w.write_all(&(shape.len() as u32).to_le_bytes()).unwrap();
    for &s in shape {
        w.write_all(&(s as u64).to_le_bytes()).unwrap();
    }
    w.write_all(&(v.len() as u64).to_le_bytes()).unwrap();
    for x in v {
        w.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn read_records(path: &Path) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut b = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut b).unwrap();
    fn take(b: &[u8], o: &mut usize, n: usize) -> Vec<u8> {
        let s = b[*o..*o + n].to_vec();
        *o += n;
        s
    }
    let mut o = 0usize;
    let mut out = Vec::new();
    while o < b.len() {
        let nl = u32::from_le_bytes(take(&b, &mut o, 4).try_into().unwrap()) as usize;
        let name = String::from_utf8(take(&b, &mut o, nl)).unwrap();
        let nd = u32::from_le_bytes(take(&b, &mut o, 4).try_into().unwrap()) as usize;
        let shape: Vec<usize> = (0..nd).map(|_| u64::from_le_bytes(take(&b, &mut o, 8).try_into().unwrap()) as usize).collect();
        let n = u64::from_le_bytes(take(&b, &mut o, 8).try_into().unwrap()) as usize;
        let v: Vec<f32> = take(&b, &mut o, 4 * n).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.push((name, shape, v));
    }
    out
}

/// The child: prefill, decode steps (logits), one dumped decode step; all to `out`.
fn child(dir: &Path, ids: &[u32], out: &Path) {
    let m = load(dir);
    let mut st = m.new_state().unwrap();
    m.forward(&mut st, &ids[..PREFILL], HeadMode::None, None).unwrap();
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
    for (s, &id) in ids[PREFILL..PREFILL + DECODE].iter().enumerate() {
        let mut row = Vec::new();
        m.forward(&mut st, &[id], HeadMode::AllLogits, Some(&mut |_, r: &[f32]| {
            row = r.to_vec();
            Ok(())
        }))
        .unwrap();
        put(&mut f, &format!("step{s:02}.logits"), &[row.len()], &row);
    }
    let d: InkTensors = m.ink_dump(&mut st, &ids[PREFILL + DECODE..PREFILL + DECODE + 1], None).unwrap();
    let mut keys: Vec<&String> = d.keys().collect();
    keys.sort();
    for k in keys {
        let (shape, v) = &d[k];
        put(&mut f, &format!("dump.{k}"), shape, v);
    }
    f.flush().unwrap();
}

fn softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = v.iter().map(|&x| (x as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a / b.max(1e-30)).ln()).sum()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        n += (*x as f64 - *y as f64).powi(2);
        d += (*y as f64).powi(2);
    }
    if d == 0.0 {
        n.sqrt()
    } else {
        (n / d).sqrt()
    }
}

#[test]
fn inkling2_real_decode_ne4_matches_ne3() {
    let Some(dir) = model_dir() else { return skip("INTARSIA_METAL_TEST_MODEL unset or not a model dir") };
    let Some(g) = golden() else { return skip("no golden/oracle short golden") };
    let ids = golden_ids(&g, "chat_system");
    assert!(ids.len() > PREFILL + DECODE, "chat_system has {} ids", ids.len());
    if let Ok(out) = std::env::var("INTARSIA_INK_NE_CHILD_OUT") {
        return child(&dir, &ids, Path::new(&out));
    }
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let exe = std::env::current_exe().unwrap();
    let mut runs = Vec::new();
    // NE 3 (the reference: bitwise the pre-merge tree), the default geometry (gated), NE 4 forced (diagnostic only)
    for ne in ["3", "default", "4"] {
        let out = tmp.join(format!("ink_ne_{ne}.bin"));
        let mut c = std::process::Command::new(&exe);
        c.args(["--exact", "inkling2_real_decode_ne4_matches_ne3", "--nocapture", "--test-threads", "1"]).env("INTARSIA_INK_NE_CHILD_OUT", &out);
        if ne == "default" {
            c.env_remove("INTARSIA_INK_MIX_NE");
        } else {
            c.env("INTARSIA_INK_MIX_NE", ne);
        }
        assert!(c.status().unwrap().success(), "NE {ne} child failed");
        runs.push(read_records(&out));
    }
    let mut worst = 0.0f64;
    let mut fails = Vec::new();
    for (tag, b) in [("default", &runs[1]), ("NE4", &runs[2])] {
        let a = &runs[0];
        assert_eq!(a.len(), b.len(), "record counts");
        for ((na, _, va), (nb, _, vb)) in a.iter().zip(b).filter(|((n, _, _), _)| n.ends_with(".logits") && n.starts_with("step")) {
            assert_eq!(na, nb);
            let k = kl(&softmax(va), &softmax(vb));
            println!("{tag} {na}: KL(NE3 || {tag}) {k:.3e}");
            if tag == "default" {
                worst = worst.max(k);
                if k > KL_MAX {
                    fails.push(format!("{na}: KL {k:.3e} > {KL_MAX:.0e}"));
                }
            }
        }
        // the dumped decode step: the tensors furthest from NE 3 (the earliest layer among them names the faulty op)
        let mut d: Vec<(f64, &String)> = a.iter().zip(b).filter(|((n, _, _), _)| n.starts_with("dump.")).map(|((na, _, va), (_, _, vb))| (rel_l2(vb, va), na)).collect();
        d.sort_by(|x, y| y.0.total_cmp(&x.0));
        for (e, na) in d.iter().take(16) {
            println!("{tag} dumped {na}: rel l2 vs NE3 {e:.3e}");
        }
    }
    println!("worst per-step KL(NE3 || default) {worst:.3e} over {DECODE} decode steps");
    assert!(fails.is_empty(), "the default mix geometry decode departs from NE 3:\n{}", fails.join("\n"));
}

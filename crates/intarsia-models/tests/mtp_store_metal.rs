//! Lane mstore: `--spec mtp` on prefix-store requests on the Metal backend (roadmap idea #8,
//! docs/results/2026-11-15-mstore-metal.md), lane mtp-store's CUDA gate (`mtp_store_cuda.rs`) ported and extended. The store
//! restores the main model's state only; the MTP head's context restarts where a request resumes (`MtpState::base`), so a
//! restored request drafts. The model is `INTARSIA_METAL_TEST_MODEL` (default the Qwen3.6-35B-A3B NVFP4 under `~/models`).
//!
//! Two chats, three turns each, through the model's chat template (thinking on), the chats alternating so every later turn
//! restores its history from the store:
//!
//! 1. reference: the single-sequence engine, store on, `--spec off`; greedy turns build the prompts, then one seeded sampled
//!    request per prompt (reported: the plain sampler's draws);
//! 2. the single-sequence engine, store on, the recipe's MTP default: every greedy reply's ids and cached tokens equal the
//!    reference's, the later turns draft after their restore; then one seeded sampled request per later prompt (restored);
//! 3. cold, per later prompt: a fresh single-sequence engine and a fresh `--max-batch 2` engine (store on, MTP default) run the
//!    seeded sampled request cold (cached 0), then the greedy one, which restores all but its last tokens (ids = reference).
//!    The restored sampled replies of 2 and 5 must equal these cold ones: a restore never changes a token, greedy or sampled;
//! 4. the single-sequence engine, store OFF, MTP on: reported only (the head's acceptance with its whole context);
//! 5. `--max-batch 2`, store on, MTP: each round's two greedy requests together, then every greedy request alone, then the
//!    seeded sampled requests alone (restored): greedy ids = reference, sampled ids = the cold batched ones of 3;
//! 6. restart: a single-sequence engine with a disk tier runs chat 0's turn 1, waits for the writer, and is dropped; a new
//!    engine on the same directory runs chat 0's turn 2, which restores from disk: ids = reference.
//!
//! Needs MTP drafting on store requests for Metal (`INTARSIA_MTP_STORE=1` while the Metal default is off). The batched arms
//! draft only where batched MTP is on for the model (`INTARSIA_BATCHED_MTP`, the recipe's `batched_mtp`: the 35B on Metal;
//! the 27B needs `INTARSIA_BATCHED_MTP=1`). Prints `MTP-STORE-METAL` lines. Opt-in: `INTARSIA_TEST_REAL_MODEL=1`; take a
//! fleetq slot on endor. `INTARSIA_TEST_GEN` sets the tokens per reply (default 160); `INTARSIA_TEST_STORE_DIR` the disk
//! tier's parent directory (default `target/mstore-run`).
#![cfg(target_os = "macos")]
use intarsia_core::engine::ChatOpts;
use intarsia_core::model::LoadOptions;
use intarsia_core::prefix_store::PrefixStoreConfig;
use intarsia_core::{Engine, GenEvent, GenRequest};
use intarsia_models::{load_engine, EngineOptions};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn enabled() -> bool {
    if std::env::var("INTARSIA_TEST_REAL_MODEL").is_ok_and(|v| v == "1") {
        return true;
    }
    eprintln!("SKIP: real-model test (set INTARSIA_TEST_REAL_MODEL=1; on endor run inside a fleetq slot)");
    false
}

fn model_dir() -> PathBuf {
    std::env::var_os("INTARSIA_METAL_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models/Qwen3.6-35B-A3B-NVFP4"))
}

fn store_dir() -> PathBuf {
    std::env::var_os("INTARSIA_TEST_STORE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/mstore-run"))
}

fn gen_tokens() -> usize {
    std::env::var("INTARSIA_TEST_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(160)
}

/// `store`: the store's config (`None`: off). `mtp`: the recipe's MTP default applies (no `--spec` given); else `--spec off`.
fn engine(store: Option<PrefixStoreConfig>, mtp: bool, max_batch: usize) -> Arc<dyn Engine> {
    let backend = intarsia_backend_metal::MetalBackend::arc().unwrap();
    let opts = EngineOptions {
        load: LoadOptions { max_seq: 16384, profile_every: 0, ..Default::default() },
        prefix_store: store,
        max_batch: Some(max_batch),
        spec_auto: mtp,
        ..Default::default()
    };
    load_engine(&model_dir(), backend, &opts).unwrap().engine
}

fn host_store() -> Option<PrefixStoreConfig> {
    Some(PrefixStoreConfig { host_bytes: 8 << 30, ..Default::default() })
}

#[derive(Clone)]
struct Out {
    ids: Vec<u32>,
    cached: usize,
    prompt: usize,
    prefill_ms: f64,
    decode_ms: f64,
}

fn chat_prompt(e: &dyn Engine, msgs: &[Value]) -> Vec<u32> {
    let opts = ChatOpts { add_generation_prompt: true, enable_thinking: Some(true), ..Default::default() };
    let text = e.apply_chat_template(msgs, &opts).unwrap();
    e.tokenize(&text, false).unwrap()
}

fn greedy(prompt: &[u32], n: usize) -> GenRequest {
    GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: 0.0, ..Default::default() }
}

/// A seeded sampled request (the Qwen3.6 thinking defaults: temperature 0.6, top-p 0.95, top-k 20), seed per prompt.
fn sampled(prompt: &[u32], n: usize, i: usize) -> GenRequest {
    GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature: 0.6, top_p: 0.95, top_k: 20, seed: Some(1000 + i as u64), ..Default::default() }
}

fn collect(rx: std::sync::mpsc::Receiver<GenEvent>, plen: usize) -> Out {
    let mut ids = vec![];
    for ev in rx.iter() {
        match ev {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { cached_tokens, prefill_ms, decode_ms, .. } => {
                return Out { ids, cached: cached_tokens, prompt: plen, prefill_ms, decode_ms };
            }
            GenEvent::Error { message } => panic!("{message}"),
        }
    }
    panic!("no Done");
}

type Spec = (u64, u64, u64);

/// (verify passes, drafted, accepted) so far.
fn spec(e: &dyn Engine) -> Spec {
    let t = e.telemetry();
    (t.spec_steps, t.spec_proposed, t.spec_accepted)
}

fn add(a: Spec, b: Spec) -> Spec {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

/// One request alone; returns the reply and the request's (passes, drafted, accepted).
fn run_req(e: &dyn Engine, req: GenRequest) -> (Out, Spec) {
    let plen = req.prompt.len();
    let s0 = spec(e);
    let out = collect(e.submit(req).unwrap(), plen);
    let s1 = spec(e);
    (out, (s1.0 - s0.0, s1.1 - s0.1, s1.2 - s0.2))
}

fn rate(s: Spec) -> f64 {
    if s.1 == 0 {
        0.0
    } else {
        s.2 as f64 / s.1 as f64
    }
}

fn tps(o: &Out) -> f64 {
    if o.decode_ms > 0.0 {
        (o.ids.len().saturating_sub(1)) as f64 / (o.decode_ms / 1e3)
    } else {
        0.0
    }
}

fn corpus(skip: usize, chars: usize) -> String {
    let t = include_str!("../../../tools/prompts/corpus_docs.txt");
    t.chars().skip(skip).take(chars).collect()
}

fn line(tag: &str, i: usize, o: &Out, s: Spec) {
    println!(
        "MTP-STORE-METAL {tag} request {i}: prompt {} cached {} | {} tokens, ttft {:.0} ms, decode {:.1} tok/s | {} passes, drafted {}, accepted {} ({:.1} %)",
        o.prompt,
        o.cached,
        o.ids.len(),
        o.prefill_ms,
        tps(o),
        s.0,
        s.1,
        s.2,
        100.0 * rate(s)
    );
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or((a.len() != b.len()).then(|| a.len().min(b.len())))
}

fn summary(tag: &str, s: Spec) {
    println!("MTP-STORE-METAL {tag}: {} passes, acceptance {:.1} % ({} / {})", s.0, 100.0 * rate(s), s.2, s.1);
}

#[test]
fn mtp_on_store_requests_restore_bitwise_metal() {
    if !enabled() {
        return;
    }
    if !model_dir().join("config.json").exists() {
        eprintln!("SKIP: no model at {}", model_dir().display());
        return;
    }
    if !intarsia_core::spec::mtp_store_for("metal") {
        eprintln!("SKIP: MTP drafting on store requests is off for Metal (set INTARSIA_MTP_STORE=1)");
        return;
    }
    let gen = gen_tokens();
    let rounds = 3;
    let sys = json!({"role": "system", "content": "You are a careful reading companion. Keep answers short."});
    let first = [
        format!("{}\n\nWhat is this passage about?", corpus(1000, 6000)),
        format!("{}\n\nSummarise.", corpus(40_000, 4000)),
    ];
    let follow = [
        [format!("{}\n\nAnd this one?", corpus(9000, 2000)), format!("{}\n\nCompare the two passages.", corpus(15_000, 1500))],
        [format!("{}\n\nAnd this one?", corpus(50_000, 2500)), "Which of the two was clearer, and why?".to_string()],
    ];
    let t0 = Instant::now();
    println!("MTP-STORE-METAL model {} gen {gen}", model_dir().display());

    // 1. the reference (store on, spec off); its greedy replies build the later turns' prompts for every arm
    let off = engine(host_store(), false, 1);
    let mut chats: Vec<Vec<Value>> = first.iter().map(|u| vec![sys.clone(), json!({"role": "user", "content": u})]).collect();
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut want: Vec<Out> = Vec::new();
    for round in 0..rounds {
        for c in 0..chats.len() {
            let p = chat_prompt(&*off, &chats[c]);
            let (o, s) = run_req(&*off, greedy(&p, gen));
            assert_eq!(s, (0, 0, 0), "the reference does not draft");
            line("ref store-on mtp-off greedy", prompts.len(), &o, s);
            if round + 1 < rounds {
                chats[c].push(json!({"role": "assistant", "content": off.detokenize(&o.ids).unwrap()}));
                chats[c].push(json!({"role": "user", "content": &follow[c][round]}));
            }
            prompts.push(p);
            want.push(o);
        }
    }
    let n_first = chats.len();
    let later: Vec<usize> = (n_first..prompts.len()).collect();
    assert!(want[n_first..].iter().all(|o| o.cached > o.prompt / 2), "the later turns restore: {:?}", want.iter().map(|o| (o.cached, o.prompt)).collect::<Vec<_>>());
    let mut want_s: Vec<Option<Out>> = vec![None; prompts.len()];
    for &i in &later {
        let (o, _) = run_req(&*off, sampled(&prompts[i], gen, i));
        line("ref store-on mtp-off sampled", i, &o, (0, 0, 0));
        want_s[i] = Some(o);
    }
    drop(off);
    println!("MTP-STORE-METAL reference done at {:.0} s", t0.elapsed().as_secs_f64());

    // 2. single-sequence engine, store on, the recipe's MTP default
    let on = engine(host_store(), true, 1);
    let mut solo_ok = true;
    let (mut cold, mut restored) = ((0, 0, 0), (0, 0, 0));
    for (i, (p, w)) in prompts.iter().zip(&want).enumerate() {
        let (o, s) = run_req(&*on, greedy(p, gen));
        line("single store-on mtp-on greedy", i, &o, s);
        let eq = o.ids == w.ids && o.cached == w.cached;
        if !eq {
            println!("MTP-STORE-METAL MISMATCH single greedy request {i}: first differing token {:?}, cached {} vs {}", first_diff(&o.ids, &w.ids), o.cached, w.cached);
        }
        solo_ok &= eq;
        let acc = if i < n_first { &mut cold } else { &mut restored };
        *acc = add(*acc, s);
    }
    let mut warm_s: Vec<Option<Out>> = vec![None; prompts.len()];
    let mut warm_s_spec = (0, 0, 0);
    for &i in &later {
        let (o, s) = run_req(&*on, sampled(&prompts[i], gen, i));
        line("single store-on mtp-on sampled (restored)", i, &o, s);
        warm_s_spec = add(warm_s_spec, s);
        warm_s[i] = Some(o);
    }
    let snap = on.telemetry().prefix_store.unwrap();
    println!("MTP-STORE-METAL single store: {}", serde_json::to_string(&snap).unwrap());
    drop(on);
    summary("single greedy cold turns", cold);
    summary("single greedy restored turns (a new turn after the restore point)", restored);
    summary("single sampled restored (all but the last tokens restored)", warm_s_spec);

    // 3. cold, per later prompt: a fresh single and a fresh batched engine run the sampled request cold, then the greedy one
    let mut cold_s: Vec<Option<Out>> = vec![None; prompts.len()];
    let mut cold_bs: Vec<Option<Out>> = vec![None; prompts.len()];
    let mut cold_ok = true;
    let (mut c_s_spec, mut c_g_spec) = ((0, 0, 0), (0, 0, 0));
    for &i in &later {
        for mb in [1usize, 2] {
            let e = engine(host_store(), true, mb);
            let (o, s) = run_req(&*e, sampled(&prompts[i], gen, i));
            line(&format!("cold mb{mb} sampled"), i, &o, s);
            assert_eq!(o.cached, 0, "a fresh engine runs cold");
            if mb == 1 {
                c_s_spec = add(c_s_spec, s);
            }
            let (g, gs) = run_req(&*e, greedy(&prompts[i], gen));
            line(&format!("cold mb{mb} greedy after the same prompt (all but the last tokens restored)"), i, &g, gs);
            if mb == 1 {
                c_g_spec = add(c_g_spec, gs);
            }
            if g.ids != want[i].ids {
                println!("MTP-STORE-METAL MISMATCH cold-engine mb{mb} greedy request {i}: first differing token {:?}", first_diff(&g.ids, &want[i].ids));
                cold_ok = false;
            }
            if mb == 1 {
                cold_s[i] = Some(o);
            } else {
                cold_bs[i] = Some(o);
            }
        }
    }
    summary("single sampled cold", c_s_spec);
    summary("single greedy, all but the last tokens restored", c_g_spec);
    let mut sampled_ok = true;
    for &i in &later {
        let (w, c) = (warm_s[i].as_ref().unwrap(), cold_s[i].as_ref().unwrap());
        let eq = w.ids == c.ids;
        if !eq {
            println!("MTP-STORE-METAL MISMATCH single sampled request {i}: restored vs cold first differing token {:?}", first_diff(&w.ids, &c.ids));
        }
        let r = want_s[i].as_ref().unwrap();
        println!(
            "MTP-STORE-METAL single sampled request {i}: restored (cached {}) == cold {eq}; cold == the plain spec-off sampler {}",
            w.cached,
            c.ids == r.ids
        );
        sampled_ok &= eq;
    }
    // controls: a second fresh engine runs the first later prompt's sampled request cold (cold == cold), a third runs its
    // greedy request cold first (greedy restored == cold with MTP on, directly)
    let i0 = later[0];
    let e = engine(host_store(), true, 1);
    let (o, s) = run_req(&*e, sampled(&prompts[i0], gen, i0));
    line("control: second cold mb1 sampled", i0, &o, s);
    let cold_twice = o.ids == cold_s[i0].as_ref().unwrap().ids;
    drop(e);
    let e = engine(host_store(), true, 1);
    let (o, s) = run_req(&*e, greedy(&prompts[i0], gen));
    line("control: cold mb1 greedy", i0, &o, s);
    assert_eq!(o.cached, 0, "a fresh engine runs cold");
    let greedy_cold_ok = o.ids == want[i0].ids;
    drop(e);
    println!("MTP-STORE-METAL control request {i0}: sampled cold == cold {cold_twice}; greedy cold (MTP on) == the restored reference {greedy_cold_ok}");

    // 4. store off, MTP on: the head follows every prompt from position 0 (reported, not asserted)
    let full = engine(None, true, 1);
    let (mut f_cold, mut f_later) = ((0, 0, 0), (0, 0, 0));
    for (i, (p, w)) in prompts.iter().zip(&want).enumerate() {
        let (o, s) = run_req(&*full, greedy(p, gen));
        line("store-off mtp-on greedy", i, &o, s);
        println!("MTP-STORE-METAL store-off request {i}: ids equal the store-on reference {}", o.ids == w.ids);
        let acc = if i < n_first { &mut f_cold } else { &mut f_later };
        *acc = add(*acc, s);
    }
    drop(full);
    summary("store-off (whole head context) first turns", f_cold);
    summary("store-off (whole head context) later turns", f_later);

    // 5. --max-batch 2, store on, MTP
    let b = engine(host_store(), true, 2);
    let mut batched_ok = true;
    let s0 = spec(&*b);
    for round in 0..rounds {
        let idx: Vec<usize> = (0..n_first).map(|c| round * n_first + c).collect();
        let rxs: Vec<_> = idx.iter().map(|&i| (i, b.submit(greedy(&prompts[i], gen)).unwrap())).collect();
        for (i, rx) in rxs {
            let o = collect(rx, prompts[i].len());
            let eq = o.ids == want[i].ids;
            println!(
                "MTP-STORE-METAL batched together greedy request {i}: prompt {} cached {} | ttft {:.0} ms, decode {:.1} tok/s | ids equal the reference {eq}",
                o.prompt,
                o.cached,
                o.prefill_ms,
                tps(&o)
            );
            batched_ok &= eq;
        }
    }
    let s1 = spec(&*b);
    let together = (s1.0 - s0.0, s1.1 - s0.1, s1.2 - s0.2);
    let mut alone = (0, 0, 0);
    let mut alone_cached = true;
    for (i, (p, w)) in prompts.iter().zip(&want).enumerate() {
        let (o, s) = run_req(&*b, greedy(p, gen));
        line("batched alone greedy", i, &o, s);
        if o.ids != w.ids {
            println!("MTP-STORE-METAL MISMATCH batched greedy request {i}: first differing token {:?}", first_diff(&o.ids, &w.ids));
            batched_ok = false;
        }
        alone_cached &= o.cached > o.prompt / 2;
        alone = add(alone, s);
    }
    let mut b_s_spec = (0, 0, 0);
    let mut b_sampled_ok = true;
    for &i in &later {
        let (o, s) = run_req(&*b, sampled(&prompts[i], gen, i));
        line("batched alone sampled (restored)", i, &o, s);
        b_s_spec = add(b_s_spec, s);
        let c = cold_bs[i].as_ref().unwrap();
        let eq = o.ids == c.ids;
        if !eq {
            println!("MTP-STORE-METAL MISMATCH batched sampled request {i}: restored vs cold first differing token {:?}", first_diff(&o.ids, &c.ids));
        }
        let r = want_s[i].as_ref().unwrap();
        println!("MTP-STORE-METAL batched sampled request {i}: restored (cached {}) == cold {eq}; == the plain spec-off sampler {}", o.cached, o.ids == r.ids);
        b_sampled_ok &= eq;
    }
    let bsnap = b.telemetry().prefix_store.unwrap();
    println!("MTP-STORE-METAL batched store: {}", serde_json::to_string(&bsnap).unwrap());
    drop(b);
    summary("batched together", together);
    summary("batched alone greedy", alone);
    summary("batched alone sampled", b_s_spec);
    let b_drafts = alone.1 > 0;
    if !b_drafts {
        println!("MTP-STORE-METAL NOTE batched engine did not draft (batched MTP off for this model: INTARSIA_BATCHED_MTP=1 turns it on)");
    }

    // 6. restart: the disk tier survives a new engine
    let dir = store_dir().join(format!("restart-{}", model_dir().file_name().unwrap().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let disk = Some(PrefixStoreConfig { host_bytes: 8 << 30, disk_bytes: 64 << 30, disk_dir: Some(dir.clone()), ..Default::default() });
    {
        let e = engine(disk.clone(), true, 1);
        let (o, s) = run_req(&*e, greedy(&prompts[0], gen));
        line("restart: first engine greedy", 0, &o, s);
        let w0 = Instant::now();
        loop {
            let sn = e.telemetry().prefix_store.unwrap();
            if (sn.disk.bytes > 0 && sn.disk_pending_bytes == 0) || w0.elapsed() > Duration::from_secs(60) {
                println!("MTP-STORE-METAL restart: disk tier {} MiB, {} objects, pending {} after {:.1} s", sn.disk.bytes >> 20, sn.disk.objects, sn.disk_pending_bytes, w0.elapsed().as_secs_f64());
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    let e = engine(disk, true, 1);
    let (o, s) = run_req(&*e, greedy(&prompts[n_first], gen));
    line("restart: second engine greedy (restored from disk)", n_first, &o, s);
    let rsnap = e.telemetry().prefix_store.unwrap();
    println!("MTP-STORE-METAL restart store: {}", serde_json::to_string(&rsnap).unwrap());
    drop(e);
    let restart_ok = o.ids == want[n_first].ids;
    if !restart_ok {
        println!("MTP-STORE-METAL MISMATCH restart request {n_first}: first differing token {:?}", first_diff(&o.ids, &want[n_first].ids));
    }
    let _ = std::fs::remove_dir_all(&dir);
    println!(
        "MTP-STORE-METAL verdict: single greedy {solo_ok}; cold-engine greedy {cold_ok}; single sampled restored == cold {sampled_ok}; batched greedy {batched_ok}; batched sampled restored == cold {b_sampled_ok}; restart {restart_ok} (cached {}, disk hits {})",
        o.cached, rsnap.disk.hits
    );
    println!("MTP-STORE-METAL total {:.0} s", t0.elapsed().as_secs_f64());

    assert!(solo_ok, "single-sequence engine: a store-on MTP-on greedy reply differs from the store-on MTP-off reply");
    assert!(snap.hits as usize >= later.len(), "single-sequence engine: the drafting requests restored: {snap:?}");
    assert!(restored.1 > 0 && restored.2 > 0, "single-sequence engine: the restored turns drafted and accepted: {restored:?}");
    assert!(greedy_cold_ok, "greedy: a cold MTP-on reply differs from the restored reference");
    assert!(cold_ok, "a fresh engine's greedy reply after a restore of all but its last tokens differs from the reference");
    assert!(cold_twice, "control: two cold seeded sampled runs differ (the sampled path is not deterministic; not a store property)");
    assert!(sampled_ok, "single-sequence engine: a restored seeded sampled reply differs from the cold one");
    assert!(batched_ok, "batched engine: a store-on MTP-on greedy reply differs from the reference");
    assert!(alone_cached && bsnap.hits > 0, "batched engine: the requests restored: {bsnap:?}");
    assert!(b_sampled_ok, "batched engine: a restored seeded sampled reply differs from the cold one");
    assert!(o.cached > 0 && rsnap.disk.hits > 0, "restart: the second engine restored from disk: cached {}, {rsnap:?}", o.cached);
    assert!(restart_ok, "restart: the reply restored from disk differs from the reference");
    assert!(snap.mtp_head_restores > 0 && rsnap.mtp_head_restores > 0, "lane mhkv: the restores brought the MTP head back (single {}, restart {})", snap.mtp_head_restores, rsnap.mtp_head_restores);
}

/// Lane mhkv (docs/results/2026-11-16-mhkv.md): the entry keeps the MTP head's cache and queued row, so a restored request
/// drafts as the same request cold. Lane mstore's 4k window case (`scripts/mstore/prompts.py`: P = system + a 4.5k-token
/// document + a question; Q = P + a fixed reply + a new question; thinking off; 256 tokens, `ignore_eos`): a fresh engine
/// runs Q cold; another runs P, an unrelated chat (it replaces the held sequence), then Q, which restores P from the store.
/// Greedy and seeded sampled: the ids are equal, and the acceptance after the restore is within `INTARSIA_TEST_ACC_TOL`
/// points (default 2) of cold. Before lane mhkv: 60 % cold, 42 % restored on the 35B.
#[test]
fn mtp_store_head_acceptance_4k_metal() {
    if !enabled() {
        return;
    }
    if !model_dir().join("config.json").exists() {
        eprintln!("SKIP: no model at {}", model_dir().display());
        return;
    }
    if !intarsia_core::spec::mtp_store_for("metal") {
        eprintln!("SKIP: MTP drafting on store requests is off for Metal (set INTARSIA_MTP_STORE=1)");
        return;
    }
    let tol: f64 = std::env::var("INTARSIA_TEST_ACC_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
    let docs = include_str!("../../../tools/prompts/corpus_docs.txt");
    let code = include_str!("../../../tools/prompts/corpus_code.txt");
    let text: String = format!("{docs}\n\n{code}").chars().take(15000).collect();
    let sys = json!({"role": "system", "content": "You are a careful technical reader. Answer from the material the user gives you."});
    let a1 = "The material describes how the project is organised and how its parts are documented: an overview of the goals, a walk through the main components and how they talk to each other, the configuration knobs with their defaults, and the procedures for building, testing and measuring a change. It closes with source code that implements part of what the documents describe. The tone is practical: each section says what to run and what to expect.";
    let q2 = "Now list eight specific details from the material above, one per line. For each, quote the relevant phrase and explain in two sentences why it matters to someone changing the code.";
    let p_msgs = vec![sys, json!({"role": "user", "content": format!("{text}\n\nSummarise the material above in one paragraph.")})];
    let mut q_msgs = p_msgs.clone();
    q_msgs.push(json!({"role": "assistant", "content": a1}));
    q_msgs.push(json!({"role": "user", "content": q2}));
    let u_msgs = vec![json!({"role": "user", "content": format!("{}\n\nWhat is this text about?", corpus(60_000, 3000))})];
    let nothink = |e: &dyn Engine, msgs: &[Value]| -> Vec<u32> {
        let opts = ChatOpts { add_generation_prompt: true, enable_thinking: Some(false), ..Default::default() };
        e.tokenize(&e.apply_chat_template(msgs, &opts).unwrap(), false).unwrap()
    };
    let n = 256;
    let g = |p: &[u32]| GenRequest { ignore_eos: true, ..greedy(p, n) };
    let sm = |p: &[u32]| GenRequest { ignore_eos: true, ..sampled(p, n, 7) };
    let t0 = Instant::now();
    let mut ok = true;
    for (kind, mk) in [("greedy", &g as &dyn Fn(&[u32]) -> GenRequest), ("sampled", &sm)] {
        let cold = engine(host_store(), true, 1);
        let q = nothink(&*cold, &q_msgs);
        let (c, cs) = run_req(&*cold, mk(&q));
        line(&format!("4k cold {kind}"), 0, &c, cs);
        assert_eq!(c.cached, 0);
        drop(cold);
        let e = engine(host_store(), true, 1);
        let p = nothink(&*e, &p_msgs);
        let u = nothink(&*e, &u_msgs);
        let (o, s) = run_req(&*e, mk(&p));
        line(&format!("4k P {kind}"), 0, &o, s);
        let (o, s) = run_req(&*e, greedy(&u, 32));
        line("4k U greedy", 0, &o, s);
        let (r, rs) = run_req(&*e, mk(&q));
        line(&format!("4k restored {kind}"), 0, &r, rs);
        let sn = e.telemetry().prefix_store.unwrap();
        drop(e);
        let (ac, ar) = (100.0 * rate(cs), 100.0 * rate(rs));
        let same = r.ids == c.ids;
        println!(
            "MTP-STORE-METAL 4k {kind}: ids restored == cold {same}; cached {} of {}; acceptance cold {ac:.1} % restored {ar:.1} % (delta {:+.1} points); head restores {}",
            r.cached,
            r.prompt,
            ar - ac,
            sn.mtp_head_restores
        );
        if !same {
            println!("MTP-STORE-METAL MISMATCH 4k {kind}: first differing token {:?}", first_diff(&r.ids, &c.ids));
        }
        ok &= same && r.cached > 4000 && sn.mtp_head_restores > 0 && (ar - ac).abs() <= tol && cs.1 > 0;
    }
    println!("MTP-STORE-METAL 4k verdict {ok} ({:.0} s)", t0.elapsed().as_secs_f64());
    assert!(ok, "4k: restored ids == cold, the head restored, acceptance within {tol} points of cold (see the lines above)");
}

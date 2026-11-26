#![allow(dead_code)]
//! Shared by `tests/inkling2_dspark_sampled.rs` and `tests/inkling2_dspark_sampled_gumbel.rs` (lane ink-sampled): sampled requests
//! (temperature > 0) drafting with Inkling-2's DSpark draft (`INTARSIA_SPEC_SAMPLED_DFLASH`, `Model::set_spec_sampled_dflash`) on the
//! tiny fixtures of `tests/inkling2_dspark.rs` (`golden/inkling2-tiny`, `golden/inkling2-tiny-dspark`).

use intarsia_core::model::{DraftScript, LoadOptions, Model};
use intarsia_core::recipe::ModelDims;
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{Engine, EngineMeta, GenEvent, GenRequest, LocalEngine, Telemetry};
use intarsia_models::safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub fn dirs() -> Option<(PathBuf, PathBuf)> {
    let g = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden");
    let (t, d) = (g.join("inkling2-tiny"), g.join("inkling2-tiny-dspark"));
    (t.join("config.json").exists() && d.join("mtp.safetensors").exists()).then_some((t, d))
}

pub fn backend() -> Arc<dyn intarsia_core::Backend> {
    match std::env::var("INTARSIA_INKLING2_BACKEND").as_deref() {
        #[cfg(target_os = "macos")]
        Ok("metal") => intarsia_backend_metal::MetalBackend::arc().unwrap(),
        Ok("cuda") => intarsia_backend_cuda::create(0).unwrap(),
        Ok("hip") => intarsia_backend_hip::create(0).unwrap(),
        Ok("cpu") | Err(_) => intarsia_backend_cpu::CpuBackend::arc(),
        Ok(other) => panic!("INTARSIA_INKLING2_BACKEND={other}: cpu, metal, cuda or hip"),
    }
}

/// The draft's env knobs are read at load: every test of a binary sets the same values (no per-step gate, no cut, so the draft
/// proposes `k` tokens every pass, as `tests/inkling2_dspark.rs` does).
pub fn gate_off() {
    static ONCE: Mutex<()> = Mutex::new(());
    let _g = ONCE.lock().unwrap();
    std::env::set_var("INTARSIA_DSPARK_TAU", "0");
    std::env::set_var("INTARSIA_DSPARK_CUT", "0");
}

/// The tiny engine: `--spec dflash --spec-k k` with the DSpark draft (`k: None`: `--spec off`, no draft), sampled DFlash speculation
/// set to `sampled` (`Model::set_spec_sampled_dflash`), the draft's proposals replaced by `script` where given.
pub fn engine(k: Option<usize>, sampled: Option<bool>, script: Option<DraftScript>) -> (LocalEngine, Arc<Telemetry>) {
    let (m, tel) = model(k, sampled, script);
    let spec = match k {
        Some(k) => SpecConfig { mode: SpecMode::Dflash, k, k_min: None },
        None => SpecConfig::default(),
    };
    let e = LocalEngine::new(m, Arc::new(super::common::ByteCodec), EngineMeta { id: "tiny-dspark".into(), eos_ids: vec![], spec, ..Default::default() }, tel.clone())
        .unwrap();
    (e, tel)
}

/// The model of [`engine`].
pub fn model(k: Option<usize>, sampled: Option<bool>, script: Option<DraftScript>) -> (Model, Arc<Telemetry>) {
    let (tdir, ddir) = dirs().unwrap();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../../recipes/inkling2.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tdir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    let src = SafeTensors::open_dir(&tdir).unwrap();
    let dflash = k.map(|_| intarsia_models::load::open_dflash(Some(&ddir), false).unwrap());
    let tel = Arc::new(Telemetry::new());
    let opts = LoadOptions { max_seq: 160, prefill_chunk: 16, graphs: false, dflash, ..Default::default() };
    let mut m = Model::load(backend(), &recipe, &dims, &src, &opts, tel.clone()).unwrap();
    if let Some(k) = k {
        m.enable_verify(k + 1).unwrap();
        m.configure_plan(&Default::default(), Some(&*intarsia_backend_cpu::CpuBackend::arc())).unwrap();
    }
    m.set_spec_sampled_dflash(sampled);
    if script.is_some() {
        m.dflash_script(script).unwrap();
    }
    (m, tel)
}

pub const PROMPTS: [&[u32]; 3] = [
    &[3, 9, 27, 81, 50, 7, 7, 1, 60, 2, 33, 33, 12],
    &[90, 1, 2, 3, 4, 5],
    &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 11, 21, 31, 41],
];

/// The sampling configurations: (temperature, top_k, top_p). The first two are the recipe's own (the checkpoint sets no top-k / top-p,
/// so serving takes top-k 0, top-p 1.0: the device sampler's rows on Metal); the third takes the host sampler everywhere.
pub const CFGS: [(f32, usize, f32); 3] = [(1.0, 0, 1.0), (0.6, 0, 1.0), (0.7, 20, 0.9)];

pub fn sreq(prompt: &[u32], n: usize, seed: u64, (temperature, top_k, top_p): (f32, usize, f32)) -> GenRequest {
    GenRequest { prompt: prompt.to_vec(), max_tokens: n, temperature, top_k, top_p, seed: Some(seed), ignore_eos: true, ..Default::default() }
}

pub fn run(e: &LocalEngine, req: GenRequest) -> Vec<u32> {
    let mut ids = vec![];
    e.generate(req, &mut |ev| match ev {
        GenEvent::Token { id, .. } => ids.push(id),
        GenEvent::Error { message } => panic!("engine error: {message}"),
        _ => {}
    })
    .unwrap();
    ids
}

/// A script that proposes the tokens of `full` (a prompt and its seeded spec-off continuation, by absolute position; the caller swaps it
/// between requests) for a varying number of tokens, then a wrong one: every accepted length 0..=k happens while the request follows
/// the reference, and once it leaves it the proposals are the reference's tokens at those positions (mostly wrong, sometimes right).
pub fn partial_script(full: Arc<Mutex<Vec<u32>>>) -> DraftScript {
    let step = Mutex::new(0usize);
    Box::new(move |seq: &[u32], kk: usize| {
        let mut s = step.lock().unwrap();
        *s += 1;
        let right = *s % (kk + 1);
        let at = seq.len();
        let full = full.lock().unwrap();
        let mut d: Vec<u32> = (0..kk).filter_map(|j| full.get(at + j).copied()).collect();
        if right < d.len() {
            d[right] = (d[right] + 1) % 250;
        }
        d
    })
}

/// Gate 2 on the fixture: for every sampling configuration, prompt, `max_tokens` and seed, the ids with sampled DSpark speculation on
/// equal `--spec off`'s (and the spec-on engine reproduces them); with the random tiny draft and with scripted proposals that are
/// partly right. Returns (verify passes, proposed, accepted) of the scripted engines.
pub fn seeded_ids_equal_spec_off(label: &str) -> (u64, u64, u64) {
    let (off, _) = engine(None, None, None);
    let (on, tel_on) = engine(Some(3), Some(true), None);
    for cfg in CFGS {
        for p in PROMPTS {
            for n in [1usize, 5, 40] {
                for seed in [1u64, 7] {
                    let a = run(&off, sreq(p, n, seed, cfg));
                    let b = run(&on, sreq(p, n, seed, cfg));
                    assert_eq!(a, b, "{label}: random draft, cfg {cfg:?} n {n} seed {seed} prompt {p:?}");
                    // (a held-back text tail goes out after the last token as one more `Token` event repeating the last id)
                    assert!(a.len() == n || (a.len() == n + 1 && a[n] == a[n - 1]), "{label}: {} ids for max_tokens {n}", a.len());
                }
            }
        }
    }
    let t = tel_on.snapshot();
    assert!(t.spec_steps > 0, "{label}: sampled requests never drafted (vacuous)");
    eprintln!("{label}: random draft: passes {} proposed {} accepted {}", t.spec_steps, t.spec_proposed, t.spec_accepted);
    let (mut steps, mut prop, mut acc) = (0, 0, 0);
    for k in [3usize, 7] {
        let full = Arc::new(Mutex::new(Vec::new()));
        let (sc, tel) = engine(Some(k), Some(true), Some(partial_script(full.clone())));
        for cfg in CFGS {
            for p in PROMPTS {
                let seed = 11;
                let reference = run(&off, sreq(p, 48, seed, cfg));
                *full.lock().unwrap() = p.iter().chain(&reference).copied().collect();
                assert_eq!(run(&sc, sreq(p, 48, seed, cfg)), reference, "{label}: scripted k {k} cfg {cfg:?} prompt {p:?}");
                // another seed with the same proposals: they are mostly wrong for it
                assert_eq!(run(&sc, sreq(p, 30, 12, cfg)), run(&off, sreq(p, 30, 12, cfg)), "{label}: scripted k {k} cfg {cfg:?} seed 12");
            }
        }
        let t = tel.snapshot();
        steps += t.spec_steps;
        prop += t.spec_proposed;
        acc += t.spec_accepted;
    }
    assert!(acc > 0 && acc < prop, "{label}: scripted proposals must be both accepted and rejected ({acc} of {prop})");
    eprintln!("{label}: scripted: passes {steps} proposed {prop} accepted {acc}");
    (steps, prop, acc)
}

/// `Q(a, x)`, the upper regularized incomplete gamma function (Numerical Recipes `gammq`): the chi-square survival function is
/// `Q(df / 2, x / 2)`.
pub fn gammq(a: f64, x: f64) -> f64 {
    fn ln_gamma(z: f64) -> f64 {
        // Lanczos, g = 7
        const C: [f64; 9] = [
            0.999_999_999_999_809_9,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_311_6e-7,
        ];
        let z = z - 1.0;
        let mut s = C[0];
        for (i, c) in C.iter().enumerate().skip(1) {
            s += c / (z + i as f64);
        }
        let t = z + 7.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + s.ln()
    }
    if x <= 0.0 {
        return 1.0;
    }
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        // series for P, Q = 1 - P
        let (mut ap, mut sum) = (a, 1.0 / a);
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - gln).exp()
    } else {
        // continued fraction (Lentz)
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (-x + a * x.ln() - gln).exp() * h
    }
}

/// Pearson chi-square of homogeneity of two count tables (2 x C): tokens whose pooled expected count is at least 5 in the smaller
/// arm are cells of their own; the others are grouped into 16 bins by `id % 16` (a fixed function of the token, so the binned laws
/// are equal when the laws are), and a bin whose expected count is still below 5 joins one remainder cell. Returns (statistic,
/// degrees of freedom, p-value).
pub fn chi2_homogeneity(a: &BTreeMap<u32, u64>, b: &BTreeMap<u32, u64>) -> (f64, usize, f64) {
    let (na, nb) = (a.values().sum::<u64>() as f64, b.values().sum::<u64>() as f64);
    let n = na + nb;
    let small = |tot: f64| tot * na.min(nb) / n < 5.0;
    let keys: std::collections::BTreeSet<u32> = a.keys().chain(b.keys()).copied().collect();
    let mut cells: Vec<[f64; 2]> = Vec::new();
    let mut bins = [[0.0f64; 2]; 16];
    for k in &keys {
        let o = [*a.get(k).unwrap_or(&0) as f64, *b.get(k).unwrap_or(&0) as f64];
        if small(o[0] + o[1]) {
            let bin = &mut bins[(*k % 16) as usize];
            bin[0] += o[0];
            bin[1] += o[1];
        } else {
            cells.push(o);
        }
    }
    let mut rest = [0.0f64; 2];
    for bin in bins {
        if small(bin[0] + bin[1]) {
            rest[0] += bin[0];
            rest[1] += bin[1];
        } else {
            cells.push(bin);
        }
    }
    if rest[0] + rest[1] > 0.0 {
        cells.push(rest);
    }
    let mut stat = 0.0;
    for o in &cells {
        let tot = o[0] + o[1];
        let e = [tot * na / n, tot * nb / n];
        stat += (o[0] - e[0]).powi(2) / e[0] + (o[1] - e[1]).powi(2) / e[1];
    }
    let df = cells.len().saturating_sub(1);
    let p = if df == 0 { 1.0 } else { gammq(df as f64 / 2.0, stat / 2.0) };
    (stat, df, p)
}

/// Gate 1 on the fixture: the law of each generated token 2..=4 (the speculative positions; token 1 comes from prefill) with sampled
/// DSpark speculation on equals plain sampling's, at each configuration of [`CFGS`] (`cfgs` picks them), `n` requests per arm with
/// disjoint seed sets. The drafts are scripted to the most frequent token of a spec-off pilot at each position (a fixed proposal per
/// position, so acceptance is that token's probability: both branches run). Returns one line per (cfg, position): the statistic, df,
/// p-value, the two arms' counts and the acceptance.
pub fn law_test(label: &str, n: u64, cfgs: &[(f32, usize, f32)]) -> Vec<String> {
    let prompt = PROMPTS[0];
    let positions = 3usize; // generated tokens 2..=4
    let (off, _) = engine(None, None, None);
    let mut lines = Vec::new();
    let mut ps = Vec::new();
    for &cfg in cfgs {
        let draw = |e: &LocalEngine, seed0: u64, count: u64| -> Vec<BTreeMap<u32, u64>> {
            let mut c = vec![BTreeMap::new(); positions];
            for s in seed0..seed0 + count {
                let ids = run(e, sreq(prompt, 1 + positions, s, cfg));
                for p in 0..positions {
                    *c[p].entry(ids[1 + p]).or_insert(0) += 1;
                }
            }
            c
        };
        let pilot = draw(&off, 5_000_000, 300);
        let mode: Vec<u32> = pilot.iter().map(|m| *m.iter().max_by_key(|(_, &c)| c).unwrap().0).collect();
        let plen = prompt.len();
        let script: DraftScript = Box::new(move |seq: &[u32], kk: usize| {
            // the drafts start at absolute position seq.len(); generated token g (1-based) sits at plen + g - 1
            (0..kk).map(|j| seq.len() + j).map(|at| at.checked_sub(plen + 1).and_then(|i| mode.get(i).copied()).unwrap_or(5)).collect()
        });
        let (on, tel) = engine(Some(3), Some(true), Some(script));
        let base = draw(&off, 0, n);
        let got = draw(&on, 1_000_000, n);
        let t = tel.snapshot();
        assert!(t.spec_steps > 0 && t.spec_accepted > 0 && t.spec_accepted < t.spec_proposed, "{label} cfg {cfg:?}: both branches must run ({t:?})");
        for pos in 0..positions {
            let (stat, df, p) = chi2_homogeneity(&base[pos], &got[pos]);
            let line = format!(
                "{label} T {} top_k {} top_p {} position {}: chi2 {stat:.2} df {df} p {p:.4}; n {n} + {n}; distinct tokens {} / {}; passes {} accepted {} of {} proposed",
                cfg.0,
                cfg.1,
                cfg.2,
                pos + 2,
                base[pos].len(),
                got[pos].len(),
                t.spec_steps,
                t.spec_accepted,
                t.spec_proposed
            );
            eprintln!("{line}");
            assert!(df >= 2, "{line}: too few cells for the test");
            lines.push(line);
            ps.push(p);
        }
    }
    // Fisher's combination over the positions (they are dependent within a request, so this is a summary, not an exact test): the
    // gate is no single p below 1e-3 and the combined p above 1e-3
    let x2: f64 = ps.iter().map(|p| -2.0 * p.max(1e-300).ln()).sum();
    let pf = gammq(ps.len() as f64, x2 / 2.0);
    let line = format!("{label}: Fisher over {} tests X2 {x2:.2} df {} p {pf:.4}; min p {:.4}", ps.len(), 2 * ps.len(), ps.iter().cloned().fold(1.0, f64::min));
    eprintln!("{line}");
    assert!(ps.iter().all(|&p| p > 1e-3) && pf > 1e-3, "{line}: the law differs");
    lines.push(line);
    lines
}

//! Constrained decoding hooks (`GenRequest::constraint`, docs/structured-output.md) on the tiny Qwen3-shaped fixture,
//! cpu backend, single-sequence and batched engines: the mask decides every constrained token, a complete grammar ends
//! the request on EOS, a lazy constraint leaves the unconstrained path alone until its trigger, speculation is off for
//! constrained requests, and unconstrained requests (alone or batched next to constrained ones) are unchanged.
mod common;
use common::*;
use intarsia_core::constrain::{ConstraintSpec, TokenConstraint, TokenMask};
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::spec::{SpecConfig, SpecMode};
use intarsia_core::{
    BatchedEngine, BatchedOptions, DType, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, Result,
    Telemetry,
};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const EOS: u32 = 299;
const VOCAB: usize = 300;

fn model(seed: u64, rows: usize, spec_k: Option<usize>, tel: &Arc<Telemetry>) -> (Model, SpecConfig) {
    let t = tiny_weights(DType::Bf16, seed);
    let mut m = Model::load(
        intarsia_backend_cpu::CpuBackend::arc(),
        &recipe(),
        &dims(),
        &t.weights,
        &LoadOptions { max_seq: 160, prefill_chunk: 16, ..Default::default() },
        tel.clone(),
    )
    .unwrap();
    let spec = match spec_k {
        Some(k) => {
            m.enable_verify(k + 1).unwrap();
            SpecConfig { mode: SpecMode::Ngram, k }
        }
        None => SpecConfig::default(),
    };
    if rows > 1 {
        m.enable_batch(rows).unwrap();
    }
    m.configure_plan(&Default::default(), None).unwrap();
    (m, spec)
}

fn meta(spec: SpecConfig) -> EngineMeta {
    EngineMeta { id: "tiny".into(), eos_ids: vec![EOS], spec, ..Default::default() }
}

fn solo(seed: u64, spec_k: Option<usize>) -> (LocalEngine, Arc<Telemetry>) {
    let tel = Arc::new(Telemetry::new());
    let (m, spec) = model(seed, 1, spec_k, &tel);
    (LocalEngine::new(m, Arc::new(ByteCodec), meta(spec), tel.clone()).unwrap(), tel)
}

fn batched(seed: u64, rows: usize) -> BatchedEngine {
    let tel = Arc::new(Telemetry::new());
    let (m, spec) = model(seed, rows, None, &tel);
    let mut o = BatchedOptions::new(rows, 16);
    o.prefix_reuse = false;
    BatchedEngine::new(m, Arc::new(ByteCodec), meta(spec), tel, o).unwrap()
}

#[derive(Debug, PartialEq)]
struct Out {
    ids: Vec<u32>,
    reason: FinishReason,
    completion_tokens: usize,
}

fn collect(rx: &Receiver<GenEvent>) -> std::result::Result<Out, String> {
    let mut ids = vec![];
    loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("engine stalled") {
            GenEvent::Token { id, .. } => ids.push(id),
            GenEvent::Done { finish_reason, completion_tokens, .. } => {
                // drop the text-only tail event (it repeats the last id)
                let real = completion_tokens - usize::from(finish_reason == FinishReason::Stop);
                ids.truncate(real);
                return Ok(Out { ids, reason: finish_reason, completion_tokens });
            }
            GenEvent::Error { message } => return Err(message),
        }
    }
}

fn run(e: &dyn Engine, req: GenRequest) -> std::result::Result<Out, String> {
    collect(&e.submit(req).unwrap())
}

/// Allows the tokens of `allowed`; after `len` tokens only EOS. `trigger`: inactive until that token. Records the
/// masks it was asked for (how many steps were constrained).
#[derive(Debug)]
struct SetSpec {
    allowed: Vec<u32>,
    len: usize,
    trigger: Option<u32>,
    fail_start: bool,
    masks: Arc<Mutex<usize>>,
}

struct SetRun {
    allowed: Vec<u32>,
    len: usize,
    trigger: Option<u32>,
    on: bool,
    n: usize,
    masks: Arc<Mutex<usize>>,
}

impl ConstraintSpec for SetSpec {
    fn start(&self) -> Result<Box<dyn TokenConstraint>> {
        if self.fail_start {
            return Err(intarsia_core::Error::Other("no grammar".into()));
        }
        Ok(Box::new(SetRun {
            allowed: self.allowed.clone(),
            len: self.len,
            trigger: self.trigger,
            on: self.trigger.is_none(),
            n: 0,
            masks: self.masks.clone(),
        }))
    }
}

impl TokenConstraint for SetRun {
    fn active(&self) -> bool {
        self.on
    }
    fn compute_mask(&mut self) -> Result<TokenMask> {
        *self.masks.lock().unwrap() += 1;
        let mut m = TokenMask::none(VOCAB);
        if self.n >= self.len {
            m.allow(EOS);
        } else {
            for &t in &self.allowed {
                m.allow(t);
            }
        }
        Ok(m)
    }
    fn commit(&mut self, tok: u32) -> Result<()> {
        if !self.on {
            self.on = Some(tok) == self.trigger;
            return Ok(());
        }
        if self.n >= self.len || !self.allowed.contains(&tok) {
            return Err(intarsia_core::Error::Other(format!("token {tok} not allowed")));
        }
        self.n += 1;
        Ok(())
    }
}

fn spec(allowed: Vec<u32>, len: usize, trigger: Option<u32>) -> (Arc<dyn ConstraintSpec>, Arc<Mutex<usize>>) {
    let masks = Arc::new(Mutex::new(0));
    (Arc::new(SetSpec { allowed, len, trigger, fail_start: false, masks: masks.clone() }), masks)
}

fn req(prompt: &str, n: usize) -> GenRequest {
    GenRequest { prompt: prompt.bytes().map(u32::from).collect(), max_tokens: n, ..Default::default() }
}

fn all_tokens() -> Vec<u32> {
    (0..VOCAB as u32).collect()
}

#[test]
fn permissive_constraint_keeps_greedy_ids() {
    // an allow-everything mask takes the host-logits path on every step; on the cpu backend its argmax must be the
    // device argmax, so the ids are those of the unconstrained run
    let (e, _) = solo(11, None);
    let base = run(&e, GenRequest { ignore_eos: true, ..req("hello there", 24) }).unwrap();
    let (c, masks) = spec(all_tokens().into_iter().filter(|&t| t != EOS).collect(), 1000, None);
    let got = run(&e, GenRequest { constraint: Some(c), ..req("hello there", 24) }).unwrap();
    assert_eq!(got.ids, base.ids);
    assert_eq!(*masks.lock().unwrap(), 24, "one mask per generated token");
}

#[test]
fn mask_decides_tokens_and_grammar_end_stops() {
    let allowed: Vec<u32> = b"0123456789".iter().map(|&b| u32::from(b)).collect();
    for e in [&solo(5, None).0 as &dyn Engine, &batched(5, 4) as &dyn Engine] {
        let (c, _) = spec(allowed.clone(), 7, None);
        let got = run(e, GenRequest { constraint: Some(c), ..req("count: ", 40) }).unwrap();
        assert_eq!(got.ids.len(), 7, "{got:?}");
        assert!(got.ids.iter().all(|t| allowed.contains(t)), "{got:?}");
        // the 8th token can only be EOS: the request stops there
        assert_eq!((got.reason, got.completion_tokens), (FinishReason::Stop, 8));
        // max_tokens still wins over the grammar
        let (c, _) = spec(allowed.clone(), 7, None);
        let got = run(e, GenRequest { constraint: Some(c), ..req("count: ", 3) }).unwrap();
        assert_eq!((got.ids.len(), got.reason), (3, FinishReason::Length));
    }
}

#[test]
fn sampled_requests_are_masked_too() {
    let (e, _) = solo(5, None);
    let allowed: Vec<u32> = b"ab".iter().map(|&b| u32::from(b)).collect();
    for seed in 0..5 {
        let (c, _) = spec(allowed.clone(), 12, None);
        let r = GenRequest { constraint: Some(c), temperature: 1.5, seed: Some(seed), ..req("x", 40) };
        let got = run(&e, r).unwrap();
        assert!(got.ids.len() == 12 && got.ids.iter().all(|t| allowed.contains(t)), "{got:?}");
    }
}

#[test]
fn lazy_constraint_waits_for_its_trigger() {
    let (e, _) = solo(7, None);
    let base = run(&e, GenRequest { ignore_eos: true, ..req("abc", 20) }).unwrap();
    // trigger on the first greedy token (after the first) that has not appeared before
    let k = (1..base.ids.len()).find(|&i| !base.ids[..i].contains(&base.ids[i])).expect("a new token");
    let trig = base.ids[k];
    let allowed = vec![u32::from(b'z')];
    let (c, masks) = spec(allowed, 4, Some(trig));
    let got = run(&e, GenRequest { constraint: Some(c), ..req("abc", 20) }).unwrap();
    assert_eq!(&got.ids[..=k], &base.ids[..=k], "unconstrained up to and including the trigger");
    assert_eq!(&got.ids[k + 1..], &[u32::from(b'z'); 4]);
    assert_eq!((got.reason, got.completion_tokens), (FinishReason::Stop, k + 6));
    assert_eq!(*masks.lock().unwrap(), 5, "masks only after the trigger (4 tokens + the EOS step)");
    // a trigger that never comes: exactly the unconstrained run, no masks
    let (c, masks) = spec(vec![], 0, Some(EOS - 1));
    let got = run(&e, GenRequest { constraint: Some(c), ignore_eos: true, ..req("abc", 20) }).unwrap();
    if !base.ids.contains(&(EOS - 1)) {
        assert_eq!(got.ids, base.ids);
        assert_eq!(*masks.lock().unwrap(), 0);
    }
}

#[test]
fn speculation_is_off_for_constrained_requests() {
    // a repetitive prompt the n-gram drafter likes; the spec engine must give the constrained request exactly what
    // the plain engine gives it, with no verify passes
    let prompt = "abcabcabcabcabcabcabc";
    let allowed: Vec<u32> = b"abc".iter().map(|&b| u32::from(b)).collect();
    let (plain, _) = solo(3, None);
    let (spec_e, tel) = solo(3, Some(4));
    let (c1, _) = spec(allowed.clone(), 30, None);
    let (c2, _) = spec(allowed, 30, None);
    // the drafter does run on this prompt unconstrained
    run(&spec_e, GenRequest { ignore_eos: true, ..req(prompt, 30) }).unwrap();
    let before = tel.snapshot().spec_steps;
    assert!(before > 0, "pick a prompt the drafter drafts on");
    let a = run(&plain, GenRequest { constraint: Some(c1), ..req(prompt, 30) }).unwrap();
    let b = run(&spec_e, GenRequest { constraint: Some(c2), ..req(prompt, 30) }).unwrap();
    assert_eq!(a, b);
    assert_eq!(tel.snapshot().spec_steps, before, "no verify pass for a constrained request");
}

#[test]
fn constrained_and_unconstrained_batch_together() {
    let prompts = ["the cat", "a dog sat", "zzz", "hello world"];
    let (s, _) = solo(9, None);
    let refs: Vec<Out> =
        prompts.iter().map(|p| run(&s, GenRequest { ignore_eos: true, ..req(p, 20) }).unwrap()).collect();
    let b = batched(9, 4);
    let allowed: Vec<u32> = b"xy".iter().map(|&b| u32::from(b)).collect();
    let rxs: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let constraint = (i % 2 == 1).then(|| spec(allowed.clone(), 10, None).0);
            b.submit(GenRequest { constraint, ignore_eos: i % 2 == 0, ..req(p, 20) }).unwrap()
        })
        .collect();
    for (i, rx) in rxs.iter().enumerate() {
        let got = collect(rx).unwrap();
        if i % 2 == 0 {
            assert_eq!(got.ids, refs[i].ids, "unconstrained request {i} changed next to constrained ones");
        } else {
            assert!(got.ids.len() == 10 && got.ids.iter().all(|t| allowed.contains(t)), "{got:?}");
            assert_eq!(got.reason, FinishReason::Stop);
        }
    }
}

#[test]
fn failing_constraint_rejects_the_request() {
    let masks = Arc::new(Mutex::new(0));
    let bad: Arc<dyn ConstraintSpec> =
        Arc::new(SetSpec { allowed: vec![], len: 0, trigger: None, fail_start: true, masks });
    let (e, _) = solo(5, None);
    let err = run(&e, GenRequest { constraint: Some(bad.clone()), ..req("x", 4) }).unwrap_err();
    assert!(err.contains("structured output") && err.contains("no grammar"), "{err}");
    let err = run(&batched(5, 2), GenRequest { constraint: Some(bad), ..req("x", 4) }).unwrap_err();
    assert!(err.contains("no grammar"), "{err}");
    // a mask that allows nothing ends the request with an error instead of sampling a forbidden token
    let (c, _) = spec(vec![], 5, None);
    let err = run(&e, GenRequest { constraint: Some(c), ..req("x", 4) }).unwrap_err();
    assert!(err.contains("allows no token"), "{err}");
    // the engine still serves afterwards
    run(&e, req("x", 2)).unwrap();
}

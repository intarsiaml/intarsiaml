#![cfg(feature = "fault-inject")]
//! The `fault-inject` hook on the mock scheduler (no GPU). The mock's decode step plays the backend: like the Metal
//! synchronize it calls `fault::take()` and fails or panics when a fault is armed. Run with
//! `cargo test -p intarsia-core --features fault-inject --test fault_inject`.
//!
//! All tests share the process-wide armed flag, so they run behind one lock.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use intarsia_core::fault::{self, Mode, Plan, Trigger};
use intarsia_core::scheduler::{BatchEngine, BatchModel, BatchSeq, Row, Scheduler, SchedulerConfig, SeqStep};
use intarsia_core::{Error, Result};

static LOCK: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _ = fault::take();
    g
}

/// Slots hold token counts only; every decode step calls the "backend sync" hook.
struct Model {
    toks: Vec<Vec<u32>>,
    steps: usize,
}

impl BatchModel for Model {
    type Mode = ();
    type Out = u32;
    fn slots(&self) -> usize {
        self.toks.len()
    }
    fn slot_cap(&self) -> usize {
        4096
    }
    fn slot_tokens(&self, slot: usize) -> &[u32] {
        &self.toks[slot]
    }
    fn truncate(&mut self, slot: usize, len: usize) -> Result<()> {
        self.toks[slot].truncate(len);
        Ok(())
    }
    fn prefill_chunk(&mut self, slot: usize, tokens: &[u32], mode: Option<()>) -> Result<Option<u32>> {
        self.toks[slot].extend_from_slice(tokens);
        Ok(mode.map(|_| 5))
    }
    fn decode_step(&mut self, rows: &[Row<()>]) -> Result<Vec<Result<u32>>> {
        self.steps += 1;
        for r in rows {
            self.toks[r.slot].push(r.token);
        }
        match fault::take() {
            Some(Mode::Err) => return Err(Error::backend("metal: command buffer failed: injected fault (mock)")),
            Some(Mode::Panic) => panic!("injected fault (mock)"),
            None => {}
        }
        Ok(rows.iter().map(|_| Ok(7)).collect())
    }
}

#[derive(Debug)]
enum Ev {
    Tok,
    Done,
    Err(String),
}

struct Seq {
    prompt: Vec<u32>,
    left: usize,
    tx: Sender<Ev>,
}

fn seq(prompt: usize, new_tokens: usize) -> (Seq, Receiver<Ev>) {
    let (tx, rx) = channel();
    (Seq { prompt: vec![3; prompt], left: new_tokens, tx }, rx)
}

impl BatchSeq<Model> for Seq {
    fn prompt(&self) -> &[u32] {
        &self.prompt
    }
    fn admit(&mut self, _cached: usize, _cap: usize) -> Result<bool> {
        Ok(true)
    }
    fn first_mode(&self) {}
    fn prepare(&mut self, _m: &mut Model, _slot: usize, _spec: bool, _batched: bool) -> Result<SeqStep<()>> {
        Ok(SeqStep::Plain(9, ()))
    }
    fn accept(&mut self, _o: u32) -> Result<()> {
        Ok(())
    }
    fn on_token(&mut self) -> Result<bool> {
        let _ = self.tx.send(Ev::Tok);
        self.left -= 1;
        Ok(self.left == 0)
    }
    fn cancelled(&self) -> bool {
        false
    }
    fn finish(&mut self, _m: &mut Model, _slot: Option<usize>, _cancelled: bool) -> Result<()> {
        let _ = self.tx.send(Ev::Done);
        Ok(())
    }
    fn fail(&mut self, e: &Error) {
        let _ = self.tx.send(Ev::Err(e.to_string()));
    }
}

fn model(slots: usize) -> Model {
    Model { toks: vec![Vec::new(); slots], steps: 0 }
}

/// The scenario of the GPU test in miniature: 3 decoding streams, a long prompt mid-prefill, 2 queued behind a full
/// house of 4 slots.
fn load(s: &mut Scheduler<Model, Seq>) -> Vec<Receiver<Ev>> {
    let mut rxs = Vec::new();
    for _ in 0..3 {
        let (q, rx) = seq(8, 10_000);
        s.submit(q).unwrap();
        rxs.push(rx);
    }
    while s.metrics().decode_steps < 2 {
        s.step();
    }
    for _ in 0..3 {
        let (q, rx) = seq(2000, 10_000);
        s.submit(q).unwrap();
        rxs.push(rx);
    }
    rxs
}

fn last(rx: &Receiver<Ev>) -> Ev {
    let mut last = None;
    while let Ok(e) = rx.recv_timeout(Duration::from_secs(20)) {
        let terminal = matches!(e, Ev::Done | Ev::Err(_));
        last = Some(e);
        if terminal {
            break;
        }
    }
    last.expect("no event")
}

fn cfg() -> SchedulerConfig {
    SchedulerConfig { max_batch: 4, ..Default::default() }
}

#[test]
fn err_fails_every_request_and_refuses_later_submits() {
    let _g = serial();
    let mut s = Scheduler::<Model, Seq>::new(model(4), cfg()).unwrap();
    s.set_fault_plan(Some(Plan::new(Mode::Err, Trigger::Mixed(3)).with_waiting(2).with_prefilled(512)));
    let rxs = load(&mut s);
    let mut n = 0;
    while s.has_work() {
        s.step();
        n += 1;
        assert!(n < 10_000, "scheduler does not terminate");
    }
    let why = s.failed().expect("engine failed").to_string();
    assert!(why.contains("injected fault"), "{why}");
    // 3 decoding + 1 prefilling + 2 queued: all six end with an error, none with Done
    for rx in &rxs {
        match last(rx) {
            Ev::Err(m) => assert!(m.contains("engine failed") && m.contains("injected fault"), "{m}"),
            e => panic!("expected an error, got {e:?}"),
        }
    }
    let m = s.metrics();
    assert_eq!((m.failed, m.finished), (6, 0));
    assert_eq!(m.running, 0);
    assert_eq!(fault::take(), None, "the armed fault was consumed");
    let (q, rx) = seq(4, 4);
    assert!(s.submit(q).is_err());
    assert!(matches!(last(&rx), Ev::Err(_)));
}

#[test]
fn after_steps_fires_on_the_chosen_step_only() {
    let _g = serial();
    let mut s = Scheduler::<Model, Seq>::new(model(4), cfg()).unwrap();
    s.set_fault_plan(Some(Plan::new(Mode::Err, Trigger::AfterSteps(5))));
    let (q, rx) = seq(4, 10_000);
    s.submit(q).unwrap();
    while s.failed().is_none() {
        s.step();
    }
    assert_eq!(s.model().steps, 6, "five steps completed, the sixth failed");
    assert!(matches!(last(&rx), Ev::Err(_)));
}

#[test]
fn unset_plan_never_fires() {
    let _g = serial();
    let mut s = Scheduler::<Model, Seq>::new(model(4), cfg()).unwrap();
    s.set_fault_plan(None);
    let rxs: Vec<_> = (0..3)
        .map(|_| {
            let (q, rx) = seq(4, 20);
            s.submit(q).unwrap();
            rx
        })
        .collect();
    while s.has_work() {
        s.step();
    }
    assert!(s.failed().is_none());
    for rx in &rxs {
        assert!(matches!(last(rx), Ev::Done));
    }
}

#[test]
fn env_plan_panic_is_caught_and_fails_the_engine() {
    let _g = serial();
    std::env::set_var("INTARSIA_FAULT_INJECT", "panic:mixed=3,waiting=2,pos=512");
    // the scheduler is built on the engine thread, so the variable stays set until the engine has failed
    let engine = BatchEngine::<Model, Seq>::spawn(model(4), cfg()).unwrap();
    let mut rxs = Vec::new();
    for _ in 0..3 {
        let (q, rx) = seq(8, 1_000_000);
        engine.submit(q).unwrap();
        rxs.push(rx);
    }
    // wait until the three decode, then add the long prompt and the queue
    while engine.metrics().decode_steps < 2 {
        std::thread::sleep(Duration::from_millis(2));
    }
    for _ in 0..3 {
        let (q, rx) = seq(2000, 1_000_000);
        engine.submit(q).unwrap();
        rxs.push(rx);
    }
    for rx in &rxs {
        match last(rx) {
            Ev::Err(m) => assert!(m.contains("injected fault"), "{m}"),
            e => panic!("expected an error, got {e:?}"),
        }
    }
    let why = engine.failed().expect("engine failed");
    assert!(why.contains("scheduler panicked") && why.contains("injected fault"), "{why}");
    let (q, rx) = seq(4, 4);
    assert!(engine.submit(q).is_err());
    assert!(matches!(last(&rx), Ev::Err(_)));
    std::env::remove_var("INTARSIA_FAULT_INJECT");
}

#[test]
fn bad_env_spec_is_refused_at_construction() {
    let _g = serial();
    std::env::set_var("INTARSIA_FAULT_INJECT", "err:nonsense");
    let r = Scheduler::<Model, Seq>::new(model(4), cfg());
    std::env::remove_var("INTARSIA_FAULT_INJECT");
    assert!(r.is_err());
}

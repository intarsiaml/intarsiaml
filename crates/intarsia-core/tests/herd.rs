//! Prefix herd (lane herd, `SchedulerConfig::prefix_herd`) in the scheduler, without a GPU: requests that share a prompt prefix
//! another request is still prefilling wait for it and restore it from the (mock) prefix store instead of prefilling it again.
//!
//! The mock model's KV is the token list of a slot, its output a hash of that list, so a request's tokens are a function of the
//! tokens its slot holds, however the prefix got there (prefilled or restored): a restore that took the wrong prefix, or a
//! slot left in a wrong state, changes the reply. The mock store holds the token prefixes a store request captured at the ends of
//! its own-schedule chunks (every `GRID` tokens, as the real store's grid does); `herd_resume` and admission restore the longest
//! one that is a prefix of the prompt.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use intarsia_core::scheduler::{BatchModel, BatchSeq, PrefillOrder, RequestId, Row, Scheduler, SchedulerConfig, SeqStep};
use intarsia_core::{Error, FinishReason, Result};

const CHUNK: usize = 8;
const GRID: usize = 16;
const MIN: usize = 8;
const V: u64 = 1000;

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

fn hash(toks: &[u32]) -> u64 {
    toks.iter().fold(7, |h, &t| mix(h, t as u64 + 1))
}

type Store = Arc<Mutex<Vec<Vec<u32>>>>;

#[derive(Default)]
struct Rec {
    /// Prefill rows run per request.
    rows: HashMap<RequestId, usize>,
}

struct Model {
    slots: Vec<Vec<u32>>,
    owner: Vec<Option<RequestId>>,
    cap: usize,
    rec: Arc<Mutex<Rec>>,
    /// Request ids whose prefill chunk fails (non-fatal).
    fail_prefill: Vec<RequestId>,
}

impl Model {
    fn new(slots: usize, cap: usize) -> (Model, Arc<Mutex<Rec>>) {
        let rec = Arc::new(Mutex::new(Rec::default()));
        (Model { slots: vec![Vec::new(); slots], owner: vec![None; slots], cap, rec: rec.clone(), fail_prefill: Vec::new() }, rec)
    }
}

impl BatchModel for Model {
    type Mode = ();
    type Out = u64;
    fn slots(&self) -> usize {
        self.slots.len()
    }
    fn slot_cap(&self) -> usize {
        self.cap
    }
    fn slot_tokens(&self, slot: usize) -> &[u32] {
        &self.slots[slot]
    }
    fn truncate(&mut self, slot: usize, len: usize) -> Result<()> {
        assert!(len <= self.slots[slot].len(), "truncate to {len} of a slot holding {}", self.slots[slot].len());
        self.slots[slot].truncate(len);
        Ok(())
    }
    fn prefill_chunk(&mut self, slot: usize, tokens: &[u32], mode: Option<()>) -> Result<Option<u64>> {
        let id = self.owner[slot].expect("prefill into an unowned slot");
        if self.fail_prefill.contains(&id) {
            return Err(Error::Other("injected prefill failure".into()));
        }
        *self.rec.lock().unwrap().rows.entry(id).or_default() += tokens.len();
        self.slots[slot].extend_from_slice(tokens);
        Ok(mode.map(|()| hash(&self.slots[slot])))
    }
    fn decode_step(&mut self, rows: &[Row<()>]) -> Result<Vec<Result<u64>>> {
        Ok(rows
            .iter()
            .map(|r| {
                self.slots[r.slot].push(r.token);
                Ok(hash(&self.slots[r.slot]))
            })
            .collect())
    }
    fn on_admit(&mut self, slot: usize, id: RequestId, _cached: usize) {
        self.owner[slot] = Some(id);
    }
}

enum Ev {
    Tok(u32),
    Done { cached: usize },
    Err(String),
}

struct Seq {
    prompt: Vec<u32>,
    max_new: usize,
    store: Store,
    /// `herd_min`: the request takes part in the herd.
    herd: bool,
    cancel: Arc<AtomicBool>,
    tx: Sender<Ev>,
    cached: usize,
    generated: usize,
    tok: u32,
    reason: Option<FinishReason>,
    /// Rows the mock store restored mid-prefill (`herd_resume`).
    resumed: usize,
}

impl Seq {
    fn new(prompt: Vec<u32>, max_new: usize, store: &Store, herd: bool) -> (Seq, Receiver<Ev>, Arc<AtomicBool>) {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let seq = Seq {
            prompt,
            max_new,
            store: store.clone(),
            herd,
            cancel: cancel.clone(),
            tx,
            cached: 0,
            generated: 0,
            tok: 0,
            reason: None,
            resumed: 0,
        };
        (seq, rx, cancel)
    }

    /// The longest stored prefix of the prompt longer than `local` and below the prompt's length.
    fn lookup(&self, local: usize) -> Option<usize> {
        let store = self.store.lock().unwrap();
        store
            .iter()
            .filter(|e| e.len() > local && e.len() < self.prompt.len() && self.prompt.starts_with(e))
            .map(|e| e.len())
            .max()
    }
}

impl BatchSeq<Model> for Seq {
    fn prompt(&self) -> &[u32] {
        &self.prompt
    }
    fn admit(&mut self, cached: usize, _cap: usize) -> Result<bool> {
        self.cached = cached;
        Ok(self.max_new > 0)
    }
    fn first_mode(&self) {}
    fn restore(&mut self, m: &mut Model, slot: usize, local: usize, _idle: bool) -> Result<usize> {
        // admission's lookup: the slot's own prefix (`local`), or a longer stored one
        match self.lookup(local) {
            Some(len) => {
                m.slots[slot] = self.prompt[..len].to_vec();
                Ok(len)
            }
            None => Ok(local),
        }
    }
    fn chunk_len(&self, _pos: usize, remaining: usize) -> Option<usize> {
        Some(remaining.min(CHUNK))
    }
    fn own_chunk(&self) -> Option<usize> {
        Some(CHUNK)
    }
    fn on_chunk(&mut self, _m: &mut Model, _slot: usize, _start: usize, end: usize, last: bool) -> Result<()> {
        if !last && end >= MIN && end % GRID == 0 && end > self.cached {
            let mut store = self.store.lock().unwrap();
            let p = self.prompt[..end].to_vec();
            if !store.contains(&p) {
                store.push(p);
            }
        }
        Ok(())
    }
    fn herd_min(&self) -> Option<usize> {
        self.herd.then_some(MIN)
    }
    fn herd_resume(&mut self, m: &mut Model, slot: usize, pos: usize) -> Result<usize> {
        match self.lookup(pos) {
            Some(len) => {
                m.slots[slot] = self.prompt[..len].to_vec();
                self.resumed = len - pos;
                self.cached = len;
                Ok(len)
            }
            None => Ok(pos),
        }
    }
    fn prepare(&mut self, _m: &mut Model, _slot: usize, _spec: bool, _busy: bool) -> Result<SeqStep<()>> {
        Ok(SeqStep::Plain(self.tok, ()))
    }
    fn accept(&mut self, out: u64) -> Result<()> {
        self.tok = (out % V) as u32;
        Ok(())
    }
    fn on_token(&mut self) -> Result<bool> {
        self.generated += 1;
        let _ = self.tx.send(Ev::Tok(self.tok));
        if self.generated >= self.max_new {
            self.reason = Some(FinishReason::Length);
            return Ok(true);
        }
        Ok(false)
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
    fn finish(&mut self, _m: &mut Model, _slot: Option<usize>, _cancelled: bool) -> Result<()> {
        let _ = self.tx.send(Ev::Done { cached: self.cached });
        Ok(())
    }
    fn fail(&mut self, err: &Error) {
        let _ = self.tx.send(Ev::Err(err.to_string()));
    }
}

/// `n` prompts sharing `shared` leading tokens, each with its own `own` tokens after them.
fn prompts(n: usize, shared: usize, own: usize) -> Vec<Vec<u32>> {
    let head: Vec<u32> = (0..shared as u32).map(|i| 10 + (i * 7 + 3) % 500).collect();
    (0..n as u32)
        .map(|k| {
            let mut p = head.clone();
            p.extend((0..own as u32).map(|i| 600 + (k * 53 + i * 11 + 1) % 300));
            p
        })
        .collect()
}

struct Run {
    toks: Vec<Vec<u32>>,
    cached: Vec<Option<usize>>,
    errors: Vec<Option<String>>,
    rows: Vec<usize>,
    metrics: intarsia_core::scheduler::BatchMetrics,
}

fn collect(rx: &Receiver<Ev>) -> (Vec<u32>, Option<usize>, Option<String>) {
    let (mut toks, mut cached, mut err) = (vec![], None, None);
    for e in rx.try_iter() {
        match e {
            Ev::Tok(t) => toks.push(t),
            Ev::Done { cached: c } => cached = Some(c),
            Ev::Err(m) => err = Some(m),
        }
    }
    (toks, cached, err)
}

/// Submit every prompt at once, step to the end (at most `limit` iterations: a hang fails the test), and report. `cancel_leader_at`:
/// set request 0's cancel flag before that iteration. `fail_leader`: request 0's prefill chunks fail.
fn run(prompts: &[Vec<u32>], herd: bool, slots: usize, cancel_leader_at: Option<u64>, fail_leader: bool) -> Run {
    let (mut m, rec) = Model::new(slots, 400);
    if fail_leader {
        m.fail_prefill.push(0);
    }
    let cfg = SchedulerConfig { max_batch: slots, prefix_reuse: false, prefix_herd: herd, ..Default::default() };
    let mut s = Scheduler::new(m, cfg).unwrap();
    let store: Store = Arc::new(Mutex::new(Vec::new()));
    let mut rxs = Vec::new();
    let mut cancels = Vec::new();
    for p in prompts {
        let (seq, rx, c) = Seq::new(p.clone(), 5, &store, true);
        s.submit(seq).unwrap();
        rxs.push(rx);
        cancels.push(c);
    }
    let mut it = 0u64;
    while s.has_work() {
        if cancel_leader_at == Some(it) {
            cancels[0].store(true, Ordering::SeqCst);
        }
        s.step();
        it += 1;
        assert!(it < 10_000, "the scheduler does not terminate (a held follower waits for ever?)");
    }
    let out: Vec<_> = rxs.iter().map(collect).collect();
    let rows = rec.lock().unwrap();
    Run {
        toks: out.iter().map(|o| o.0.clone()).collect(),
        cached: out.iter().map(|o| o.1).collect(),
        errors: out.iter().map(|o| o.2.clone()).collect(),
        rows: (0..prompts.len() as u64).map(|id| rows.rows.get(&id).copied().unwrap_or(0)).collect(),
        metrics: s.metrics(),
    }
}

/// Without the herd every request prefills the shared prefix itself: the baseline the other tests are measured against.
#[test]
fn herd_off_every_request_prefills_the_shared_prefix() {
    let ps = prompts(4, 64, 6);
    let r = run(&ps, false, 4, None, false);
    assert_eq!(r.rows, vec![70; 4], "each request prefills its whole prompt");
    assert_eq!(r.metrics.herd_holds + r.metrics.herd_restores + r.metrics.herd_tokens, 0);
    assert!(r.errors.iter().all(Option::is_none));
    assert!(r.cached.iter().all(|c| *c == Some(0)));
}

/// 4 simultaneous requests, one new shared prefix: the leader prefills it, the followers restore it and prefill only their own
/// tokens. Every reply is the one the same request gets with the herd off.
#[test]
fn herd_followers_restore_instead_of_prefilling() {
    let ps = prompts(4, 64, 6);
    let off = run(&ps, false, 4, None, false);
    let on = run(&ps, true, 4, None, false);
    assert_eq!(on.rows, vec![70, 6, 6, 6], "the followers prefill only their own 6 tokens");
    assert_eq!(on.cached, vec![Some(0), Some(64), Some(64), Some(64)]);
    assert_eq!(on.metrics.herd_restores, 3);
    assert_eq!(on.metrics.herd_tokens, 3 * 64);
    assert!(on.metrics.herd_holds > 0);
    assert_eq!(on.toks, off.toks, "replies are unchanged");
    assert!(on.toks.iter().all(|t| t.len() == 5));
    assert!(on.errors.iter().all(Option::is_none));
}

/// Fewer slots than requests: the late ones are admitted after the leader finished (they restore at admission, as before), and
/// replies are still the herd-off ones.
#[test]
fn herd_with_a_queue_matches_herd_off() {
    let ps = prompts(6, 64, 6);
    let off = run(&ps, false, 2, None, false);
    let on = run(&ps, true, 2, None, false);
    assert_eq!(on.toks, off.toks);
    let (a, b): (usize, usize) = (on.rows.iter().sum(), off.rows.iter().sum());
    assert!(a < b, "the herd prefilled {a} rows, off {b}");
}

/// A leader that is cancelled during its prefill releases its followers, which restore what it had captured and prefill the rest.
#[test]
fn herd_cancelled_leader_does_not_hang_followers() {
    let ps = prompts(4, 64, 6);
    let off = run(&ps, false, 4, None, false);
    // the leader prefills one 8-token chunk per iteration: 3 chunks (24 tokens, one capture at 16) are done at iteration 3
    let on = run(&ps, true, 4, Some(3), false);
    assert!(on.toks[0].is_empty(), "the cancelled leader generated nothing");
    for k in 1..4 {
        assert_eq!(on.toks[k], off.toks[k], "follower {k}");
        assert!(on.errors[k].is_none());
    }
    assert!(on.rows[0] < 70, "the leader stopped early: {}", on.rows[0]);
    // follower 1 took over as the leader of the rest: nobody prefilled the shared prefix twice beyond what was captured
    assert_eq!(on.cached[1], Some(16), "restored what the cancelled leader had captured");
    let total: usize = on.rows.iter().sum();
    assert!(total < 70 + 3 * 70, "followers reused the leader's work ({total} rows)");
}

/// A leader whose prefill fails releases its followers the same way.
#[test]
fn herd_failed_leader_does_not_hang_followers() {
    let ps = prompts(3, 64, 6);
    let off = run(&ps, false, 3, None, true);
    let on = run(&ps, true, 3, None, true);
    assert!(on.errors[0].as_deref().is_some_and(|e| e.contains("injected")), "{:?}", on.errors);
    for k in 1..3 {
        assert!(on.errors[k].is_none(), "follower {k}: {:?}", on.errors[k]);
        assert_eq!(on.toks[k], off.toks[k], "follower {k}");
        assert_eq!(on.toks[k].len(), 5);
    }
}

/// Prompts that share fewer than `min_tokens` leading tokens are no herd: nobody waits, nothing is restored.
#[test]
fn herd_unrelated_prompts_are_untouched() {
    let ps = prompts(4, 4, 40);
    let off = run(&ps, false, 4, None, false);
    let on = run(&ps, true, 4, None, false);
    assert_eq!(on.rows, off.rows);
    assert_eq!(on.toks, off.toks);
    assert_eq!(on.metrics.herd_holds, 0);
    assert_eq!(on.metrics.herd_restores, 0);
}

/// A request that does not use the store (`herd_min` is `None`) is never held and never a leader.
#[test]
fn herd_requests_without_the_store_are_not_held() {
    let ps = prompts(3, 64, 6);
    let (m, rec) = Model::new(3, 400);
    let cfg = SchedulerConfig { max_batch: 3, prefix_reuse: false, prefix_herd: true, ..Default::default() };
    let mut s = Scheduler::new(m, cfg).unwrap();
    let store: Store = Arc::new(Mutex::new(Vec::new()));
    let mut rxs = Vec::new();
    for p in &ps {
        let (seq, rx, _c) = Seq::new(p.clone(), 5, &store, false);
        s.submit(seq).unwrap();
        rxs.push(rx);
    }
    let mut it = 0;
    while s.has_work() {
        s.step();
        it += 1;
        assert!(it < 10_000);
    }
    assert_eq!(s.metrics().herd_holds, 0);
    assert_eq!(rec.lock().unwrap().rows.values().copied().collect::<Vec<_>>().len(), 3);
    assert!(rxs.iter().all(|rx| collect(rx).0.len() == 5));
}

/// A request that arrives while the leader is part-way through: admission restores what the leader has captured so far, the request
/// waits for the rest, then restores that too.
#[test]
fn herd_late_follower_restores_twice() {
    let ps = prompts(2, 64, 6);
    let (m, rec) = Model::new(2, 400);
    let cfg = SchedulerConfig { max_batch: 2, prefix_reuse: false, prefix_herd: true, ..Default::default() };
    let mut s = Scheduler::new(m, cfg).unwrap();
    let store: Store = Arc::new(Mutex::new(Vec::new()));
    let (lead, rx0, _) = Seq::new(ps[0].clone(), 5, &store, true);
    s.submit(lead).unwrap();
    // 5 iterations: 40 prompt tokens prefilled (captures at 16 and 32)
    for _ in 0..5 {
        s.step();
    }
    let (late, rx1, _) = Seq::new(ps[1].clone(), 5, &store, true);
    s.submit(late).unwrap();
    let mut it = 0;
    while s.has_work() {
        s.step();
        it += 1;
        assert!(it < 10_000);
    }
    let rows = rec.lock().unwrap();
    assert_eq!(rows.rows[&0], 70);
    assert_eq!(rows.rows[&1], 6, "the late request prefilled only its own tokens");
    let (t0, _, e0) = collect(&rx0);
    let (t1, c1, e1) = collect(&rx1);
    assert!(e0.is_none() && e1.is_none());
    assert_eq!((t0.len(), t1.len(), c1), (5, 5, Some(64)));
    let m = s.metrics();
    assert_eq!(m.herd_restores, 1, "one mid-prefill restore, after the admission one");
    assert_eq!(m.herd_tokens, 64 - 32);
}

/// Lane sjf with the herd: under `PrefillOrder::Shortest` a follower with far fewer tokens left than its leader is still held for the
/// leader (never picked just because it is shorter), restores the shared prefix when released, and every reply is the herd-off one.
#[test]
fn herd_followers_stay_held_under_shortest() {
    let shared = prompts(1, 64, 0).remove(0);
    let mut ps = Vec::new();
    for (k, own) in [(0u32, 240usize), (1, 2), (2, 3)] {
        let mut p = shared.clone();
        p.extend((0..own as u32).map(|i| 600 + (k * 53 + i * 11 + 1) % 300));
        ps.push(p);
    }
    let go = |herd: bool, order: PrefillOrder| {
        let (m, rec) = Model::new(3, 400);
        let cfg = SchedulerConfig { max_batch: 3, prefix_reuse: false, prefix_herd: herd, prefill_order: order, ..Default::default() };
        let mut s = Scheduler::new(m, cfg).unwrap();
        let store: Store = Arc::new(Mutex::new(Vec::new()));
        let rxs: Vec<_> = ps
            .iter()
            .map(|p| {
                let (seq, rx, _) = Seq::new(p.clone(), 5, &store, true);
                s.submit(seq).unwrap();
                rx
            })
            .collect();
        let mut it = 0;
        while s.has_work() {
            s.step();
            it += 1;
            assert!(it < 10_000, "no termination");
        }
        let toks: Vec<_> = rxs.iter().map(|r| collect(r).0).collect();
        let rows: Vec<usize> = (0..3u64).map(|id| rec.lock().unwrap().rows.get(&id).copied().unwrap_or(0)).collect();
        (toks, rows, s.metrics())
    };
    let (off_toks, _, _) = go(false, PrefillOrder::Fifo);
    let (fifo_toks, fifo_rows, _) = go(true, PrefillOrder::Fifo);
    let (short_toks, short_rows, m) = go(true, PrefillOrder::Shortest);
    assert_eq!(fifo_toks, off_toks);
    assert_eq!(short_toks, off_toks, "replies are unchanged");
    assert_eq!(fifo_rows, vec![304, 2, 3]);
    assert_eq!(short_rows, fifo_rows, "the followers were held for the leader under shortest too and prefilled only their own tokens");
    assert_eq!(m.herd_restores, 2);
}

/// Lane hsjf: `herd` and `sjf` together, `n` store requests sharing a prefix at iteration 0 and one unrelated short prompt submitted
/// at iteration `short_at`. Returns, per request (the short one last), the iteration of its first token and of its last, and its
/// tokens.
fn herd_plus_short(slots: usize, herd: bool, order: PrefillOrder, n: usize, short_at: u64) -> (Vec<(u64, u64)>, Vec<Vec<u32>>) {
    let ps = prompts(n, 320, 8);
    let short: Vec<u32> = (0..20u32).map(|i| 900 + (i * 3) % 90).collect();
    assert!(short.iter().all(|t| *t >= 900), "no leading tokens shared with the herd");
    let (m, _rec) = Model::new(slots, 400);
    let cfg = SchedulerConfig { max_batch: slots, prefix_reuse: false, prefix_herd: herd, prefill_order: order, ..Default::default() };
    let mut s = Scheduler::new(m, cfg).unwrap();
    let store: Store = Arc::new(Mutex::new(Vec::new()));
    let mut rxs = Vec::new();
    for p in &ps {
        let (seq, rx, _) = Seq::new(p.clone(), 5, &store, true);
        s.submit(seq).unwrap();
        rxs.push(rx);
    }
    let mut toks: Vec<Vec<u32>> = vec![Vec::new(); n + 1];
    let mut span = vec![(u64::MAX, 0u64); n + 1];
    let mut it = 0u64;
    loop {
        if it == short_at {
            let (seq, rx, _) = Seq::new(short.clone(), 5, &store, true);
            s.submit(seq).unwrap();
            rxs.push(rx);
        }
        if it > short_at && !s.has_work() {
            break;
        }
        s.step();
        for (k, rx) in rxs.iter().enumerate() {
            let before = toks[k].len();
            toks[k].extend(collect(rx).0);
            if toks[k].len() > before {
                span[k].0 = span[k].0.min(it);
                span[k].1 = it;
            }
        }
        it += 1;
        assert!(it < 10_000, "no termination");
    }
    (span, toks)
}

/// Lane hsjf: with herd and sjf both on and a free slot, a short unrelated prompt that arrives while the herd leader is mid-prefill
/// is prefilled before the leader's remaining tokens (it overtakes the leader; the held followers do not matter), where the herd
/// alone leaves it behind everyone (fifo). Every reply is the herd-off fifo one.
#[test]
fn herd_and_shortest_short_prompt_overtakes_the_leader() {
    let (n, short_at) = (4usize, 6u64);
    let (_, base) = herd_plus_short(n + 1, false, PrefillOrder::Fifo, n, short_at);
    let (fifo, fifo_toks) = herd_plus_short(n + 1, true, PrefillOrder::Fifo, n, short_at);
    let (both, both_toks) = herd_plus_short(n + 1, true, PrefillOrder::Shortest, n, short_at);
    // herd alone, fifo: the leader (request 0) prefills its 264 tokens first, the short prompt is last
    eprintln!("first/last token iteration (short last): fifo {fifo:?}, both {both:?}");
    assert!(fifo[n].0 > fifo[0].0, "fifo: short {:?} leader {:?}", fifo[n], fifo[0]);
    // both on: the short prompt (20 tokens) gets its chunks right away, its first token comes long before the leader's
    assert!(both[n].0 < both[0].0, "both on: short {:?} leader {:?}", both[n], both[0]);
    assert!(both[n].0 <= short_at + 6, "the short prompt is picked within a few iterations: {:?}", both[n]);
    assert_eq!(fifo_toks, base, "replies are unchanged (herd, fifo)");
    assert_eq!(both_toks, base, "replies are unchanged (herd + shortest)");
}

/// Lane hsjf, the mflip finding (8 herd prompts + 1 short one at `--max-batch 8`: the short prompt's TTFT equalled the herd's end):
/// with as many slots as herd requests the short prompt has no slot, whatever the prefill order. Admission is FIFO and nobody is
/// preempted, so it is admitted when the first herd request finishes decoding, after the leader has prefilled. Not a
/// `prefill_pick` effect: the same run with one more slot has it overtake the leader.
#[test]
fn herd_and_shortest_short_prompt_waits_for_a_slot_when_the_herd_fills_them() {
    let (n, short_at) = (4usize, 6u64);
    let (full, full_toks) = herd_plus_short(n, true, PrefillOrder::Shortest, n, short_at);
    let (free, free_toks) = herd_plus_short(n + 1, true, PrefillOrder::Shortest, n, short_at);
    eprintln!("first/last token iteration (short last): full {full:?}, free {free:?}");
    let first_herd_end = (0..n).map(|k| full[k].1).min().unwrap();
    assert!(full[n].0 > first_herd_end, "no slot before a herd request ended: short {:?}, herd {:?}", full[n], &full[..n]);
    assert!(full[n].0 > full[0].0, "and follows the leader's first token: {:?} vs {:?}", full[n], full[0]);
    assert!(free[n].0 < free[0].0, "with a free slot it overtakes the leader: {:?} vs {:?}", free[n], free[0]);
    assert_eq!(full_toks, free_toks, "slot count does not change any reply");
}

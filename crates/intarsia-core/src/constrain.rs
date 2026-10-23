//! Constrained decoding hooks (structured output, docs/structured-output.md).
//!
//! The engine knows nothing about grammars: a request carries a [`ConstraintSpec`] (compiled by the server, e.g. a
//! JSON schema through llguidance), each sequence starts its own [`TokenConstraint`] from it, and the sampler masks
//! the logits with the [`TokenMask`] the constraint computes for every step. Unconstrained requests never touch this
//! module, so their path (and their greedy ids) is unchanged.

use crate::error::{Error, Result};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Opt-in per-phase timing of the constrained step (`INTARSIA_CONSTRAIN_PROFILE=1`): cumulative nanoseconds in the
/// mask worker, the wait for it, the thread spawn, `commit`, the mask apply and the sampler, printed to stderr every
/// 100 constrained steps. Diagnostic only; off it costs one cached bool test per phase.
pub(crate) mod prof {
    use super::*;

    pub(crate) static COMPUTE: AtomicU64 = AtomicU64::new(0);
    pub(crate) static WAIT: AtomicU64 = AtomicU64::new(0);
    pub(crate) static SPAWN: AtomicU64 = AtomicU64::new(0);
    pub(crate) static COMMIT: AtomicU64 = AtomicU64::new(0);
    pub(crate) static APPLY: AtomicU64 = AtomicU64::new(0);
    pub(crate) static SAMPLE: AtomicU64 = AtomicU64::new(0);
    pub(crate) static STEPS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("INTARSIA_CONSTRAIN_PROFILE").is_ok_and(|v| v != "0" && !v.is_empty()))
    }

    /// Start a timer when profiling is on.
    pub(crate) fn t0() -> Option<Instant> {
        on().then(Instant::now)
    }

    /// Add the time since `t` to `acc`.
    pub(crate) fn add(acc: &AtomicU64, t: Option<Instant>) {
        if let Some(t) = t {
            acc.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    /// Count a constrained step; every 100th prints the cumulative totals.
    pub(crate) fn step() {
        if !on() {
            return;
        }
        let n = STEPS.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 100 == 0 {
            let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
            eprintln!(
                "constrain-profile steps={n} compute_ns={} wait_ns={} spawn_ns={} commit_ns={} apply_ns={} sample_ns={}",
                g(&COMPUTE),
                g(&WAIT),
                g(&SPAWN),
                g(&COMMIT),
                g(&APPLY),
                g(&SAMPLE)
            );
        }
    }
}

/// The allowed set for one sampling step: bit `i % 32` of word `i / 32` allows token `i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenMask {
    pub words: Vec<u32>,
}

impl TokenMask {
    /// No token allowed, room for `n` tokens.
    pub fn none(n: usize) -> TokenMask {
        TokenMask { words: vec![0; n.div_ceil(32)] }
    }

    pub fn allow(&mut self, tok: u32) {
        let i = tok as usize;
        if let Some(w) = self.words.get_mut(i / 32) {
            *w |= 1 << (i % 32);
        }
    }

    pub fn allows(&self, tok: u32) -> bool {
        let i = tok as usize;
        self.words.get(i / 32).is_some_and(|w| w >> (i % 32) & 1 == 1)
    }

    /// Set every disallowed logit to `-inf` (tokens past the mask are disallowed). Returns whether any finite logit is
    /// left, i.e. whether the sampler still has something to choose.
    pub fn apply(&self, logits: &mut [f32]) -> bool {
        let mut any = false;
        for (wi, chunk) in logits.chunks_mut(32).enumerate() {
            let w = self.words.get(wi).copied().unwrap_or(0);
            if w == u32::MAX {
                any |= chunk.iter().any(|x| x.is_finite());
                continue;
            }
            if w == 0 {
                chunk.fill(f32::NEG_INFINITY);
                continue;
            }
            for (b, x) in chunk.iter_mut().enumerate() {
                if w >> b & 1 == 0 {
                    *x = f32::NEG_INFINITY;
                } else if x.is_finite() {
                    any = true;
                }
            }
        }
        any
    }
}

/// A compiled constraint (grammar + token table), shared by every sequence that uses it. `Debug` + `Clone` (through
/// `Arc`) so it can ride on a [`crate::engine::GenRequest`].
pub trait ConstraintSpec: Send + Sync + std::fmt::Debug {
    /// Fresh per-sequence state at the first generated token.
    fn start(&self) -> Result<Box<dyn TokenConstraint>>;
}

/// Per-sequence constraint state.
pub trait TokenConstraint: Send {
    /// Whether the next token is constrained. A lazy constraint (e.g. tool calls under `tool_choice: auto`) waits for
    /// its trigger token; while inactive the request decodes exactly like an unconstrained one (device argmax).
    fn active(&self) -> bool;
    /// The allowed set for the next token. Called only while [`TokenConstraint::active`], possibly on a helper thread
    /// while the forward pass runs.
    fn compute_mask(&mut self) -> Result<TokenMask>;
    /// Feed the token that was chosen (every generated token except the stop / EOS token that ends the request).
    fn commit(&mut self, tok: u32) -> Result<()>;
}

/// A sequence's running constraint: the state plus at most one mask being computed in the background, so the CPU
/// work overlaps the step's forward pass. The mask is computed on one helper thread per sequence, started at the first
/// prefetch and kept for the sequence's life (a thread per step cost more than the mask itself).
pub(crate) struct ConstraintRun {
    state: Arc<Mutex<Box<dyn TokenConstraint>>>,
    /// Wakes the worker for one mask; dropping it ends the worker.
    work: Option<Sender<()>>,
    done: Option<Receiver<Result<TokenMask>>>,
    /// A mask has been requested and not yet taken.
    pending: bool,
    /// No thread could be started: masks are computed inline.
    inline: bool,
}

impl ConstraintRun {
    pub(crate) fn new(state: Box<dyn TokenConstraint>) -> ConstraintRun {
        ConstraintRun { state: Arc::new(Mutex::new(state)), work: None, done: None, pending: false, inline: false }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Box<dyn TokenConstraint>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wait for the prefetched mask, if one is being computed.
    fn join(&mut self) -> Result<Option<TokenMask>> {
        if !self.pending {
            return Ok(None);
        }
        self.pending = false;
        let t = prof::t0();
        let m = self
            .done
            .as_ref()
            .and_then(|rx| rx.recv().ok())
            .ok_or_else(|| Error::Other("constraint mask worker died".into()))?;
        prof::add(&prof::WAIT, t);
        m.map(Some)
    }

    fn start_worker(&mut self) -> bool {
        let (work_tx, work_rx) = channel::<()>();
        let (done_tx, done_rx) = channel();
        let st = self.state.clone();
        let spawned = std::thread::Builder::new().name("intarsia-mask".into()).spawn(move || {
            while work_rx.recv().is_ok() {
                let t = prof::t0();
                let m = st.lock().unwrap_or_else(|e| e.into_inner()).compute_mask();
                prof::add(&prof::COMPUTE, t);
                if done_tx.send(m).is_err() {
                    break;
                }
            }
        });
        if spawned.is_err() {
            return false;
        }
        self.work = Some(work_tx);
        self.done = Some(done_rx);
        true
    }

    pub(crate) fn active(&self) -> bool {
        // a mask is only ever prefetched for an active constraint
        self.pending || self.lock().active()
    }

    /// Start computing the next step's mask on the helper thread (no-op when inactive or already started; when no
    /// thread can be started, [`ConstraintRun::mask`] computes it inline).
    pub(crate) fn prefetch(&mut self) {
        if self.pending || self.inline || !self.lock().active() {
            return;
        }
        let ts = prof::t0();
        if self.work.is_none() && !self.start_worker() {
            self.inline = true;
            return;
        }
        let sent = self.work.as_ref().is_some_and(|w| w.send(()).is_ok());
        prof::add(&prof::SPAWN, ts);
        self.pending = sent;
    }

    /// The mask for this step: the prefetched one, or computed now.
    pub(crate) fn mask(&mut self) -> Result<TokenMask> {
        if let Some(m) = self.join()? {
            return Ok(m);
        }
        self.lock().compute_mask()
    }

    pub(crate) fn commit(&mut self, tok: u32) -> Result<()> {
        // a prefetched mask nobody took (cannot happen in step order, but never commit under a running worker)
        let _ = self.join();
        let t = prof::t0();
        let r = self.lock().commit(tok);
        prof::add(&prof::COMMIT, t);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_bits_and_apply() {
        let mut m = TokenMask::none(70);
        assert_eq!(m.words.len(), 3);
        m.allow(3);
        m.allow(64);
        m.allow(1000); // out of range: ignored
        assert!(m.allows(3) && m.allows(64) && !m.allows(4) && !m.allows(1000));
        let mut l: Vec<f32> = (0..70).map(|i| i as f32).collect();
        assert!(m.apply(&mut l));
        for (i, x) in l.iter().enumerate() {
            assert_eq!(x.is_finite(), i == 3 || i == 64, "token {i}");
        }
        // logits wider than the mask: the tail is disallowed
        let mut l = vec![0.0f32; 100];
        assert!(m.apply(&mut l));
        assert!(l[96..].iter().all(|x| *x == f32::NEG_INFINITY));
        // nothing allowed
        let mut l = vec![1.0f32; 10];
        assert!(!TokenMask::none(10).apply(&mut l));
    }

    #[test]
    fn full_words_keep_logits() {
        let m = TokenMask { words: vec![u32::MAX] };
        let mut l = vec![0.5f32; 32];
        assert!(m.apply(&mut l));
        assert!(l.iter().all(|x| *x == 0.5));
    }

    /// Allows only token `next`; commit advances it.
    struct Counter {
        next: u32,
        on: bool,
    }
    impl TokenConstraint for Counter {
        fn active(&self) -> bool {
            self.on
        }
        fn compute_mask(&mut self) -> Result<TokenMask> {
            let mut m = TokenMask::none(16);
            m.allow(self.next);
            Ok(m)
        }
        fn commit(&mut self, tok: u32) -> Result<()> {
            self.next = tok + 1;
            Ok(())
        }
    }

    #[test]
    fn prefetch_matches_sync() {
        let mut a = ConstraintRun::new(Box::new(Counter { next: 2, on: true }));
        let mut b = ConstraintRun::new(Box::new(Counter { next: 2, on: true }));
        for _ in 0..5 {
            a.prefetch();
            a.prefetch(); // idempotent
            let (ma, mb) = (a.mask().unwrap(), b.mask().unwrap());
            assert_eq!(ma, mb);
            let t = (0..16).find(|&t| ma.allows(t)).unwrap();
            a.commit(t).unwrap();
            b.commit(t).unwrap();
        }
        // an unused prefetch does not lose the state
        a.prefetch();
        a.commit(9).unwrap();
        assert!(a.mask().unwrap().allows(10));
        // inactive: no prefetch
        let mut c = ConstraintRun::new(Box::new(Counter { next: 0, on: false }));
        c.prefetch();
        assert!(!c.active() && !c.pending);
    }
}

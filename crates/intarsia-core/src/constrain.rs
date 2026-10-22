//! Constrained decoding hooks (structured output, docs/structured-output.md).
//!
//! The engine knows nothing about grammars: a request carries a [`ConstraintSpec`] (compiled by the server, e.g. a
//! JSON schema through llguidance), each sequence starts its own [`TokenConstraint`] from it, and the sampler masks
//! the logits with the [`TokenMask`] the constraint computes for every step. Unconstrained requests never touch this
//! module, so their path (and their greedy ids) is unchanged.

use crate::error::{Error, Result};
use std::sync::mpsc::{channel, Receiver};
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
/// work overlaps the step's forward pass.
pub(crate) struct ConstraintRun {
    state: Arc<Mutex<Box<dyn TokenConstraint>>>,
    job: Option<Receiver<Result<TokenMask>>>,
}

impl ConstraintRun {
    pub(crate) fn new(state: Box<dyn TokenConstraint>) -> ConstraintRun {
        ConstraintRun { state: Arc::new(Mutex::new(state)), job: None }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Box<dyn TokenConstraint>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wait for the prefetched mask, if one is being computed.
    fn join(&mut self) -> Result<Option<TokenMask>> {
        let Some(rx) = self.job.take() else { return Ok(None) };
        let t = prof::t0();
        let m = rx.recv().map_err(|_| Error::Other("constraint mask worker died".into()))?;
        prof::add(&prof::WAIT, t);
        m.map(Some)
    }

    pub(crate) fn active(&self) -> bool {
        // a mask is only ever prefetched for an active constraint
        self.job.is_some() || self.lock().active()
    }

    /// Start computing the next step's mask on a helper thread (no-op when inactive or already started; when no thread
    /// can be started, [`ConstraintRun::mask`] computes it inline).
    pub(crate) fn prefetch(&mut self) {
        if self.job.is_some() || !self.lock().active() {
            return;
        }
        let (tx, rx) = channel();
        let st = self.state.clone();
        let ts = prof::t0();
        let spawned = std::thread::Builder::new().name("intarsia-mask".into()).spawn(move || {
            let t = prof::t0();
            let m = st.lock().unwrap_or_else(|e| e.into_inner()).compute_mask();
            prof::add(&prof::COMPUTE, t);
            let _ = tx.send(m);
        });
        prof::add(&prof::SPAWN, ts);
        if spawned.is_ok() {
            self.job = Some(rx);
        }
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
        assert!(!c.active() && c.job.is_none());
    }
}

//! Fault injection for the continuous-batching engine (cargo feature `fault-inject`, off by default).
//!
//! The scheduler owns a [`Plan`] (read once from `INTARSIA_FAULT_INJECT` when the scheduler is built) and shows it every
//! batched decode step through [`Plan::observe`]. When the plan's trigger matches, the scheduler [`arm`]s the process-wide
//! fault. The next `Backend::synchronize` that reaches [`take`] fails: the Metal backend returns `Error::Backend` exactly
//! where a command buffer that finished in the `Error` status would surface ([`Mode::Err`]), or panics ([`Mode::Panic`]).
//! The plan fires once.
//!
//! Spec grammar (`INTARSIA_FAULT_INJECT`): `<mode>:<trigger>[,waiting=Q][,pos=P]`
//!
//! * mode: `err` (backend error) or `panic`
//! * trigger `after=N`: the decode step after `N` completed ones (the `N+1`-th call of `decode_step`)
//! * trigger `mixed=R`: the first decode step with at least `R` rows while a request is mid-prefill (its next chunk runs
//!   in the same iteration, after the decode step)
//! * `waiting=Q` (either trigger): additionally at least `Q` requests are queued behind the full slot set
//! * `pos=P` (either trigger): additionally some prefilling request has already prefilled at least `P` prompt tokens
//!   (the failure then hits a prompt that is part-way through its chunks, not one that has not started)
//!
//! With the feature off this module is not compiled, so a normal build carries no hook.

use std::sync::atomic::{AtomicU8, Ordering};

/// How the armed fault shows up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The synchronize that reaches it returns a backend error.
    Err,
    /// The synchronize that reaches it panics.
    Panic,
}

/// When a [`Plan`] fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// After this many decode steps completed.
    AfterSteps(u64),
    /// First decode step with at least this many rows and a prefilling request.
    Mixed(usize),
}

/// One planned fault.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub mode: Mode,
    pub trigger: Trigger,
    /// Also require at least this many queued requests.
    pub min_waiting: usize,
    /// Also require a prefilling request that has prefilled at least this many prompt tokens.
    pub min_prefilled: usize,
    steps_seen: u64,
    fired: bool,
}

impl Plan {
    pub fn new(mode: Mode, trigger: Trigger) -> Self {
        Plan { mode, trigger, min_waiting: 0, min_prefilled: 0, steps_seen: 0, fired: false }
    }

    /// Also require at least `q` queued requests.
    pub fn with_waiting(mut self, q: usize) -> Self {
        self.min_waiting = q;
        self
    }

    /// Also require a prefilling request with at least `p` prompt tokens prefilled.
    pub fn with_prefilled(mut self, p: usize) -> Self {
        self.min_prefilled = p;
        self
    }

    /// Parse a spec (see the module docs).
    pub fn parse(spec: &str) -> Result<Plan, String> {
        let (mode, rest) =
            spec.split_once(':').ok_or_else(|| format!("fault spec '{spec}': expected <mode>:<trigger>"))?;
        let mode = match mode.trim() {
            "err" => Mode::Err,
            "panic" => Mode::Panic,
            m => return Err(format!("fault spec '{spec}': mode '{m}' is not err or panic")),
        };
        let mut trigger = None;
        let (mut min_waiting, mut min_prefilled) = (0, 0);
        for part in rest.split(',') {
            let (k, v) =
                part.split_once('=').ok_or_else(|| format!("fault spec '{spec}': '{part}' is not key=value"))?;
            let n: u64 = v.trim().parse().map_err(|_| format!("fault spec '{spec}': '{v}' is not a number"))?;
            match k.trim() {
                "after" => trigger = Some(Trigger::AfterSteps(n)),
                "mixed" if n >= 1 => trigger = Some(Trigger::Mixed(n as usize)),
                "mixed" => return Err(format!("fault spec '{spec}': mixed needs at least 1 row")),
                "waiting" => min_waiting = n as usize,
                "pos" => min_prefilled = n as usize,
                k => return Err(format!("fault spec '{spec}': unknown key '{k}' (after, mixed, waiting, pos)")),
            }
        }
        let trigger = trigger.ok_or_else(|| format!("fault spec '{spec}': needs after=N or mixed=R"))?;
        Ok(Plan::new(mode, trigger).with_waiting(min_waiting).with_prefilled(min_prefilled))
    }

    /// The plan in `INTARSIA_FAULT_INJECT`, if set and non-empty.
    pub fn from_env() -> Result<Option<Plan>, String> {
        match std::env::var("INTARSIA_FAULT_INJECT") {
            Ok(s) if !s.trim().is_empty() => Plan::parse(s.trim()).map(Some),
            _ => Ok(None),
        }
    }

    /// Whether the plan has fired.
    pub fn fired(&self) -> bool {
        self.fired
    }

    /// Show the plan the decode step about to run: `rows` decoding rows, `prefilling` requests mid-prefill, `waiting`
    /// queued requests, the furthest of which has prefilled `prefilled` prompt tokens. Returns the mode when the plan
    /// fires now (once).
    pub fn observe(&mut self, rows: usize, prefilling: usize, waiting: usize, prefilled: usize) -> Option<Mode> {
        if self.fired {
            return None;
        }
        let step = self.steps_seen;
        self.steps_seen += 1;
        let hit = match self.trigger {
            Trigger::AfterSteps(n) => step >= n,
            Trigger::Mixed(r) => rows >= r && prefilling >= 1,
        };
        if hit && waiting >= self.min_waiting && prefilled >= self.min_prefilled {
            self.fired = true;
            Some(self.mode)
        } else {
            None
        }
    }
}

const NONE: u8 = 0;
const ERR: u8 = 1;
const PANIC: u8 = 2;
static ARMED: AtomicU8 = AtomicU8::new(NONE);

/// Make the next [`take`] return `mode`.
pub fn arm(mode: Mode) {
    ARMED.store(if mode == Mode::Err { ERR } else { PANIC }, Ordering::SeqCst);
}

/// Consume the armed fault, if any (called by the backend's synchronize).
pub fn take() -> Option<Mode> {
    match ARMED.swap(NONE, Ordering::SeqCst) {
        ERR => Some(Mode::Err),
        PANIC => Some(Mode::Panic),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_specs() {
        assert_eq!(Plan::parse("err:after=20").unwrap(), Plan::new(Mode::Err, Trigger::AfterSteps(20)));
        assert_eq!(
            Plan::parse("panic:mixed=7,waiting=2,pos=512").unwrap(),
            Plan::new(Mode::Panic, Trigger::Mixed(7)).with_waiting(2).with_prefilled(512)
        );
        for bad in ["", "err", "boom:after=1", "err:after", "err:after=x", "err:mixed=0", "err:waiting=1", "err:foo=1"]
        {
            assert!(Plan::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn after_steps_fires_once_on_the_next_step() {
        let mut p = Plan::new(Mode::Err, Trigger::AfterSteps(3));
        let hits: Vec<_> = (0..8).map(|_| p.observe(2, 0, 0, 0)).collect();
        assert_eq!(hits, [None, None, None, Some(Mode::Err), None, None, None, None]);
        assert!(p.fired());
    }

    #[test]
    fn mixed_needs_rows_a_prefilling_request_and_the_extra_conditions() {
        let mut p = Plan::new(Mode::Panic, Trigger::Mixed(7)).with_waiting(1).with_prefilled(512);
        assert_eq!(p.observe(7, 0, 3, 0), None, "no prefill in flight");
        assert_eq!(p.observe(6, 1, 3, 600), None, "too few rows");
        assert_eq!(p.observe(7, 1, 0, 600), None, "nobody queued");
        assert_eq!(p.observe(7, 1, 3, 256), None, "prefill not far enough");
        assert_eq!(p.observe(8, 2, 1, 512), Some(Mode::Panic));
        assert_eq!(p.observe(8, 2, 1, 900), None, "fires once");
    }

    #[test]
    fn arm_take_is_one_shot() {
        // the only test that touches the process-wide flag in this crate's lib tests
        assert_eq!(take(), None);
        arm(Mode::Err);
        assert_eq!(take(), Some(Mode::Err));
        assert_eq!(take(), None);
        arm(Mode::Panic);
        assert_eq!(take(), Some(Mode::Panic));
    }
}

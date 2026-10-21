//! Device-side token feedback for greedy single-sequence decode (lane T1, docs/results/2026-10-21-t1-token-feedback.md).
//!
//! The eager decode step reads its sampled token on the host before the next step can be encoded, so the device
//! idles for the host wake, the encode of the next step's first buffer and its launch (0.39 ms of a 28.6 ms step on
//! Metal at 4k). With feedback the argmax of step N stays on device in a two-slot ring inside `ws.tok` (slots 1 and 2;
//! slot 0 stays the plain path's) and step N+1's embedding reads it from there: the engine encodes and submits step
//! N+1 before it reads step N's token ([`Backend::fence`] / [`Backend::read_after`] wait only for step N).
//!
//! Same kernels, same inputs: only the buffer the embedding reads its id from and where command buffers end change.
//! A step whose input token was not read yet is committed to the [`SequenceState`] with [`UNREAD`] in `tokens`; the
//! engine replaces it once the token is read, and rolls the step back (truncate) when the sequence stops before it.

use super::*;
use std::sync::atomic::Ordering;

/// `SequenceState::tokens` entry of a fed token the host has not read yet (never a valid id).
pub(crate) const UNREAD: u32 = u32::MAX;

/// The two ring slots (i32 elements of `ws.tok`) a feedback step's argmax goes to.
pub(crate) const RING: [usize; 2] = [1, 2];

/// The initial switch: the backend's `(supported, default)` with `INTARSIA_TOKEN_FEEDBACK=0|1` overriding the default.
pub(super) fn initial(backend: (bool, bool), env: Option<&str>) -> bool {
    let (supported, default) = backend;
    supported
        && match env.map(str::trim) {
            Some("1") | Some("on") | Some("true") => true,
            Some("0") | Some("off") | Some("false") => false,
            _ => default,
        }
}

/// Where a feedback step's input token comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Feed {
    /// A token the host knows (uploaded to `ws.ids`, as on the plain path).
    Host(u32),
    /// The argmax a previous feedback step left in this ring slot.
    Slot(usize),
}

/// A feedback step not yet read: its fence (`None` while it is held, [`Model::feedback_release`]) and the ring slot
/// its argmax goes to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pending {
    pub(crate) fence: Option<u64>,
    pub(crate) slot: usize,
}

/// Dispatches of a speculative step submitted before the previous token is read (`INTARSIA_TOKEN_FEEDBACK_HOLD`); the
/// rest is held until the token says the step is needed, so a stop wastes only these. They must cover the host's wake,
/// token handling and the release (about 0.3 ms on Metal); 16 dispatches of the 8B are about 1 ms of GPU work.
/// 0 submits the whole step at once.
pub(crate) const HOLD: usize = 16;

pub(super) fn hold_from_env(env: Option<&str>) -> usize {
    env.and_then(|v| v.trim().parse().ok()).unwrap_or(HOLD)
}

impl Model {
    /// Turn device-side token feedback on or off for greedy decode; returns whether it is on (never on a backend
    /// without support).
    pub fn set_token_feedback(&self, on: bool) -> bool {
        let on = on && self.backend.token_feedback().0;
        self.token_feedback.store(on, Ordering::Relaxed);
        on
    }

    /// Whether greedy decode uses device-side token feedback where it is safe ([`Model::feedback_ok`]).
    pub fn token_feedback(&self) -> bool {
        self.token_feedback.load(Ordering::Relaxed)
    }

    /// Whether a greedy decode on `st` may take feedback steps: switched on, one device (no tensor-parallel peers), no
    /// captured graph replay (CUDA / HIP keep their path), attention-only state (a discarded step must leave nothing
    /// that truncation cannot roll back), not a dry run.
    pub(crate) fn feedback_ok(&self, st: &SequenceState) -> bool {
        self.token_feedback()
            && self.peers.is_empty()
            && st.peers.is_empty()
            && st.id != 0
            && !st.has_recurrent()
            && !self.graphs.usable(self)
            && self.recorder.lock().unwrap().is_none()
    }

    /// Encode one greedy decode step at position `st.len`. Its input is `feed`, its argmax goes to ring slot `out`.
    /// `held`: submit only its first [`HOLD`] dispatches and hold the rest ([`Model::feedback_release`] or
    /// [`Model::feedback_discard`]); otherwise submit it all and fence it. Commits the position to `st` (with [`UNREAD`]
    /// as the token for a slot feed).
    pub(crate) fn feedback_step(&self, st: &mut SequenceState, feed: Feed, out: usize, held: bool) -> Result<Pending> {
        if !RING.contains(&out) || feed == Feed::Slot(out) || matches!(feed, Feed::Slot(s) if !RING.contains(&s)) {
            return Err(Error::invalid(format!("token feedback: bad ring slots {feed:?} -> {out}")));
        }
        Self::check_forward(st, &[0], HeadMode::LastArgmax, false)?;
        let profile = self.profile_every > 0 && self.telemetry.steps_total() % self.profile_every as u64 == 0;
        self.telemetry.decode_eager_step(profile);
        let held = held && self.feedback_hold > 0;
        if held {
            self.backend.hold(0, self.feedback_hold)?;
        }
        let ring = self.view(&self.ws.tok, DType::I32, &[4]);
        let hn = match feed {
            Feed::Host(id) => self.run_layers(Seqs::One(st), &[id], profile, None, true, None)?,
            Feed::Slot(s) => self.run_layers(Seqs::One(st), &[0], profile, None, false, Some(&ring.rows(s, 1)))?,
        };
        self.head_to_device_into(&hn, 0, profile, ring.rows(out, 1))?;
        let fence = if held { None } else { Some(self.backend.fence(0)?) };
        st.commit(&[match feed {
            Feed::Host(id) => id,
            Feed::Slot(_) => UNREAD,
        }]);
        self.telemetry.step();
        Ok(Pending { fence, slot: out })
    }

    /// Submit the held rest of `p` (a step [`Model::feedback_step`] held) and fence it.
    pub(crate) fn feedback_release(&self, p: Pending) -> Result<Pending> {
        if p.fence.is_some() {
            return Ok(p);
        }
        self.backend.release(0)?;
        Ok(Pending { fence: Some(self.backend.fence(0)?), slot: p.slot })
    }

    /// Drop the held rest of a step that is not needed (its committed position is rolled back by the caller).
    pub(crate) fn feedback_discard(&self) -> Result<()> {
        self.backend.discard(0)
    }

    /// Wait for the step behind `p` (not for later ones) and read its token. `p` must be released.
    pub(crate) fn feedback_read(&self, p: Pending) -> Result<u32> {
        let fence = p.fence.ok_or_else(|| Error::invalid("token feedback: read of a held step"))?;
        let mut b = [0u8; 4];
        self.backend.read_after(fence, &self.ws.tok, p.slot * 4, &mut b, 0)?;
        let t = i32::from_le_bytes(b) as u32;
        if t as usize >= self.dims.vocab {
            return Err(Error::backend(format!("token feedback: argmax {t} >= vocab {}", self.dims.vocab)));
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_switch() {
        assert!(!initial((false, true), Some("1")), "never on without backend support");
        assert!(!initial((true, false), None));
        assert!(initial((true, true), None));
        assert!(initial((true, false), Some("1")));
        assert!(!initial((true, true), Some("0")));
        assert!(initial((true, true), Some("junk")));
    }
}

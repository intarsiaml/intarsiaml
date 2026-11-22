//! Host-side step timeline (`INTARSIA_HOST_PROFILE=1`, off by default; one relaxed atomic load per span when off).
//!
//! The scheduler thread runs every iteration serially: reap, admit, plan the decode rows, run the model step (graph launch,
//! then a synchronising download), sample, detokenize, stream, then one prefill chunk. Each phase is a [`Cat`]; the spans are
//! summed per iteration, the iteration is classed by what ran in it ([`Kind`]), and every 256 iterations of a class (and at
//! [`flush`]) one table line goes to stderr: mean ms per iteration of that class and per category. GPU-busy time of a decode
//! step is `Launch + Sync`; everything else in a decode-only iteration is the host gap the GPU idles through.
//!
//! The solo decode loop (`engine::run_seq`, what `generate` and a `serve` without `--max-batch` run) is profiled the same way
//! (lane ink-hostgap): one iteration per decode step (`on_token`, `prepare`, the step as `step-setup` net of `sync`, `step-tail`
//! and `softcap`, `accept` = the host sampler), its `sync` the logits download's wait (plain path) or the token read of a
//! token-feedback step (there `launch` is the encode of the next step, held after its first dispatches, and `gpu-idle-gap` the
//! host time from the token read to the release of that step, which the GPU covers with the dispatches it already has).
//!
//! A speculative cycle of that loop (the eager draft + verify of `Seq::prepare_with`) is its own class, `spec-cycle`: `draft` and
//! `verify` (device time and syncs included) out of `prepare`, `rows` the tokens the cycle emitted, one iteration from one cycle
//! to the next (the `on_token` of its tokens included), and `gpu-idle-gap` the host time from the verify's token download to the
//! next draft.
//!
//! The profile only measures: it changes no output and no schedule.
use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Cat {
    /// cancellation reaping and queue admission (`Scheduler::step` phases 1-2)
    ReapAdmit,
    /// planning the decode rows: `Seq::prepare` per row (draft planning, head mode, structured-output mask)
    Prepare,
    /// `SlotModel::decode_step` outside `Model::forward_batch` (row validation, borrow splitting)
    StepSetup,
    /// `forward_batch` before the launch: argument checks, token / row-table uploads
    Upload,
    /// graph replay call (or the eager launch of the layers when no graph is usable) and its telemetry
    Launch,
    /// the synchronising download of the step's tokens: the wait for the GPU
    Sync,
    /// `forward_batch` after the sync: building `StepOut`s, committing the tokens to the slot states, telemetry
    StepTail,
    /// `Seq::accept` per row (token from the step output, sampler for host-sampled rows)
    Accept,
    /// scheduler metrics (TTFT / ITL histograms under their lock)
    Metrics,
    /// `Seq::on_token` outside the next two: stop ids, context push, grammar commit, stop matcher
    OnToken,
    /// incremental detokenization of the new token
    Detok,
    /// handing the `GenEvent::Token` to the request's channel
    Send,
    /// finishing requests (`end_active`: final flush, `Done` event, telemetry)
    Finish,
    /// the prefill chunk of the iteration, end to end
    Prefill,
    /// the host logit soft-cap of a step that reads its logits (Gemma 4, Inkling-2; `Model::softcap_logits`), inside `StepTail`
    Softcap,
    /// a speculative cycle's draft (`Model::dflash_draft` / `mtp_draft`, host and device, its syncs included), inside `Prepare`
    Draft,
    /// a speculative cycle's verify pass, acceptance and rollback (`Seq::verify`, ending in the token download), inside `Prepare`
    Verify,
}
pub const NCAT: usize = 17;
const NAMES: [&str; NCAT] = [
    "reap+admit", "prepare", "step-setup", "upload", "launch", "sync", "step-tail", "accept", "metrics", "on_token", "detok", "send", "finish", "prefill", "softcap",
    "draft", "verify",
];

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
enum Kind {
    /// a decode step and no prefill chunk
    DecodeOnly,
    /// a decode step and a prefill chunk
    Mixed,
    /// a prefill chunk and no decode step
    PrefillOnly,
    /// a speculative draft + verify cycle of the solo decode loop and no prefill chunk (`rows` = tokens the cycle emitted)
    Spec,
}
const NKIND: usize = 4;
const KIND_NAMES: [&str; NKIND] = ["decode-only", "decode+prefill", "prefill-only", "spec-cycle"];

#[derive(Default)]
struct Acc {
    iters: u64,
    wall_ns: u128,
    cat_ns: [u128; NCAT],
    /// time between the end of one decode step's sync and the next one's launch, decode-only iterations that follow each
    /// other: the span the GPU has nothing queued
    gap_ns: u128,
    gaps: u64,
    rows: u64,
    since_print: u64,
}

#[derive(Default)]
struct Prof {
    cur: [u128; NCAT],
    acc: [Acc; NKIND],
    decoded: bool,
    prefilled: bool,
    /// a speculative cycle ran ([`spec_cycle`])
    spec: bool,
    rows: usize,
    /// instant the last decode step's sync returned
    last_sync_end: Option<Instant>,
    /// the running iteration marked a sync end (an iteration that did not leaves no `last_sync_end` for the next one: a stale
    /// mark would count everything since as a gap)
    synced_this: bool,
    /// the previous iteration was decode-only or a spec cycle (the gap is only meaningful between two of those)
    prev_decode_only: bool,
    gap_this: Option<u128>,
}

thread_local! {
    static P: RefCell<Prof> = RefCell::new(Prof::default());
}
static ENABLED: AtomicU8 = AtomicU8::new(2);

/// Whether `INTARSIA_HOST_PROFILE=1` is set (read once).
#[inline]
pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("INTARSIA_HOST_PROFILE").is_ok_and(|v| v == "1");
            ENABLED.store(on as u8, Ordering::Relaxed);
            on
        }
    }
}

/// A running span; adds its elapsed time to the category when dropped.
pub struct Span(Option<(Cat, Instant)>);

#[inline]
pub fn span(c: Cat) -> Span {
    Span(if enabled() { Some((c, Instant::now())) } else { None })
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((c, t)) = self.0.take() {
            add_ns(c, t.elapsed().as_nanos());
        }
    }
}

/// Adds `ns` to category `c` of the running iteration (what a span does when it drops; also the seam the tests feed).
#[inline]
fn add_ns(c: Cat, ns: u128) {
    P.with(|p| p.borrow_mut().cur[c as usize] += ns);
}

/// Marks the instant the step's launch begins (call at the top of the launch span's work).
#[inline]
pub fn mark_launch_begin() {
    if !enabled() {
        return;
    }
    P.with(|p| {
        let mut p = p.borrow_mut();
        if let (Some(t), true) = (p.last_sync_end, p.prev_decode_only) {
            p.gap_this = Some(t.elapsed().as_nanos());
        }
    });
}

/// Marks the instant the step's synchronising download returned.
#[inline]
pub fn mark_sync_end() {
    if !enabled() {
        return;
    }
    P.with(|p| {
        let mut p = p.borrow_mut();
        p.last_sync_end = Some(Instant::now());
        p.synced_this = true;
    });
}

/// A decode step ran with `rows` rows.
#[inline]
pub fn decoded(rows: usize) {
    if !enabled() {
        return;
    }
    P.with(|p| {
        let mut p = p.borrow_mut();
        p.decoded = true;
        p.rows = p.rows.max(rows);
    });
}

/// A prefill chunk ran.
#[inline]
pub fn prefilled() {
    if !enabled() {
        return;
    }
    P.with(|p| p.borrow_mut().prefilled = true);
}

/// A speculative draft + verify cycle ran and emitted `tokens` tokens (the solo decode loop, lane ink-hostgap).
#[inline]
pub fn spec_cycle(tokens: usize) {
    if !enabled() {
        return;
    }
    P.with(|p| {
        let mut p = p.borrow_mut();
        p.spec = true;
        p.rows = p.rows.max(tokens);
    });
}

/// The running iteration has run a decode step, a prefill chunk or a spec cycle (what [`iter_end`] folds; the solo loop keeps an
/// iteration open across the passes that only emit tokens a cycle already verified).
pub fn pending() -> bool {
    enabled() && P.with(|p| {
        let p = p.borrow();
        p.decoded || p.prefilled || p.spec
    })
}

/// Start of an iteration: zero the per-iteration sums.
#[inline]
pub fn iter_begin() {
    if !enabled() {
        return;
    }
    P.with(|p| {
        let mut p = p.borrow_mut();
        p.cur = [0; NCAT];
        p.decoded = false;
        p.prefilled = false;
        p.spec = false;
        p.synced_this = false;
        p.rows = 0;
        p.gap_this = None;
    });
}

/// End of an iteration of `wall_ns` that did work: fold it into its class.
pub fn iter_end(wall_ns: u128) {
    if !enabled() {
        return;
    }
    let line = P.with(|p| {
        let mut p = p.borrow_mut();
        let k = match (p.decoded || p.spec, p.prefilled) {
            (true, false) if p.spec => Kind::Spec,
            (true, false) => Kind::DecodeOnly,
            (true, true) => Kind::Mixed,
            (false, true) => Kind::PrefillOnly,
            (false, false) => return None,
        };
        let mut cur = p.cur;
        // `StepSetup` is measured around the whole `decode_step`, with the model step inside it
        let inner = cur[Cat::Launch as usize] + cur[Cat::Sync as usize] + cur[Cat::StepTail as usize];
        cur[Cat::StepSetup as usize] = cur[Cat::StepSetup as usize].saturating_sub(inner);
        // `Launch` is measured around the whole call, with the uploads inside it
        cur[Cat::Launch as usize] = cur[Cat::Launch as usize].saturating_sub(cur[Cat::Upload as usize]);
        // `OnToken` is measured around the whole call; Detok and Send are inside it
        cur[Cat::OnToken as usize] = cur[Cat::OnToken as usize].saturating_sub(cur[Cat::Detok as usize] + cur[Cat::Send as usize]);
        // `StepTail` holds the logits soft-cap
        cur[Cat::StepTail as usize] = cur[Cat::StepTail as usize].saturating_sub(cur[Cat::Softcap as usize]);
        // `Prepare` holds a spec cycle's draft and verify
        cur[Cat::Prepare as usize] = cur[Cat::Prepare as usize].saturating_sub(cur[Cat::Draft as usize] + cur[Cat::Verify as usize]);
        let gap = p.gap_this;
        let rows = p.rows as u64;
        p.prev_decode_only = matches!(k, Kind::DecodeOnly | Kind::Spec);
        if !p.synced_this {
            p.last_sync_end = None;
        }
        let a = &mut p.acc[k as usize];
        a.iters += 1;
        a.since_print += 1;
        a.wall_ns += wall_ns;
        a.rows += rows;
        for (s, c) in a.cat_ns.iter_mut().zip(cur) {
            *s += c;
        }
        if let Some(g) = gap {
            a.gap_ns += g;
            a.gaps += 1;
        }
        if a.since_print >= 256 {
            // one window per line: the accumulator restarts
            let line = render(k as usize, a);
            *a = Acc::default();
            Some(line)
        } else {
            None
        }
    });
    if let Some(l) = line {
        eprintln!("{l}");
    }
}

fn render(k: usize, a: &Acc) -> String {
    let n = a.iters.max(1) as f64;
    let ms = |x: u128| x as f64 / n / 1e6;
    let mut s = format!(
        "host-profile {:<14} iters {:>6} rows {:>5.2} wall {:>7.3} ms |",
        KIND_NAMES[k],
        a.iters,
        a.rows as f64 / n,
        ms(a.wall_ns)
    );
    let mut acc_sum = 0u128;
    for (i, name) in NAMES.iter().enumerate() {
        s += &format!(" {name} {:.3}", ms(a.cat_ns[i]));
        acc_sum += a.cat_ns[i];
    }
    s += &format!(" | other {:.3}", ms(a.wall_ns.saturating_sub(acc_sum)));
    if a.gaps > 0 {
        s += &format!(" | gpu-idle-gap {:.3} ms ({} gaps)", a.gap_ns as f64 / a.gaps as f64 / 1e6, a.gaps);
    }
    s
}

/// Print (and restart) every class that has iterations: the scheduler calls it when it goes idle, so a burst's tail is not lost.
pub fn flush() {
    if !enabled() {
        return;
    }
    let lines: Vec<String> = P.with(|p| {
        let mut p = p.borrow_mut();
        let mut v = Vec::new();
        for k in 0..NKIND {
            if p.acc[k].iters > 0 {
                v.push(render(k, &p.acc[k]));
                p.acc[k] = Acc::default();
            }
        }
        p.prev_decode_only = false;
        p.last_sync_end = None;
        v
    });
    for l in lines {
        eprintln!("{l}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests flip the process-wide `ENABLED`: one at a time.
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The iteration is classed by what ran in it, per-category time adds up across iterations, nested categories are
    /// netted out of their enclosing one, and the gap between two decode-only iterations is counted once. Durations are
    /// fed explicitly through `add_ns` (the call a dropping span makes) so no wall-clock thresholds are involved;
    /// `mark_launch_begin`/`mark_sync_end` only feed the gap, which is asserted by count.
    #[test]
    fn spans_fold_into_the_iteration_class() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        const MS: u128 = 1_000_000;
        ENABLED.store(1, Ordering::Relaxed);
        P.with(|p| *p.borrow_mut() = Prof::default());
        for _ in 0..2 {
            iter_begin();
            // `StepSetup` wraps launch (2 ms) and sync (3 ms) plus 0.5 ms of its own
            mark_launch_begin();
            add_ns(Cat::Launch, 2 * MS);
            add_ns(Cat::Sync, 3 * MS);
            mark_sync_end();
            add_ns(Cat::StepSetup, 5 * MS + MS / 2);
            // `OnToken` wraps `Detok`, nothing of its own
            add_ns(Cat::OnToken, MS);
            add_ns(Cat::Detok, MS);
            decoded(4);
            iter_end(8 * MS);
        }
        P.with(|p| {
            let p = p.borrow();
            let a = &p.acc[Kind::DecodeOnly as usize];
            assert_eq!((a.iters, a.rows), (2, 8));
            assert_eq!(a.cat_ns[Cat::Sync as usize], 6 * MS, "{:?}", a.cat_ns);
            assert_eq!(a.cat_ns[Cat::Launch as usize], 4 * MS, "{:?}", a.cat_ns);
            // `StepSetup` was measured around launch and sync: only its own remainder is left
            assert_eq!(a.cat_ns[Cat::StepSetup as usize], MS, "{:?}", a.cat_ns);
            assert_eq!(a.cat_ns[Cat::OnToken as usize], 0, "{:?}", a.cat_ns);
            assert_eq!(a.gaps, 1, "one gap, between the two iterations");
            assert_eq!(p.acc[Kind::Mixed as usize].iters, 0);
        });
        let l = P.with(|p| render(Kind::DecodeOnly as usize, &p.borrow().acc[Kind::DecodeOnly as usize]));
        assert!(l.contains("decode-only") && l.contains("gpu-idle-gap"), "{l}");
        ENABLED.store(2, Ordering::Relaxed);
    }

    /// A gap is counted only from a sync end marked in the previous iteration: one whose step was not split into launch and sync
    /// (a path without the hooks) leaves none, so the next launch does not count everything since an older sync as a gap.
    #[test]
    fn a_gap_needs_a_sync_in_the_previous_iteration() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        ENABLED.store(1, Ordering::Relaxed);
        P.with(|p| *p.borrow_mut() = Prof::default());
        // synced, unsynced, then two launches: only the one after a synced iteration counts
        for synced in [true, false, true, true] {
            iter_begin();
            mark_launch_begin();
            if synced {
                mark_sync_end();
            }
            decoded(1);
            iter_end(1_000_000);
        }
        P.with(|p| assert_eq!(p.borrow().acc[Kind::DecodeOnly as usize].gaps, 1));
        ENABLED.store(2, Ordering::Relaxed);
    }

    /// A spec cycle (lane ink-hostgap) is its own class: draft and verify come out of `prepare`, `rows` is the cycle's tokens,
    /// an iteration with only emitted tokens is not folded (`pending` false, the solo loop keeps it open), and the gap between
    /// two cycles is counted.
    #[test]
    fn spec_cycles_are_their_own_class() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        const MS: u128 = 1_000_000;
        ENABLED.store(1, Ordering::Relaxed);
        P.with(|p| *p.borrow_mut() = Prof::default());
        for tokens in [3usize, 1] {
            iter_begin();
            add_ns(Cat::OnToken, MS);
            assert!(!pending(), "emitting a verified token is not a step");
            mark_launch_begin();
            add_ns(Cat::Draft, 2 * MS);
            add_ns(Cat::Verify, 5 * MS);
            mark_sync_end();
            // `Prepare` wraps draft and verify plus 1 ms of its own
            add_ns(Cat::Prepare, 8 * MS);
            spec_cycle(tokens);
            assert!(pending());
            iter_end(10 * MS);
        }
        P.with(|p| {
            let p = p.borrow();
            let a = &p.acc[Kind::Spec as usize];
            assert_eq!((a.iters, a.rows), (2, 4));
            assert_eq!(a.cat_ns[Cat::Prepare as usize], 2 * MS, "{:?}", a.cat_ns);
            assert_eq!(a.cat_ns[Cat::Draft as usize], 4 * MS, "{:?}", a.cat_ns);
            assert_eq!(a.cat_ns[Cat::Verify as usize], 10 * MS, "{:?}", a.cat_ns);
            assert_eq!(a.gaps, 1, "one gap, between the two cycles");
            assert_eq!(p.acc[Kind::DecodeOnly as usize].iters, 0);
        });
        let l = P.with(|p| render(Kind::Spec as usize, &p.borrow().acc[Kind::Spec as usize]));
        assert!(l.contains("spec-cycle") && l.contains(" draft ") && l.contains(" verify "), "{l}");
        ENABLED.store(2, Ordering::Relaxed);
    }
}

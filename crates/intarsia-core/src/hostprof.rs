//! Host-side step timeline (`INTARSIA_HOST_PROFILE=1`, off by default; one relaxed atomic load per span when off).
//!
//! The scheduler thread runs every iteration serially: reap, admit, plan the decode rows, run the model step (graph launch,
//! then a synchronising download), sample, detokenize, stream, then one prefill chunk. Each phase is a [`Cat`]; the spans are
//! summed per iteration, the iteration is classed by what ran in it ([`Kind`]), and every 256 iterations of a class (and at
//! [`flush`]) one table line goes to stderr: mean ms per iteration of that class and per category. GPU-busy time of a decode
//! step is `Launch + Sync`; everything else in a decode-only iteration is the host gap the GPU idles through.
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
}
pub const NCAT: usize = 14;
const NAMES: [&str; NCAT] =
    ["reap+admit", "prepare", "step-setup", "upload", "launch", "sync", "step-tail", "accept", "metrics", "on_token", "detok", "send", "finish", "prefill"];

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
enum Kind {
    /// a decode step and no prefill chunk
    DecodeOnly,
    /// a decode step and a prefill chunk
    Mixed,
    /// a prefill chunk and no decode step
    PrefillOnly,
}
const NKIND: usize = 3;
const KIND_NAMES: [&str; NKIND] = ["decode-only", "decode+prefill", "prefill-only"];

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
    rows: usize,
    /// instant the last decode step's sync returned
    last_sync_end: Option<Instant>,
    /// the previous iteration was decode-only (the gap is only meaningful between two of those)
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
            let ns = t.elapsed().as_nanos();
            P.with(|p| p.borrow_mut().cur[c as usize] += ns);
        }
    }
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
    P.with(|p| p.borrow_mut().last_sync_end = Some(Instant::now()));
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
        let k = match (p.decoded, p.prefilled) {
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
        let gap = p.gap_this;
        let rows = p.rows as u64;
        p.prev_decode_only = matches!(k, Kind::DecodeOnly);
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

    /// Spans add up per category, the iteration is classed by what ran in it, nested categories are netted out, and the
    /// gap between two decode-only iterations is the time from one sync to the next launch.
    #[test]
    fn spans_fold_into_the_iteration_class() {
        ENABLED.store(1, Ordering::Relaxed);
        for it in 0..2 {
            iter_begin();
            {
                let _outer = span(Cat::StepSetup);
                {
                    mark_launch_begin();
                    let _l = span(Cat::Launch);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                {
                    let _s = span(Cat::Sync);
                    std::thread::sleep(std::time::Duration::from_millis(3));
                }
                mark_sync_end();
            }
            {
                let _t = span(Cat::OnToken);
                let _d = span(Cat::Detok);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            decoded(4);
            iter_end(8_000_000);
            let _ = it;
        }
        P.with(|p| {
            let p = p.borrow();
            let a = &p.acc[Kind::DecodeOnly as usize];
            assert_eq!((a.iters, a.rows), (2, 8));
            assert!(a.cat_ns[Cat::Sync as usize] >= 6_000_000, "{:?}", a.cat_ns);
            // `StepSetup` was measured around launch and sync: only its own remainder is left
            assert!(a.cat_ns[Cat::StepSetup as usize] < 2_000_000, "{:?}", a.cat_ns);
            assert_eq!(a.gaps, 1, "one gap, between the two iterations");
            assert_eq!(p.acc[Kind::Mixed as usize].iters, 0);
        });
        let l = P.with(|p| render(Kind::DecodeOnly as usize, &p.borrow().acc[Kind::DecodeOnly as usize]));
        assert!(l.contains("decode-only") && l.contains("gpu-idle-gap"), "{l}");
        ENABLED.store(2, Ordering::Relaxed);
    }
}

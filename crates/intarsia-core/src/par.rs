//! A small persistent thread pool for the host's per-token row work (lane ink-hostgap): the logit soft-cap and the sampler over
//! a 200k-entry logits row cost 3.5 ms per sampled Inkling-2-Mini token on one core while the GPU waits
//! (docs/results/2026-11-21-inkling2-hostgap.md). Work is split into tasks by index and every task writes only its own outputs,
//! so a result never depends on which thread ran which task: the callers get the same bits as the one-thread loop.
//!
//! Whether the callers use the pool ([`enabled`]): `INTARSIA_HOST_THREADS` (read once) is the threads that share a task list, the
//! calling thread included, for every model; `1` (or `0`) turns the pool off. Unset: the model decides at load
//! ([`set_model_default`]: on for Inkling-2, off for every other model until its own gates pass) with the smaller of 8 and
//! `available_parallelism` threads.
//!
//! One caller at a time uses the workers; a second, concurrent caller runs its tasks itself (same result). Workers spin for
//! [`SPIN`] after a task list before they block, so the phases of one token do not pay a wake-up each.
//!
//! Pre-wake (lane ink-hostgap item 7, `INTARSIA_HOST_PREWAKE=1`): a decode step's row work follows a GPU wait of about 15 ms,
//! long enough for the parked workers' cores to idle, so its first task list paid the wake-ups and a cold start. A caller about
//! to block calls [`prewake`]: the workers sleep until [`LEAD`] before the expected end of the wait (the last measured
//! `prewake` -> `run` interval), then spin until a task list arrives or [`MARGIN`] past the expected end, then park again.
//! Which thread runs which task is unchanged in kind, so results are the same bits. `INTARSIA_HOST_QOS=1` also gives the
//! workers the user-interactive QoS class (macOS: the performance cores).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long an idle worker spins for the next task list before it blocks.
const SPIN: Duration = Duration::from_micros(300);
/// Pre-woken workers start spinning this long before the expected end of the caller's wait ([`prewake`]).
pub const LEAD: Duration = Duration::from_micros(700);
/// ... and stop spinning (park) this long after it, if no task list came.
pub const MARGIN: Duration = Duration::from_micros(1500);
/// A `prewake` -> `run` interval longer than this is not a decode step's wait (no update of the estimate).
const EST_MAX: Duration = Duration::from_millis(250);

/// Nanoseconds since the first call (never 0, so 0 can mean "unset").
fn now_ns() -> u64 {
    static BASE: OnceLock<Instant> = OnceLock::new();
    BASE.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
}

/// `INTARSIA_HOST_PREWAKE=1` (read once; default off until its endor gate): [`prewake`] does something, and the pool is on.
pub fn prewake_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("INTARSIA_HOST_PREWAKE").is_ok_and(|v| v.trim() == "1")) && enabled()
}

/// `INTARSIA_HOST_QOS=1` (read once, when the pool starts; default off until its endor gate): the workers run at the
/// user-interactive QoS class (macOS only; elsewhere nothing).
fn qos_on() -> bool {
    std::env::var("INTARSIA_HOST_QOS").is_ok_and(|v| v.trim() == "1")
}

/// The caller is about to block on work (a GPU step) whose result the next [`run`] processes: have the workers hot by then
/// (module docs). The first call only starts the measurement; later ones use the last `prewake` -> `run` interval (an average
/// of the last two). Nothing when the pool is off. Results never depend on it.
pub fn prewake() {
    let Some(p) = pool() else { return };
    let now = now_ns();
    p.pre_t0.store(now, Ordering::Relaxed);
    let est = p.est.load(Ordering::Relaxed);
    if est == 0 {
        return;
    }
    p.hot_at.store((now + est).saturating_sub(LEAD.as_nanos() as u64).max(now), Ordering::Relaxed);
    p.hot_until.store(now + est + MARGIN.as_nanos() as u64, Ordering::Relaxed);
    let mut st = p.st.lock().unwrap_or_else(|e| e.into_inner());
    st.wake += 1;
    drop(st);
    p.work.notify_all();
}

/// The current estimate of the caller's wait (`None` before the first measured `prewake` -> `run`); for tests and benches.
pub fn prewake_estimate() -> Option<Duration> {
    pool().map(|p| p.est.load(Ordering::Relaxed)).filter(|&e| e > 0).map(Duration::from_nanos)
}

/// `INTARSIA_HOST_THREADS` as given (read once): `None` when unset or not a number.
fn env_threads() -> Option<usize> {
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| std::env::var("INTARSIA_HOST_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()))
}

/// Threads that share a task list, the caller included: `INTARSIA_HOST_THREADS`, else the smaller of 8 and the cores (at least 1).
pub fn threads() -> usize {
    env_threads().unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()).min(8)).max(1)
}

/// The loaded model's default for [`enabled`] when `INTARSIA_HOST_THREADS` is unset.
static MODEL_DEFAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set by `Model::load`: whether this model's host row work uses the pool when `INTARSIA_HOST_THREADS` is unset (Inkling-2 on,
/// every other model off). Results never depend on it.
pub fn set_model_default(on: bool) {
    MODEL_DEFAULT.store(on, Ordering::Relaxed);
}

/// Whether the host row work (`sampler::softcap`, `sampler::sample_row_in`) splits a long row across the pool: more than one
/// [`threads`], and `INTARSIA_HOST_THREADS` set, or unset and the model's default on.
pub fn enabled() -> bool {
    threads() > 1 && (env_threads().is_some() || MODEL_DEFAULT.load(Ordering::Relaxed))
}

/// Run `f(i)` for every `i` in `0..n` on the pool and the calling thread; returns when all have returned. A panic in a task is
/// re-raised here once every task has finished.
pub fn run(n: usize, f: &(dyn Fn(usize) + Sync)) {
    match pool() {
        Some(p) if n > 1 => p.run(n, f),
        _ => run_inline(n, f),
    }
}

/// Every task on the calling thread, in order, with [`run`]'s panic behaviour (all tasks run, then the panic is re-raised).
fn run_inline(n: usize, f: &(dyn Fn(usize) + Sync)) {
    let mut panicked = false;
    for i in 0..n {
        panicked |= std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i))).is_err();
    }
    if panicked {
        panic!("a host pool task panicked");
    }
}

thread_local! {
    static ON_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// This thread is one of the pool's workers (not a caller of [`run`]).
pub fn on_worker() -> bool {
    ON_WORKER.with(|w| w.get())
}

/// `*mut T` that tasks share; each task writes only the elements it owns.
#[derive(Clone, Copy)]
pub(crate) struct SharedMut<T>(pub(crate) *mut T);
// SAFETY: the tasks of one `run` write disjoint elements, and `run` returns only after all of them have finished.
unsafe impl<T> Send for SharedMut<T> {}
unsafe impl<T> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    /// The pointer to element `i` (a method, so a closure captures the whole `SharedMut`, which is `Sync`, not its field).
    ///
    /// # Safety
    /// `i` is in bounds of the allocation, and no other task touches element `i` during the `run`.
    #[inline]
    pub(crate) unsafe fn at(&self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

#[derive(Clone, Copy)]
struct Job(*const (dyn Fn(usize) + Sync + 'static));
// SAFETY: the pointee is `Sync`, and it is dereferenced only while its `run` is waiting for the tasks.
unsafe impl Send for Job {}

struct State {
    job: Option<Job>,
    n: usize,
    next: usize,
    left: usize,
    gen: u64,
    panicked: bool,
    /// [`prewake`] calls so far
    wake: u64,
}

struct Pool {
    caller: Mutex<()>,
    st: Mutex<State>,
    work: Condvar,
    done: Condvar,
    gen: AtomicU64,
    /// [`prewake`]: its time ([`now_ns`], 0 = none since the last `run`), the wait estimate (ns, 0 = none yet), and when the
    /// workers start and stop spinning ([`now_ns`])
    pre_t0: AtomicU64,
    est: AtomicU64,
    hot_at: AtomicU64,
    hot_until: AtomicU64,
}

fn pool() -> Option<&'static Pool> {
    static P: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *P.get_or_init(|| {
        let n = threads();
        if n <= 1 {
            return None;
        }
        let p: &'static Pool = Box::leak(Box::new(Pool {
            caller: Mutex::new(()),
            st: Mutex::new(State { job: None, n: 0, next: 0, left: 0, gen: 0, panicked: false, wake: 0 }),
            work: Condvar::new(),
            done: Condvar::new(),
            gen: AtomicU64::new(0),
            pre_t0: AtomicU64::new(0),
            est: AtomicU64::new(0),
            hot_at: AtomicU64::new(0),
            hot_until: AtomicU64::new(0),
        }));
        let qos = qos_on();
        for w in 1..n {
            if std::thread::Builder::new().name(format!("intarsia-host-{w}")).spawn(move || p.worker(qos)).is_err() {
                // fewer workers is still correct: the caller takes what is left
                break;
            }
        }
        Some(p)
    })
}

impl Pool {
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        let Ok(_one) = self.caller.try_lock() else {
            run_inline(n, f);
            return;
        };
        // the first task list after a `prewake`: the caller's wait, for the next estimate (the mean of the last two)
        let t0 = self.pre_t0.swap(0, Ordering::Relaxed);
        if t0 != 0 {
            let d = now_ns().saturating_sub(t0);
            if d > 0 && d < EST_MAX.as_nanos() as u64 {
                let e = self.est.load(Ordering::Relaxed);
                self.est.store(if e == 0 { d } else { (e + d) / 2 }, Ordering::Relaxed);
            }
        }
        // SAFETY: only the lifetime is erased. Workers call the pointer for a task they claimed while this job is current, and
        // this function returns only after `left` reached 0 (every claimed task returned) and the job was cleared.
        let job = Job(unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync + 'static)>(f) });
        let mut st = self.st.lock().unwrap_or_else(|e| e.into_inner());
        st.job = Some(job);
        st.n = n;
        st.next = 0;
        st.left = n;
        st.panicked = false;
        st.gen += 1;
        self.gen.store(st.gen, Ordering::Release);
        self.work.notify_all();
        while st.next < st.n {
            let i = st.next;
            st.next += 1;
            drop(st);
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i))).is_ok();
            st = self.st.lock().unwrap_or_else(|e| e.into_inner());
            st.panicked |= !ok;
            st.left -= 1;
        }
        while st.left > 0 {
            st = self.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.job = None;
        let panicked = st.panicked;
        drop(st);
        if panicked {
            panic!("a host pool task panicked");
        }
    }

    fn worker(&self, qos: bool) {
        ON_WORKER.with(|w| w.set(true));
        if qos {
            set_user_interactive_qos();
        }
        let mut seen = 0u64;
        let mut woke = 0u64;
        // pre-woken: spin until `hot_until` instead of for `SPIN`
        let mut hot = false;
        'idle: loop {
            let t0 = Instant::now();
            let mut spins = 0u32;
            while self.gen.load(Ordering::Acquire) == seen {
                std::hint::spin_loop();
                spins = spins.wrapping_add(1);
                if spins % 64 == 0 {
                    let over = if hot { now_ns() >= self.hot_until.load(Ordering::Relaxed) } else { t0.elapsed() > SPIN };
                    if over {
                        break;
                    }
                }
            }
            hot = false;
            let mut st = self.st.lock().unwrap_or_else(|e| e.into_inner());
            while st.gen == seen {
                if st.wake != woke {
                    // a `prewake`: sleep (a task list still wakes it) until `hot_at`, then spin until `hot_until`
                    woke = st.wake;
                    let (at, now) = (self.hot_at.load(Ordering::Relaxed), now_ns());
                    if at > now {
                        st = self
                            .work
                            .wait_timeout_while(st, Duration::from_nanos(at - now), |s| s.gen == seen && s.wake == woke)
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                        if st.gen == seen && st.wake != woke {
                            continue; // a newer prewake: take its times
                        }
                    }
                    if st.gen == seen {
                        drop(st);
                        hot = true;
                        continue 'idle;
                    }
                    break;
                }
                st = self.work.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            seen = st.gen;
            while let (Some(job), true) = (st.job, st.next < st.n) {
                let i = st.next;
                st.next += 1;
                drop(st);
                // SAFETY: see `Pool::run`: the job is current and its caller waits for this task.
                let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*job.0)(i) })).is_ok();
                st = self.st.lock().unwrap_or_else(|e| e.into_inner());
                st.panicked |= !ok;
                st.left -= 1;
                if st.left == 0 {
                    self.done.notify_one();
                }
            }
        }
    }
}

/// The calling thread's QoS class becomes user-interactive (macOS; elsewhere nothing).
fn set_user_interactive_qos() {
    #[cfg(target_os = "macos")]
    // SAFETY: sets the calling thread's own QoS class; no pointers involved.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Every index runs exactly once, back to back and from concurrent callers (the second runs inline), and a panicking task
    /// is re-raised after the others ran.
    #[test]
    fn every_task_runs_once() {
        for n in [0usize, 1, 2, 7, 64, 1000] {
            for _ in 0..20 {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                run(n, &|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n {n}");
            }
        }
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..50 {
                        let sum = AtomicUsize::new(0);
                        run(100, &|i| {
                            sum.fetch_add(i, Ordering::Relaxed);
                        });
                        assert_eq!(sum.load(Ordering::Relaxed), 4950);
                    }
                });
            }
        });
        let ran = AtomicUsize::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(16, &|i| {
                ran.fetch_add(1, Ordering::Relaxed);
                assert!(i != 5, "task 5");
            })
        }));
        assert!(r.is_err());
        assert_eq!(ran.load(Ordering::Relaxed), 16);
        // the pool still works after a panic
        let sum = AtomicUsize::new(0);
        run(10, &|i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 45);
    }

    /// `prewake` before a wait changes nothing about the result (every task once), measures the wait, and survives a prewake
    /// with no task list after it, back-to-back prewakes, and a task list during the workers' sleep.
    #[test]
    fn prewake_keeps_every_task_once_and_measures_the_wait() {
        let check = |n: usize| {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n {n}");
        };
        for wait_ms in [3u64, 3, 3, 0, 5, 1] {
            prewake();
            std::thread::sleep(Duration::from_millis(wait_ms));
            check(64);
            check(7);
        }
        if pool().is_some() {
            let e = prewake_estimate().expect("measured");
            assert!(e < Duration::from_millis(100), "{e:?}");
        }
        // a prewake with no task list after it, then two in a row, then a task list at once (the workers are asleep)
        prewake();
        std::thread::sleep(Duration::from_millis(8));
        prewake();
        prewake();
        check(100);
        std::thread::sleep(Duration::from_millis(1));
        check(3);
    }
}

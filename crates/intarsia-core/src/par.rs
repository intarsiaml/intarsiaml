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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long an idle worker spins for the next task list before it blocks.
const SPIN: Duration = Duration::from_micros(300);

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
}

struct Pool {
    caller: Mutex<()>,
    st: Mutex<State>,
    work: Condvar,
    done: Condvar,
    gen: AtomicU64,
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
            st: Mutex::new(State { job: None, n: 0, next: 0, left: 0, gen: 0, panicked: false }),
            work: Condvar::new(),
            done: Condvar::new(),
            gen: AtomicU64::new(0),
        }));
        for w in 1..n {
            if std::thread::Builder::new().name(format!("intarsia-host-{w}")).spawn(move || p.worker()).is_err() {
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

    fn worker(&self) {
        let mut seen = 0u64;
        loop {
            let t0 = Instant::now();
            let mut spins = 0u32;
            while self.gen.load(Ordering::Acquire) == seen {
                std::hint::spin_loop();
                spins = spins.wrapping_add(1);
                if spins % 64 == 0 && t0.elapsed() > SPIN {
                    break;
                }
            }
            let mut st = self.st.lock().unwrap_or_else(|e| e.into_inner());
            while st.gen == seen {
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
}

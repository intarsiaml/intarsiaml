//! Portable backend: blocking `pread`s on a pool of worker threads (or inline with zero threads).
//! Used on macOS and in tests, and as the Linux fallback when io_uring is unavailable. Workers sleep on a
//! condition variable, so an idle pool pays a thread wake-up per batch (TensorFold measured ~0.6 ms on GB10);
//! the io_uring backend avoids that.

use super::{Backend, Completion, ReadOp};
use std::collections::VecDeque;
use std::io;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

struct Shared {
    queue: Mutex<State>,
    work: Condvar,
    done: Condvar,
}

struct State {
    jobs: VecDeque<ReadOp>,
    finished: Vec<Completion>,
    stop: bool,
}

pub struct Pool {
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    in_flight: usize,
    depth: usize,
    /// Inline mode (no threads): completions of ops already executed in `submit`.
    inline_done: Vec<Completion>,
}

impl Pool {
    pub fn new(threads: usize, depth: usize) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(State { jobs: VecDeque::new(), finished: Vec::new(), stop: false }),
            work: Condvar::new(),
            done: Condvar::new(),
        });
        let threads = (0..threads)
            .map(|_| {
                let s = shared.clone();
                std::thread::spawn(move || worker(&s))
            })
            .collect();
        Self { shared, threads, in_flight: 0, depth: depth.max(1), inline_done: Vec::new() }
    }
}

fn run(op: &ReadOp) -> Completion {
    // SAFETY: the submitter reserved `buf[..len]` for this op.
    let buf = unsafe { std::slice::from_raw_parts_mut(op.buf, op.len as usize) };
    // like a kernel read: a short count at end of file (an aligned span may run past it), -errno on error
    let mut done = 0usize;
    while done < buf.len() {
        // SAFETY: buf[done..] is valid writable memory.
        let n = unsafe {
            libc::pread(op.fd, buf[done..].as_mut_ptr().cast(), buf.len() - done, (op.offset + done as u64) as libc::off_t)
        };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Completion { tag: op.tag, res: -(e.raw_os_error().unwrap_or(libc::EIO) as i64) };
        }
        if n == 0 {
            break;
        }
        done += n as usize;
    }
    Completion { tag: op.tag, res: done as i64 }
}

fn worker(s: &Shared) {
    loop {
        let op = {
            let mut st = s.queue.lock().unwrap();
            loop {
                if st.stop {
                    return;
                }
                if let Some(op) = st.jobs.pop_front() {
                    break op;
                }
                st = s.work.wait(st).unwrap();
            }
        };
        let c = run(&op);
        let mut st = s.queue.lock().unwrap();
        st.finished.push(c);
        s.done.notify_all();
    }
}

impl Backend for Pool {
    fn submit(&mut self, ops: &[ReadOp]) -> io::Result<()> {
        if self.threads.is_empty() {
            for op in ops {
                self.inline_done.push(run(op));
            }
        } else {
            let mut st = self.shared.queue.lock().unwrap();
            st.jobs.extend(ops.iter().copied());
            if ops.len() == 1 {
                self.shared.work.notify_one();
            } else {
                self.shared.work.notify_all();
            }
        }
        self.in_flight += ops.len();
        Ok(())
    }

    fn reap(&mut self, out: &mut Vec<Completion>, wait: bool, spin: Duration) -> io::Result<()> {
        if self.threads.is_empty() {
            self.in_flight -= self.inline_done.len();
            out.append(&mut self.inline_done);
            return Ok(());
        }
        if self.in_flight == 0 {
            return Ok(());
        }
        let start = Instant::now();
        let mut st = self.shared.queue.lock().unwrap();
        loop {
            if !st.finished.is_empty() || !wait {
                break;
            }
            if start.elapsed() < spin {
                drop(st);
                std::hint::spin_loop();
                st = self.shared.queue.lock().unwrap();
            } else {
                st = self.shared.done.wait(st).unwrap();
            }
        }
        self.in_flight -= st.finished.len();
        out.append(&mut st.finished);
        Ok(())
    }

    fn in_flight(&self) -> usize {
        self.in_flight
    }
    fn depth(&self) -> usize {
        self.depth
    }
    fn name(&self) -> &'static str {
        if self.threads.is_empty() {
            "pread-inline"
        } else {
            "pread-pool"
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut st = self.shared.queue.lock().unwrap();
            // finish what is queued: the buffers belong to the caller and must not be written after we return
            while !st.jobs.is_empty() {
                drop(st);
                std::thread::yield_now();
                st = self.shared.queue.lock().unwrap();
            }
            st.stop = true;
            self.shared.work.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

//! io_uring backend (Linux): reads are submitted from the caller's thread and reaped by the thread that waits
//! (bounded busy-poll of the completion queue, then a blocking `io_uring_enter`), so no worker thread has to be
//! woken. Optional SQPOLL moves submission to a kernel thread (costs a core while busy).

use super::{Backend, Completion, ReadOp};
use io_uring::{opcode, types, IoUring};
use std::io;
use std::time::{Duration, Instant};

pub struct Uring {
    ring: IoUring,
    in_flight: usize,
    depth: usize,
    sqpoll: bool,
    /// Completions drained while submitting (completion queue full); handed out by the next `reap`.
    stash: Vec<Completion>,
}

impl Uring {
    pub fn new(depth: usize, sqpoll_idle_ms: Option<u32>) -> io::Result<Self> {
        let entries = depth.clamp(8, 4096).next_power_of_two() as u32;
        let mut b = IoUring::builder();
        if let Some(idle) = sqpoll_idle_ms {
            b.setup_sqpoll(idle);
        }
        let ring = b.build(entries)?;
        Ok(Self { ring, in_flight: 0, depth: entries as usize, sqpoll: sqpoll_idle_ms.is_some(), stash: Vec::new() })
    }
}

impl Backend for Uring {
    fn submit(&mut self, ops: &[ReadOp]) -> io::Result<()> {
        let mut i = 0;
        while i < ops.len() {
            {
                let mut sq = self.ring.submission();
                while i < ops.len() {
                    let op = &ops[i];
                    let e = opcode::Read::new(types::Fd(op.fd), op.buf, op.len).offset(op.offset).build().user_data(op.tag);
                    // SAFETY: op.buf[..len] stays reserved and alive until this op's completion is reaped.
                    if unsafe { sq.push(&e) }.is_err() {
                        break;
                    }
                    i += 1;
                }
            }
            loop {
                match self.ring.submit() {
                    Ok(_) => break,
                    Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                    Err(e) if e.raw_os_error() == Some(libc::EBUSY) || e.raw_os_error() == Some(libc::EAGAIN) => {
                        // completion queue full: drain it into the stash, then submit again
                        for cqe in self.ring.completion() {
                            self.stash.push(Completion { tag: cqe.user_data(), res: cqe.result() as i64 });
                        }
                        std::hint::spin_loop();
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        self.in_flight += ops.len();
        Ok(())
    }

    fn reap(&mut self, out: &mut Vec<Completion>, wait: bool, spin: Duration) -> io::Result<()> {
        let start = Instant::now();
        if !self.stash.is_empty() {
            self.in_flight -= self.stash.len();
            out.append(&mut self.stash);
            return Ok(());
        }
        loop {
            let before = out.len();
            for cqe in self.ring.completion() {
                out.push(Completion { tag: cqe.user_data(), res: cqe.result() as i64 });
            }
            let got = out.len() - before;
            self.in_flight -= got;
            if got > 0 || !wait || self.in_flight == 0 {
                return Ok(());
            }
            if self.sqpoll || start.elapsed() < spin {
                std::hint::spin_loop();
                continue;
            }
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => {}
                Err(e) => return Err(e),
            }
        }
    }

    fn in_flight(&self) -> usize {
        self.in_flight
    }
    fn depth(&self) -> usize {
        self.depth
    }
    fn name(&self) -> &'static str {
        if self.sqpoll {
            "io_uring+sqpoll"
        } else {
            "io_uring"
        }
    }
}

impl Drop for Uring {
    fn drop(&mut self) {
        // every read in flight lands before its buffer can go away
        while self.in_flight > 0 {
            let mut v = Vec::new();
            if self.reap(&mut v, true, Duration::ZERO).is_err() {
                break;
            }
        }
    }
}

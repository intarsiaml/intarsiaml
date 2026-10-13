//! Liveness: the engine step counter is the heartbeat. A busy engine whose counter
//! stops moving is stalled (a wedged CUDA context keeps the HTTP side alive while
//! the GPU is dead, so `/metrics` alone proves nothing).

use intarsia_core::telemetry::TelemetrySnapshot;
use intarsia_core::Engine;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Exit codes of a server child, read by the supervisor.
pub const EXIT_STALLED: i32 = 86;
pub const EXIT_FAILED: i32 = 87;
pub const EXIT_ORPHANED: i32 = 88;
/// Configuration or model load error: restarting cannot help.
pub const EXIT_FATAL: i32 = 78;

struct Seen {
    busy: bool,
    steps: u64,
    since: Instant,
}

/// Tracks how long the engine has been busy without completing a step.
pub struct Progress {
    stall_after: Duration,
    seen: Mutex<Seen>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Ok,
    /// Busy, but no step for this long.
    Stalled(Duration),
    Failed(String),
}

impl Progress {
    pub fn new(stall_after: Duration) -> Progress {
        Progress { stall_after, seen: Mutex::new(Seen { busy: false, steps: 0, since: Instant::now() }) }
    }

    /// Time since the engine last made progress while it has work; zero when idle.
    pub fn observe(&self, snap: &TelemetrySnapshot) -> Duration {
        let busy = snap.running + snap.waiting > 0;
        let mut g = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        if snap.steps_total != g.steps || busy != g.busy {
            g.steps = snap.steps_total;
            g.since = Instant::now();
        }
        g.busy = busy;
        if busy {
            g.since.elapsed()
        } else {
            Duration::ZERO
        }
    }

    pub fn status(&self, engine: &dyn Engine, snap: &TelemetrySnapshot) -> (Status, Duration) {
        let idle_for = self.observe(snap);
        if let Some(why) = engine.failed() {
            return (Status::Failed(why), idle_for);
        }
        if !self.stall_after.is_zero() && idle_for > self.stall_after {
            return (Status::Stalled(idle_for), idle_for);
        }
        (Status::Ok, idle_for)
    }
}

/// Background watchdog. With `exit_on_trip` the process exits with a distinct code so a
/// supervisor can restart it; a wedged driver can hang `exit`, so an abort backs it up.
pub fn spawn_watchdog(
    engine: std::sync::Arc<dyn Engine>,
    progress: std::sync::Arc<Progress>,
    metrics: std::sync::Arc<crate::metrics::Metrics>,
    exit_on_trip: bool,
    parent_pid: Option<u32>,
) {
    let r = std::thread::Builder::new().name("intarsia-watchdog".into()).spawn(move || loop {
        std::thread::sleep(Duration::from_millis(500));
        if let Some(p) = parent_pid {
            #[cfg(unix)]
            if std::os::unix::process::parent_id() != p {
                eprintln!("[watchdog] supervisor {p} is gone; exiting");
                die(EXIT_ORPHANED);
            }
            #[cfg(not(unix))]
            let _ = p;
        }
        let snap = engine.telemetry();
        let (status, idle_for) = progress.status(&*engine, &snap);
        metrics.set_health(matches!(status, Status::Failed(_)), idle_for);
        match status {
            Status::Ok => {}
            Status::Stalled(d) if exit_on_trip => {
                eprintln!(
                    "[watchdog] no engine step for {:.0}s with {} running and {} waiting; exiting for restart",
                    d.as_secs_f64(),
                    snap.running,
                    snap.waiting
                );
                die(EXIT_STALLED);
            }
            Status::Failed(why) if exit_on_trip => {
                eprintln!("[watchdog] engine failed ({why}); exiting for restart");
                // Let in-flight error responses flush first.
                std::thread::sleep(Duration::from_secs(2));
                die(EXIT_FAILED);
            }
            _ => {}
        }
    });
    if let Err(e) = r {
        eprintln!("warning: could not start the watchdog thread: {e}");
    }
}

fn die(code: i32) -> ! {
    let _ = std::thread::Builder::new().name("intarsia-abort".into()).spawn(|| {
        std::thread::sleep(Duration::from_secs(5));
        std::process::abort();
    });
    std::process::exit(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(steps: u64, running: u32, waiting: u32) -> TelemetrySnapshot {
        TelemetrySnapshot { steps_total: steps, running, waiting, ..Default::default() }
    }

    #[test]
    fn progress_tracks_stalls() {
        let p = Progress::new(Duration::from_secs(1));
        assert_eq!(p.observe(&snap(5, 0, 0)), Duration::ZERO, "idle never stalls");
        std::thread::sleep(Duration::from_millis(30));
        assert!(p.observe(&snap(5, 1, 0)) < Duration::from_millis(30), "busy clock starts when idle ended");
        std::thread::sleep(Duration::from_millis(40));
        assert!(p.observe(&snap(5, 1, 0)) >= Duration::from_millis(40));
        assert!(p.observe(&snap(6, 1, 0)) < Duration::from_millis(40), "a step resets the clock");
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(p.observe(&snap(6, 0, 0)), Duration::ZERO);
    }
}

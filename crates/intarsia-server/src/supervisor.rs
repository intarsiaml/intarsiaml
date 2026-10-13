//! Process supervisor (ARCHITECTURE 4): `intarsia serve` runs the real server as a child and
//! restarts it with backoff when it exits or when its watchdog trips (the child exits with
//! [`crate::EXIT_STALLED`]). A wedged CUDA context has sticky errors, so a fresh process is the
//! only reliable recovery.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

pub struct SupervisorConfig {
    pub exe: PathBuf,
    pub args: Vec<OsString>,
    /// First restart delay; doubles per consecutive failure up to `backoff_max`.
    pub backoff_min: Duration,
    pub backoff_max: Duration,
    /// A child that ran this long counts as healthy: the backoff starts over.
    pub healthy_after: Duration,
    /// Give up after this many restarts (None = never).
    pub max_restarts: Option<u32>,
}

impl SupervisorConfig {
    pub fn new(exe: PathBuf, args: Vec<OsString>) -> SupervisorConfig {
        SupervisorConfig {
            exe,
            args,
            backoff_min: Duration::from_secs(1),
            backoff_max: Duration::from_secs(30),
            healthy_after: Duration::from_secs(60),
            max_restarts: None,
        }
    }
}

static SIGNAL: AtomicI32 = AtomicI32::new(0);

extern "C" fn on_signal(sig: libc::c_int) {
    SIGNAL.store(sig, Ordering::SeqCst);
}

fn install_handlers() {
    for sig in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
        unsafe {
            libc::signal(sig, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
        }
    }
}

fn describe(status: &std::process::ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(c), _) => match c {
            c if c == crate::EXIT_STALLED => format!("exit {c} (watchdog: engine stalled)"),
            c if c == crate::EXIT_FAILED => format!("exit {c} (engine worker failed)"),
            c if c == crate::EXIT_ORPHANED => format!("exit {c} (supervisor lost)"),
            c => format!("exit {c}"),
        },
        (None, Some(s)) => format!("killed by signal {s}"),
        _ => "unknown status".into(),
    }
}

/// Stop the child: SIGTERM, then SIGKILL after `grace`.
fn terminate(child: &mut Child, grace: Duration) -> Option<std::process::ExitStatus> {
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
    }
    let t0 = Instant::now();
    while t0.elapsed() < grace {
        if let Ok(Some(st)) = child.try_wait() {
            return Some(st);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    child.wait().ok()
}

/// Run the child until it exits cleanly or fatally; restart it otherwise. Returns the process
/// exit code for the supervisor. Blocks.
pub fn supervise(cfg: SupervisorConfig) -> i32 {
    install_handlers();
    let mut backoff = cfg.backoff_min;
    let mut restarts = 0u32;
    loop {
        let t0 = Instant::now();
        let mut child = match Command::new(&cfg.exe)
            .args(&cfg.args)
            .env("INTARSIA_SUPERVISOR_PID", std::process::id().to_string())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[supervisor] cannot start {}: {e}", cfg.exe.display());
                return 1;
            }
        };
        eprintln!("[supervisor] started server pid {}", child.id());
        let status = loop {
            if SIGNAL.load(Ordering::SeqCst) != 0 {
                eprintln!("[supervisor] signal {}: stopping the server", SIGNAL.load(Ordering::SeqCst));
                let st = terminate(&mut child, Duration::from_secs(30));
                return st.and_then(|s| s.code()).unwrap_or(0);
            }
            match child.try_wait() {
                Ok(Some(st)) => break st,
                Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                Err(e) => {
                    eprintln!("[supervisor] wait failed: {e}");
                    return 1;
                }
            }
        };
        let ran = t0.elapsed();
        match status.code() {
            Some(0) => return 0,
            // 2 = bad command line (clap), 78 = model/config error: a restart cannot help.
            Some(c) if c == 2 || c == crate::EXIT_FATAL => {
                eprintln!("[supervisor] server failed to start ({}); not restarting", describe(&status));
                return c;
            }
            _ => {}
        }
        if ran >= cfg.healthy_after {
            backoff = cfg.backoff_min;
        }
        restarts += 1;
        if cfg.max_restarts.is_some_and(|m| restarts > m) {
            eprintln!("[supervisor] server died ({}); giving up after {} restarts", describe(&status), restarts - 1);
            return status.code().unwrap_or(1);
        }
        eprintln!(
            "[supervisor] server died after {:.1}s: {}; restart #{restarts} in {:.1}s",
            ran.as_secs_f64(),
            describe(&status),
            backoff.as_secs_f64()
        );
        let wake = Instant::now() + backoff;
        while Instant::now() < wake {
            if SIGNAL.load(Ordering::SeqCst) != 0 {
                return 0;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        backoff = (backoff * 2).min(cfg.backoff_max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(script: &str) -> SupervisorConfig {
        let mut c = SupervisorConfig::new("/bin/sh".into(), vec!["-c".into(), script.into()]);
        c.backoff_min = Duration::from_millis(10);
        c.backoff_max = Duration::from_millis(40);
        c
    }

    #[test]
    fn restarts_until_clean_exit() {
        let dir = std::env::temp_dir().join(format!("intarsia-sup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("count");
        // dies with the watchdog code twice, then exits cleanly
        let script = format!(
            "n=$(cat {0} 2>/dev/null || echo 0); echo $((n+1)) > {0}; [ $n -ge 2 ] && exit 0; exit {1}",
            f.display(),
            crate::EXIT_STALLED
        );
        assert_eq!(supervise(cfg(&script)), 0);
        assert_eq!(std::fs::read_to_string(&f).unwrap().trim(), "3");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fatal_codes_are_not_retried() {
        assert_eq!(supervise(cfg(&format!("exit {}", crate::EXIT_FATAL))), crate::EXIT_FATAL);
    }

    #[test]
    fn gives_up_after_max_restarts() {
        let mut c = cfg("exit 3");
        c.max_restarts = Some(2);
        assert_eq!(supervise(c), 3);
    }
}

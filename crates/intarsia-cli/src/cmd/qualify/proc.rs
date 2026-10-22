//! Subprocess helpers for the qualify pipeline: run a command with its output captured to files and a hard
//! timeout, and keep a model server alive for the duration of a bench stage (and never longer).

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

/// Result of one captured subprocess.
#[derive(Debug, Clone)]
pub struct Run {
    pub code: Option<i32>,
    pub timed_out: bool,
    pub wall_s: f64,
    /// stdout (the command's results: JSON, gate lines) and stderr (diagnostics) files.
    pub out: PathBuf,
    pub err: PathBuf,
}

impl Run {
    pub fn ok(&self) -> bool {
        self.code == Some(0) && !self.timed_out
    }
    pub fn stdout(&self) -> String {
        std::fs::read_to_string(&self.out).unwrap_or_default()
    }
    pub fn stderr(&self) -> String {
        std::fs::read_to_string(&self.err).unwrap_or_default()
    }
    /// The last `n` non-empty lines of stderr (falling back to stdout), for failure reasons.
    pub fn tail(&self, n: usize) -> String {
        let e = self.stderr();
        let text = if e.trim().is_empty() { self.stdout() } else { e };
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        lines[lines.len().saturating_sub(n)..].join("\n")
    }
    pub fn describe(&self) -> String {
        if self.timed_out {
            format!("timed out after {:.0}s", self.wall_s)
        } else {
            match self.code {
                Some(c) => format!("exit {c}"),
                None => "killed by a signal".to_string(),
            }
        }
    }
}

#[cfg(unix)]
fn own_group(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
}
#[cfg(not(unix))]
fn own_group(_cmd: &mut Command) {}

/// Ask the kernel to kill the child when this process dies (Linux), so an aborted qualify run cannot leave a model
/// server holding the GPU.
#[cfg(target_os = "linux")]
fn die_with_parent(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: prctl is async-signal-safe and only sets this child's own death signal.
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            Ok(())
        });
    }
}
#[cfg(not(target_os = "linux"))]
fn die_with_parent(_cmd: &mut Command) {}

#[cfg(unix)]
fn signal_group(pid: u32, sig: i32) {
    // SAFETY: plain kill(2) on the process group we created for the child.
    unsafe {
        libc::kill(-(pid as i32), sig);
    }
}
#[cfg(not(unix))]
fn signal_group(_pid: u32, _sig: i32) {}

#[cfg(unix)]
const SIGTERM: i32 = libc::SIGTERM;
#[cfg(unix)]
const SIGKILL: i32 = libc::SIGKILL;
#[cfg(not(unix))]
const SIGTERM: i32 = 15;
#[cfg(not(unix))]
const SIGKILL: i32 = 9;

fn stdio_files(stem: &Path) -> Result<(File, File, PathBuf, PathBuf)> {
    if let Some(d) = stem.parent() {
        std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
    }
    let out = stem.with_extension("out");
    let err = stem.with_extension("err");
    let fo = File::create(&out).with_context(|| format!("creating {}", out.display()))?;
    let fe = File::create(&err).with_context(|| format!("creating {}", err.display()))?;
    Ok((fo, fe, out, err))
}

/// Process group of the subprocess [`run_captured`] is waiting for (0: none), for [`forward_termination`].
static LIVE_PGID: AtomicI32 = AtomicI32::new(0);
/// Process group of the model server a [`Server`] owns (0: none).
static SERVER_PGID: AtomicI32 = AtomicI32::new(0);

/// Make SIGTERM / SIGINT of this process also stop the subprocess [`run_captured`] is waiting for (its own process
/// group, so a plain kill of the parent would orphan it; `die_with_parent` only exists on Linux). Used by the
/// long-running `intarsia onboard` and by `intarsia qualify` (so a killed onboard takes the whole chain down).
#[cfg(unix)]
pub fn forward_termination() {
    extern "C" fn handler(sig: libc::c_int) {
        // SAFETY: kill(2) and _exit(2) are async-signal-safe.
        unsafe {
            for pg in [LIVE_PGID.load(Ordering::SeqCst), SERVER_PGID.load(Ordering::SeqCst)] {
                if pg > 0 {
                    libc::kill(-pg, libc::SIGTERM);
                }
            }
            libc::_exit(128 + sig);
        }
    }
    // SAFETY: installs a handler that only calls async-signal-safe functions.
    unsafe {
        libc::signal(libc::SIGTERM, handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, handler as *const () as libc::sighandler_t);
    }
}
#[cfg(not(unix))]
pub fn forward_termination() {}

/// Run `cmd` to completion. stdout goes to `<stem>.out`, stderr to `<stem>.err`; the child runs in its own process
/// group and is killed (SIGTERM, then SIGKILL) when `timeout` passes.
pub fn run_captured(mut cmd: Command, stem: &Path, timeout: Duration) -> Result<Run> {
    let (fo, fe, out, err) = stdio_files(stem)?;
    cmd.stdin(Stdio::null()).stdout(Stdio::from(fo)).stderr(Stdio::from(fe));
    own_group(&mut cmd);
    die_with_parent(&mut cmd);
    let t0 = Instant::now();
    let mut child = cmd.spawn().with_context(|| format!("starting {:?}", cmd.get_program()))?;
    let pid = child.id();
    LIVE_PGID.store(pid as i32, Ordering::SeqCst);
    let (code, timed_out) = loop {
        if let Some(st) = child.try_wait()? {
            break (st.code(), false);
        }
        if t0.elapsed() > timeout {
            signal_group(pid, SIGTERM);
            let t1 = Instant::now();
            while t1.elapsed() < Duration::from_secs(15) && child.try_wait()?.is_none() {
                std::thread::sleep(Duration::from_millis(200));
            }
            signal_group(pid, SIGKILL);
            let _ = child.kill();
            let _ = child.wait();
            break (None, true);
        }
        std::thread::sleep(Duration::from_millis(150));
    };
    LIVE_PGID.store(0, Ordering::SeqCst);
    Ok(Run { code, timed_out, wall_s: t0.elapsed().as_secs_f64(), out, err })
}

/// A model server (`intarsia serve`) owned by one stage. Dropping it stops the process group and waits for it.
pub struct Server {
    child: Child,
    pub url: String,
    pub err: PathBuf,
}

impl Server {
    pub fn start(mut cmd: Command, stem: &Path, url: String) -> Result<Server> {
        let (fo, fe, _out, err) = stdio_files(stem)?;
        cmd.stdin(Stdio::null()).stdout(Stdio::from(fo)).stderr(Stdio::from(fe));
        own_group(&mut cmd);
        die_with_parent(&mut cmd);
        let child = cmd.spawn().with_context(|| format!("starting the server {:?}", cmd.get_program()))?;
        SERVER_PGID.store(child.id() as i32, Ordering::SeqCst);
        Ok(Server { child, url, err })
    }

    /// Poll `/health` until it answers 200. Fails early when the process exits.
    pub fn wait_healthy(&mut self, timeout: Duration) -> Result<f64> {
        let t0 = Instant::now();
        let agent = intarsia_bench::http::agent(Some(Duration::from_secs(3)));
        loop {
            if let Some(st) = self.child.try_wait()? {
                bail!("the server exited with {st} before it became healthy:\n{}", tail_of(&self.err, 12));
            }
            if let Ok(r) = agent.get(&format!("{}/health", self.url)).call() {
                if r.status().as_u16() == 200 {
                    return Ok(t0.elapsed().as_secs_f64());
                }
            }
            if t0.elapsed() > timeout {
                bail!(
                    "the server was not healthy after {:.0}s:\n{}",
                    t0.elapsed().as_secs_f64(),
                    tail_of(&self.err, 12)
                );
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    fn stop(&mut self) {
        SERVER_PGID.store(0, Ordering::SeqCst);
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return;
        }
        let pid = self.child.id();
        signal_group(pid, SIGTERM);
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(60) {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        signal_group(pid, SIGKILL);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn tail_of(path: &Path, n: usize) -> String {
    let t = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = t.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// A free TCP port on the loopback interface, starting the search at `start`.
pub fn free_port(start: u16) -> u16 {
    (start..start.saturating_add(200)).find(|p| std::net::TcpListener::bind(("127.0.0.1", *p)).is_ok()).unwrap_or(start)
}

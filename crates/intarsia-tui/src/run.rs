//! The two drivers: the interactive terminal loop, and `--snapshot` (headless: feed the same `App`, render to a text frame).

use crate::app::App;
use crate::feed::{self, Event};
use crate::ui::{self, Theme};
use anyhow::Result;
use intarsia_bench::http;
use ratatui::crossterm::event::{self, Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use std::io::{IsTerminal, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

pub struct TuiOptions {
    /// Server origin, e.g. `http://127.0.0.1:8090` (a trailing `/v1` is accepted).
    pub url: String,
    pub light: bool,
    /// Let thinking models think in the chat (default: `enable_thinking: false`).
    pub thinking: bool,
    pub max_tokens: usize,
    /// Headless mode: print text frames to stdout and exit instead of taking over the terminal.
    pub snapshot: Option<SnapshotOptions>,
    /// Exit by itself after this long (smoke tests of the interactive mode).
    pub exit_after: Option<Duration>,
}

#[derive(Clone, Debug)]
pub struct SnapshotOptions {
    /// Send this chat message once telemetry and the plan have arrived.
    pub chat: Option<String>,
    pub width: u16,
    pub height: u16,
    /// Without a chat: how long to watch before the single frame.
    pub hold: Duration,
    /// With a chat: capture the "mid-stream" frame once this many chunks have streamed.
    pub mid_tokens: u64,
    /// Give up waiting for the server after this long and print what there is.
    pub timeout: Duration,
}

impl Default for SnapshotOptions {
    fn default() -> Self {
        SnapshotOptions {
            chat: None,
            width: 150,
            height: 48,
            hold: Duration::from_secs(3),
            mid_tokens: 12,
            timeout: Duration::from_secs(60),
        }
    }
}

struct Session {
    app: App,
    rx: Receiver<Event>,
    tx: Sender<Event>,
    origin: String,
    t0: Instant,
    thinking: bool,
    max_tokens: usize,
}

impl Session {
    fn new(opts: &TuiOptions) -> Result<Session> {
        http::check_scheme(&opts.url)?;
        let origin = http::origin(&opts.url);
        let (tx, rx) = mpsc::channel();
        feed::spawn_all(&origin, tx.clone());
        Ok(Session { app: App::new(&origin), rx, tx, origin, t0: Instant::now(), thinking: opts.thinking, max_tokens: opts.max_tokens })
    }

    fn now(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    fn pump(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            let t = self.now();
            match ev {
                Event::Snap(s) => self.app.ingest_snapshot(s, t),
                Event::Status(s) => self.app.status = s,
                Event::Plan(p) => self.app.ingest_plan(p),
                Event::Metrics(m) => self.app.ingest_metrics(&m, t),
                Event::Model(m) => self.app.chat_model = m,
                Event::Chat(c) => self.app.chat_event(c),
            }
        }
    }

    /// Send the input line as a chat turn (no-op when empty or a turn is in flight).
    fn send(&mut self) {
        if let Some(history) = self.app.submit() {
            let model = if self.app.chat_model.is_empty() { self.app.model.clone() } else { self.app.chat_model.clone() };
            feed::spawn_chat(&self.origin, &model, &history, self.max_tokens, self.thinking, self.tx.clone());
        }
    }
}

pub fn run(opts: TuiOptions) -> Result<()> {
    let theme = if opts.light { Theme::light() } else { Theme::dark() };
    let mut s = Session::new(&opts)?;
    match opts.snapshot.clone() {
        Some(so) => snapshot(&mut s, &theme, &so),
        None => interactive(&mut s, &theme, opts.exit_after),
    }
}

fn interactive(s: &mut Session, theme: &Theme, exit_after: Option<Duration>) -> Result<()> {
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        anyhow::bail!("intarsia tui needs a terminal on stdin and stdout; use --snapshot for text frames without one");
    }
    let mut term = ratatui::init();
    let res = (|| -> Result<()> {
        loop {
            s.pump();
            term.draw(|f| ui::draw(f, &s.app, theme))?;
            if s.app.quit || exit_after.is_some_and(|d| s.t0.elapsed() >= d) {
                return Ok(());
            }
            if event::poll(Duration::from_millis(50))? {
                if let TermEvent::Key(k) = event::read()? {
                    if k.kind != KeyEventKind::Press {
                        continue;
                    }
                    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                    match k.code {
                        KeyCode::Esc => s.app.quit = true,
                        KeyCode::Char('c') if ctrl => s.app.quit = true,
                        KeyCode::Char('l') if ctrl => {
                            s.app.chat.clear();
                            s.app.chat_scroll = 0;
                        }
                        KeyCode::Char(c) if !ctrl => s.app.input.push(c),
                        KeyCode::Backspace => {
                            s.app.input.pop();
                        }
                        KeyCode::Enter => s.send(),
                        KeyCode::PageUp => s.app.chat_scroll += 5,
                        KeyCode::PageDown => s.app.chat_scroll = s.app.chat_scroll.saturating_sub(5),
                        _ => {}
                    }
                }
            }
        }
    })();
    ratatui::restore();
    res
}

/// Headless run. With a chat message: wait for telemetry + plan, send it, print a frame once `mid_tokens` chunks have
/// streamed, one just after the reply finished (rates still non-zero) and one a little later (latency percentiles in).
/// Without: print one frame after `hold`.
fn snapshot(s: &mut Session, theme: &Theme, so: &SnapshotOptions) -> Result<()> {
    let start = Instant::now();
    let mut out = std::io::stdout();
    let mut frame = |s: &Session, label: &str| -> Result<()> {
        writeln!(out, "=== frame: {label} (t={:.1}s) ===", s.now())?;
        write!(out, "{}", ui::render_text(&s.app, theme, so.width, so.height))?;
        writeln!(out)?;
        out.flush()?;
        Ok(())
    };
    let ready = |s: &Session| s.app.snap.is_some() && s.app.plan_known;
    // phase 1: wait for the server
    let mut ready_at: Option<Instant> = None;
    while start.elapsed() < so.timeout {
        s.pump();
        if ready(s) && ready_at.is_none() {
            ready_at = Some(Instant::now());
        }
        // a short settle so the first rates/percentiles have two samples
        if ready_at.is_some_and(|t| t.elapsed() >= Duration::from_millis(1500)) {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let Some(chat) = so.chat.clone() else {
        let until = Instant::now() + so.hold;
        while Instant::now() < until {
            s.pump();
            std::thread::sleep(Duration::from_millis(50));
        }
        return frame(s, "idle");
    };
    s.app.input = chat;
    s.send();
    let (mut mid, mut done_at, mut after_done) = (false, None::<Instant>, false);
    while start.elapsed() < so.timeout {
        s.pump();
        if !mid && s.app.stream_chunks >= so.mid_tokens {
            mid = true;
            frame(s, "mid-stream")?;
        }
        if !s.app.streaming && done_at.is_none() {
            done_at = Some(Instant::now());
        }
        if let Some(d) = done_at {
            if !mid {
                mid = true;
                frame(s, "mid-stream (reply was shorter than --mid-tokens)")?;
            }
            // the rate window is ~1.2 s: look while the finished request's tokens are still inside it
            if !after_done && d.elapsed() >= Duration::from_millis(400) {
                after_done = true;
                frame(s, "reply finished")?;
            }
            if d.elapsed() >= Duration::from_millis(3200) {
                return frame(s, "settled (latency percentiles)");
            }
        }
        std::thread::sleep(Duration::from_millis(40));
    }
    frame(s, "timeout")
}

//! `intarsia top`: terminal dashboard fed by the server's `/events` SSE stream
//! (`TelemetrySnapshot`, docs/telemetry.md). `--demo` synthesises snapshots instead.

use crate::http;
use crate::snapshot::{Device, OpStat, Snapshot};
use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};
use ratatui::Frame;
use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

const HIST: usize = 300;

// ---------------------------------------------------------------------------------------------
// Model

#[derive(Clone, Debug, Default)]
pub struct Loads {
    pub api: f64,
    pub scheduler: f64,
    pub engine: f64,
    pub kernels: f64,
    pub device: f64,
}

#[derive(Default)]
pub struct App {
    pub snap: Option<Snapshot>,
    prev: Option<(Instant, Snapshot)>,
    pub decode_hist: VecDeque<f64>,
    pub prefill_hist: VecDeque<f64>,
    pub steps_hist: VecDeque<f64>,
    pub steps_rate: f64,
    peak_steps: f64,
    pub op_share: BTreeMap<String, f64>,
    pub loads: Loads,
    pub status: String,
    pub demo: bool,
}

fn push(h: &mut VecDeque<f64>, v: f64) {
    if h.len() >= HIST {
        h.pop_front();
    }
    h.push_back(v);
}

/// Estimated total op time: sampled time scaled up by calls/timed_calls.
fn est_ms(o: &OpStat) -> f64 {
    if o.timed_calls == 0 {
        0.0
    } else {
        o.time_ms * (o.calls.max(o.timed_calls) as f64 / o.timed_calls as f64)
    }
}

fn ema(old: f64, new: f64, a: f64) -> f64 {
    old + (new - old) * a
}

impl App {
    pub fn new(demo: bool) -> Self {
        App { demo, status: "connecting...".into(), ..Default::default() }
    }

    pub fn ingest(&mut self, s: Snapshot, now: Instant) {
        push(&mut self.decode_hist, s.last_decode_tok_s);
        push(&mut self.prefill_hist, s.last_prefill_tok_s);
        let mut busy = None;
        if let Some((t0, p)) = &self.prev {
            let dt = now.duration_since(*t0).as_secs_f64();
            if dt > 1e-3 {
                let rate = s.steps_total.saturating_sub(p.steps_total) as f64 / dt;
                self.steps_rate = ema(self.steps_rate, rate, 0.4);
                // Per-op time delta (estimated) -> shares and kernel busy fraction.
                let mut deltas: BTreeMap<String, f64> = BTreeMap::new();
                for (k, o) in &s.ops {
                    let before = p.ops.get(k).map(est_ms).unwrap_or(0.0);
                    deltas.insert(k.clone(), (est_ms(o) - before).max(0.0));
                }
                let total: f64 = deltas.values().sum();
                if total > 1e-6 {
                    for (k, d) in &deltas {
                        let sh = self.op_share.entry(k.clone()).or_insert(*d / total);
                        *sh = ema(*sh, *d / total, 0.3);
                    }
                    busy = Some((total / (dt * 1000.0)).clamp(0.0, 1.0));
                } else {
                    busy = Some(0.0);
                }
            }
        } else {
            let total: f64 = s.ops.values().map(est_ms).sum();
            if total > 0.0 {
                for (k, o) in &s.ops {
                    self.op_share.insert(k.clone(), est_ms(o) / total);
                }
            }
        }
        push(&mut self.steps_hist, self.steps_rate);
        self.peak_steps = (self.peak_steps * 0.995).max(self.steps_rate);
        let engine = if self.peak_steps > 1.0 { (self.steps_rate / self.peak_steps).clamp(0.0, 1.0) } else { 0.0 };
        let q = (s.running + s.waiting) as f64;
        let sched = if s.waiting > 0 {
            0.6 + 0.4 * (s.waiting as f64 / 4.0).min(1.0)
        } else if s.running > 0 {
            0.35
        } else {
            0.0
        };
        let kv = if s.kv_total_bytes > 0 { s.kv_used_bytes as f64 / s.kv_total_bytes as f64 } else { 0.0 };
        let l = &mut self.loads;
        l.api = ema(l.api, (q / 4.0).min(1.0), 0.5);
        l.scheduler = ema(l.scheduler, sched, 0.5);
        l.engine = ema(l.engine, engine, 0.5);
        // Without sampled op timing (cpu-ref samples every step, GPUs 1/16) fall back to the engine load.
        l.kernels = ema(l.kernels, busy.filter(|b| *b > 0.0).unwrap_or(engine), 0.5);
        l.device = ema(l.device, kv, 0.5);
        self.status = "connected".into();
        self.prev = Some((now, s.clone()));
        self.snap = Some(s);
    }
}

// ---------------------------------------------------------------------------------------------
// Colours

fn lerp(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    let f = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t).round() as u8;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

/// Dark slate -> amber -> red as `load` goes 0 -> 1.
pub fn heat(load: f64) -> Color {
    let l = load.clamp(0.0, 1.0);
    let (r, g, b) = if l < 0.5 {
        lerp((38, 44, 56), (214, 160, 40), l / 0.5)
    } else {
        lerp((214, 160, 40), (232, 64, 56), (l - 0.5) / 0.5)
    };
    Color::Rgb(r, g, b)
}

fn fg_for(load: f64) -> Color {
    if load > 0.35 {
        Color::Rgb(16, 18, 22)
    } else {
        Color::Rgb(190, 196, 206)
    }
}

const DIM: Color = Color::Rgb(120, 128, 142);
const ACCENT: Color = Color::Rgb(94, 190, 220);
const BG: Color = Color::Rgb(14, 16, 20);

fn block<'a>(title: &'a str) -> Block<'a> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(58, 64, 76)))
        .title(Span::styled(format!(" {title} "), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)))
}

// ---------------------------------------------------------------------------------------------
// Drawing

fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else {
        format!("{:.0} kB", b / 1e3)
    }
}

fn fmt_uptime(s: f64) -> String {
    let s = s as u64;
    format!("{}h{:02}m{:02}s", s / 3600, s / 60 % 60, s % 60)
}

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    f.render_widget(Block::default().style(Style::default().bg(BG)), area);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(6),
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(6),
            Constraint::Length(1),
        ])
        .split(area);
    let empty = Snapshot::default();
    let s = app.snap.as_ref().unwrap_or(&empty);
    draw_header(f, rows[0], app, s);
    draw_sparks(f, rows[1], app, s);
    draw_gauges(f, rows[2], s);
    draw_ops(f, rows[3], app);
    draw_arch(f, rows[4], app);
    let foot = if app.demo { " DEMO DATA  |  q / Esc / Ctrl-C quit" } else { " q / Esc / Ctrl-C quit" };
    f.render_widget(Paragraph::new(foot).style(Style::default().fg(DIM)), rows[5]);
}

fn draw_header(f: &mut Frame, a: Rect, app: &App, s: &Snapshot) {
    let d: &Device = &s.device;
    let fin = |k: &str| s.finished.get(k).copied().unwrap_or(0);
    let l1 = Line::from(vec![
        Span::styled(if s.model.is_empty() { "(no model yet)".to_string() } else { s.model.clone() }, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::styled("   ", Style::default()),
        Span::styled(
            format!("{} [{} {}]{}", d.name, d.backend, d.arch, if d.unified { " unified" } else { "" }),
            Style::default().fg(ACCENT),
        ),
        Span::styled(format!("   up {}", fmt_uptime(s.uptime_s)), Style::default().fg(DIM)),
        Span::styled(format!("   {}", app.status), Style::default().fg(if app.status == "connected" { Color::Rgb(110, 200, 120) } else { Color::Rgb(230, 170, 60) })),
    ]);
    let l2 = Line::from(Span::styled(
        format!(
            "steps/s {:>6.1}   requests {}  (stop {} length {} cancelled {} error {})   tokens in {} (cached {}) out {}   weights {}   mem free {} / {}",
            app.steps_rate, s.requests_total, fin("stop"), fin("length"), fin("cancelled"), fin("error"),
            s.prompt_tokens_total, s.cached_tokens_total, s.generation_tokens_total, human_bytes(s.weights_bytes),
            human_bytes(d.free_mem_bytes), human_bytes(d.total_mem_bytes)
        ),
        Style::default().fg(Color::Rgb(200, 206, 216)),
    ));
    f.render_widget(Paragraph::new(vec![l1, l2]).block(block("intarsia top")), a);
}

fn spark(f: &mut Frame, a: Rect, title: String, hist: &VecDeque<f64>, color: Color) {
    let inner_w = a.width.saturating_sub(2) as usize;
    let skip = hist.len().saturating_sub(inner_w.max(1));
    // x10 keeps one decimal of resolution in the integer sparkline.
    let data: Vec<u64> = hist.iter().skip(skip).map(|v| (v.max(0.0) * 10.0) as u64).collect();
    f.render_widget(Sparkline::default().block(block(&title)).data(data).style(Style::default().fg(color)), a);
}

fn draw_sparks(f: &mut Frame, a: Rect, app: &App, s: &Snapshot) {
    let c = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(34), Constraint::Percentage(33), Constraint::Percentage(33)])
        .split(a);
    spark(f, c[0], format!("decode tok/s  {:.1}", s.last_decode_tok_s), &app.decode_hist, Color::Rgb(110, 200, 120));
    spark(f, c[1], format!("prefill tok/s  {:.0}   ttft {:.0} ms", s.last_prefill_tok_s, s.last_ttft_ms), &app.prefill_hist, Color::Rgb(94, 160, 232));
    spark(f, c[2], format!("engine steps/s  {:.1}", app.steps_rate), &app.steps_hist, Color::Rgb(214, 160, 40));
}

fn gauge(f: &mut Frame, a: Rect, title: &str, ratio: f64, label: String, color: Color) {
    f.render_widget(
        Gauge::default()
            .block(block(title))
            .gauge_style(Style::default().fg(color).bg(Color::Rgb(30, 34, 42)))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(Span::styled(label, Style::default().fg(Color::White))),
        a,
    );
}

fn draw_gauges(f: &mut Frame, a: Rect, s: &Snapshot) {
    let c = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(34), Constraint::Percentage(33), Constraint::Percentage(33)])
        .split(a);
    let kvb = if s.kv_total_bytes > 0 { s.kv_used_bytes as f64 / s.kv_total_bytes as f64 } else { 0.0 };
    gauge(f, c[0], "KV cache bytes", kvb, format!("{} / {}  {:.0}%", human_bytes(s.kv_used_bytes), human_bytes(s.kv_total_bytes), kvb * 100.0), heat(kvb));
    let kvt = if s.kv_capacity_tokens > 0 { s.kv_used_tokens as f64 / s.kv_capacity_tokens as f64 } else { 0.0 };
    gauge(f, c[1], "KV cache tokens", kvt, format!("{} / {} tokens", s.kv_used_tokens, s.kv_capacity_tokens), heat(kvt));
    let q = s.running + s.waiting;
    let r = if q == 0 { 0.0 } else { s.running as f64 / q as f64 };
    gauge(f, c[2], "requests", r, format!("running {}   waiting {}", s.running, s.waiting), if s.waiting > 0 { Color::Rgb(230, 170, 60) } else { Color::Rgb(110, 200, 120) });
}

fn draw_ops(f: &mut Frame, a: Rect, app: &App) {
    let mut ops: Vec<(&String, f64)> = app.op_share.iter().map(|(k, v)| (k, *v)).collect();
    ops.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    let cap = a.height.saturating_sub(2) as usize;
    let bar_w = (a.width as usize).saturating_sub(2 + 14 + 8 + 14).max(4);
    let mut lines = vec![];
    for (name, share) in ops.into_iter().take(cap) {
        let n = (share * bar_w as f64).round() as usize;
        let per = app
            .snap
            .as_ref()
            .and_then(|s| s.ops.get(name))
            .filter(|o| o.timed_calls > 0)
            .map(|o| format!("{:>8.3} ms/call", o.time_ms / o.timed_calls as f64))
            .unwrap_or_default();
        lines.push(Line::from(vec![
            Span::styled(format!("{name:<13} "), Style::default().fg(Color::Rgb(200, 206, 216))),
            Span::styled("█".repeat(n), Style::default().fg(heat(0.15 + share * 1.6))),
            Span::styled("░".repeat(bar_w.saturating_sub(n)), Style::default().fg(Color::Rgb(36, 40, 50))),
            Span::styled(format!(" {:>5.1}%", share * 100.0), Style::default().fg(Color::White)),
            Span::styled(format!(" {per}"), Style::default().fg(DIM)),
        ]));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled("no op timings yet (send a request)", Style::default().fg(DIM))));
    }
    f.render_widget(Paragraph::new(lines).block(block("kernel time share (live)")), a);
}

fn draw_arch(f: &mut Frame, a: Rect, app: &App) {
    let l = &app.loads;
    let dev = app.snap.as_ref().map(|s| s.device.name.clone()).filter(|n| !n.is_empty()).unwrap_or_else(|| "device".into());
    let nodes: [(&str, f64, String); 5] = [
        ("API", l.api, "HTTP /v1".into()),
        ("scheduler", l.scheduler, "queue".into()),
        ("engine", l.engine, "steps".into()),
        ("kernels", l.kernels, "launches".into()),
        (&dev, l.device, "KV fill".into()),
    ];
    let outer = block("architecture (colour = live load)");
    let inner = outer.inner(a);
    f.render_widget(outer, a);
    let mut cons = vec![];
    for i in 0..nodes.len() {
        cons.push(Constraint::Fill(1));
        if i + 1 < nodes.len() {
            cons.push(Constraint::Length(3));
        }
    }
    let cells = Layout::default().direction(Direction::Horizontal).constraints(cons).split(inner);
    for (i, (name, load, sub)) in nodes.iter().enumerate() {
        let r = cells[i * 2];
        let st = Style::default().bg(heat(*load)).fg(fg_for(*load));
        let txt = vec![
            Line::from(Span::styled(name.to_string(), st.add_modifier(Modifier::BOLD))),
            Line::from(Span::styled(format!("{sub} {:>3.0}%", load * 100.0), st)),
        ];
        f.render_widget(Paragraph::new(txt).alignment(Alignment::Center).style(st), r);
        if i + 1 < nodes.len() {
            let ar = cells[i * 2 + 1];
            f.render_widget(
                Paragraph::new(vec![Line::from(""), Line::from(" -> ")]).alignment(Alignment::Center).style(Style::default().fg(DIM)),
                ar,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Data sources

pub enum Msg {
    Snap(Snapshot),
    Status(String),
}

/// Follow `{origin}/events` forever, reconnecting on errors.
pub fn spawn_sse(url: &str, tx: Sender<Msg>) {
    let origin = http::origin(url);
    std::thread::spawn(move || {
        let agent = http::agent(None);
        loop {
            let res = (|| -> Result<()> {
                let resp = agent.get(format!("{origin}/events")).header("Accept", "text/event-stream").call()?;
                if !resp.status().is_success() {
                    anyhow::bail!("GET /events: HTTP {}", resp.status().as_u16());
                }
                for line in BufReader::new(resp.into_body().into_reader()).lines() {
                    let line = line?;
                    let Some(p) = line.strip_prefix("data:") else { continue };
                    if let Ok(s) = serde_json::from_str::<Snapshot>(p.trim()) {
                        if tx.send(Msg::Snap(s)).is_err() {
                            return Ok(());
                        }
                    }
                }
                anyhow::bail!("stream closed")
            })();
            if let Err(e) = res {
                if tx.send(Msg::Status(format!("{e}; retrying"))).is_err() {
                    return;
                }
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    });
}

/// Synthetic engine: a request every ~8 s (prefill burst, then decode), so every widget moves.
pub struct Demo {
    t: f64,
    snap: Snapshot,
    rng: crate::speed::Rng,
}

const DEMO_OPS: [(&str, f64); 10] = [
    ("matmul", 0.62), ("attn_decode", 0.12), ("logits", 0.08), ("rmsnorm", 0.04), ("silu_glu", 0.04),
    ("rope", 0.03), ("qk_norm", 0.025), ("kv_append", 0.02), ("embedding", 0.01), ("argmax", 0.005),
];

impl Demo {
    pub fn new() -> Self {
        let snap = Snapshot {
            schema_version: 1,
            model: "Qwen/Qwen3-8B (demo)".into(),
            device: Device { name: "NVIDIA GB10".into(), backend: "cuda".into(), arch: "sm_121".into(), total_mem_bytes: 130_000_000_000, free_mem_bytes: 52_000_000_000, unified: true },
            weights_bytes: 16_400_000_000,
            kv_total_bytes: 603_979_776,
            kv_capacity_tokens: 4096,
            ..Default::default()
        };
        Demo { t: 0.0, snap, rng: crate::speed::Rng::from_clock() }
    }

    pub fn step(&mut self, dt: f64) -> Snapshot {
        self.t += dt;
        let cycle = self.t % 9.0;
        let s = &mut self.snap;
        let j = (self.rng.next() % 1000) as f64 / 1000.0;
        s.uptime_s = self.t;
        let prefill = cycle < 1.6;
        let decode = (1.6..8.0).contains(&cycle);
        if cycle < dt + 1e-9 {
            s.requests_total += 1;
        }
        s.running = (prefill || decode) as u32;
        s.waiting = if (prefill || decode) && ((self.t / 9.0) as u64) % 3 == 1 { 1 + (cycle > 4.0) as u32 } else { 0 };
        let steps = if prefill { dt * (12.0 + 8.0 * j) } else if decode { dt * (22.0 + 4.0 * j) } else { 0.0 };
        s.steps_total += steps.round() as u64;
        let ms_per_s = if prefill { 950.0 } else if decode { 700.0 } else { 0.0 };
        for (name, w) in DEMO_OPS {
            let e = s.ops.entry(name.to_string()).or_default();
            let calls = (steps * 36.0 * w * 4.0).round() as u64;
            e.calls += calls;
            e.timed_calls += calls / 16;
            e.time_ms += ms_per_s * dt * w / 16.0 * (0.9 + 0.2 * j);
        }
        if prefill {
            s.last_prefill_tok_s = 780.0 + 120.0 * j;
            s.prompt_tokens_total += (dt * s.last_prefill_tok_s) as u64;
            s.kv_used_tokens = (s.kv_used_tokens + (dt * s.last_prefill_tok_s) as u64).min(s.kv_capacity_tokens);
        } else if decode {
            s.last_decode_tok_s = 23.0 + 2.5 * j;
            s.generation_tokens_total += (dt * s.last_decode_tok_s) as u64;
            s.kv_used_tokens = (s.kv_used_tokens + (dt * s.last_decode_tok_s) as u64).min(s.kv_capacity_tokens);
            s.last_ttft_ms = 130.0 + 20.0 * j;
        } else if cycle >= 8.0 && cycle < 8.0 + dt + 1e-9 {
            *s.finished.entry("stop".into()).or_insert(0) += 1;
            s.kv_used_tokens = 0;
        }
        s.kv_used_bytes = s.kv_used_tokens * (s.kv_total_bytes / s.kv_capacity_tokens.max(1));
        s.timestamp_ms = (self.t * 1000.0) as u64;
        s.clone()
    }
}

// ---------------------------------------------------------------------------------------------
// Entry point

pub struct TopOptions {
    pub url: String,
    pub demo: bool,
    pub exit_after: Option<Duration>,
}

pub fn run(opts: TopOptions) -> Result<()> {
    let (tx, rx): (Sender<Msg>, Receiver<Msg>) = mpsc::channel();
    if opts.demo {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut d = Demo::new();
            // Warm the history so the first frame is not empty.
            loop {
                std::thread::sleep(Duration::from_millis(200));
                if tx.send(Msg::Snap(d.step(0.2))).is_err() {
                    return;
                }
            }
        });
    } else {
        http::check_scheme(&opts.url)?;
        spawn_sse(&opts.url, tx);
    }
    let mut app = App::new(opts.demo);
    let mut term = ratatui::init();
    let started = Instant::now();
    let res = (|| -> Result<()> {
        loop {
            while let Ok(m) = rx.try_recv() {
                match m {
                    Msg::Snap(s) => app.ingest(s, Instant::now()),
                    Msg::Status(t) => app.status = t,
                }
            }
            term.draw(|f| draw(f, &app))?;
            if opts.exit_after.is_some_and(|d| started.elapsed() >= d) {
                return Ok(());
            }
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(k) = event::read()? {
                    if k.kind == KeyEventKind::Press
                        && (matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
                            || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL)))
                    {
                        return Ok(());
                    }
                }
            }
        }
    })();
    ratatui::restore();
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn text(t: &Terminal<TestBackend>) -> String {
        t.backend().buffer().content().iter().map(|c| c.symbol()).collect()
    }

    #[test]
    fn demo_renders_every_widget() {
        let mut app = App::new(true);
        let mut d = Demo::new();
        let t0 = Instant::now();
        for i in 0..80 {
            let s = d.step(0.2);
            app.ingest(s, t0 + Duration::from_millis(200 * i));
        }
        let mut term = Terminal::new(TestBackend::new(140, 40)).unwrap();
        term.draw(|f| draw(f, &app)).unwrap();
        let t = text(&term);
        for needle in ["decode tok/s", "prefill tok/s", "KV cache bytes", "kernel time share", "scheduler", "engine", "kernels", "matmul", "DEMO"] {
            assert!(t.contains(needle), "missing {needle}");
        }
        assert!(app.op_share["matmul"] > 0.3);
        assert!(app.loads.engine > 0.0);
        // narrow terminals must not panic
        let mut small = Terminal::new(TestBackend::new(50, 12)).unwrap();
        small.draw(|f| draw(f, &app)).unwrap();
    }

    #[test]
    fn heat_is_monotone_in_brightness() {
        let Color::Rgb(a, _, _) = heat(0.0) else { panic!() };
        let Color::Rgb(b, _, _) = heat(0.6) else { panic!() };
        assert!(b > a);
    }
}

//! Rendering: the architecture map with live overlays, the metrics pane (sparklines + latency percentiles) and the chat.
//!
//! Pure function of [`App`] and a [`Theme`]: no clocks, no I/O, so `TestBackend` snapshots are deterministic.

use crate::app::{App, Quantiles};
use crate::chat::Role;
use crate::layout::{Node, NodeKind, Row};
use ratatui::backend::TestBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Sparkline};
use ratatui::{Frame, Terminal};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub bg: Color,
    pub fg: Color,
    pub dim: Color,
    pub border: Color,
    pub accent: Color,
    pub good: Color,
    pub warn: Color,
    pub bad: Color,
    heat0: (u8, u8, u8),
    heat1: (u8, u8, u8),
    heat2: (u8, u8, u8),
    palette: [Color; 6],
}

impl Theme {
    /// Dark slate: the default.
    pub fn dark() -> Theme {
        Theme {
            bg: Color::Rgb(14, 16, 20),
            fg: Color::Rgb(206, 212, 222),
            dim: Color::Rgb(120, 128, 142),
            border: Color::Rgb(58, 64, 76),
            accent: Color::Rgb(94, 190, 220),
            good: Color::Rgb(110, 200, 140),
            warn: Color::Rgb(232, 178, 80),
            bad: Color::Rgb(232, 100, 96),
            heat0: (40, 46, 56),
            heat1: (232, 178, 80),
            heat2: (232, 100, 96),
            palette: [
                Color::Rgb(94, 190, 220),
                Color::Rgb(180, 140, 230),
                Color::Rgb(110, 200, 140),
                Color::Rgb(232, 178, 80),
                Color::Rgb(232, 120, 150),
                Color::Rgb(140, 170, 240),
            ],
        }
    }

    pub fn light() -> Theme {
        Theme {
            bg: Color::Rgb(246, 247, 249),
            fg: Color::Rgb(30, 34, 42),
            dim: Color::Rgb(110, 118, 132),
            border: Color::Rgb(180, 186, 198),
            accent: Color::Rgb(0, 112, 150),
            good: Color::Rgb(30, 130, 70),
            warn: Color::Rgb(170, 110, 0),
            bad: Color::Rgb(190, 50, 50),
            heat0: (214, 218, 226),
            heat1: (214, 150, 40),
            heat2: (200, 60, 56),
            palette: [
                Color::Rgb(0, 112, 150),
                Color::Rgb(120, 70, 190),
                Color::Rgb(30, 130, 70),
                Color::Rgb(170, 110, 0),
                Color::Rgb(190, 50, 100),
                Color::Rgb(50, 90, 200),
            ],
        }
    }

    /// Slate -> amber -> red as `load` goes 0 -> 1.
    pub fn heat(&self, load: f64) -> Color {
        let l = load.clamp(0.0, 1.0);
        let mix = |a: (u8, u8, u8), b: (u8, u8, u8), t: f64| {
            let f = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t).round() as u8;
            (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
        };
        let c = if l < 0.5 { mix(self.heat0, self.heat1, l * 2.0) } else { mix(self.heat1, self.heat2, (l - 0.5) * 2.0) };
        Color::Rgb(c.0, c.1, c.2)
    }

    fn glyph_color(&self, g: char) -> Color {
        self.palette[(g as usize).wrapping_sub('A' as usize) % self.palette.len()]
    }
}

/// Render `app` into a `w` x `h` off-screen terminal and return the text, one line per row (trailing blanks trimmed).
/// This is what `intarsia tui --snapshot` prints and what the snapshot tests compare.
pub fn render_text(app: &App, theme: &Theme, w: u16, h: u16) -> String {
    let mut term = Terminal::new(TestBackend::new(w, h)).expect("test backend");
    term.draw(|f| draw(f, app, theme)).expect("draw");
    let buf = term.backend().buffer();
    let mut out = String::new();
    for y in 0..h {
        let mut line = String::new();
        for x in 0..w {
            line.push_str(buf[(x, y)].symbol());
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

pub fn draw(f: &mut Frame, app: &App, th: &Theme) {
    let area = f.area();
    f.render_widget(Block::default().style(Style::default().bg(th.bg).fg(th.fg)), area);
    if area.height < 4 || area.width < 20 {
        f.render_widget(Paragraph::new("terminal too small"), area);
        return;
    }
    let [head, body, foot] = rects(Layout::vertical([Constraint::Length(1), Constraint::Min(3), Constraint::Length(1)]).split(area));
    draw_header(f, head, app, th);
    if area.width >= 100 {
        let [left, right] = rects(Layout::horizontal([Constraint::Percentage(56), Constraint::Percentage(44)]).split(body));
        let [metrics, chat] = rects(Layout::vertical([Constraint::Length(15), Constraint::Min(6)]).split(right));
        draw_map(f, left, app, th);
        draw_metrics(f, metrics, app, th);
        draw_chat(f, chat, app, th);
    } else {
        let [map, metrics, chat] =
            rects(Layout::vertical([Constraint::Percentage(40), Constraint::Length(15), Constraint::Min(6)]).split(body));
        draw_map(f, map, app, th);
        draw_metrics(f, metrics, app, th);
        draw_chat(f, chat, app, th);
    }
    draw_footer(f, foot, app, th);
}

fn rects<const N: usize>(r: impl AsRef<[Rect]>) -> [Rect; N] {
    let r = r.as_ref();
    std::array::from_fn(|i| r.get(i).copied().unwrap_or_default())
}

fn dim(th: &Theme) -> Style {
    Style::default().fg(th.dim)
}

fn pane<'a>(title: String, focus: bool, th: &Theme) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if focus { th.accent } else { th.border }))
        .title(Span::styled(format!(" {title} "), Style::default().fg(th.accent).add_modifier(Modifier::BOLD)))
}

fn draw_header(f: &mut Frame, a: Rect, app: &App, th: &Theme) {
    let (backend, device) = match (&app.plan, &app.snap) {
        (Some(p), _) if !p.backend.is_empty() => (p.backend.clone(), p.device.clone()),
        (_, Some(s)) => (s.device.backend.clone(), s.device.name.clone()),
        _ => (String::new(), String::new()),
    };
    let state = if app.snap.is_some() { "live".to_string() } else { app.status.clone() };
    let state_color = if app.snap.is_some() { th.good } else { th.warn };
    let line = Line::from(vec![
        Span::styled(" intarsia tui ", Style::default().fg(th.bg).bg(th.accent).add_modifier(Modifier::BOLD)),
        Span::raw(format!(" {}  ", app.url)),
        Span::styled(format!("{}  ", if app.model.is_empty() { "-" } else { &app.model }), Style::default().fg(th.fg).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{backend} {device}  "), dim(th)),
        Span::styled(state, Style::default().fg(state_color)),
    ]);
    f.render_widget(Paragraph::new(line), a);
}

fn draw_footer(f: &mut Frame, a: Rect, app: &App, th: &Theme) {
    let t = if app.streaming {
        " streaming...  PgUp/PgDn scroll chat  Ctrl-L clear  Esc quit"
    } else {
        " Enter send  /clear or Ctrl-L clear chat  PgUp/PgDn scroll chat  Esc / Ctrl-C quit"
    };
    f.render_widget(Paragraph::new(t).style(dim(th)), a);
}

// ---- values -----------------------------------------------------------------------------------------------------------

pub fn fmt_rate(v: f64) -> String {
    if v >= 100.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

/// A `last_*` telemetry value: 0 means no request has finished yet.
pub fn fmt_last(v: f64) -> String {
    if v > 0.0 {
        fmt_rate(v)
    } else {
        "n/a".into()
    }
}

pub fn fmt_secs(v: Option<f64>) -> String {
    match v {
        None => "n/a".into(),
        Some(s) if s < 1.0 => format!("{:.0}ms", s * 1e3),
        Some(s) => format!("{s:.2}s"),
    }
}

pub fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.1}GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1}MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0}kB", b / 1e3)
    } else {
        format!("{b:.0}B")
    }
}

fn fmt_q(q: &Quantiles) -> String {
    if q.count <= 0.0 {
        return "n/a (no requests yet)".into();
    }
    format!(
        "p50 {}  p95 {}  p99 {}  ({}, n={:.0})",
        fmt_secs(q.p50),
        fmt_secs(q.p95),
        fmt_secs(q.p99),
        if q.windowed { "30s" } else { "all" },
        q.count
    )
}

fn trunc(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(w.saturating_sub(1)).collect();
        t.push('~');
        t
    }
}

fn kv_line(app: &App) -> Option<(f64, String, String)> {
    let s = app.snap.as_ref()?;
    let frac = if s.kv_capacity_tokens > 0 {
        s.kv_used_tokens as f64 / s.kv_capacity_tokens as f64
    } else if s.kv_total_bytes > 0 {
        s.kv_used_bytes as f64 / s.kv_total_bytes as f64
    } else {
        0.0
    };
    Some((
        frac,
        format!("{} / {} tok ({:.1}%)", s.kv_used_tokens, s.kv_capacity_tokens, frac * 100.0),
        format!("{} of {}", human_bytes(s.kv_used_bytes), human_bytes(s.kv_total_bytes)),
    ))
}

fn op_share_text(app: &App, op: &str) -> String {
    if !app.ops_timed {
        "n/a".into()
    } else {
        format!("{:.1}%", app.op_share.get(op).copied().unwrap_or(0.0) * 100.0)
    }
}

/// Live lines for a node and whether it is active now. Anything the server does not export reads `n/a`.
fn overlay(n: &Node, app: &App) -> (Vec<String>, bool) {
    let mut lines = n.detail.clone();
    let Some(s) = app.snap.as_ref() else {
        lines.push("waiting for telemetry".into());
        return (lines, false);
    };
    let est = if app.rate_gen_est { "~" } else { "" };
    let live = match n.kind {
        NodeKind::Queue => {
            lines.push(format!("running {} · waiting {}", s.running, s.waiting));
            let depth = app.batch.queue_depth.map_or("n/a".to_string(), |d| format!("{d:.0}"));
            lines.push(format!("batch queue depth {depth} · requests {}", s.requests_total));
            s.running + s.waiting > 0
        }
        NodeKind::Prefill => {
            lines.push(format!("now {} tok/s", fmt_rate(app.rate_prefill)));
            lines.push(format!("last {} tok/s · cached {}", fmt_last(s.last_prefill_tok_s), s.cached_tokens_total));
            app.rate_prefill > 0.0
        }
        NodeKind::Decode => {
            lines.push(format!("now {est}{} tok/s · last {}", fmt_rate(app.rate_gen), fmt_last(s.last_decode_tok_s)));
            let batch = app.batch.avg_rows.map_or("n/a".to_string(), |r| format!("{r:.1}"));
            lines.push(format!("batch size {batch} · steps {}/s", fmt_rate(app.rate_steps)));
            app.rate_gen > 0.0
        }
        NodeKind::LayerGroup => {
            lines.push("time share n/a (per-op below)".into());
            s.running > 0 && app.rate_steps > 0.0
        }
        NodeKind::Op => app.op_share.get(&n.ops[0]).is_some_and(|x| *x > 0.0),
        NodeKind::Kv => {
            if let Some((_, tok, bytes)) = kv_line(app) {
                lines.push(tok);
                lines.push(bytes);
            }
            if !n.ops.is_empty() {
                lines.push(n.ops.iter().map(|o| format!("{o} {}", op_share_text(app, o))).collect::<Vec<_>>().join(" · "));
            }
            s.kv_used_tokens > 0
        }
        NodeKind::Sampler => {
            if n.ops.is_empty() {
                lines.push("ops n/a".into());
            } else {
                lines.push(n.ops.iter().map(|o| format!("{o} {}", op_share_text(app, o))).collect::<Vec<_>>().join(" · "));
            }
            app.rate_gen > 0.0
        }
        NodeKind::Output => {
            lines.push(format!("generated {} tok", s.generation_tokens_total));
            let fin: Vec<String> = s.finished.iter().filter(|(_, v)| **v > 0).map(|(k, v)| format!("{k} {v}")).collect();
            lines.push(if fin.is_empty() { "finished none yet".into() } else { format!("finished {}", fin.join(" · ")) });
            app.rate_gen > 0.0
        }
    };
    (lines, live)
}

// ---- architecture map -------------------------------------------------------------------------------------------------

fn draw_map(f: &mut Frame, area: Rect, app: &App, th: &Theme) {
    let title = match &app.plan {
        Some(p) => format!("architecture · {} layers · exactness {}", app.map.layer_count, if p.exactness.is_empty() { "-" } else { &p.exactness }),
        None if app.plan_known => "architecture · no plan (built from telemetry op names)".to_string(),
        None => "architecture · waiting for plan".to_string(),
    };
    let block = pane(title, false, th);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let mut y = inner.y;
    let bottom = inner.bottom();
    let active = app.rate_steps > 0.0;
    for (ri, row) in app.map.rows.iter().enumerate() {
        if ri > 0 {
            if y >= bottom {
                break;
            }
            let c = if active { th.accent } else { th.border };
            f.render_widget(Paragraph::new("▼").centered().style(Style::default().fg(c)), Rect { x: inner.x, y, width: inner.width, height: 1 });
            y += 1;
        }
        let is_ops = row.nodes.first().is_some_and(|n| n.kind == NodeKind::Op);
        y = if is_ops { draw_ops(f, inner, y, row, app, th) } else { draw_boxes(f, inner, y, row, app, th) };
        if row.nodes.first().is_some_and(|n| n.kind == NodeKind::LayerGroup) && y < bottom {
            y = draw_strip(f, inner, y, app, th);
        }
        if y >= bottom {
            break;
        }
    }
    if y >= bottom && app.map.rows.len() > 1 {
        // the map is taller than the pane: say so on the last line instead of silently clipping
        let r = Rect { x: inner.x, y: bottom - 1, width: inner.width, height: 1 };
        f.render_widget(Paragraph::new("... (enlarge the terminal to see the rest)").style(dim(th)), r);
    }
}

fn draw_boxes(f: &mut Frame, inner: Rect, mut y: u16, row: &Row, app: &App, th: &Theme) -> u16 {
    let bottom = inner.bottom();
    let cols = row.nodes.len().min((inner.width / 26).max(1) as usize).max(1);
    for chunk in row.nodes.chunks(cols) {
        let ov: Vec<(Vec<String>, bool)> = chunk.iter().map(|n| overlay(n, app)).collect();
        let h = ov.iter().map(|(l, _)| l.len()).max().unwrap_or(0) as u16 + 2;
        if y + h > bottom {
            return bottom;
        }
        let cells = Layout::horizontal(vec![Constraint::Ratio(1, chunk.len() as u32); chunk.len()])
            .split(Rect { x: inner.x, y, width: inner.width, height: h });
        for ((n, (lines, live)), r) in chunk.iter().zip(&ov).zip(cells.iter()) {
            let tcol = n.glyph.map_or(th.fg, |g| th.glyph_color(g));
            let b = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(if *live { th.accent } else { th.border }))
                .title(Span::styled(
                    format!(" {} ", trunc(&n.title, r.width.saturating_sub(4) as usize)),
                    Style::default().fg(if *live { th.accent } else { tcol }).add_modifier(Modifier::BOLD),
                ));
            let ir = b.inner(*r);
            f.render_widget(b, *r);
            for (i, l) in lines.iter().take(ir.height as usize).enumerate() {
                let st = if i < n.detail.len() { dim(th) } else { Style::default().fg(th.fg) };
                f.render_widget(
                    Paragraph::new(trunc(l, ir.width as usize)).style(st),
                    Rect { x: ir.x, y: ir.y + i as u16, width: ir.width, height: 1 },
                );
            }
        }
        y += h;
    }
    y
}

/// One glyph per layer, coloured by block signature, wrapped to the pane width.
fn draw_strip(f: &mut Frame, inner: Rect, mut y: u16, app: &App, th: &Theme) -> u16 {
    let w = inner.width.saturating_sub(2).max(1) as usize;
    for chunk in app.map.layer_strip.chunks(w) {
        if y >= inner.bottom() {
            return y;
        }
        let spans: Vec<Span> = chunk.iter().map(|g| Span::styled(g.to_string(), Style::default().fg(th.glyph_color(*g)))).collect();
        f.render_widget(Paragraph::new(Line::from(spans)), Rect { x: inner.x + 1, y, width: inner.width.saturating_sub(1), height: 1 });
        y += 1;
    }
    y
}

fn draw_ops(f: &mut Frame, inner: Rect, mut y: u16, row: &Row, app: &App, th: &Theme) -> u16 {
    let bottom = inner.bottom();
    if y >= bottom {
        return bottom;
    }
    let note = if app.snap.is_none() {
        String::new()
    } else if app.ops_timed {
        " · share of sampled op time".to_string()
    } else {
        " · time share n/a: server op profiling is off (--profile-every N)".to_string()
    };
    f.render_widget(
        Paragraph::new(format!("kernels ({}){note}", row.nodes.len())).style(dim(th)),
        Rect { x: inner.x + 1, y, width: inner.width.saturating_sub(1), height: 1 },
    );
    y += 1;
    let cols = ((inner.width / 30).max(1) as usize).min(row.nodes.len()).max(1);
    let cw = inner.width / cols as u16;
    let name_w = row.nodes.iter().map(|n| n.title.chars().count()).max().unwrap_or(8).clamp(8, (cw as usize).saturating_sub(20).max(8));
    for chunk in row.nodes.chunks(cols) {
        if y >= bottom {
            return bottom;
        }
        for (i, n) in chunk.iter().enumerate() {
            let share = app.op_share.get(&n.ops[0]).copied().unwrap_or(0.0);
            let bar_w = cw.saturating_sub(name_w as u16 + 1 + 7 + 2).min(10) as usize;
            let filled = ((share * bar_w as f64 * 2.0).ceil() as usize).min(bar_w); // x2: shares are mostly small
            let mut spans = vec![Span::styled(format!("{:<name_w$} ", trunc(&n.title, name_w)), Style::default().fg(th.fg))];
            spans.push(Span::styled(format!("{:>6} ", op_share_text(app, &n.ops[0])), if app.ops_timed { Style::default().fg(th.heat(share * 1.5)) } else { dim(th) }));
            if app.ops_timed {
                spans.push(Span::styled("█".repeat(filled), Style::default().fg(th.heat(share * 1.5))));
                spans.push(Span::styled("░".repeat(bar_w - filled), Style::default().fg(th.border)));
            }
            f.render_widget(
                Paragraph::new(Line::from(spans)),
                Rect { x: inner.x + 1 + i as u16 * cw, y, width: cw.saturating_sub(1), height: 1 },
            );
        }
        y += 1;
    }
    y
}

// ---- metrics ------------------------------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn spark(f: &mut Frame, a: Rect, label: &str, value: String, hist: &VecDeque<f64>, scale: f64, color: Color, th: &Theme) {
    let lw = 22.min(a.width / 2);
    let [l, r] = rects(Layout::horizontal([Constraint::Length(lw), Constraint::Min(1)]).split(a));
    f.render_widget(
        Paragraph::new(Line::from(vec![Span::styled(format!("{label:<10}"), dim(th)), Span::styled(trunc(&value, lw.saturating_sub(10) as usize), Style::default().fg(th.fg))])),
        l,
    );
    let n = r.width as usize;
    let data: Vec<u64> = hist.iter().skip(hist.len().saturating_sub(n)).map(|v| (v.max(0.0) * scale).round() as u64).collect();
    let max = data.iter().copied().max().unwrap_or(0).max(1);
    f.render_widget(Sparkline::default().data(data).max(max).style(Style::default().fg(color)), r);
}

fn draw_metrics(f: &mut Frame, area: Rect, app: &App, th: &Theme) {
    let block = pane("metrics".to_string(), false, th);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let Some(s) = app.snap.as_ref() else {
        f.render_widget(Paragraph::new(format!("waiting for {}/events ...", app.url)).style(dim(th)), inner);
        return;
    };
    let est = if app.rate_gen_est { "~" } else { "" };
    let rows = Layout::vertical(vec![Constraint::Length(1); inner.height.min(13) as usize]).split(inner);
    let row = |i: usize| rows.get(i).copied();
    let sparks: [(&str, String, &VecDeque<f64>, f64, Color); 4] = [
        ("decode", format!("{est}{} tok/s", fmt_rate(app.rate_gen)), &app.hist_gen, 10.0, th.good),
        ("prefill", format!("{} tok/s", fmt_rate(app.rate_prefill)), &app.hist_prefill, 10.0, th.accent),
        ("waiting", format!("{} (run {})", s.waiting, s.running), &app.hist_waiting, 10.0, th.warn),
        ("KV used", format!("{:.1}%", kv_line(app).map_or(0.0, |k| k.0 * 100.0)), &app.hist_kv, 10.0, th.heat(0.55)),
    ];
    for (i, (label, value, hist, scale, color)) in sparks.into_iter().enumerate() {
        if let Some(r) = row(i) {
            spark(f, r, label, value, hist, scale, color, th);
        }
    }
    let batch = &app.batch;
    let na = || "n/a".to_string();
    let text: Vec<(String, String)> = vec![
        ("TTFT".into(), if app.metrics_ok { fmt_q(&app.ttft) } else { "n/a (/metrics unreachable)".into() }),
        ("ITL".into(), if app.metrics_ok { fmt_q(&app.itl) } else { "n/a".into() }),
        ("last req".into(), format!("TTFT {} · decode {} tok/s · prefill {} tok/s", fmt_secs((s.last_ttft_ms > 0.0).then(|| s.last_ttft_ms / 1e3)), fmt_last(s.last_decode_tok_s), fmt_last(s.last_prefill_tok_s))),
        ("batch".into(), match batch.avg_rows {
            Some(r) => format!("avg {r:.1} rows · step {} ms", batch.step_ewma_ms.map_or_else(na, |m| format!("{m:.1}"))),
            None => "n/a (single-sequence engine or no decode yet)".into(),
        }),
        ("queue".into(), format!("running {} · waiting {} · depth max {}", s.running, s.waiting, batch.queue_depth_max.map_or_else(na, |d| format!("{d:.0}")))),
        ("steps".into(), format!("{}/s · total {}", fmt_rate(app.rate_steps), s.steps_total)),
        ("tokens".into(), format!("prompt {} · cached {} · generated {}", s.prompt_tokens_total, s.cached_tokens_total, s.generation_tokens_total)),
        ("memory".into(), if s.device.total_mem_bytes > 0 {
            format!("weights {} · free {} of {}", human_bytes(s.weights_bytes), human_bytes(s.device.free_mem_bytes), human_bytes(s.device.total_mem_bytes))
        } else {
            format!("weights {} · device memory n/a", human_bytes(s.weights_bytes))
        }),
        ("uptime".into(), format!("{:.0}s · {} requests", s.uptime_s, s.requests_total)),
    ];
    for (i, (k, v)) in text.into_iter().enumerate() {
        if let Some(r) = row(4 + i) {
            let line = Line::from(vec![Span::styled(format!("{k:<10}"), dim(th)), Span::raw(trunc(&v, r.width.saturating_sub(10) as usize))]);
            f.render_widget(Paragraph::new(line), r);
        }
    }
}

// ---- chat ---------------------------------------------------------------------------------------------------------------

/// Greedy word wrap on character counts; keeps explicit newlines.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for para in text.split('\n') {
        let mut cur = String::new();
        for word in para.split(' ') {
            let mut word = word.to_string();
            while word.chars().count() > width {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(word.chars().take(width).collect());
                word = word.chars().skip(width).collect();
            }
            let need = cur.chars().count() + usize::from(!cur.is_empty()) + word.chars().count();
            if need > width && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&word);
        }
        out.push(cur);
    }
    out
}

fn draw_chat(f: &mut Frame, area: Rect, app: &App, th: &Theme) {
    let mut title = format!("chat · {}", if app.model.is_empty() { "-" } else { &app.model });
    if app.streaming {
        title.push_str(&format!(" · streaming {} tok", app.stream_chunks));
    } else if let Some((tok, secs)) = app.last_chat {
        title.push_str(&format!(" · last {tok} tok in {secs:.1}s ({} tok/s)", fmt_rate(if secs > 0.0 { tok as f64 / secs } else { 0.0 })));
    }
    let block = pane(title, true, th);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 2 {
        return;
    }
    let [hist, input] = rects(Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(inner));
    let w = hist.width as usize;
    let mut lines: Vec<Line> = Vec::new();
    if app.chat.is_empty() {
        lines.push(Line::styled("Talk to the model: type below, Enter sends.", dim(th)));
    }
    for m in &app.chat {
        let (tag, style) = match m.role {
            Role::User => ("you ", Style::default().fg(th.accent).add_modifier(Modifier::BOLD)),
            Role::Assistant => ("ai  ", Style::default().fg(th.good).add_modifier(Modifier::BOLD)),
            Role::Note => ("!   ", Style::default().fg(th.bad)),
        };
        for (i, l) in wrap(&m.text, w.saturating_sub(4)).into_iter().enumerate() {
            lines.push(Line::from(vec![Span::styled(if i == 0 { tag } else { "    " }, style), Span::raw(l)]));
        }
    }
    let h = hist.height as usize;
    let max_scroll = lines.len().saturating_sub(h);
    let scroll = app.chat_scroll.min(max_scroll);
    let end = lines.len() - scroll;
    let start = end.saturating_sub(h);
    f.render_widget(Paragraph::new(lines[start..end].to_vec()), hist);
    let text = if app.input.is_empty() {
        Line::from(vec![Span::styled("> ", Style::default().fg(th.accent)), Span::styled("message, Enter to send", dim(th))])
    } else {
        let shown = trunc_left(&app.input, input.width.saturating_sub(4) as usize);
        Line::from(vec![Span::styled("> ", Style::default().fg(th.accent)), Span::raw(shown), Span::styled("▌", Style::default().fg(th.accent))])
    };
    f.render_widget(Paragraph::new(text), input);
}

fn trunc_left(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n <= w {
        s.to_string()
    } else {
        s.chars().skip(n - w).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_keeps_words_and_newlines() {
        assert_eq!(wrap("hello big world", 9), vec!["hello big", "world"]);
        assert_eq!(wrap("a\nb", 5), vec!["a", "b"]);
        assert_eq!(wrap("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
        assert_eq!(wrap("", 4), vec![""]);
    }

    #[test]
    fn formatters() {
        assert_eq!(fmt_secs(Some(0.1404)), "140ms");
        assert_eq!(fmt_secs(Some(1.234)), "1.23s");
        assert_eq!(fmt_secs(None), "n/a");
        assert_eq!(fmt_rate(24.14), "24.1");
        assert_eq!(fmt_rate(850.2), "850");
        assert_eq!(human_bytes(604_000_000), "604.0MB");
    }

    #[test]
    fn heat_is_monotone_in_brightness_and_themes_differ() {
        let t = Theme::dark();
        let Color::Rgb(a, _, _) = t.heat(0.0) else { panic!() };
        let Color::Rgb(b, _, _) = t.heat(0.6) else { panic!() };
        assert!(b > a);
        assert_ne!(Theme::dark().bg, Theme::light().bg);
    }

    #[test]
    fn tiny_terminals_do_not_panic() {
        let app = App::new("http://x");
        for (w, h) in [(10, 3), (20, 4), (50, 12), (99, 30), (100, 8), (300, 100)] {
            let _ = render_text(&app, &Theme::dark(), w, h);
        }
    }
}

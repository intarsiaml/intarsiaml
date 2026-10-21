//! `intarsia tui`: terminal UI for a running server (live architecture map, streaming chat, metrics).
//!
//! The UI lives in the `intarsia-tui` crate; this is the CLI wrapper. Built only with the `tui` cargo feature (on by
//! default; `--no-default-features` leaves it out).

use anyhow::{anyhow, Result};
use intarsia_tui::{SnapshotOptions, TuiOptions};
use std::time::Duration;

#[derive(clap::Args, Debug)]
pub struct TuiArgs {
    /// URL of a running `intarsia serve`.
    #[arg(long, default_value = "http://127.0.0.1:8090")]
    pub url: String,
    /// Light theme (the default is dark).
    #[arg(long)]
    pub light: bool,
    /// Let thinking models think in the chat (default: thinking off, `enable_thinking: false`).
    #[arg(long)]
    pub thinking: bool,
    /// `max_tokens` of a chat turn.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,
    /// Headless: render the UI off-screen, print text frames to stdout and exit (no terminal needed; for CI, docs, logs).
    #[arg(long)]
    pub snapshot: bool,
    /// With `--snapshot`: send this chat message and print frames mid-stream, right after the reply and once settled.
    #[arg(long, value_name = "TEXT", requires = "snapshot")]
    pub chat: Option<String>,
    /// With `--snapshot`: frame size as WxH.
    #[arg(long, value_name = "WxH", default_value = "150x48", requires = "snapshot")]
    pub size: String,
    /// With `--snapshot` and no `--chat`: seconds to watch before the frame.
    #[arg(long, value_name = "SECS", default_value_t = 3.0, requires = "snapshot")]
    pub hold: f64,
    /// With `--snapshot --chat`: capture the mid-stream frame after this many streamed chunks.
    #[arg(long, value_name = "N", default_value_t = 12, requires = "snapshot")]
    pub mid_tokens: u64,
    /// With `--snapshot`: stop waiting for the server after this many seconds.
    #[arg(long, value_name = "SECS", default_value_t = 60.0, requires = "snapshot")]
    pub timeout: f64,
    /// Exit by itself after this many seconds (smoke tests of the interactive mode).
    #[arg(long, value_name = "SECS")]
    pub exit_after: Option<f64>,
}

fn parse_size(s: &str) -> Result<(u16, u16)> {
    let (w, h) = s.split_once(['x', 'X']).ok_or_else(|| anyhow!("--size wants WxH, e.g. 150x48"))?;
    let (w, h): (u16, u16) = (w.trim().parse()?, h.trim().parse()?);
    if w < 20 || h < 4 {
        return Err(anyhow!("--size {s} is too small (at least 20x4)"));
    }
    Ok((w, h))
}

pub fn run(a: TuiArgs) -> Result<()> {
    let snapshot = if a.snapshot {
        let (width, height) = parse_size(&a.size)?;
        Some(SnapshotOptions {
            chat: a.chat,
            width,
            height,
            hold: Duration::from_secs_f64(a.hold),
            mid_tokens: a.mid_tokens,
            timeout: Duration::from_secs_f64(a.timeout),
        })
    } else {
        None
    };
    intarsia_tui::run(TuiOptions {
        url: a.url,
        light: a.light,
        thinking: a.thinking,
        max_tokens: a.max_tokens,
        snapshot,
        exit_after: a.exit_after.map(Duration::from_secs_f64),
    })
}

#[cfg(test)]
mod tests {
    use super::parse_size;

    #[test]
    fn size_parses() {
        assert_eq!(parse_size("150x48").unwrap(), (150, 48));
        assert_eq!(parse_size("80X24").unwrap(), (80, 24));
        assert!(parse_size("80").is_err());
        assert!(parse_size("5x2").is_err());
    }
}

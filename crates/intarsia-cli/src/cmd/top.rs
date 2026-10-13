//! `intarsia top`: terminal dashboard fed by the server's `/events` SSE stream.
//!
//! The dashboard itself lives in `intarsia_bench::top`; this is the CLI wrapper.

use anyhow::Result;
use std::time::Duration;

#[derive(clap::Args, Debug)]
pub struct TopArgs {
    /// URL of a running `intarsia serve`.
    #[arg(long, default_value = "http://127.0.0.1:8090")]
    pub url: String,
    /// Feed the dashboard synthetic snapshots instead of connecting to a server.
    #[arg(long)]
    pub demo: bool,
    /// Exit by itself after this many seconds (smoke tests).
    #[arg(long, value_name = "SECS")]
    pub exit_after: Option<f64>,
}

pub fn run(args: TopArgs) -> Result<()> {
    intarsia_bench::top::run(intarsia_bench::top::TopOptions {
        url: args.url,
        demo: args.demo,
        exit_after: args.exit_after.map(Duration::from_secs_f64),
    })
}

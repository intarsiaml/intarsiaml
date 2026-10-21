//! `intarsia tui`: a terminal UI for a running `intarsia serve`.
//!
//! Three panes: a live architecture map (request queue, prefill/decode, the layer stack from the model plan, kernels, KV
//! cache, sampler, output) overlaid with live metrics, a metrics pane with sparklines and TTFT/ITL percentiles, and a
//! streaming chat against the server's OpenAI endpoint. The data sources are the ones the web UI's map uses
//! (`/v1/intarsia/plan`, `/events`) plus `/metrics`; nothing is invented, and what the server does not export reads `n/a`.
//!
//! The map is data-driven ([`layout`]): layer groups and kernel nodes come from the plan, so a new block kind or op shows
//! up without changes here. `--snapshot` renders the same UI off-screen and prints it as text (no TTY needed).

pub mod app;
pub mod chat;
pub mod feed;
pub mod layout;
pub mod prom;
mod run;
pub mod ui;

pub use run::{run, SnapshotOptions, TuiOptions};

//! Speed benchmark client and terminal dashboard.
//!
//! Both talk plain HTTP to a running server: `speed` drives any OpenAI-compatible
//! `/v1/chat/completions` endpoint (so it can also bench vLLM for comparison), `top` follows
//! the engine's `/events` SSE stream.

#![recursion_limit = "256"] // record.rs builds large json! literals

pub mod http;
pub mod record;
pub mod snapshot;
pub mod reuse;
pub mod speed;
pub mod top;

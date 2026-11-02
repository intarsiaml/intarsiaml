//! DeepSeek-V4.1 (Flash) in the engine (lane 1a skeleton, docs/plans/2026-10-31-dsv41-unpark.md): the op set, the config,
//! the model graph and the per-sequence state layout. The ops and their contract: [`ops`] and docs/design/dsv41-ops.md.

pub mod ops;
pub mod cases;
pub mod engram;
pub mod config;
pub mod graph;
pub mod plan;
pub mod runtime;

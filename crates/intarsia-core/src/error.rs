//! Error type shared by the core crates.

use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A backend (device) call failed: allocation, launch, copy, sync.
    #[error("backend error: {0}")]
    Backend(String),
    /// Shapes, dtypes or attributes of an op call are inconsistent.
    #[error("invalid op call: {0}")]
    InvalidCall(String),
    /// No admitted kernel implementation matches (op, backend, arch, shape).
    #[error("no kernel for {0}")]
    NoKernel(String),
    /// Feature not available in this build / phase (for example an unimplemented block kind).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Model description, recipe or weights problem.
    #[error("model error: {0}")]
    Model(String),
    /// Out of memory according to the memory plan.
    #[error("memory plan: {0}")]
    Memory(String),
    /// Request rejected before running (too long, bad parameters).
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("tokenizer/template error: {0}")]
    Text(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn backend(msg: impl fmt::Display) -> Self {
        Error::Backend(msg.to_string())
    }
    pub fn invalid(msg: impl fmt::Display) -> Self {
        Error::InvalidCall(msg.to_string())
    }
    pub fn model(msg: impl fmt::Display) -> Self {
        Error::Model(msg.to_string())
    }
}

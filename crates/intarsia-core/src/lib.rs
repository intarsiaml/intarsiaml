//! Intarsia core: tensors, the [`backend::Backend`] trait, the op/kernel
//! registry with its admission harness, the generic decoder, sequence state,
//! memory planning, sampling, telemetry and the [`engine::Engine`] API.
//!
//! See `docs/ARCHITECTURE.md` sections 3, 4, 6 and 13.

pub mod admission;
pub mod backend;
pub mod comm;
pub mod constrain;
pub mod dtype;
pub mod engine;
pub mod error;
pub mod exec;
#[cfg(feature = "fault-inject")]
pub mod fault;
pub mod fusion;
pub mod gdn;
pub mod hostprof;
pub mod memplan;
pub mod model;
pub mod ops;
pub mod plan;
pub mod prefix_store;
pub mod quant;
pub mod recipe;
pub mod registry;
pub mod rng;
pub mod rope;
pub mod sampler;
pub mod scheduler;
pub mod spec;
pub mod state;
pub mod telemetry;
pub mod tensor;
pub mod tp_remote;
pub mod weights;

pub use backend::{Backend, Buffer, DeviceInfo, MemKind, StreamId, TensorRef};
pub use comm::{Comm, CommStats, HostCommGroup};
pub use dtype::DType;
pub use engine::{BatchedEngine, BatchedOptions, ChatOpts, Engine, EngineMeta, FinishReason, GenEvent, GenRequest, LocalEngine, ModelInfo, Prewarm, ScriptStep, TextCodec};
pub use error::{Error, Result};
pub use ops::{Op, OpAttrs, OpCall, ShapeInfo};
pub use registry::{Admission, Exactness, KernelImpl, Registry, Tol};
pub use telemetry::{Telemetry, TelemetrySnapshot};
pub use tensor::HostTensor;

//! Inkling-2-Mini (`inkling2_loop`, thinkingmachines/Inkling-2-Mini) in the engine: the op set (`Op::AttnResMix` ..
//! `Op::ConvMix`, discriminants 67..=74) and its contract. Port spec: docs/plans/2026-11-21-inkling2-port.md (section a
//! has the formulas, f the gap list); the model block is `crate::model::inkling2`.

pub mod cases;
pub mod config;
pub mod ops;

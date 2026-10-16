//! The `intarsia` command line tool as a library so other crates (and tests)
//! can reuse the model/backend loading helpers.

// The cuda and hip kernel libraries export identical `ik_*` C symbols (the shared kernel ABI); linking both would
// silently resolve each launcher to whichever archive the linker meets first.
#[cfg(all(feature = "cuda", feature = "hip"))]
compile_error!("the `cuda` and `hip` features are mutually exclusive: both backends export the same ik_* C symbols; build one binary per backend");

pub mod cmd;
pub mod common;

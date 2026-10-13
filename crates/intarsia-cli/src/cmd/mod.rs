//! One module per subcommand so agents working on different commands never
//! touch the same file. Each module exports an `Args` struct (clap) and `run`.

pub mod accuracy;
pub mod bench;
pub mod doctor;
pub mod generate;
pub mod kernels;
pub mod plan;
pub mod serve;
pub mod top;

use anyhow::Result;
use clap::{Parser, Subcommand};
use intarsia_cli::cmd;

/// IntarsiaML: a modular LLM inference engine.
#[derive(Parser)]
#[command(name = "intarsia", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Complete a prompt with a local model.
    Generate(cmd::generate::GenerateArgs),
    /// Kernel registry tools: admission tests against the cpu-ref oracle.
    Kernels(cmd::kernels::KernelsArgs),
    /// Serve OpenAI / Anthropic compatible APIs.
    Serve(cmd::serve::ServeArgs),
    /// Golden-logit accuracy gate.
    Accuracy(cmd::accuracy::AccuracyArgs),
    /// Measure a golden's own reference noise (a second run or another precision), offline.
    RefNoise(cmd::accuracy::RefNoiseArgs),
    /// Speed benchmark.
    Bench(cmd::bench::BenchArgs),
    /// New model, one command: recipe, smoke, quick tier (published), then the in-depth tier. Resumable.
    Onboard(cmd::onboard::OnboardArgs),
    /// Host fingerprint and known-pitfall checks.
    Doctor(cmd::doctor::DoctorArgs),
    /// Show the resolved execution plan (kernel per op) for a model.
    Plan(cmd::plan::PlanArgs),
    /// Recipe tools: scaffold a recipe for a new model.
    Recipe(cmd::recipe::RecipeArgs),
    /// One-shot model qualification: quick numbers first, then the in-depth run, back to back.
    Qualify(cmd::qualify::QualifyArgs),
    /// Terminal dashboard for a running server.
    Top(cmd::top::TopArgs),
    /// Rank 1 of a cross-process tensor-parallel group (rank 0 runs any command with `--tp-remote`).
    TpWorker(intarsia_cli::common::TpWorkerArgs),
    /// Terminal UI for a running server: live architecture map, streaming chat, metrics (feature `tui`).
    #[cfg(feature = "tui")]
    Tui(cmd::tui::TuiArgs),
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Command::Generate(a) => cmd::generate::run(a),
        Command::Kernels(a) => cmd::kernels::run(a),
        Command::Onboard(a) => cmd::onboard::run(a),
        Command::Serve(a) => cmd::serve::run(a),
        Command::Accuracy(a) => cmd::accuracy::run(a),
        Command::RefNoise(a) => cmd::accuracy::run_ref_noise(a),
        Command::Bench(a) => cmd::bench::run(a),
        Command::Doctor(a) => cmd::doctor::run(a),
        Command::Plan(a) => cmd::plan::run(a),
        Command::Recipe(a) => cmd::recipe::run(a),
        Command::Qualify(a) => cmd::qualify::run(a),
        Command::Top(a) => cmd::top::run(a),
        Command::TpWorker(a) => intarsia_cli::common::run_tp_worker(a),
        #[cfg(feature = "tui")]
        Command::Tui(a) => cmd::tui::run(a),
    }
}

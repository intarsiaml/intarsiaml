//! `intarsia kernels`: list registered kernels and run the admission tests.

use crate::common::open_backend;
use anyhow::{bail, Result};
use clap::{Args, Subcommand};
use intarsia_core::admission::{run_admission, AdmissionOpts, SweepConfig};
use intarsia_core::{Backend, Op};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct KernelsArgs {
    #[command(subcommand)]
    pub cmd: KernelsCmd,
}

#[derive(Subcommand, Debug)]
pub enum KernelsCmd {
    /// Run every registered implementation against the cpu-ref oracle over a shape sweep.
    Test(TestArgs),
    /// List the kernels a backend registers.
    List(ListArgs),
}

#[derive(Args, Debug)]
pub struct TestArgs {
    /// Backend under test (the oracle is always cpu-ref).
    #[arg(long, default_value = "cpu")]
    pub backend: String,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Take sweep shapes from this model directory's config.json (default: Qwen3-0.6B-like).
    #[arg(long, short = 'm')]
    pub model: Option<PathBuf>,
    /// Smaller sweep.
    #[arg(long)]
    pub quick: bool,
    /// Only this op (e.g. matmul, attn_decode).
    #[arg(long)]
    pub op: Option<String>,
    /// Only this kernel name.
    #[arg(long)]
    pub kernel: Option<String>,
    /// Print every case, not just failures.
    #[arg(long, short = 'v')]
    pub verbose: bool,
    /// Emit the report as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct ListArgs {
    #[arg(long, default_value = "cpu")]
    pub backend: String,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
}

pub fn run(args: KernelsArgs) -> Result<()> {
    match args.cmd {
        KernelsCmd::Test(a) => test(a),
        KernelsCmd::List(a) => list(a),
    }
}

fn list(a: ListArgs) -> Result<()> {
    let b = open_backend(&a.backend, a.device)?;
    let info = b.info();
    println!("{} ({}/{})", info.name, info.backend, info.arch);
    println!("{:<14} {:<26} {:<13} {:<5} {}", "op", "kernel", "exactness", "ref", "tolerance");
    for k in b.kernels() {
        println!(
            "{:<14} {:<26} {:<13} {:<5} max_abs={:.1e} rel_l2={:.1e}",
            k.op.as_str(),
            k.name,
            k.exact.as_str(),
            if k.is_reference { "yes" } else { "" },
            k.tol.max_abs,
            k.tol.rel_l2
        );
    }
    Ok(())
}

fn test(a: TestArgs) -> Result<()> {
    let reference = intarsia_backend_cpu::CpuBackend::arc();
    let candidate: std::sync::Arc<dyn Backend> = open_backend(&a.backend, a.device)?;
    let mut sweep = SweepConfig { quick: a.quick, ..Default::default() };
    if let Some(dir) = &a.model {
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let keys = intarsia_core::recipe::ConfigKeys::default();
        let d = intarsia_core::recipe::ModelDims::from_hf_config(&cfg, &keys).map_err(|e| anyhow::anyhow!("{e}"))?;
        sweep.hidden = d.hidden;
        sweep.n_heads = d.n_heads;
        sweep.n_kv_heads = d.n_kv_heads;
        sweep.head_dim = d.head_dim;
        sweep.intermediate = d.intermediate;
        sweep.vocab = d.vocab;
        sweep.rope_theta = d.rope_theta;
    }
    let only_ops = match &a.op {
        Some(o) => Some(vec![Op::parse(o).ok_or_else(|| anyhow::anyhow!("unknown op '{o}'"))?]),
        None => None,
    };
    let t0 = std::time::Instant::now();
    let report = run_admission(
        &*reference,
        &*candidate,
        &AdmissionOpts { sweep: Some(sweep), only_ops, only_kernel: a.kernel.clone() },
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    if a.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", report.render(a.verbose));
        println!("{} kernels, {} cases in {:.1}s", report.kernels.len(), report.rows.len(), t0.elapsed().as_secs_f64());
    }
    // An admission run that exercised nothing must not look like a pass (a filter that matched no kernel, a backend
    // whose kernels all skipped): say SKIPPED per kernel and fail when no kernel ran a single case.
    let skipped: Vec<&str> = report.kernels.iter().filter(|k| k.cases == 0).map(|k| k.kernel.as_str()).collect();
    for k in &skipped {
        eprintln!("SKIPPED: kernel {k} ran 0 cases on backend '{}'", a.backend);
    }
    let ran = report.kernels.len() - skipped.len();
    if ran == 0 {
        bail!(
            "no kernel ran any case on backend '{}' ({} registered kernel(s) matched the filters): nothing was tested",
            a.backend,
            report.kernels.len()
        );
    }
    if !report.all_pass() {
        bail!("kernel admission failed");
    }
    Ok(())
}

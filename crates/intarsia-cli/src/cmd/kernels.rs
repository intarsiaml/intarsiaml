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
            "{:<14} {:<26} {:<13} {:<5} max_abs={:.1e} rel_l2={:.1e}{}",
            k.op.as_str(),
            k.name,
            k.exact.as_str(),
            if k.is_reference { "yes" } else { "" },
            k.tol.max_abs,
            k.tol.rel_l2,
            k.tol_f32.map_or(String::new(), |t| format!("  f32 out: max_abs={:.1e} rel_l2={:.1e}", t.max_abs, t.rel_l2))
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
        // the model's recipe names its config keys (`[config]`: LFM2 spells `rms_norm_eps` as `norm_eps`); no recipe: the HF names
        let keys = intarsia_models::recipe::select_recipe(&cfg, &[]).map(|(_, r)| r.config).unwrap_or_default();
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
    // whose kernels all skipped): say SKIPPED per kernel and fail when no kernel ran a single case. A kernel with no case in
    // this sweep (e.g. --quick has no shape it accepts) is untested, not failed: it stays `Untested` for plan-time admission,
    // which tests it on the plan's own shapes, so it does not fail the command (it used to, through `all_pass`).
    let skipped: Vec<&str> = report.kernels.iter().filter(|k| k.cases == 0).map(|k| k.kernel.as_str()).collect();
    for k in &skipped {
        eprintln!("SKIPPED: kernel {k} ran 0 cases on backend '{}'", a.backend);
    }
    match verdict(&report.kernels) {
        Verdict::NothingRan => bail!(
            "no kernel ran any case on backend '{}' ({} registered kernel(s) matched the filters): nothing was tested",
            a.backend,
            report.kernels.len()
        ),
        Verdict::Failed(n) => bail!("kernel admission failed ({n} kernel(s) REJECTED)"),
        Verdict::Pass { skipped: 0 } => {}
        Verdict::Pass { skipped } => println!(
            "incomplete: {skipped} kernel(s) had no cases in this sweep{} (untested, not failed; plan-time admission tests them)",
            if a.quick { " (--quick)" } else { "" }
        ),
    }
    Ok(())
}

/// Outcome of an admission run for the exit status: any kernel that ran cases and failed one (REJECTED) fails it, a run in
/// which no kernel ran a case fails it, kernels with no case (SKIPPED) alone do not.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Pass { skipped: usize },
    Failed(usize),
    NothingRan,
}

fn verdict(kernels: &[intarsia_core::admission::KernelSummary]) -> Verdict {
    let skipped = kernels.iter().filter(|k| k.cases == 0).count();
    if skipped == kernels.len() {
        return Verdict::NothingRan;
    }
    let rejected = kernels.iter().filter(|k| k.cases > 0 && !k.admitted).count();
    if rejected > 0 {
        Verdict::Failed(rejected)
    } else {
        Verdict::Pass { skipped }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use intarsia_core::admission::KernelSummary;

    fn k(cases: usize, failed: usize) -> KernelSummary {
        KernelSummary {
            op: Op::Matmul,
            kernel: format!("k{cases}_{failed}"),
            backend: "cuda".into(),
            cases,
            failed,
            worst_max_abs: 0.0,
            worst_rel_l2: 0.0,
            admitted: cases > 0 && failed == 0, // as run_admission sets it
        }
    }

    #[test]
    fn skipped_kernels_alone_do_not_fail_the_run() {
        // the CUDA --quick sweep of main: 39 ADMITTED, 7 SKIPPED, 0 FAIL -> exit 0 with the "incomplete" line
        let mut ks: Vec<_> = (0..39).map(|_| k(5, 0)).collect();
        ks.extend((0..7).map(|_| k(0, 0)));
        assert_eq!(verdict(&ks), Verdict::Pass { skipped: 7 });
        assert_eq!(verdict(&[k(3, 0)]), Verdict::Pass { skipped: 0 });
    }

    #[test]
    fn a_rejected_kernel_or_an_empty_run_still_fails() {
        assert_eq!(verdict(&[k(5, 0), k(5, 1), k(0, 0)]), Verdict::Failed(1));
        assert_eq!(verdict(&[k(5, 2), k(4, 4)]), Verdict::Failed(2));
        assert_eq!(verdict(&[k(0, 0), k(0, 0)]), Verdict::NothingRan);
        assert_eq!(verdict(&[]), Verdict::NothingRan);
    }
}

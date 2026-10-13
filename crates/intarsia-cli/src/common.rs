//! Flags and helpers shared by every command that loads a model.

use anyhow::{bail, Context, Result};
use clap::Args;
use intarsia_core::model::LoadOptions;
use intarsia_core::{Backend, DType};
use intarsia_models::{ChatTemplateSource, EngineOptions, Loaded};
use std::path::PathBuf;
use std::sync::Arc;

/// Model + backend selection. Embed in a command with `#[command(flatten)]`.
#[derive(Args, Clone, Debug)]
pub struct ModelArgs {
    /// Model directory (config.json, tokenizer.json, *.safetensors).
    #[arg(long, short = 'm')]
    pub model: PathBuf,
    /// Backend: `cpu` (reference oracle) or `cuda` (build with `--features cuda`).
    #[arg(long, default_value = "cpu")]
    pub backend: String,
    /// Device ordinal for GPU backends.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Maximum context length in tokens (clamped by measured free memory).
    #[arg(long, default_value_t = 4096)]
    pub max_seq: usize,
    /// Fraction of measured free memory the KV cache may use.
    #[arg(long, default_value_t = 0.5)]
    pub kv_fraction: f32,
    /// Total memory this process may use, in GiB (weights + workspace + KV). The KV cache is
    /// sized to what is left, so a fleetq `--mem` lease decides the context, not the box's free memory.
    #[arg(long)]
    pub mem_budget: Option<f64>,
    /// FP8 projections: `static` (W8A8, checkpoint input_scale) or `none` (W8A16); default: the recipe's.
    #[arg(long)]
    pub fp8_act: Option<String>,
    /// Tokens per forward pass (prefill chunk).
    #[arg(long, default_value_t = 256)]
    pub prefill_chunk: usize,
    /// KV cache dtype override (bf16, f16, f32, fp8_e4m3); default from the recipe.
    #[arg(long)]
    pub kv_dtype: Option<String>,
    /// Time every Nth step per op (default: every step on cpu, every 16th on GPUs; 0 = never).
    #[arg(long)]
    pub profile_every: Option<u32>,
    /// Run the single-token decode step eagerly instead of replaying a captured CUDA graph (A/B and debugging).
    #[arg(long)]
    pub no_graphs: bool,
    /// Recipe file (default: matched by architecture from recipes/ and built-ins).
    #[arg(long)]
    pub recipe: Option<PathBuf>,
    /// Extra directory with recipe *.toml files (searched before built-ins).
    #[arg(long)]
    pub recipe_dir: Vec<PathBuf>,
    /// Jinja chat template file (overrides the model's templates).
    #[arg(long)]
    pub chat_template_file: Option<PathBuf>,
    /// Use the chat template in tokenizer_config.json even if chat_template.jinja exists.
    #[arg(long)]
    pub chat_template_from_config: bool,
    /// Model id reported to clients (default: derived from the directory).
    #[arg(long)]
    pub model_id: Option<String>,
    /// Run every op with the cpu-ref reference kernels (`ref_*`, f64 oracle) instead of the fast ones.
    /// Fails at load on backends that register no reference kernels (cuda).
    #[arg(long)]
    pub reference: bool,
    /// Run kernel admission on the exact shapes of this model's plan at load (results are cached per
    /// host under ~/.cache/intarsia/admission/); `serve` always does.
    #[arg(long)]
    pub admit: bool,
    /// Do not read or write the admission cache.
    #[arg(long)]
    pub no_admission_cache: bool,
    /// Only admitted kernels are selectable (set by `serve`).
    #[arg(skip)]
    pub require_admitted: bool,
}

pub fn open_backend(name: &str, device: usize) -> Result<Arc<dyn Backend>> {
    match name {
        "cpu" => Ok(intarsia_backend_cpu::CpuBackend::arc()),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                intarsia_backend_cuda::create(device).map_err(|e| anyhow::anyhow!("{e}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device;
                bail!("this binary was built without the `cuda` feature (cargo build -p intarsia-cli --features cuda)")
            }
        }
        other => bail!("unknown backend '{other}' (cpu, cuda)"),
    }
}

impl ModelArgs {
    pub fn open_backend(&self) -> Result<Arc<dyn Backend>> {
        open_backend(&self.backend, self.device)
    }

    pub fn engine_options(&self) -> Result<EngineOptions> {
        let kv_dtype = self.kv_dtype.as_deref().map(DType::parse).transpose().map_err(|e| anyhow::anyhow!("{e}"))?;
        let chat_template = if let Some(p) = &self.chat_template_file {
            ChatTemplateSource::File(p.clone())
        } else if self.chat_template_from_config {
            ChatTemplateSource::TokenizerConfig
        } else {
            ChatTemplateSource::Auto
        };
        Ok(EngineOptions {
            load: LoadOptions {
                max_seq: self.max_seq,
                kv_fraction: self.kv_fraction,
                prefill_chunk: self.prefill_chunk,
                act_dtype: None,
                kv_dtype,
                profile_every: self.profile_every.unwrap_or(if self.backend == "cpu" { 1 } else { 16 }),
                mem_budget: self.mem_budget.map(|g| (g * (1u64 << 30) as f64) as u64),
                fp8_act: self.fp8_act.as_deref().map(intarsia_core::quant::Fp8Act::parse).transpose().map_err(|e| anyhow::anyhow!("{e}"))?,
                graphs: !self.no_graphs,
            },
            recipe: self.recipe.clone(),
            recipe_dirs: self.recipe_dir.clone(),
            chat_template,
            model_id: self.model_id.clone(),
            plan: intarsia_core::plan::PlanOptions {
                admit: self.admit || self.require_admitted,
                require_admitted: self.require_admitted,
                force_reference: self.reference,
                cache_dir: None,
                no_cache: self.no_admission_cache,
            },
        })
    }

    /// Production serving: admit the plan's kernels at load and select nothing else.
    pub fn for_serving(mut self) -> Self {
        self.admit = true;
        self.require_admitted = true;
        self
    }

    /// Open the backend and load the model into a ready engine.
    pub fn load(&self) -> Result<Loaded> {
        let backend = self.open_backend()?;
        let opts = self.engine_options()?;
        intarsia_models::load_engine(&self.model, backend, &opts)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {}", self.model.display()))
    }
}

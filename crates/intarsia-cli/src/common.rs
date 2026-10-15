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
    /// Backend: `cpu` (reference oracle), `cuda` (build with `--features cuda`), `hip` (`--features hip`) or `metal`
    /// (macOS, `--features metal`).
    #[arg(long, default_value = "cpu")]
    pub backend: String,
    /// Device ordinal for GPU backends.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Tensor-parallel degree: shard the model over this many devices (attention heads, KV heads and the FFN are split,
    /// every layer ends in an all-reduce). Devices are `--devices`, default `0..tp`.
    #[arg(long, default_value_t = 1)]
    pub tp: usize,
    /// Device ordinals for `--tp` > 1 (comma separated, one per rank).
    #[arg(long, value_delimiter = ',')]
    pub devices: Vec<usize>,
    /// Collective for `--tp` > 1: `auto` (the backend's device-to-device all-reduce when it has one) or `host`
    /// (staged through host memory: slow, works everywhere, a fallback and a cross-check).
    #[arg(long, default_value = "auto")]
    pub comm: String,
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
    /// Tokens per forward pass (prefill chunk); default: the recipe's value for the backend, then its base value, then 256.
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    /// KV cache dtype override: `bf16` (the recipe default), `fp8` / `fp8_e4m3` and `int8` (block-scaled, one bf16 scale per 32
    /// elements; about half the KV bytes each; Metal only, opt-in, accuracy and speed in
    /// docs/results/2026-10-15-metal-kv-quant.md; fp8 is also taken by the cpu oracle), `f16`, `f32` (cpu oracle).
    /// Backends that have no kernels for the dtype refuse it at start.
    #[arg(long)]
    pub kv_dtype: Option<String>,
    /// Time every Nth step per op (default: every step on cpu, every 16th on GPUs, never for `serve` on metal; 0 = never).
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
    /// Speculative decoding: `off` (default) or `ngram` (prompt lookup over prompt + generated ids, verified in one
    /// multi-row pass whose rows are bitwise the sequential decode steps). Metal only; greedy requests only (sampled
    /// requests fall back to plain decode).
    #[arg(long, default_value = "off")]
    pub spec: String,
    /// Most draft tokens per verify pass with `--spec ngram` (1..=6).
    #[arg(long, default_value_t = intarsia_core::spec::SPEC_K_DEFAULT)]
    pub spec_k: usize,
    /// Concurrent sequences the engine decodes together (continuous batching). Default: the recipe's `max_batch`.
    /// Only 1 is implemented so far; larger values are rejected until the batching scheduler lands
    /// (docs/research/continuous-batching-design.md, M1).
    #[arg(long)]
    pub max_batch: Option<usize>,
    /// Where the resolved prefill chunk came from; set by [`ModelArgs::pin_prefill_chunk`] once the model is loaded.
    #[arg(skip)]
    pub prefill_chunk_source: Option<String>,
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
        "hip" => {
            #[cfg(feature = "hip")]
            {
                intarsia_backend_hip::create(device).map_err(|e| anyhow::anyhow!("{e}"))
            }
            #[cfg(not(feature = "hip"))]
            {
                let _ = device;
                bail!("this binary was built without the `hip` feature (cargo build -p intarsia-cli --features hip)")
            }
        }
        "metal" => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                if device != 0 {
                    bail!("the metal backend drives the system default GPU only (--device {device} is not supported)");
                }
                intarsia_backend_metal::MetalBackend::arc().map_err(|e| anyhow::anyhow!("{e}"))
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                let _ = device;
                bail!("this binary was built without the Metal backend (macOS only: cargo build -p intarsia-cli --features metal)")
            }
        }
        other => bail!("unknown backend '{other}' (cpu, cuda, hip, metal)"),
    }
}

/// Backends and collectives for a tensor-parallel group (`ordinals[r]` is rank `r`).
pub fn open_group(name: &str, ordinals: &[usize], comm: &str) -> Result<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn intarsia_core::Comm>>)> {
    match comm {
        "auto" | "host" => {}
        other => bail!("unknown --comm '{other}' (auto, host)"),
    }
    if name == "metal" && ordinals.len() > 1 {
        bail!("the metal backend has one GPU (the system default); --tp {} needs {} devices", ordinals.len(), ordinals.len());
    }
    if name == "hip" && comm == "auto" {
        #[cfg(feature = "hip")]
        {
            return intarsia_backend_hip::create_group(ordinals).map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let backends = ordinals.iter().map(|&o| open_backend(name, o)).collect::<Result<Vec<_>>>()?;
    let comms = intarsia_core::HostCommGroup::new(backends.clone());
    Ok((backends, comms))
}

impl ModelArgs {
    pub fn open_backend(&self) -> Result<Arc<dyn Backend>> {
        // the metal backend lists its int8-KV kernels only when an int8 cache is requested (see its `kernels()`)
        if self.backend == "metal" && self.kv_dtype.as_deref().map(DType::parse).is_some_and(|d| matches!(d, Ok(DType::I8))) {
            std::env::set_var("INTARSIA_METAL_KV_INT8", "1");
        }
        open_backend(&self.backend, self.device)
    }

    /// The device ordinals of the tensor-parallel group.
    pub fn tp_devices(&self) -> Result<Vec<usize>> {
        if self.tp == 0 {
            bail!("--tp must be at least 1");
        }
        if self.tp == 1 {
            return Ok(vec![self.device]);
        }
        let devs = if self.devices.is_empty() { (0..self.tp).collect() } else { self.devices.clone() };
        if devs.len() != self.tp {
            bail!("--tp {} needs {} device ordinals, --devices lists {}", self.tp, self.tp, devs.len());
        }
        let mut sorted = devs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != devs.len() && self.backend != "cpu" {
            bail!("--devices {devs:?} repeats a device");
        }
        Ok(devs)
    }

    pub fn engine_options(&self) -> Result<EngineOptions> {
        let kv_dtype = self.kv_dtype.as_deref().map(DType::parse).transpose().map_err(|e| anyhow::anyhow!("{e}"))?;
        // KV cache dtypes: bf16 everywhere (the default and the only one the cuda / hip kernels take); fp8 (e4m3) and int8
        // (block-scaled) on metal, fp8 also on the cpu oracle; f16 / f32 on the cpu oracle.
        let kv_ok = match kv_dtype {
            None | Some(DType::Bf16) => true,
            Some(DType::Fp8E4m3) => matches!(self.backend.as_str(), "cpu" | "metal"),
            Some(DType::I8) => self.backend == "metal",
            Some(DType::F16 | DType::F32) => self.backend == "cpu",
            Some(_) => false,
        };
        if !kv_ok {
            bail!(
                "--kv-dtype {}: not supported on the {} backend (bf16 everywhere; fp8 / int8 on metal; fp8, f16, f32 on cpu)",
                self.kv_dtype.as_deref().unwrap_or(""),
                self.backend
            );
        }
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
                prefill_chunk: self.prefill_chunk.unwrap_or(intarsia_core::recipe::DEFAULT_PREFILL_CHUNK),
                act_dtype: None,
                kv_dtype,
                profile_every: self.profile_every.unwrap_or(if self.backend == "cpu" { 1 } else { 16 }),
                mem_budget: self.mem_budget.map(|g| (g * (1u64 << 30) as f64) as u64),
                fp8_act: self.fp8_act.as_deref().map(intarsia_core::quant::Fp8Act::parse).transpose().map_err(|e| anyhow::anyhow!("{e}"))?,
                graphs: !self.no_graphs,
            },
            prefill_chunk: self.prefill_chunk,
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
            spec: intarsia_core::spec::SpecConfig::parse(&self.spec, self.spec_k).map_err(|e| anyhow::anyhow!("{e}"))?,
            max_batch: self.max_batch,
        })
    }

    /// Fix the prefill chunk to what the load resolved (flag, recipe backend value, recipe base value, 256), so the
    /// commands that record it ([`ModelArgs::chunk`], [`ModelArgs::chunk_source`]) record the value that ran.
    pub fn pin_prefill_chunk(&mut self, c: &intarsia_core::recipe::PrefillChunk) {
        self.prefill_chunk = Some(c.value);
        self.prefill_chunk_source = Some(c.source.clone());
    }

    /// The prefill chunk in effect (the resolved one once pinned, else the flag, else the 256 default).
    pub fn chunk(&self) -> usize {
        self.prefill_chunk.unwrap_or(intarsia_core::recipe::DEFAULT_PREFILL_CHUNK)
    }

    /// Where [`ModelArgs::chunk`] came from.
    pub fn chunk_source(&self) -> String {
        match (&self.prefill_chunk_source, self.prefill_chunk) {
            (Some(s), _) => s.clone(),
            (None, Some(_)) => "--prefill-chunk".into(),
            (None, None) => "default".into(),
        }
    }

    /// Production serving: admit the plan's kernels at load and select nothing else.
    pub fn for_serving(mut self) -> Self {
        // int8 KV cache: its kernels have no cpu-oracle counterpart (the oracle has no block-scaled cache), and `plan::admit` builds its
        // cases from the model's own calls, so the oracle fails them and the kernels are Rejected. Serving an int8 cache therefore runs
        // without admission (the kernels are covered by crates/intarsia-backend-metal/tests/kv_i8.rs and the batch identity test;
        // docs/results/2026-10-15-metal-kv-quant.md); everything else keeps `admit` + `require_admitted`.
        let int8 = self.kv_dtype.as_deref().map(DType::parse).is_some_and(|d| matches!(d, Ok(DType::I8)));
        if int8 {
            eprintln!("warning: --kv-dtype int8: serving without kernel admission (no oracle for a block-scaled KV cache)");
        }
        self.admit = !int8;
        self.require_admitted = !int8;
        // Metal has no decode graphs, so a profiled step synchronises around every op: at the default N = 16 that
        // costs about 12% of served decode. CUDA/HIP keep their default (graph decode profiles far less often).
        if self.profile_every.is_none() && self.backend == "metal" {
            self.profile_every = Some(0);
        }
        self
    }

    /// Open the backend and load the model into a ready engine.
    pub fn load(&self) -> Result<Loaded> {
        let opts = self.engine_options()?;
        if self.tp > 1 {
            let (backends, comms) = open_group(&self.backend, &self.tp_devices()?, &self.comm)?;
            return intarsia_models::load_engine_tp(&self.model, backends, comms, &opts)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("loading {} with tp={}", self.model.display(), self.tp));
        }
        let backend = self.open_backend()?;
        intarsia_models::load_engine(&self.model, backend, &opts)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {}", self.model.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::ModelArgs;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        model: ModelArgs,
    }

    fn profile_every(args: &[&str], serving: bool) -> u32 {
        let mut a = Cli::parse_from([&["t", "-m", "/m"], args].concat()).model;
        if serving {
            a = a.for_serving();
        }
        a.engine_options().unwrap().load.profile_every
    }

    #[test]
    fn serve_on_metal_does_not_profile_by_default() {
        assert_eq!(profile_every(&["--backend", "metal"], true), 0);
        assert_eq!(profile_every(&["--backend", "metal", "--profile-every", "16"], true), 16);
        assert_eq!(profile_every(&["--backend", "metal"], false), 16);
        assert_eq!(profile_every(&["--backend", "cuda"], true), 16);
        assert_eq!(profile_every(&["--backend", "cpu"], true), 1);
    }
}

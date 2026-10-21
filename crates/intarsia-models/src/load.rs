//! One-call model loading: directory -> ready [`LocalEngine`].

use crate::chat_template::ChatTemplateSource;
use crate::recipe::{load_recipe_file, select_recipe};
use crate::safetensors::SafeTensors;
use crate::tokenizer::HfCodec;
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::plan::PlanOptions;
use intarsia_core::quant::scheme::QuantScheme;
use intarsia_core::recipe::{ModelDims, PrefillChunk, Recipe};
use intarsia_core::backend::HostMemory;
use intarsia_core::weights::{WeightInfo, WeightSource};
use intarsia_core::{Backend, Comm, Engine, EngineMeta, Error, LocalEngine, Result, Telemetry};

/// Requests the batching scheduler may hold waiting for a slot when the caller names no `EngineOptions::max_queue`
/// (library use). `intarsia serve` passes `--max-queue`, which sizes the queue to `max_queue + max_batch`.
const BATCH_QUEUE: usize = 1024;

/// The batched engine's queue bound for the server's `--max-queue` (`None`: [`BATCH_QUEUE`]).
fn batch_queue(max_queue: Option<usize>, max_batch: usize) -> usize {
    max_queue.map_or(BATCH_QUEUE, |q| q.saturating_add(max_batch))
}

#[cfg(test)]
mod queue_tests {
    use super::*;

    #[test]
    fn engine_queue_holds_everything_the_server_admits() {
        assert_eq!(batch_queue(None, 4), BATCH_QUEUE);
        assert_eq!(batch_queue(Some(64), 4), 68);
        // a --max-queue above the old fixed 1024 is no longer cut short
        assert_eq!(batch_queue(Some(2000), 8), 2008);
        assert_eq!(batch_queue(Some(usize::MAX), 8), usize::MAX);
    }
}
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct EngineOptions {
    pub load: LoadOptions,
    /// Explicit prefill chunk (`--prefill-chunk`). `None`: the recipe's value for the backend, then its base value,
    /// then 256; `load.prefill_chunk` is overwritten with the result.
    pub prefill_chunk: Option<usize>,
    /// Explicit recipe file (otherwise matched by architecture).
    pub recipe: Option<PathBuf>,
    /// Extra directories searched for recipes before the built-ins.
    pub recipe_dirs: Vec<PathBuf>,
    pub chat_template: ChatTemplateSource,
    /// Name reported as the model id (default: directory name).
    pub model_id: Option<String>,
    /// Plan resolution: `--reference`, admission on the plan's shapes, `require_admitted`.
    pub plan: PlanOptions,
    /// Speculative decoding (`--spec`, `--spec-k`); default off. Metal only (milestone G1).
    pub spec: intarsia_core::spec::SpecConfig,
    /// Concurrent sequences (`--max-batch`). `None`: the recipe's `max_batch`, clamped to what the engine implements.
    pub max_batch: Option<usize>,
    /// The server's `--max-queue`, for sizing the batched engine's request queue to `max_queue + max_batch` (the most
    /// requests the server lets wait inside the engine). `None`: [`BATCH_QUEUE`].
    pub max_queue: Option<usize>,
    /// The server's `--spec-off-at` (batched engine only; `None`: the scheduler default).
    pub spec_off_at: Option<usize>,
    /// The server's `--busy-chunk` (batched engine only; `None`: the scheduler default, 256).
    pub busy_chunk: Option<usize>,
}

fn read_json(p: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(p).map_err(|e| Error::model(format!("reading {}: {e}", p.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::model(format!("{}: {e}", p.display())))
}

/// Recipe + dimensions for a model directory.
pub fn read_model_spec(dir: &Path, opts: &EngineOptions) -> Result<(String, Recipe, ModelDims, Value)> {
    let cfg = read_json(&dir.join("config.json"))?;
    let (src, recipe) = match &opts.recipe {
        Some(p) => (p.display().to_string(), load_recipe_file(p)?),
        None => select_recipe(&cfg, &opts.recipe_dirs)?,
    };
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config)?;
    Ok((src, recipe, dims, cfg))
}

/// EOS ids from `config.json` and `generation_config.json`.
pub fn eos_ids(dir: &Path, cfg: &Value) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let mut take = |v: Option<&Value>| match v {
        Some(Value::Number(n)) => out.extend(n.as_u64().map(|x| x as u32)),
        Some(Value::Array(a)) => out.extend(a.iter().filter_map(|x| x.as_u64().map(|x| x as u32))),
        _ => {}
    };
    take(cfg.get("eos_token_id"));
    take(intarsia_core::recipe::text_config(cfg).get("eos_token_id"));
    if let Ok(g) = read_json(&dir.join("generation_config.json")) {
        take(g.get("eos_token_id"));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// `hf_quant_config.json` of a model directory, parsed. `Ok(None)` when the file does not exist; a file that exists
/// but cannot be parsed is an error (a checkpoint whose declared scheme we cannot read must not load unchecked).
/// ExLlamaV3 checkpoints declare `quant_method = "exl3"` in `config.json` or a sidecar `quantization_config.json`.
pub fn read_quant_scheme(dir: &Path) -> Result<Option<QuantScheme>> {
    let p = dir.join("hf_quant_config.json");
    if !p.exists() {
        // AMD Quark declares its scheme in config.json (`quantization_config.quant_method = "quark"`, top level or
        // under text_config); other quant_methods keep the old behaviour (tensor detection only)
        let c = dir.join("config.json");
        if !c.exists() {
            return Ok(None);
        }
        let cfg = read_json(&c)?;
        let qc = cfg.get("quantization_config").or_else(|| cfg.get("text_config").and_then(|t| t.get("quantization_config")));
        return match qc {
            Some(q) if q.get("quant_method").and_then(serde_json::Value::as_str) == Some("quark") => QuantScheme::parse_quark(q).map(Some),
            _ => read_exl3_scheme(dir),
        };
    }
    QuantScheme::parse(&read_json(&p)?).map(Some)
}

/// The EXL3 scheme of `config.json` or `quantization_config.json` (`quant_method = "exl3"`), `None` for any other.
fn read_exl3_scheme(dir: &Path) -> Result<Option<QuantScheme>> {
    for f in ["config.json", "quantization_config.json"] {
        let p = dir.join(f);
        if !p.exists() {
            continue;
        }
        if let Some(fields) = intarsia_core::quant::exl3::config_fields(&read_json(&p)?) {
            return QuantScheme::parse_exl3(&fields).map(Some);
        }
    }
    Ok(None)
}

/// The checkpoint's tensors plus the quantisation scheme it declares, so `Model::load` can check every linear
/// layer against `hf_quant_config.json` (declared algorithm implemented? agrees with the stored tensors?).
pub struct SchemedSource {
    inner: SafeTensors,
    scheme: Option<QuantScheme>,
}

impl SchemedSource {
    pub fn open(dir: &Path) -> Result<SchemedSource> {
        Ok(SchemedSource { inner: SafeTensors::open_dir(dir)?, scheme: read_quant_scheme(dir)? })
    }
    pub fn scheme(&self) -> Option<&QuantScheme> {
        self.scheme.as_ref()
    }
}

impl WeightSource for SchemedSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }
    fn info(&self, name: &str) -> Option<WeightInfo> {
        self.inner.info(name)
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.inner.bytes(name)
    }
    fn host_memory(&self, name: &str) -> Option<(Arc<dyn HostMemory>, usize, usize)> {
        self.inner.host_memory(name)
    }
    fn quant_scheme(&self) -> Option<&QuantScheme> {
        self.scheme.as_ref()
    }
}

pub struct Loaded {
    pub engine: Arc<dyn Engine>,
    /// Sequences the engine decodes together (1: the single-sequence engine).
    pub max_batch: usize,
    pub recipe_source: String,
    /// The prefill chunk the model was loaded with and where it came from.
    pub prefill_chunk: PrefillChunk,
    pub chat_template_source: Option<String>,
}

pub fn load_engine(dir: &Path, backend: Arc<dyn Backend>, opts: &EngineOptions) -> Result<Loaded> {
    load_engine_tp(dir, vec![backend], Vec::new(), opts)
}

/// Like [`load_engine`] for a tensor-parallel group: `backends[r]` is rank `r`, `comms[r]` its collective handle
/// (`comms` is empty for a single backend).
pub fn load_engine_tp(
    dir: &Path,
    backends: Vec<Arc<dyn Backend>>,
    comms: Vec<Arc<dyn Comm>>,
    opts: &EngineOptions,
) -> Result<Loaded> {
    let (recipe_source, recipe, dims, cfg) = read_model_spec(dir, opts)?;
    // Batching is opt-in: without `--max-batch` the engine decodes one sequence at a time whatever the recipe declares.
    let max_batch = recipe.resolve_max_batch(
        opts.max_batch,
        if opts.max_batch.is_some() { intarsia_core::engine::SERVE_MAX_BATCH } else { 1 },
    )?;
    if max_batch.value > 1 {
        // fail before reading any weights: continuous batching is Metal only (row-exact multi-row kernels)
        let b = backends.first().map_or("cpu", |b| b.name());
        if b != "metal" {
            return Err(Error::Unsupported(format!(
                "--max-batch {} needs the Metal backend (row-exact multi-row kernels; backend is {b}); use --max-batch 1",
                max_batch.value
            )));
        }
    }
    let weights = SchemedSource::open(dir)?;
    let telemetry = Arc::new(Telemetry::new());
    let prefill_chunk = recipe.resolve_prefill_chunk(opts.prefill_chunk, backends.first().map_or("cpu", |b| b.name()));
    eprintln!("prefill chunk {}", prefill_chunk.describe());
    let load_opts = LoadOptions { prefill_chunk: prefill_chunk.value, ..opts.load.clone() };
    let mut model = if backends.len() == 1 {
        Model::load(backends.into_iter().next().unwrap(), &recipe, &dims, &weights, &load_opts, telemetry.clone())?
    } else {
        Model::load_tp(backends, comms, &recipe, &dims, &weights, &load_opts, telemetry.clone())?
    };
    // deviations from the declared scheme that are not errors: logged once, at load
    if let Some(note) = weights.scheme().and_then(|s| s.kv_note(model.kv)) {
        eprintln!("quantisation scheme note: {note}");
    }
    if let (Some(s), Some(ours)) = (weights.scheme(), opts.load.mxfp4_act) {
        if let Some(note) = s.mxfp4_act_note(ours) {
            eprintln!("quantisation scheme note: {note}");
        }
    }
    // the admission oracle is always the cpu backend (its `ref_*` kernels)
    let oracle = intarsia_backend_cpu::CpuBackend::arc();
    if opts.spec.enabled() {
        // G1: the row-exact verify kernels (bitwise the 1-row decode launches per row) exist on Metal only
        if model.backend.name() != "metal" || model.tp_world() > 1 {
            return Err(Error::Unsupported(format!(
                "--spec {:?} needs the Metal backend without tensor parallelism (backend {}, tp {})",
                opts.spec.mode,
                model.backend.name(),
                model.tp_world()
            )));
        }
        model.enable_verify(opts.spec.verify_rows())?;
    }
    if max_batch.value > 1 {
        // Batching needs the row-exact multi-row kernels (Metal). `enable_batch` exempts the cpu backend (its kernels
        // loop over rows), but the batched engine serves only backends that register them.
        if !model.batch_capable() {
            return Err(Error::Unsupported(format!(
                "--max-batch {} needs row-exact multi-row kernels, which the {} backend does not provide (continuous \
                 batching is Metal only); use --max-batch 1",
                max_batch.value,
                model.backend.name()
            )));
        }
        // refuses a tensor-parallel or non-dense model
        model.enable_batch(max_batch.value)?;
    }
    let summary = model.configure_plan(&opts.plan, Some(&*oracle))?;
    if opts.spec.enabled() {
        let rec = model.dry_run(intarsia_core::plan::Phase::Verify, opts.spec.verify_rows(), 0)?;
        let mut picks: Vec<String> = Vec::new();
        for c in &rec.calls {
            let p = format!("{}={}", c.call.op, c.kernel);
            if !picks.contains(&p) {
                picks.push(p);
            }
        }
        eprintln!(
            "speculative decoding: {:?}, up to {} drafts per verify pass; verify kernels: {}",
            opts.spec.mode,
            opts.spec.k,
            picks.join(" ")
        );
    }
    if opts.plan.admit {
        eprintln!(
            "admission on plan shapes: {} kernel(s) checked, {} case(s) run, {} cached{}",
            summary.kernels_checked,
            summary.cases_run,
            summary.cases_cached,
            if summary.rejected.is_empty() {
                String::new()
            } else {
                format!(", REJECTED: {}", summary.rejected.iter().map(|(k, w)| format!("{k} ({w})")).collect::<Vec<_>>().join("; "))
            }
        );
    }
    let codec = HfCodec::load(dir, &opts.chat_template)?;
    let chat_template_source = codec.template_source().map(String::from);
    let id = opts.model_id.clone().unwrap_or_else(|| {
        // snapshot dirs end in a hash; prefer the repo name from ".../models--Org--Name/snapshots/<hash>"
        let comps: Vec<_> = dir.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
        comps.iter().rev().find_map(|c| c.strip_prefix("models--").map(|r| r.replacen("--", "/", 1))).unwrap_or_else(
            || dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "model".into()),
        )
    });
    let meta = EngineMeta {
        spec: opts.spec,
        id,
        eos_ids: eos_ids(dir, &cfg),
        reasoning_parser: recipe.model.reasoning_parser.clone(),
        tool_parser: recipe.model.tool_parser.clone(),
    };
    let engine: Arc<dyn Engine> = if max_batch.value > 1 {
        // the server admits at most max_queue + max_batch requests, so a queue of that size never fills first
        let queue = batch_queue(opts.max_queue, max_batch.value);
        let mut bo = intarsia_core::BatchedOptions::new(max_batch.value, queue);
        if let Some(n) = opts.spec_off_at {
            bo.spec_off_at = n;
        }
        if let Some(n) = opts.busy_chunk {
            bo.busy_chunk = n;
        }
        eprintln!("continuous batching: up to {} sequences ({})", max_batch.value, max_batch.source);
        Arc::new(intarsia_core::BatchedEngine::new(model, Arc::new(codec), meta, telemetry, bo)?)
    } else {
        Arc::new(LocalEngine::new(model, Arc::new(codec), meta, telemetry)?)
    };
    Ok(Loaded { engine, recipe_source, prefill_chunk, chat_template_source, max_batch: max_batch.value })
}

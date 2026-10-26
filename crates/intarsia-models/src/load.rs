//! One-call model loading: directory -> ready [`LocalEngine`].

use crate::chat_template::ChatTemplateSource;
use crate::recipe::{load_recipe_file, select_recipe};
use crate::safetensors::SafeTensors;
use crate::tokenizer::HfCodec;
use intarsia_core::model::{LoadOptions, Model};
use intarsia_core::plan::PlanOptions;
use intarsia_core::quant::compressed_tensors::CtNvfp4;
use intarsia_core::quant::scheme::{QuantScheme, CT_NVFP4A16};
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
    /// `--spec-draft DIR`: the DFlash draft checkpoint (`--spec dflash`, `model/dflash.rs`).
    pub spec_draft: Option<PathBuf>,
    /// `--dcoco`: with `--spec dflash`, the DCOCO corrector picks the drafts (needs the DCOCO checkpoint).
    pub dcoco: bool,
    /// Concurrent sequences (`--max-batch`). `None`: the recipe's `max_batch`, clamped to what the engine implements.
    pub max_batch: Option<usize>,
    /// The server's `--max-queue`, for sizing the batched engine's request queue to `max_queue + max_batch` (the most
    /// requests the server lets wait inside the engine). `None`: [`BATCH_QUEUE`].
    pub max_queue: Option<usize>,
    /// The server's `--spec-off-at` (batched engine only; `None`: the scheduler default).
    pub spec_off_at: Option<usize>,
    /// No `--spec` was given (the CLI): the loader may switch `--spec mtp` on where the recipe makes it the default for this backend
    /// under tensor parallelism (`[tuning.<backend>] mtp_default_tp_k`, in-process ranks, one sequence; `INTARSIA_SPEC_DEFAULT=0`
    /// turns the default off). `false` (the library default): `spec` is taken as it is.
    pub spec_auto: bool,
    /// The server's `--busy-chunk` (batched engine only; `None`: the scheduler default, 256).
    pub busy_chunk: Option<usize>,
    /// Content-addressed prefix store (`--prefix-store`, docs/results/2026-10-22-pc-prefix-store.md). `None`: the
    /// `INTARSIA_PREFIX_STORE` environment variable decides (unset: off). Single-sequence engine only.
    pub prefix_store: Option<intarsia_core::prefix_store::PrefixStoreConfig>,
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
            Some(q) if QuantScheme::is_compressed_tensors_nvfp4(q) => QuantScheme::parse_compressed_tensors(q).map(Some),
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
    inner: Tensors,
    scheme: Option<QuantScheme>,
}

/// The safetensors as stored, or, for a compressed-tensors NVFP4A16 checkpoint, read under the ModelOpt tensor names
/// (`intarsia_core::quant::compressed_tensors`).
enum Tensors {
    Plain(SafeTensors),
    CompressedTensors(CtNvfp4<SafeTensors>),
}

impl SchemedSource {
    pub fn open(dir: &Path) -> Result<SchemedSource> {
        let scheme = read_quant_scheme(dir)?;
        let st = SafeTensors::open_dir(dir)?;
        let inner = if scheme.as_ref().and_then(|s| s.quant_algo.as_deref()) == Some(CT_NVFP4A16) {
            Tensors::CompressedTensors(CtNvfp4::new(st)?)
        } else {
            Tensors::Plain(st)
        };
        Ok(SchemedSource { inner, scheme })
    }
    pub fn scheme(&self) -> Option<&QuantScheme> {
        self.scheme.as_ref()
    }
    fn src(&self) -> &dyn WeightSource {
        match &self.inner {
            Tensors::Plain(s) => s,
            Tensors::CompressedTensors(s) => s,
        }
    }
}

impl WeightSource for SchemedSource {
    fn names(&self) -> Vec<String> {
        self.src().names()
    }
    fn info(&self, name: &str) -> Option<WeightInfo> {
        self.src().info(name)
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.src().bytes(name)
    }
    fn host_memory(&self, name: &str) -> Option<(Arc<dyn HostMemory>, usize, usize)> {
        self.src().host_memory(name)
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

/// What both processes of a cross-process tensor-parallel group must agree on (`intarsia_core::tp_remote::handshake`):
/// the model, the recipe, the load and plan options and the group size.
fn tp_fingerprint(dir: &Path, recipe_source: &str, load_opts: &LoadOptions, opts: &EngineOptions, world: usize) -> String {
    format!("{} | recipe {recipe_source} | {load_opts:?} | {:?} | world {world}", dir.display(), opts.plan)
}

/// Like [`load_engine`] for rank 0 of a 2-way tensor-parallel group whose rank 1 runs in another process
/// ([`serve_remote_rank`] on the other end of `stream`, docs/design/35b-tp2.md). `comm` must already join both processes.
pub fn load_engine_remote(
    dir: &Path,
    backend: Arc<dyn Backend>,
    comm: Arc<dyn Comm>,
    stream: std::net::TcpStream,
    opts: &EngineOptions,
) -> Result<Loaded> {
    load_engine_group(dir, Group::Remote { backend, comm, stream }, opts)
}

/// Rank `rank` (> 0) of a tensor-parallel group in its own process: load this rank's shard, agree the KV capacity with
/// rank 0 over `stream`, then run rank 0's commands (`intarsia_core::tp_remote::serve`) until it disconnects. Returns the
/// number of forward passes served.
pub fn serve_remote_rank(
    dir: &Path,
    backend: Arc<dyn Backend>,
    comm: Arc<dyn Comm>,
    mut stream: std::net::TcpStream,
    opts: &EngineOptions,
) -> Result<u64> {
    let (rank, world) = (comm.rank(), comm.world());
    if rank == 0 || world < 2 {
        return Err(Error::invalid(format!("a served rank is rank 1.. of a group (got rank {rank} of {world})")));
    }
    let (recipe_source, recipe, dims, _cfg) = read_model_spec(dir, opts)?;
    let weights = SchemedSource::open(dir)?;
    let prefill_chunk = recipe.resolve_prefill_chunk(opts.prefill_chunk, backend.name());
    let load_opts = LoadOptions { prefill_chunk: prefill_chunk.value, ..opts.load.clone() };
    let tp = intarsia_core::model::TpRank { rank, world, comm: Some(comm) };
    let mut model = Model::load_shard(backend, &recipe, &dims, &weights, &load_opts, Arc::new(Telemetry::new()), tp)?;
    let fp = tp_fingerprint(dir, &recipe_source, &load_opts, opts, world);
    let cap = intarsia_core::tp_remote::handshake(&mut stream, &fp, model.plan.kv_cap_tokens)?;
    model.clamp_kv_cap(cap)?;
    let oracle = intarsia_backend_cpu::CpuBackend::arc();
    let summary = model.configure_plan(&opts.plan, Some(&*oracle))?;
    if opts.plan.admit {
        eprintln!(
            "admission on plan shapes (rank {rank}): {} kernel(s) checked, {} case(s) run, {} cached, rejected: {:?}",
            summary.kernels_checked, summary.cases_run, summary.cases_cached, summary.rejected
        );
    }
    eprintln!("tensor-parallel rank {rank} of {world}: serving rank 0 ({} tokens of context)", model.plan.kv_cap_tokens);
    let n = intarsia_core::tp_remote::serve(&model, stream);
    let g = model.graph_stats();
    eprintln!(
        "tensor-parallel rank {rank}: decode graph enabled={} captures={} replays={}",
        g.enabled, g.captures, g.replays
    );
    n
}

enum Group {
    Local(Vec<Arc<dyn Backend>>, Vec<Arc<dyn Comm>>),
    Remote { backend: Arc<dyn Backend>, comm: Arc<dyn Comm>, stream: std::net::TcpStream },
}

/// Like [`load_engine`] for a tensor-parallel group: `backends[r]` is rank `r`, `comms[r]` its collective handle
/// (`comms` is empty for a single backend).
pub fn load_engine_tp(
    dir: &Path,
    backends: Vec<Arc<dyn Backend>>,
    comms: Vec<Arc<dyn Comm>>,
    opts: &EngineOptions,
) -> Result<Loaded> {
    load_engine_group(dir, Group::Local(backends, comms), opts)
}

/// The DFlash draft checkpoint `dir` (`--spec-draft`): its weights and `config.json` (`dflash_config`).
pub fn open_dflash(dir: Option<&Path>, dcoco: bool) -> Result<intarsia_core::model::DflashLoad> {
    let dir = dir.ok_or_else(|| Error::invalid("--spec dflash needs --spec-draft DIR (the draft checkpoint)"))?;
    let cfg_path = dir.join("config.json");
    let cfg: Value = serde_json::from_slice(
        &std::fs::read(&cfg_path).map_err(|e| Error::model(format!("{}: {e}", cfg_path.display())))?,
    )
    .map_err(|e| Error::model(format!("{}: {e}", cfg_path.display())))?;
    let top = |k: &str| -> Result<usize> {
        cfg.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| Error::model(format!("{}: '{k}' missing", cfg_path.display())))
    };
    let d = cfg.get("dflash_config").ok_or_else(|| Error::model(format!("{}: not a DFlash draft (no dflash_config)", cfg_path.display())))?;
    let dn = |k: &str| d.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
    let layer_ids: Vec<usize> = d
        .get("target_layer_ids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
        .ok_or_else(|| Error::model(format!("{}: dflash_config.target_layer_ids missing", cfg_path.display())))?;
    let mask_id = d
        .get("mask_token_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::model(format!("{}: dflash_config.mask_token_id missing", cfg_path.display())))? as u32;
    let config = intarsia_core::model::DflashConfig {
        layer_ids,
        mask_id,
        block: dn("block_size"),
        hidden: top("hidden_size")?,
        n_layers: top("num_hidden_layers")?,
        intermediate: top("intermediate_size")?,
        n_heads: top("num_attention_heads")?,
        n_kv_heads: top("num_key_value_heads")?,
        head_dim: top("head_dim")?,
        vocab: top("vocab_size")?,
        topk: dn("dcoco_candidate_topk"),
        emb_dim: dn("dcoco_emb_dim"),
        state_dim: dn("dcoco_state_dim"),
        mlp_dim: dn("dcoco_mlp_dim"),
        v2: cfg.get("architectures").and_then(Value::as_array).is_some_and(|a| a.iter().any(|x| x.as_str() == Some("DFlash2DraftModel"))),
        rms_eps: cfg.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
        rope_theta: cfg
            .get("rope_theta")
            .and_then(Value::as_f64)
            .or_else(|| cfg.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(Value::as_f64))
            .unwrap_or(10000.0) as f32,
        sliding_window: if cfg.get("use_sliding_window").and_then(Value::as_bool) == Some(false) {
            0
        } else {
            cfg.get("sliding_window").and_then(Value::as_u64).unwrap_or(0) as usize
        },
        conv_kernel: dn("conv_kernel_size"),
        conv_group: dn("conv_group_size"),
        selector_rank: dn("selector_rank"),
        selector_top_k: dn("selector_top_k"),
    };
    if config.v2 {
        // what this port implements (docs/results/2026-10-26-dfh-hip-dflash.md): non-causal sliding-window layers, default rope
        let types = cfg.get("layer_types").and_then(Value::as_array);
        let all_sliding = types.is_some_and(|a| a.iter().all(|x| x.as_str() == Some("sliding_attention")));
        let rope_default = cfg
            .get("rope_parameters")
            .or_else(|| cfg.get("rope_scaling"))
            .and_then(|r| r.get("rope_type"))
            .and_then(Value::as_str)
            .is_none_or(|t| t == "default");
        let causal = cfg.get("is_causal").and_then(Value::as_bool).unwrap_or(false);
        if !all_sliding || !rope_default || causal || config.sliding_window == 0 {
            return Err(Error::Unsupported(format!(
                "{}: DFlash2 drafts with sliding-window, non-causal layers and default rope only",
                cfg_path.display()
            )));
        }
    }
    let src: Arc<dyn WeightSource> = Arc::new(SchemedSource::open(dir)?);
    Ok(intarsia_core::model::DflashLoad { src, config, dcoco })
}

/// The recipe's tensor-parallel MTP default (lane dfh; the Qwen3.8-27B on HIP at tp > 1, user decision 2026-10-26 11:07): `Some(k)`
/// when no `--spec` was given (`spec_auto`), the group is `world > 1` in-process ranks on `backend`, one sequence (MTP does not
/// batch), the recipe sets `[tuning.<backend>] mtp_default_tp_k = k`, and `INTARSIA_SPEC_DEFAULT` is not `0`.
pub fn tp_mtp_default(opts: &EngineOptions, recipe: &intarsia_core::recipe::Recipe, backend: &str, world: usize, local: bool) -> Option<usize> {
    if !opts.spec_auto
        || opts.spec.mode != intarsia_core::spec::SpecMode::Off
        || opts.load.mtp
        || world < 2
        || !local
        || opts.max_batch.is_some_and(|n| n > 1)
        || std::env::var("INTARSIA_SPEC_DEFAULT").is_ok_and(|v| v == "0")
    {
        return None;
    }
    recipe.resolve_mtp_default_tp(backend)
}

fn load_engine_group(dir: &Path, group: Group, opts: &EngineOptions) -> Result<Loaded> {
    let backends: Vec<Arc<dyn Backend>> = match &group {
        Group::Local(b, _) => b.clone(),
        Group::Remote { backend, .. } => vec![backend.clone()],
    };
    // `--spec mtp` under tensor parallelism (lane dfh): in-process ranks only. Every rank loads its shard of the head and runs it in
    // lockstep with rank 0, which drafts (from its share of a vocab-parallel head, or a whole copy) and verifies across the group; a
    // rank in another process is not driven through the drafting passes, so that group is refused before any weight is read.
    if matches!(group, Group::Remote { .. }) && (opts.spec.mode == intarsia_core::spec::SpecMode::Mtp || opts.load.mtp) {
        return Err(Error::Unsupported(
            "--spec mtp with a tensor-parallel rank in another process: the MTP drafter runs on in-process ranks only (--tp N on one host)"
                .into(),
        ));
    }
    let (recipe_source, recipe, dims, cfg) = read_model_spec(dir, opts)?;
    // The recipe's tensor-parallel MTP default (lane dfh; the Qwen3.8-27B on HIP at tp > 1, user decision 2026-10-26 11:07): only
    // without --spec, on in-process ranks, at one sequence (MTP does not batch), unless INTARSIA_SPEC_DEFAULT=0.
    let defaulted;
    let local = matches!(group, Group::Local(..));
    let opts = match backends.first().and_then(|b0| tp_mtp_default(opts, &recipe, b0.name(), backends.len(), local).map(|k| (b0, k))) {
        Some((b0, k)) => {
            eprintln!(
                "--spec mtp --spec-k {k}: the default of recipe {} on {} at tp={} ([tuning.{}] mtp_default_tp_k; --spec off or \
                 INTARSIA_SPEC_DEFAULT=0 turns it off)",
                recipe.model.family,
                b0.name(),
                backends.len(),
                b0.name()
            );
            let mut o = opts.clone();
            o.spec = intarsia_core::spec::SpecConfig { mode: intarsia_core::spec::SpecMode::Mtp, k };
            o.load.mtp = true;
            defaulted = o;
            &defaulted
        }
        None => opts,
    };
    // Batching is opt-in: without `--max-batch` the engine decodes one sequence at a time whatever the recipe declares.
    // The most rows are the backend's (`Backend::batch_row_cap`: Metal 8, CUDA 16).
    let backend_name = backends.first().map_or("cpu", |b| b.name());
    let backend_cap = backends.first().map_or(intarsia_core::engine::SERVE_MAX_BATCH, |b| b.batch_row_cap());
    let max_batch = recipe.resolve_max_batch(opts.max_batch, if opts.max_batch.is_some() { backend_cap } else { 1 })?;
    // Hybrid models (GatedDeltaNet, MoE, gated attention) batch on CUDA and HIP whenever `--max-batch N` asks (its `_rows` kernels are
    // proven row-exact, docs/results/2026-10-23-rb2-cuda-batching.md); the cpu reference needs the explicit opt-in
    // `INTARSIA_RECURRENT_BATCH=1` (docs/results/2026-10-23-rb-recurrent-batching.md). Every other backend is refused.
    let hybrid_opt_in = std::env::var("INTARSIA_RECURRENT_BATCH").is_ok_and(|v| v == "1");
    let hybrid_batch = matches!(backend_name, "cuda" | "hip") || (hybrid_opt_in && backend_name == "cpu");
    if max_batch.value > 1 {
        // fail before reading any weights: continuous batching needs row-exact multi-row kernels (Metal, CUDA, HIP)
        if !matches!(backend_name, "metal" | "cuda" | "hip") && !hybrid_batch {
            return Err(Error::Unsupported(format!(
                "--max-batch {} needs the Metal, CUDA or HIP backend (row-exact multi-row kernels; backend is {backend_name}); use --max-batch 1",
                max_batch.value
            )));
        }
        // MTP drafting and the prefix store stay at one sequence (needs per-row hidden outputs and k x 63 MB of spec rows per
        // sequence): refuse before the weights are read, not after
        if opts.spec.mode == intarsia_core::spec::SpecMode::Mtp {
            return Err(Error::Unsupported("--spec mtp with the batching engine (use --max-batch 1)".into()));
        }
    }
    // `--spec dflash`: the draft checkpoint, read and checked before the target's weights
    let dflash = if opts.spec.mode == intarsia_core::spec::SpecMode::Dflash {
        if max_batch.value > 1 {
            return Err(Error::Unsupported("--spec dflash with the batching engine (use --max-batch 1)".into()));
        }
        let dl = open_dflash(opts.spec_draft.as_deref(), opts.dcoco)?;
        // DFlash (dense Qwen3 targets): cpu and CUDA. DFlash2 (lane dfh): cpu and HIP, which have its conv and sliding-window kernels.
        let ok = if dl.config.v2 { matches!(backend_name, "cpu" | "hip") } else { matches!(backend_name, "cpu" | "cuda") };
        if !ok {
            return Err(Error::Unsupported(format!(
                "--spec dflash with a {} draft on the {backend_name} backend ({} only so far)",
                if dl.config.v2 { "DFlash2" } else { "DFlash" },
                if dl.config.v2 { "cpu and hip" } else { "cpu and cuda" }
            )));
        }
        Some(dl)
    } else {
        if opts.spec_draft.is_some() || opts.dcoco {
            return Err(Error::invalid("--spec-draft and --dcoco need --spec dflash"));
        }
        None
    };
    let weights = SchemedSource::open(dir)?;
    let telemetry = Arc::new(Telemetry::new());
    let prefill_chunk = recipe.resolve_prefill_chunk(opts.prefill_chunk, backends.first().map_or("cpu", |b| b.name()));
    eprintln!("prefill chunk {}", prefill_chunk.describe());
    let mtp = opts.spec.mode == intarsia_core::spec::SpecMode::Mtp;
    let load_opts = LoadOptions {
        prefill_chunk: prefill_chunk.value,
        mtp: opts.load.mtp || mtp,
        mtp_k: if mtp { opts.spec.k } else { opts.load.mtp_k },
        dflash,
        ..opts.load.clone()
    };
    let mut model = match group {
        Group::Local(backends, comms) if backends.len() == 1 => {
            let _ = comms;
            Model::load(backends.into_iter().next().unwrap(), &recipe, &dims, &weights, &load_opts, telemetry.clone())?
        }
        Group::Local(backends, comms) => Model::load_tp(backends, comms, &recipe, &dims, &weights, &load_opts, telemetry.clone())?,
        Group::Remote { backend, comm, mut stream } => {
            let world = comm.world();
            if comm.rank() != 0 || world != 2 {
                return Err(Error::invalid(format!("a remote tensor-parallel group is driven by rank 0 of 2 (got rank {} of {world})", comm.rank())));
            }
            let tp = intarsia_core::model::TpRank { rank: 0, world, comm: Some(comm.clone()) };
            let mut m = Model::load_shard(backend, &recipe, &dims, &weights, &load_opts, telemetry.clone(), tp)?;
            let fp = tp_fingerprint(dir, &recipe_source, &load_opts, opts, world);
            let cap = intarsia_core::tp_remote::handshake(&mut stream, &fp, m.plan.kv_cap_tokens)?;
            m.clamp_kv_cap(cap)?;
            m.attach_remote(intarsia_core::tp_remote::RemotePeer::new(stream, comm)?)?;
            eprintln!("tensor parallel x{world} across processes: rank 1 is remote ({} tokens of context)", m.plan.kv_cap_tokens);
            m
        }
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
    // the MXFP4 activation mode the projections run (same rule as Model::load), so every report records it
    let declared = weights.scheme().and_then(|s| s.mxfp4_act());
    if declared.is_some() || opts.load.mxfp4_act.is_some() {
        let eff = intarsia_core::quant::mxfp4::effective_act(opts.load.mxfp4_act, declared);
        let src = if opts.load.mxfp4_act.is_some() { "--mxfp4-act" } else { "declared by the checkpoint" };
        eprintln!("mxfp4 activations: {} ({src})", eff.name());
    }
    // the admission oracle is always the cpu backend (its `ref_*` kernels)
    let oracle = intarsia_backend_cpu::CpuBackend::arc();
    if opts.spec.enabled() {
        // G1: the row-exact verify kernels (bitwise the 1-row decode launches per row) exist on Metal only. `--spec mtp`
        // also runs elsewhere (opt-in, measured): there a verify row's argmax can differ from the 1-row decode's, and the
        // emitted token is the verify row's (docs/results/2026-10-22-mtp-spec-decode.md).
        if opts.spec.mode == intarsia_core::spec::SpecMode::Dflash {
            if model.tp_world() != 1 && !model.has_dflash() {
                return Err(Error::Unsupported("--spec dflash under tensor parallelism without the draft on rank 0".into()));
            }
            eprintln!(
                "--spec dflash on {}: verify rows run the row-exact GEMV kernels and the attention one decode call per row (bitwise the single-row decode per row)",
                model.backend.name()
            );
        } else if mtp && model.tp_world() > 1 {
            // in-process tensor parallelism (lane dfh): every rank runs the verify pass on the row-exact kernels and the drafter's
            // passes in lockstep with rank 0 (`Model::mtp_draft`); a rank in another process was refused above
            if !model.has_mtp() {
                return Err(Error::Unsupported("--spec mtp under tensor parallelism without the MTP head".into()));
            }
            eprintln!(
                "--spec mtp on {} with tp={}: verify rows run the row-exact kernels on every rank (bitwise the single-row decode per row)",
                model.backend.name(),
                model.tp_world()
            );
        } else if mtp && model.backend.name() != "metal" && model.tp_world() == 1 && model.batch_capable() {
            // CUDA since lane rb2: the verify rows take the `_rows` kernels (G-B: bitwise the one-row kernel per row)
            eprintln!("--spec mtp on {}: verify rows run the row-exact kernels (bitwise the single-row decode per row)", model.backend.name());
        } else if mtp && model.backend.name() != "metal" && model.tp_world() == 1 {
            eprintln!(
                "--spec mtp on {}: no row-exact verify kernels here, so greedy output can differ from --spec off where a \
                 verify row's argmax differs from the single-row decode's",
                model.backend.name()
            );
        } else if model.backend.name() != "metal" || model.tp_world() > 1 {
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
        // Metal: only the hybrid 35B opts in (row-exact fp8 / nvfp4 / expert-grouped kernels, lane mb); a dense model keeps the
        // path it always had
        if hybrid_batch || (backend_name == "metal" && model.has_hybrid_blocks()) {
            model.allow_recurrent_batch()?;
        }
        // Batching needs the row-exact multi-row kernels (Metal, CUDA, HIP). `enable_batch` exempts the cpu backend (its kernels
        // loop over rows), but the batched engine serves only backends that register them.
        if !model.batch_capable() && !(hybrid_opt_in && model.backend.name() == "cpu") {
            return Err(Error::Unsupported(format!(
                "--max-batch {} needs row-exact multi-row kernels, which the {} backend does not provide (continuous \
                 batching is Metal, CUDA and HIP only); use --max-batch 1",
                max_batch.value,
                model.backend.name()
            )));
        }
        // refuses a non-dense model without the opt-in, a remote tensor-parallel rank, and --tp > 1 on any backend but hip (and cpu)
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
        prefix_store: match &opts.prefix_store {
            Some(c) => Some(c.clone()),
            None => intarsia_core::prefix_store::PrefixStoreConfig::from_env()?,
        },
    };
    if meta.prefix_store.is_some() && max_batch.value > 1 {
        return Err(Error::invalid("the prefix store needs the single-sequence engine (--max-batch 1)"));
    }
    let engine: Arc<dyn Engine> = if max_batch.value > 1 {
        // the server admits at most max_queue + max_batch requests, so a queue of that size never fills first
        let queue = batch_queue(opts.max_queue, max_batch.value);
        let mut bo = intarsia_core::BatchedOptions::new(max_batch.value, queue);
        if let Some(n) = opts.spec_off_at {
            bo.spec_off_at = n;
        }
        if let Some(n) = opts.busy_chunk {
            bo.busy_chunk = n;
        } else if model.recurrent_batch() && matches!(backend_name, "cuda" | "hip" | "metal") {
            // The busy chunk is the solo chunk (2048 for the 35B on CUDA and Metal, 4096 on HIP, lane hb): then a request that prefills beside decoding sequences
            // runs the same chunk schedule as its solo run, and its state, and every token after it, is bitwise the solo
            // engine's for every prompt length. A smaller busy chunk changes the GEMM tiling of the prefill: 1024 against 2048
            // measured mean 21-way KL 7.87e-3 on the 35B (2.5x the acc-short ref-noise floor 3.161e-3, 4.1% top-1 flips),
            // docs/results/2026-10-23-rb2-cuda-batching.md section 11. `--busy-chunk 1024` is the opt-in speed knob.
            bo.busy_chunk = model.prefill_chunk();
        }
        eprintln!("continuous batching: up to {} sequences ({}), busy prefill chunk {}", max_batch.value, max_batch.source, bo.busy_chunk);
        Arc::new(intarsia_core::BatchedEngine::new(model, Arc::new(codec), meta, telemetry, bo)?)
    } else {
        Arc::new(LocalEngine::new(model, Arc::new(codec), meta, telemetry)?)
    };
    Ok(Loaded { engine, recipe_source, prefill_chunk, chat_template_source, max_batch: max_batch.value })
}

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
    /// Tensor parallelism across processes / hosts (cuda, `--tp 2`): this process is rank 0 on `--device`, rank 1 is an
    /// `intarsia tp-worker` listening at HOST:PORT, started with the same model flags. The ranks join an NCCL
    /// communicator (libnccl from `INTARSIA_NCCL_LIB`); this connection carries only what to run
    /// (docs/design/35b-tp2.md).
    #[arg(long, value_name = "HOST:PORT")]
    pub tp_remote: Option<String>,
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
    /// MXFP4 projections: `dynamic` (W4A4, Quark's declared scheme), `none` (W4A16) or `fp8` (W4A8: per-token e4m3 activations, opt-in, HIP only); default: the recipe's `[defaults] mxfp4_act` (the Qwen3.8-27B: `none`), else what the checkpoint declares.
    #[arg(long)]
    pub mxfp4_act: Option<String>,
    /// Tokens per forward pass (prefill chunk); default: the recipe's value for the backend, then its base value, then 256.
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    /// KV cache dtype override (bf16, f16, f32, fp8_e4m3); default from the recipe.
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
    /// Speculative decoding: `off` (default), `ngram` (prompt lookup over prompt + generated ids, verified in one
    /// multi-row pass whose rows are bitwise the sequential decode steps; Metal only) or `mtp` (the checkpoint's
    /// multi-token-prediction head drafts, recipe `[mtp]`: Qwen3.5/3.6 MoE and dense Qwen3.8; single sequence; measured on
    /// CUDA, HIP and Metal, and on the CPU backend in the tests). Bitwise to `off` where the backend has row-exact verify
    /// kernels (HIP also runs the verify attention one decode call per row, and replays draft + verify as one graph unless
    /// `--no-graphs` or `INTARSIA_SPEC_GRAPH=0`); elsewhere self-consistent greedy over the verify rows. Greedy requests only
    /// (sampled requests fall back to plain decode). Unset: `off`, except where the recipe makes MTP the default for this
    /// backend (`[tuning.<backend>] mtp_default_tp_k` under tensor parallelism: the Qwen3.8-27B on HIP at `--tp 2`; `mtp_default_k`
    /// on one rank: the Qwen3.6-35B-A3B on CUDA); `--spec off`
    /// or `INTARSIA_SPEC_DEFAULT=0` opts out.
    ///
    /// `--spec auto --spec-draft DIR` (lane asp): loads both the MTP head (2 drafts per pass) and the DFlash2 draft (`--spec-k`, default 7)
    /// and picks the drafter per pass by measured tokens per ms (running acceptance times the pass cost, an idle drafter probed now and
    /// then); the greedy output is `--spec off`'s. `INTARSIA_SPEC_AUTO=mtp|dflash|alt:N` pins a drafter or switches every N passes.
    #[arg(long)]
    pub spec: Option<String>,
    /// Most draft tokens per verify pass with `--spec ngram` / `mtp` (1..=6).
    /// Default: 4 for `ngram`, 2 for `mtp`. Without `--spec`, on a backend where the recipe's tensor-parallel MTP default applies,
    /// it sets that default's draft length (`--spec-k 3` alone runs MTP k3); elsewhere it has no effect and warns.
    #[arg(long)]
    pub spec_k: Option<usize>,
    /// With `--spec dflash`: the draft checkpoint directory (`TD-Labs/Qwen3-8B-DCOCO` layout: `config.json` with
    /// `dflash_config`, `model.safetensors`). CUDA and the CPU backend, single sequence, greedy requests; the target's
    /// embedding and lm_head are used for the draft.
    #[arg(long)]
    pub spec_draft: Option<PathBuf>,
    /// With `--spec dflash`: the DCOCO corrector (arXiv 2610.12570) picks each block's tokens from the draft's top-16
    /// candidates per slot; needs the DCOCO checkpoint. Greedy output is unchanged, only the acceptance moves.
    #[arg(long)]
    pub dcoco: bool,
    /// With `--spec mtp`: draft over the first N lm_head rows (token ids 0..N, the frequent end of a BPE vocabulary) instead of
    /// all of them, so each draft costs N/V of a head sweep (about 11% of a 27B pass at k=2 is the two draft sweeps). Output is
    /// unchanged (the verify pass keeps the full head); only acceptance can move, down when the true token is outside 0..N.
    /// N a multiple of 8, below the vocabulary size (else the full head is used, with a warning at load). Needs `--spec mtp`.
    /// Default: the recipe's `[mtp] draft_vocab` (65536 for the Qwen3.8-27B recipe, `qwen3_5_vl`; the full head for every other
    /// recipe). `--mtp-draft-vocab 0` forces the full head.
    #[arg(long, value_parser = parse_draft_vocab)]
    pub mtp_draft_vocab: Option<usize>,
    /// Concurrent sequences the engine decodes together (continuous batching). Not a flag of this group: only
    /// `serve --max-batch` sets it, because the other commands need the single-sequence engine.
    #[arg(skip)]
    pub max_batch: Option<usize>,
    /// `serve --max-queue`, which sizes the batched engine's request queue. Set by `serve` only.
    #[arg(skip)]
    pub max_queue: Option<usize>,
    /// `serve --spec-off-at`, batched engine only. `serve` only.
    #[arg(skip)]
    pub spec_off_at: Option<usize>,
    /// Metal: keep the weight wiring from decaying while the server idles (`MetalBackend::set_keepwarm`). Default off;
    /// [`ModelArgs::for_serving`] turns it on, `INTARSIA_METAL_KEEPWARM=0` / `=1` / `=<ms>` override ([`keepwarm_period_ms`]);
    /// `INTARSIA_METAL_KEEPWARM_MODE=blit` keeps the files warm with GPU touches instead of an `MTLResidencySet`.
    #[arg(skip)]
    pub keepwarm: Option<bool>,
    /// `serve --busy-chunk`, batched engine only. `serve` only.
    #[arg(skip)]
    pub busy_chunk: Option<usize>,
    /// `serve --prefix-store`. `None`: `INTARSIA_PREFIX_STORE` decides (unset: off).
    #[arg(skip)]
    pub prefix_store: Option<intarsia_core::prefix_store::PrefixStoreConfig>,
    /// Where the resolved prefill chunk came from; set by [`ModelArgs::pin_prefill_chunk`] once the model is loaded.
    #[arg(skip)]
    pub prefill_chunk_source: Option<String>,
}

/// Keep-warm refresh period (ms) of the Metal backend, 0 = off: the driver drops its wiring of a weight file after about
/// a second unused (GPU touches: 1.0 s held, 1.2 s decayed; residency set: 2 s held, 5 s decayed), so 400 ms leaves room
/// for a late tick. `INTARSIA_METAL_KEEPWARM` wins over `default` (set by [`ModelArgs::for_serving`]; `None` = off, so every
/// other command behaves as it did before the switch existed): `0` / `off` / `false` / `no` = off, `1` / `on` / `true` /
/// `yes` = the default period, any other number = a period in ms clamped to [`KEEPWARM_MIN_MS`]..=[`KEEPWARM_MAX_MS`]
/// (with a warning: 2 ms would be ~500 touches/s, 5000 ms loses the wiring and the next use pays 210..230 ms). Anything
/// else is warned about and ignored.
pub fn keepwarm_period_ms(default: Option<bool>, env: Option<&str>) -> u64 {
    let from_default = if default == Some(true) { KEEPWARM_DEFAULT_MS } else { 0 };
    let Some(raw) = env else { return from_default };
    let v = raw.trim().to_ascii_lowercase();
    match v.as_str() {
        "0" | "off" | "false" | "no" => 0,
        "1" | "on" | "true" | "yes" => KEEPWARM_DEFAULT_MS,
        _ => match v.parse::<u64>() {
            Ok(ms) if (KEEPWARM_MIN_MS..=KEEPWARM_MAX_MS).contains(&ms) => ms,
            Ok(ms) => {
                let c = ms.clamp(KEEPWARM_MIN_MS, KEEPWARM_MAX_MS);
                eprintln!("warning: INTARSIA_METAL_KEEPWARM={ms} is outside {KEEPWARM_MIN_MS}..{KEEPWARM_MAX_MS} ms, using {c}");
                c
            }
            Err(_) => {
                eprintln!("warning: INTARSIA_METAL_KEEPWARM={raw:?} is not 0/1/on/off/true/false/yes/no or a period in ms, ignored");
                from_default
            }
        },
    }
}

const KEEPWARM_DEFAULT_MS: u64 = 400;
pub const KEEPWARM_MIN_MS: u64 = 50;
pub const KEEPWARM_MAX_MS: u64 = 1500;

pub fn open_backend(name: &str, device: usize) -> Result<Arc<dyn Backend>> {
    open_backend_with(name, device, None)
}

/// [`open_backend`] with the keep-warm default of the caller (only the Metal backend has one).
pub fn open_backend_with(name: &str, device: usize, keepwarm: Option<bool>) -> Result<Arc<dyn Backend>> {
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let _ = keepwarm;
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
                let be = intarsia_backend_metal::MetalBackend::new().map_err(|e| anyhow::anyhow!("{e}"))?;
                let mode = match std::env::var("INTARSIA_METAL_KEEPWARM_MODE").ok().as_deref() {
                    Some("blit") => intarsia_backend_metal::KeepMode::Blit,
                    _ => intarsia_backend_metal::KeepMode::Residency,
                };
                be.set_keepwarm_mode(keepwarm_period_ms(keepwarm, std::env::var("INTARSIA_METAL_KEEPWARM").ok().as_deref()), mode);
                Ok(Arc::new(be))
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
        "nccl" => bail!("--comm nccl joins processes: use it with --tp-remote (rank 0) and `intarsia tp-worker` (rank 1)"),
        other => bail!("unknown --comm '{other}' (auto, host, nccl)"),
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
        open_backend_with(&self.backend, self.device, self.keepwarm)
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
        // Two ranks on one GPU are refused: the CUDA backend keeps one cuBLAS handle and workspace per device, so two ranks
        // on two streams of the same device race on it (measured: tp=2 one-box self-golden KL moved run to run,
        // docs/results/2026-10-21-35b-tp2-cuda.md). `INTARSIA_TP_SAME_DEVICE=1` allows it with the host communicator,
        // for debugging only (results are not reproducible).
        let same_ok = self.comm == "host" && std::env::var_os("INTARSIA_TP_SAME_DEVICE").is_some_and(|v| v == "1");
        if sorted.len() != devs.len() && self.backend != "cpu" && !same_ok {
            bail!("--devices {devs:?} repeats a device (two ranks on one GPU race on its cuBLAS workspace)");
        }
        Ok(devs)
    }

    /// `--spec` as given, `off` when unset (the loader may still apply the recipe's tensor-parallel default, `spec_auto`).
    pub fn spec_mode(&self) -> &str {
        self.spec.as_deref().unwrap_or("off")
    }

    pub fn engine_options(&self) -> Result<EngineOptions> {
        if let Some(n) = self.mtp_draft_vocab.filter(|&n| n > 0).filter(|_| self.spec_mode() != "mtp") {
            anyhow::bail!("--mtp-draft-vocab {n} needs --spec mtp (it narrows the MTP drafter's head; --spec is {})", self.spec_mode());
        }
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
                prefill_chunk: self.prefill_chunk.unwrap_or(intarsia_core::recipe::DEFAULT_PREFILL_CHUNK),
                act_dtype: None,
                kv_dtype,
                profile_every: self.profile_every.unwrap_or(if self.backend == "cpu" { 1 } else { 16 }),
                mem_budget: self.mem_budget.map(|g| (g * (1u64 << 30) as f64) as u64),
                fp8_act: self.fp8_act.as_deref().map(intarsia_core::quant::Fp8Act::parse).transpose().map_err(|e| anyhow::anyhow!("{e}"))?,
                mxfp4_act: self.mxfp4_act.as_deref().map(intarsia_core::quant::Mxfp4Act::parse).transpose().map_err(|e| anyhow::anyhow!("{e}"))?,
                graphs: !self.no_graphs,
                tp_fusion: tp_fusion(std::env::var_os("INTARSIA_TP_FUSION").as_deref(), &self.backend, self.tp),
                tp_vocab: tp_vocab(std::env::var_os("INTARSIA_TP_VOCAB").as_deref(), &self.backend, self.tp, &self.comm),
                tp_arnorm: tp_arnorm(std::env::var_os("INTARSIA_TP_ARNORM").as_deref(), &self.backend, self.tp),
                mtp: self.spec_mode() == "mtp",
                mtp_k: 0, // set from --spec-k by the loader
                mtp_draft_vocab: self.mtp_draft_vocab,
                dflash: None, // set from --spec-draft by the loader
                prefill_macro: None, // INTARSIA_PREFILL_MACRO (Model::load)
                row_invariant_prefill: None, // INTARSIA_ROW_INVARIANT_PREFILL (Model::load)
            },
            spec_draft: self.spec_draft.clone(),
            dcoco: self.dcoco,
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
            spec: intarsia_core::spec::SpecConfig::parse(
                self.spec_mode(),
                self.spec_k.unwrap_or(if self.spec_mode() == "mtp" {
                    intarsia_core::spec::MTP_K_DEFAULT
                } else if self.spec_mode() == "dflash" {
                    intarsia_core::spec::DFLASH_K_DEFAULT
                } else if self.spec_mode() == "auto" {
                    intarsia_core::spec::AUTO_DFLASH_K_DEFAULT
                } else {
                    intarsia_core::spec::SPEC_K_DEFAULT
                }),
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?,
            max_batch: self.max_batch,
            max_queue: self.max_queue,
            spec_off_at: self.spec_off_at,
            // no --spec given (a lone --spec-k only sets the draft length of that default): the loader may apply the recipe's tensor-parallel MTP default
            spec_auto: self.spec.is_none(),
            spec_k_given: self.spec_k.is_some(),
            busy_chunk: self.busy_chunk,
            prefix_store: self.prefix_store.clone(),
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
        self.admit = true;
        self.require_admitted = true;
        // Metal has no decode graphs, so a profiled step synchronises around every op: at the default N = 16 that
        // costs about 12% of served decode. CUDA/HIP keep their default (graph decode profiles far less often).
        if self.profile_every.is_none() && self.backend == "metal" {
            self.profile_every = Some(0);
        }
        // The driver drops its wiring of the weight files about a second after their last use, so the first request after
        // a pause pays ~170 ms for it again (Qwen3-8B): a server keeps it warm. `generate`, `plan`, `doctor` and the accuracy
        // tools use the weights at once and exit, they keep the old behaviour.
        if self.keepwarm.is_none() {
            self.keepwarm = Some(true);
        }
        self
    }

    /// Open the backend and load the model into a ready engine.
    pub fn load(&self) -> Result<Loaded> {
        let opts = self.engine_options()?;
        if let Some(addr) = &self.tp_remote {
            let (backend, comm, stream) = self.join_remote_group(addr)?;
            return intarsia_models::load_engine_remote(&self.model, backend, comm, stream, &opts)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("loading {} with tp={} (rank 1 at {addr})", self.model.display(), self.tp));
        }
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

    /// Rank 0 of a cross-process group: connect to the rank-1 worker at `addr` (retrying while it starts), send it a fresh
    /// NCCL id, and join the communicator on `--device` (blocks until the worker has joined too).
    fn join_remote_group(&self, addr: &str) -> Result<(Arc<dyn Backend>, Arc<dyn intarsia_core::Comm>, std::net::TcpStream)> {
        if self.tp != 2 || self.backend != "cuda" || !matches!(self.comm.as_str(), "auto" | "nccl") {
            bail!("--tp-remote needs --tp 2 --backend cuda and the nccl communicator (got --tp {} --backend {} --comm {})", self.tp, self.backend, self.comm);
        }
        #[cfg(feature = "cuda")]
        {
            use std::io::Write;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
            let mut stream = loop {
                match std::net::TcpStream::connect(addr) {
                    Ok(s) => break s,
                    Err(e) if std::time::Instant::now() < deadline => {
                        let _ = e;
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                    Err(e) => bail!("cannot reach the tp-worker at {addr}: {e}"),
                }
            };
            let id = intarsia_backend_cuda::nccl::unique_id().map_err(|e| anyhow::anyhow!("{e}"))?;
            stream.write_all(&id)?;
            stream.flush()?;
            eprintln!("tensor parallel: rank 0 joining NCCL {} with the worker at {addr}", intarsia_backend_cuda::nccl::version().unwrap_or_default());
            let (backend, comm) = intarsia_backend_cuda::nccl::create(self.device, 0, 2, id).map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok((backend, comm, stream))
        }
        #[cfg(not(feature = "cuda"))]
        {
            bail!("--tp-remote needs a build with the CUDA backend (--features cuda); address {addr}")
        }
    }
}

/// `intarsia tp-worker`: rank 1 of a cross-process tensor-parallel group (docs/design/35b-tp2.md).
#[derive(clap::Args, Debug)]
pub struct TpWorkerArgs {
    /// Address to listen on for rank 0 (`--tp-remote` of the driving process).
    #[arg(long, value_name = "HOST:PORT")]
    pub listen: String,
    /// Rank 0 is `intarsia serve`: load with the server's settings (admission on plan shapes, admitted kernels only), as
    /// rank 0 does.
    #[arg(long)]
    pub serving: bool,
    /// The same model flags as rank 0 (both processes must load the same model with the same options; the handshake
    /// checks it).
    #[command(flatten)]
    pub model: ModelArgs,
}

pub fn run_tp_worker(a: TpWorkerArgs) -> Result<()> {
    if a.model.tp != 2 || a.model.backend != "cuda" {
        bail!("tp-worker serves rank 1 of --tp 2 on --backend cuda (got --tp {} --backend {})", a.model.tp, a.model.backend);
    }
    let model = if a.serving { a.model.clone().for_serving() } else { a.model.clone() };
    let opts = model.engine_options()?;
    let listener = std::net::TcpListener::bind(&a.listen).with_context(|| format!("listening on {}", a.listen))?;
    eprintln!("tp-worker: waiting for rank 0 on {}", a.listen);
    let (mut stream, peer) = listener.accept()?;
    eprintln!("tp-worker: rank 0 connected from {peer}");
    #[cfg(feature = "cuda")]
    {
        use std::io::Read;
        let mut id = [0u8; 128];
        stream.read_exact(&mut id)?;
        let (backend, comm) = intarsia_backend_cuda::nccl::create(a.model.device, 1, 2, id).map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("tp-worker: joined NCCL {} as rank 1 on device {}", intarsia_backend_cuda::nccl::version().unwrap_or_default(), a.model.device);
        let n = intarsia_models::serve_remote_rank(&a.model.model, backend, comm, stream, &opts).map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("tp-worker: rank 0 disconnected after {n} forward passes");
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (&mut stream, opts);
        bail!("tp-worker needs a build with the CUDA backend (--features cuda)")
    }
}

/// Row-local fusion under tensor parallelism (`LoadOptions::tp_fusion`; acts only at world > 1). On by default for CUDA
/// (user decision 2026-10-21 17:13, docs/results/2026-10-21-tp2-cuda-graphs.md) and for HIP at tp > 1 (user decisions 2026-10-21,
/// after the soak in docs/results/2026-10-21-hs-hip-fusion-soak.md, and 2026-10-25, gate vs tp1: docs/results/2026-10-25-tp2-hip.md),
/// off elsewhere. `INTARSIA_TP_FUSION=0`
/// opts out; any other value asks for it (the load refuses it on a backend without `Backend::fusion_under_tp`).
fn tp_fusion(env: Option<&std::ffi::OsStr>, backend: &str, tp: usize) -> bool {
    match env {
        Some(v) => v != "0",
        None => backend == "cuda" || (backend == "hip" && tp > 1),
    }
}

/// Vocab-parallel lm_head (`LoadOptions::tp_vocab`; acts only at world > 1): each rank computes half of the logits and the
/// halves are all-gathered into rank 0's logits, bitwise those of the replicated head. On by default for HIP at tp=2 on the
/// peer-to-peer communicator (`--comm auto`; the host-staged one would pay a host round trip per step), off elsewhere.
/// `INTARSIA_TP_VOCAB=0` opts out; any other value asks for it (the model keeps the replicated head, with a plan note, when
/// the communicator or the head cannot do it).
fn tp_vocab(env: Option<&std::ffi::OsStr>, backend: &str, tp: usize, comm: &str) -> bool {
    match env {
        Some(v) => v != "0",
        None => backend == "hip" && tp == 2 && comm == "auto",
    }
}

/// Whether HIP tp=2 fuses each block-end all-reduce with the residual add + rmsnorm after it by default. OFF until the GPU
/// gates of `scripts/tp2-hip/stages-arnorm.sh` (bitwise ids and dumps, speed) are recorded; flipping it is this one constant.
const TP_ARNORM_DEFAULT_ON: bool = false;

/// Fused all-reduce + residual add + rmsnorm under tensor parallelism (`LoadOptions::tp_arnorm`; acts at world 2 with fusion
/// on and a communicator that has it, today HIP's peer-to-peer one). `INTARSIA_TP_ARNORM=0` opts out, any other value asks
/// for it; unset follows [`TP_ARNORM_DEFAULT_ON`] for HIP at tp=2 and is off elsewhere.
fn tp_arnorm(env: Option<&std::ffi::OsStr>, backend: &str, tp: usize) -> bool {
    match env {
        Some(v) => v != "0",
        None => TP_ARNORM_DEFAULT_ON && backend == "hip" && tp == 2,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tp_vocab_default_on_for_hip_tp2_p2p_only_and_env_overrides() {
        use std::ffi::OsStr;
        assert!(super::tp_vocab(None, "hip", 2, "auto"));
        assert!(!super::tp_vocab(None, "hip", 2, "host"));
        assert!(!super::tp_vocab(None, "hip", 1, "auto"));
        assert!(!super::tp_vocab(None, "hip", 4, "auto"));
        assert!(!super::tp_vocab(None, "cuda", 2, "auto"));
        assert!(!super::tp_vocab(None, "metal", 2, "auto"));
        assert!(!super::tp_vocab(None, "cpu", 2, "host"));
        assert!(!super::tp_vocab(Some(OsStr::new("0")), "hip", 2, "auto"));
        assert!(super::tp_vocab(Some(OsStr::new("1")), "cpu", 2, "host"));
        assert!(super::tp_vocab(Some(OsStr::new("1")), "hip", 2, "host"));
    }

    #[test]
    fn tp_arnorm_env_overrides_and_the_default_is_hip_tp2_only() {
        use std::ffi::OsStr;
        assert_eq!(super::tp_arnorm(None, "hip", 2), super::TP_ARNORM_DEFAULT_ON);
        assert!(!super::tp_arnorm(None, "hip", 1));
        assert!(!super::tp_arnorm(None, "cuda", 2));
        assert!(!super::tp_arnorm(None, "cpu", 2));
        assert!(!super::tp_arnorm(Some(OsStr::new("0")), "hip", 2));
        assert!(super::tp_arnorm(Some(OsStr::new("1")), "hip", 2));
    }

    #[test]
    fn tp_fusion_default_on_for_cuda_and_hip_tp2_and_env_overrides() {
        use std::ffi::OsStr;
        assert!(super::tp_fusion(None, "cuda", 2));
        assert!(super::tp_fusion(None, "hip", 2));
        assert!(!super::tp_fusion(None, "hip", 1));
        assert!(super::tp_fusion(None, "hip", 4)); // HIP's collective is two ranks; a larger group is refused at load, not here
        assert!(!super::tp_fusion(None, "metal", 2));
        assert!(!super::tp_fusion(None, "cpu", 2));
        assert!(!super::tp_fusion(Some(OsStr::new("0")), "cuda", 2));
        assert!(!super::tp_fusion(Some(OsStr::new("0")), "hip", 2));
        // an explicit request is kept everywhere; the load refuses it on a backend that cannot
        assert!(super::tp_fusion(Some(OsStr::new("1")), "hip", 2));
        assert!(super::tp_fusion(Some(OsStr::new("1")), "hip", 1));
    }

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

    #[test]
    fn keepwarm_is_on_for_serve_only_and_the_env_overrides_both_ways() {
        use super::keepwarm_period_ms;
        let args = |serving: bool| {
            let mut a = Cli::parse_from(["t", "-m", "/m", "--backend", "metal"]).model;
            if serving {
                a = a.for_serving();
            }
            a.keepwarm
        };
        assert_eq!(args(false), None, "non-serve commands carry no default");
        assert_eq!(args(true), Some(true));
        assert_eq!(keepwarm_period_ms(args(false), None), 0);
        assert_eq!(keepwarm_period_ms(args(true), None), 400);
        assert_eq!(keepwarm_period_ms(args(true), Some("0")), 0);
        assert_eq!(keepwarm_period_ms(args(false), Some("1")), 400);
        assert_eq!(keepwarm_period_ms(args(false), Some("250")), 250);
        // words: off means off on serve too, on means on elsewhere (case and spaces ignored)
        for off in ["off", "OFF", "false", "no", " No "] {
            assert_eq!(keepwarm_period_ms(args(true), Some(off)), 0, "{off:?} on serve");
            assert_eq!(keepwarm_period_ms(args(false), Some(off)), 0, "{off:?}");
        }
        for on in ["on", "On", "true", "yes"] {
            assert_eq!(keepwarm_period_ms(args(false), Some(on)), 400, "{on:?}");
            assert_eq!(keepwarm_period_ms(args(true), Some(on)), 400, "{on:?} on serve");
        }
        // clamped to 50..=1500
        assert_eq!(keepwarm_period_ms(args(true), Some("2")), 50);
        assert_eq!(keepwarm_period_ms(args(true), Some("49")), 50);
        assert_eq!(keepwarm_period_ms(args(true), Some("50")), 50);
        assert_eq!(keepwarm_period_ms(args(true), Some("1500")), 1500);
        assert_eq!(keepwarm_period_ms(args(true), Some("5000")), 1500);
        // anything else is ignored: the default stands
        assert_eq!(keepwarm_period_ms(args(true), Some("")), 400);
        assert_eq!(keepwarm_period_ms(args(true), Some("maybe")), 400);
        assert_eq!(keepwarm_period_ms(args(false), Some("yess")), 0);
        assert_eq!(keepwarm_period_ms(args(true), Some("-5")), 400);
    }

    #[derive(Parser)]
    struct Gen {
        #[command(flatten)]
        a: crate::cmd::generate::GenerateArgs,
    }
    #[derive(Parser)]
    struct Acc {
        #[command(flatten)]
        a: crate::cmd::accuracy::AccuracyArgs,
    }
    #[derive(Parser)]
    struct Srv {
        #[command(flatten)]
        a: crate::cmd::serve::ServeArgs,
    }

    #[test]
    fn max_batch_is_a_serve_only_flag() {
        use clap::error::ErrorKind;
        let g = Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--max-batch", "4"]);
        assert_eq!(g.err().map(|e| e.kind()), Some(ErrorKind::UnknownArgument));
        let a = Acc::try_parse_from(["t", "-m", "/m", "--max-batch", "4"]);
        assert_eq!(a.err().map(|e| e.kind()), Some(ErrorKind::UnknownArgument));
        let s = Srv::try_parse_from(["t", "-m", "/m", "--max-batch", "4", "--max-queue", "2000"]).unwrap().a;
        assert_eq!((s.max_batch, s.max_queue), (Some(4), 2000));
        assert_eq!(s.model.max_batch, None, "serve copies the flag into the model args when it loads");
    }

    #[test]
    fn spec_off_at_is_a_serve_flag_with_a_checked_range() {
        let s = Srv::try_parse_from(["t", "-m", "/m"]).unwrap().a;
        assert_eq!(s.spec_off_at, 4, "default matches SchedulerConfig");
        let s = Srv::try_parse_from(["t", "-m", "/m", "--spec-off-at", "6"]).unwrap().a;
        assert_eq!(s.spec_off_at, 6);
        assert!(Srv::try_parse_from(["t", "-m", "/m", "--spec-off-at", "10"]).is_err());
        let g = Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--spec-off-at", "4"]);
        assert!(g.is_err());
    }

    #[test]
    fn mtp_draft_vocab_needs_spec_mtp_and_a_multiple_of_eight() {
        // not a multiple of 8: refused when the command line is parsed
        assert!(Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--spec", "mtp", "--mtp-draft-vocab", "65537"]).is_err());
        // without --spec mtp it would silently do nothing: an error from engine_options
        let g = Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--mtp-draft-vocab", "65536"]).unwrap().a;
        let e = g.model.engine_options().err().expect("refused").to_string();
        assert!(e.contains("--spec mtp"), "{e}");
        let g = Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--spec", "ngram", "--mtp-draft-vocab", "65536"]).unwrap().a;
        assert!(g.model.engine_options().is_err());
        // with it, and unset: fine
        let g = Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--spec", "mtp", "--mtp-draft-vocab", "65536"]).unwrap().a;
        assert_eq!(g.model.engine_options().unwrap().load.mtp_draft_vocab, Some(65536));
        // unset: the recipe decides (None); an explicit 0 forces the full head and is fine with any --spec
        let g = Gen::try_parse_from(["t", "-m", "/m", "-p", "x"]).unwrap().a;
        assert_eq!(g.model.engine_options().unwrap().load.mtp_draft_vocab, None);
        let g = Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--mtp-draft-vocab", "0"]).unwrap().a;
        assert_eq!(g.model.engine_options().unwrap().load.mtp_draft_vocab, Some(0));
        let g = Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--spec", "mtp", "--mtp-draft-vocab", "0"]).unwrap().a;
        assert_eq!(g.model.engine_options().unwrap().load.mtp_draft_vocab, Some(0));
    }

    #[test]
    fn busy_chunk_is_a_serve_flag_with_a_checked_range() {
        let s = Srv::try_parse_from(["t", "-m", "/m"]).unwrap().a;
        // unset: the engine's default (256; 1024 for hybrid MoE on CUDA, resolved at load)
        assert_eq!(s.busy_chunk, None);
        assert_eq!(intarsia_core::engine::BUSY_PREFILL_CHUNK, 256);
        let s = Srv::try_parse_from(["t", "-m", "/m", "--busy-chunk", "320"]).unwrap().a;
        assert_eq!(s.busy_chunk, Some(320));
        assert!(Srv::try_parse_from(["t", "-m", "/m", "--busy-chunk", "15"]).is_err());
        assert!(Srv::try_parse_from(["t", "-m", "/m", "--busy-chunk", "2049"]).is_err());
        assert!(Gen::try_parse_from(["t", "-m", "/m", "-p", "x", "--busy-chunk", "320"]).is_err());
    }
}

/// `--mtp-draft-vocab`: 0 (the full head) or a positive multiple of 8 (the head's row-prefix kernels need whole 8-row groups).
fn parse_draft_vocab(s: &str) -> std::result::Result<usize, String> {
    let n: usize = s.trim().parse().map_err(|e| format!("not a number: {e}"))?;
    if n % 8 != 0 {
        return Err(format!("{n} is not a multiple of 8"));
    }
    Ok(n)
}

#[cfg(test)]
mod draft_vocab_tests {
    use super::parse_draft_vocab;

    #[test]
    fn draft_vocab_is_zero_or_a_multiple_of_eight() {
        assert_eq!(parse_draft_vocab("0"), Ok(0));
        assert_eq!(parse_draft_vocab("65536"), Ok(65536));
        assert!(parse_draft_vocab("65537").is_err());
        assert!(parse_draft_vocab("-8").is_err());
        assert!(parse_draft_vocab("x").is_err());
    }
}

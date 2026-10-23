//! `intarsia recipe new`: infer a recipe from a Hugging Face model directory.
//!
//! The command reads `config.json` (merging `text_config` of multimodal wrappers), the quantisation metadata and the
//! tensor names of the checkpoint (`model.safetensors.index.json`, or the headers of the `*.safetensors` files: never
//! tensor data), then
//!
//! * checks whether an existing recipe (`--recipe-dir` directories first, then the built-ins) already covers the model:
//!   its architecture matches, every tensor template of its blocks resolves in the checkpoint, the config passes
//!   `check_supported` and the quantisation scheme is one the engine implements. If so nothing is written;
//! * otherwise writes a recipe scaffold in which every inferred field carries a comment with its source and everything
//!   that could not be mapped is a `# TODO(recipe-new): ...` marker; the command then fails so that a half-finished
//!   recipe is never mistaken for a finished one.
//!
//! [`analyse`] is the pure part (no printing, no writes) so that `intarsia qualify` can call it.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand};
use intarsia_core::quant::scheme::{Algo, QuantScheme};
use intarsia_core::quant::{BF16, EXL3, FP8_E4M3, MXFP4, NVFP4};
use intarsia_core::recipe::{text_config, ConfigKeys, ModelDims, Recipe};
use intarsia_core::Error as CoreError;
use intarsia_models::recipe::all_recipes;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct RecipeArgs {
    #[command(subcommand)]
    pub cmd: RecipeCmd,
}

#[derive(Subcommand, Debug)]
pub enum RecipeCmd {
    /// Infer a recipe from a Hugging Face model directory.
    New(NewArgs),
}

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Hugging Face model directory (config.json, optional hf_quant_config.json, *.safetensors or the index).
    #[arg(long, short = 'm')]
    pub model: PathBuf,
    /// Output file (default: recipes/<family>.toml). Given explicitly, the scaffold is written even when an
    /// existing recipe already covers the model.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Extra directory with recipe *.toml files (searched before built-ins) when looking for an existing recipe.
    #[arg(long)]
    pub recipe_dir: Vec<PathBuf>,
    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,
}

/// What `recipe new` found out about a model directory.
#[derive(Clone, Debug)]
pub struct Analysis {
    /// Source of an existing recipe that already handles this model (`builtin:qwen3`, or the path of a recipe file
    /// found in a `--recipe-dir`); `None` when a new recipe is needed.
    pub covered_by: Option<String>,
    /// Recipe family derived from the config's `model_type` (the default output name is `recipes/<family>.toml`).
    pub family: String,
    /// What the scaffold could not map, one line each, in file order. Each also appears as a
    /// `# TODO(recipe-new): ...` marker in [`Analysis::toml`]. Empty: the scaffold is complete.
    pub todos: Vec<String>,
    /// The scaffold recipe, built whether or not an existing recipe covers the model.
    pub toml: String,
    /// Quantisation of the checkpoint's linear layers: `bf16` / `f16` / `f32`, `fp8_e4m3`, `nvfp4`, `nvfp4+fp8`
    /// (mixed), or `unknown(<what>)`.
    pub quant: String,
    /// `(resolved, total)` distinct tensor names of the covering recipe (else of the closest recipe), expanded for the
    /// first and last layer that uses each block, plus the top-level tensors.
    pub tensor_templates_resolved: (usize, usize),
    /// What was inferred and why the closest recipe was chosen.
    pub notes: Vec<String>,
}

pub fn run(args: RecipeArgs) -> Result<()> {
    match args.cmd {
        RecipeCmd::New(a) => new_cmd(a),
    }
}

fn new_cmd(a: NewArgs) -> Result<()> {
    let an = analyse(&a.model, &a.recipe_dir)?;
    let explicit = a.out.is_some();
    let (r, t) = an.tensor_templates_resolved;
    match &an.covered_by {
        Some(src) => println!(
            "covered: {src} already handles this model ({r}/{t} tensor templates resolve, quant: {}); no new recipe needed",
            an.quant
        ),
        None => println!(
            "not covered: no existing recipe handles this model (closest: {r}/{t} tensor templates resolve, quant: {})",
            an.quant
        ),
    }
    if an.covered_by.is_some() && !explicit {
        return Ok(());
    }
    for n in &an.notes {
        println!("  note: {n}");
    }
    let path = a.out.clone().unwrap_or_else(|| {
        // `recipes/<family>.toml` is usually the family's existing recipe, which (we are here) does not cover this model,
        // e.g. a VL wrapper of a known text family: do not default onto it, take the first free `<family>_<n>.toml`
        let base = PathBuf::from("recipes").join(format!("{}.toml", an.family));
        if !base.exists() {
            return base;
        }
        (2..100).map(|n| PathBuf::from("recipes").join(format!("{}_{n}.toml", an.family))).find(|p| !p.exists()).unwrap_or(base)
    });
    if path.exists() && !a.force {
        bail!(
            "{} exists; refusing to overwrite it. It may be another model's recipe: pass --out recipes/<new name>.toml \
             (--force overwrites it)",
            path.display()
        );
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(&path, &an.toml).with_context(|| format!("writing {}", path.display()))?;
    println!("wrote {} ({} TODO(recipe-new) item(s))", path.display(), an.todos.len());
    if an.todos.is_empty() {
        println!(
            "next: intarsia generate --model {} --recipe {} --backend cpu --prompt \"The capital of France is\" --max-tokens 16",
            a.model.display(),
            path.display()
        );
        return Ok(());
    }
    eprintln!("{} item(s) need manual work:", an.todos.len());
    for (i, t) in an.todos.iter().enumerate() {
        eprintln!("  {}. {t}", i + 1);
    }
    bail!("{} item(s) need manual work, see the TODO(recipe-new) markers in {}", an.todos.len(), path.display())
}

// ---------------------------------------------------------------------------------------------------------------
// checkpoint inventory

/// Tensor names (and dtype / shape where a safetensors header was read) of a checkpoint directory.
struct Inventory {
    names: BTreeSet<String>,
    info: BTreeMap<String, (String, Vec<usize>)>,
    source: String,
}

/// Name, dtype and shape of every tensor in a safetensors file; reads the 8-byte length and the JSON header only.
fn read_header(path: &Path) -> Result<Vec<(String, String, Vec<usize>)>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).with_context(|| format!("{}: reading the safetensors header length", path.display()))?;
    let n = u64::from_le_bytes(len);
    if n > 256 << 20 {
        bail!("{}: safetensors header of {n} bytes is implausible (not a safetensors file?)", path.display());
    }
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf).with_context(|| format!("{}: reading the safetensors header", path.display()))?;
    let v: Value =
        serde_json::from_slice(&buf).with_context(|| format!("{}: parsing the safetensors header", path.display()))?;
    let obj = v.as_object().ok_or_else(|| anyhow!("{}: safetensors header is not an object", path.display()))?;
    let mut out = Vec::new();
    for (k, t) in obj {
        if k == "__metadata__" {
            continue;
        }
        let dtype = t.get("dtype").and_then(Value::as_str).unwrap_or("?").to_string();
        let shape = t
            .get("shape")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
            .unwrap_or_default();
        out.push((k.clone(), dtype, shape));
    }
    Ok(out)
}

impl Inventory {
    fn open(dir: &Path) -> Result<Inventory> {
        let mut inv = Inventory { names: BTreeSet::new(), info: BTreeMap::new(), source: String::new() };
        let index = dir.join("model.safetensors.index.json");
        if index.is_file() {
            let text = std::fs::read_to_string(&index).with_context(|| format!("reading {}", index.display()))?;
            let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", index.display()))?;
            let wm = v
                .get("weight_map")
                .and_then(Value::as_object)
                .ok_or_else(|| anyhow!("{}: no weight_map object", index.display()))?;
            let mut files = BTreeSet::new();
            for (name, f) in wm {
                inv.names.insert(name.clone());
                if let Some(f) = f.as_str() {
                    files.insert(f.to_string());
                }
            }
            let mut read = 0;
            for f in &files {
                let p = dir.join(f);
                if p.is_file() {
                    for (n, d, s) in read_header(&p)? {
                        inv.names.insert(n.clone());
                        inv.info.insert(n, (d, s));
                    }
                    read += 1;
                }
            }
            inv.source = if read > 0 {
                format!("model.safetensors.index.json ({} tensors) + {read}/{} shard headers", wm.len(), files.len())
            } else {
                format!(
                    "model.safetensors.index.json ({} tensors; shards not present, so no dtypes or shapes)",
                    wm.len()
                )
            };
        } else {
            let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
                .with_context(|| format!("reading {}", dir.display()))?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
                .collect();
            files.sort();
            if files.is_empty() {
                bail!("{}: neither model.safetensors.index.json nor *.safetensors files found", dir.display());
            }
            for p in &files {
                for (n, d, s) in read_header(p)? {
                    inv.names.insert(n.clone());
                    inv.info.insert(n, (d, s));
                }
            }
            inv.source = format!("safetensors headers of {} file(s) ({} tensors)", files.len(), inv.names.len());
        }
        inv.add_exl3_views();
        Ok(inv)
    }

    /// An EXL3 module has no `.weight`: give it a virtual one (dtype `EXL3`, the logical `[N, K]` from the trellis
    /// `[K/16, N/16, 16*bits]`), so templates resolve and shapes check like any other linear layer.
    fn add_exl3_views(&mut self) {
        let mods: Vec<String> = self.names.iter().filter_map(|n| n.strip_suffix(".trellis").map(String::from)).collect();
        for m in mods {
            let w = format!("{m}.weight");
            if self.has(&w) {
                continue;
            }
            self.names.insert(w.clone());
            if let Some((_, s)) = self.info.get(&format!("{m}.trellis")).filter(|(_, s)| s.len() == 3).cloned() {
                self.info.insert(w, ("EXL3".to_string(), vec![16 * s[1], 16 * s[0]]));
            }
        }
    }

    fn has(&self, n: &str) -> bool {
        self.names.contains(n)
    }

    /// The tensor a recipe template names: the name itself or `name.weight` (as the model loader resolves it).
    fn resolve(&self, n: &str) -> Option<String> {
        if self.has(n) {
            return Some(n.to_string());
        }
        let w = format!("{n}.weight");
        self.has(&w).then_some(w)
    }

    fn dtype(&self, n: &str) -> Option<&str> {
        self.info.get(n).map(|(d, _)| d.as_str())
    }

    fn shape(&self, n: &str) -> Option<&[usize]> {
        self.info.get(n).map(|(_, s)| s.as_slice())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// layer groups

/// Path segments that introduce a stack of numbered layers.
const LAYER_WORDS: &[&str] = &["layers", "h", "blocks"];
/// Name segments (prefixes) of components that are not part of the text decoder.
const NON_TEXT: &[&str] = &["visual", "vision", "mtp", "multi_modal", "audio", "speech", "talker"];
/// Trailing tensor names of a module (`<module>.weight`, ...).
const PARAMS: &[&str] = &[
    "weight",
    "bias",
    "weight_scale",
    "weight_scale_2",
    "weight_scale_inv",
    "input_scale",
    "qweight",
    "qzeros",
    "scales",
    "g_idx",
    "k_scale",
    "v_scale",
    "weight_packed",
    "weight_shape",
    // EXL3 (ExLlamaV3): the trellis, the input / output scales (or packed signs) and the codebook markers
    "trellis",
    "suh",
    "svh",
    "su",
    "sv",
    "mcg",
    "mul1",
];

fn is_non_text(name: &str) -> bool {
    name.split('.').any(|s| NON_TEXT.iter().any(|p| s.starts_with(p)))
}

/// `model.layers.7.self_attn.q_proj.weight` -> (`model.layers`, 7, `self_attn.q_proj.weight`).
fn layer_split(name: &str) -> Option<(String, usize, String)> {
    let segs: Vec<&str> = name.split('.').collect();
    for k in 0..segs.len().saturating_sub(2) {
        if LAYER_WORDS.contains(&segs[k]) {
            if let Ok(i) = segs[k + 1].parse::<usize>() {
                return Some((segs[..=k].join("."), i, segs[k + 2..].join(".")));
            }
        }
    }
    None
}

fn split_param(rel: &str) -> (&str, &str) {
    match rel.rsplit_once('.') {
        Some((m, l)) if PARAMS.contains(&l) => (m, l),
        _ => (rel, ""),
    }
}

#[derive(Clone, Debug)]
struct LayerGroup {
    /// `model.language_model.layers`
    prefix: String,
    indices: BTreeSet<usize>,
}

impl LayerGroup {
    fn template(&self, rel: &str) -> String {
        format!("{}.{{i}}.{rel}", self.prefix)
    }
    fn concrete(&self, layer: usize, rel: &str) -> String {
        format!("{}.{layer}.{rel}", self.prefix)
    }
    /// The module that holds `embed_tokens` / `norm`: the prefix without the trailing `.layers`.
    fn base(&self) -> &str {
        self.prefix.rsplit_once('.').map(|(b, _)| b).unwrap_or("")
    }
}

// ---------------------------------------------------------------------------------------------------------------
// block detection inside one layer

#[derive(Clone, Debug, PartialEq)]
struct AttnT {
    parent: String,
    q: String,
    k: String,
    v: String,
    o: String,
    q_norm: Option<String>,
    k_norm: Option<String>,
    /// Module names (`<parent>.q_proj`, ...) whose `.bias` tensor exists: Qwen2-style projection biases.
    q_bias: Option<String>,
    k_bias: Option<String>,
    v_bias: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct GdnT {
    parent: String,
    in_qkv: String,
    in_z: String,
    in_b: String,
    in_a: String,
    conv: String,
    a_log: String,
    dt_bias: String,
    norm: String,
    out: String,
}

#[derive(Clone, Debug, PartialEq)]
struct FfnT {
    parent: String,
    gate: String,
    up: String,
    down: String,
}

#[derive(Clone, Debug, PartialEq)]
struct MoeT {
    parent: String,
}

#[derive(Clone, Debug)]
enum Blk {
    Attn(AttnT),
    Gdn(GdnT),
    Ffn(FfnT),
    Moe(MoeT),
}

impl Blk {
    fn name(&self) -> &'static str {
        match self {
            Blk::Attn(_) => "attn",
            Blk::Gdn(_) => "gdn",
            Blk::Ffn(_) => "ffn",
            Blk::Moe(_) => "moe",
        }
    }
}

/// A TODO attached to the recipe table it belongs to (`layer`, `attn`, `gdn`, `ffn`, `moe`, `norms`, `top`).
#[derive(Clone, Debug)]
struct Issue {
    at: &'static str,
    text: String,
}

fn issue(at: &'static str, text: impl Into<String>) -> Issue {
    Issue { at, text: text.into() }
}

/// Tensors of one layer, grouped into modules (`self_attn.q_proj` with the params `weight`, `weight_scale`, ...).
/// Bare tensors (`linear_attn.A_log`) are modules with the empty param.
struct LayerView {
    modules: BTreeMap<String, BTreeSet<String>>,
}

impl LayerView {
    fn new(rels: &[String]) -> LayerView {
        let mut modules: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for r in rels {
            let (m, p) = split_param(r);
            modules.entry(m.to_string()).or_default().insert(p.to_string());
        }
        LayerView { modules }
    }
}

struct LayerDet {
    blocks: Vec<Blk>,
    /// `pre_attn` -> layer-level norm module.
    pre: BTreeMap<String, String>,
    issues: Vec<Issue>,
    /// Layer-relative tensor names no block claims.
    unclaimed: Vec<String>,
}

const PRE_MIXER: &[&str] =
    &["input_layernorm", "attention_norm", "ln_1", "pre_attention_layernorm", "norm1", "input_norm"];
const PRE_FFN: &[&str] =
    &["pre_feedforward_layernorm", "post_attention_layernorm", "ffn_norm", "ln_2", "pre_mlp_layernorm", "norm2"];

fn detect_layer(view: &LayerView) -> LayerDet {
    let mods = &view.modules;
    let mut issues: Vec<Issue> = Vec::new();
    let mut claimed: BTreeSet<String> = BTreeSet::new();
    let mut parents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for m in mods.keys() {
        if let Some((p, leaf)) = m.rsplit_once('.') {
            parents.entry(p.to_string()).or_default().insert(leaf.to_string());
        }
    }

    // sparse MoE: a module with `<parent>.experts.<n>.*`
    let mut moe_parent: Option<String> = None;
    let mut fused_experts: Option<String> = None;
    for m in mods.keys() {
        if let Some(pos) = m.find(".experts.") {
            let first = m[pos + 9..].split('.').next().unwrap_or("");
            let p = m[..pos].to_string();
            if !first.is_empty() && first.bytes().all(|b| b.is_ascii_digit()) {
                moe_parent.get_or_insert(p);
            } else if fused_experts.is_none() {
                fused_experts = Some(p);
            }
        }
    }
    if moe_parent.is_none() {
        if let Some(p) = fused_experts.take() {
            issues.push(issue(
                "moe",
                format!(
                    "layer.moe: experts of {p}.experts are stored fused (e.g. experts.gate_up_proj as one tensor); the moe block loads \
                     per-expert tensors experts.{{e}}.{{gate,up,down}}_proj, so the checkpoint needs converting or the block extending"
                ),
            ));
            moe_parent = Some(p);
        }
    }

    let mut attn: Option<AttnT> = None;
    let mut gdn: Option<GdnT> = None;
    let mut ffn: Option<FfnT> = None;
    let mut moe: Option<MoeT> = None;
    if let Some(p) = &moe_parent {
        for m in mods.keys() {
            if m.starts_with(&format!("{p}.experts."))
                || m.starts_with(&format!("{p}.shared_expert."))
                || *m == format!("{p}.gate")
                || *m == format!("{p}.shared_expert_gate")
            {
                claimed.insert(m.clone());
            }
        }
        if !mods.contains_key(&format!("{p}.gate")) {
            issues.push(issue("moe", format!("layer.moe: router tensor {p}.gate not found in the layer")));
        }
        moe = Some(MoeT { parent: p.clone() });
    }

    for (p, leaves) in &parents {
        if let Some(mp) = &moe_parent {
            if p == mp || p.starts_with(&format!("{mp}.")) {
                continue;
            }
        }
        let l = |x: &str| leaves.contains(x);
        let child = |x: &str| format!("{p}.{x}");
        if l("q_proj") && l("k_proj") && l("v_proj") && l("o_proj") {
            if attn.is_none() {
                let opt = |x: &str| l(x).then(|| child(x));
                let has_bias = |x: &str| mods.get(&child(x)).is_some_and(|s| s.contains("bias")).then(|| child(x));
                attn = Some(AttnT {
                    parent: p.clone(),
                    q: child("q_proj"),
                    k: child("k_proj"),
                    v: child("v_proj"),
                    o: child("o_proj"),
                    q_norm: opt("q_norm"),
                    k_norm: opt("k_norm"),
                    q_bias: has_bias("q_proj"),
                    k_bias: has_bias("k_proj"),
                    v_bias: has_bias("v_proj"),
                });
            }
        } else if l("in_proj_qkv")
            && l("in_proj_z")
            && l("in_proj_b")
            && l("in_proj_a")
            && l("conv1d")
            && l("A_log")
            && l("dt_bias")
            && l("norm")
            && l("out_proj")
        {
            if gdn.is_none() {
                gdn = Some(GdnT {
                    parent: p.clone(),
                    in_qkv: child("in_proj_qkv"),
                    in_z: child("in_proj_z"),
                    in_b: child("in_proj_b"),
                    in_a: child("in_proj_a"),
                    conv: child("conv1d"),
                    a_log: child("A_log"),
                    dt_bias: child("dt_bias"),
                    norm: child("norm"),
                    out: child("out_proj"),
                });
            }
        } else if l("in_proj_qkvz") || l("in_proj_ba") {
            issues.push(issue(
                "layer",
                format!(
                    "layer.gdn: {p} stores the GatedDeltaNet projections fused (in_proj_qkvz / in_proj_ba, Qwen3-Next layout); the gdn \
                     block expects in_proj_qkv / in_proj_z / in_proj_b / in_proj_a"
                ),
            ));
        } else if l("in_proj") && (l("A_log") || l("D")) {
            issues.push(issue(
                "layer",
                format!("layer: {p} looks like a Mamba / SSM mixer (in_proj + A_log); no such block kind is implemented, a new block is needed (docs/adding-a-model.md section 4)"),
            ));
        } else if l("q_a_proj") || l("kv_a_proj_with_mqa") || l("kv_b_proj") || l("q_a_layernorm") {
            issues.push(issue(
                "layer",
                format!("layer.attn: {p} is multi-head latent attention (MLA: q_a_proj / kv_a_proj_with_mqa / kv_b_proj); only attention kind \"gqa\" is implemented, a new block is needed"),
            ));
        } else if l("qkv_proj") || l("query_key_value") || l("c_attn") {
            issues.push(issue(
                "layer",
                format!("layer.attn: {p} has a fused qkv projection; the gqa block needs separate q_proj / k_proj / v_proj / o_proj tensors"),
            ));
        } else if l("gate_proj") && l("up_proj") && l("down_proj") {
            if ffn.is_none() {
                ffn = Some(FfnT {
                    parent: p.clone(),
                    gate: child("gate_proj"),
                    up: child("up_proj"),
                    down: child("down_proj"),
                });
            }
        } else if l("gate_up_proj") {
            issues.push(issue(
                "layer",
                format!("layer.ffn: {p} has a fused gate_up_proj; the dense ffn block needs separate gate_proj / up_proj / down_proj tensors"),
            ));
        }
    }

    let mut mixers: Vec<Blk> = Vec::new();
    if let Some(a) = attn {
        for m in
            [Some(&a.q), Some(&a.k), Some(&a.v), Some(&a.o), a.q_norm.as_ref(), a.k_norm.as_ref()].into_iter().flatten()
        {
            claimed.insert(m.clone());
        }
        // q / k / v biases are mapped in the recipe (`q_bias` / `k_bias` / `v_bias`); an o_proj bias is not implemented
        if mods.get(&a.o).is_some_and(|s| s.contains("bias")) {
            issues.push(issue(
                "attn",
                format!("layer.attn: {}.bias present (o_proj bias, text_config.attention_bias): the output projection bias is not implemented", a.o),
            ));
        }
        mixers.push(Blk::Attn(a));
    }
    if let Some(g) = gdn {
        for m in [&g.in_qkv, &g.in_z, &g.in_b, &g.in_a, &g.conv, &g.a_log, &g.dt_bias, &g.norm, &g.out] {
            claimed.insert(m.clone());
        }
        mixers.push(Blk::Gdn(g));
    }
    let mut ffns: Vec<Blk> = Vec::new();
    if let Some(m) = moe {
        ffns.push(Blk::Moe(m));
    }
    if let Some(f) = ffn {
        for m in [&f.gate, &f.up, &f.down] {
            claimed.insert(m.clone());
        }
        ffns.push(Blk::Ffn(f));
    }
    if mixers.is_empty() {
        let names: Vec<&String> = parents.keys().collect();
        issues.push(issue(
            "layer",
            format!("layer: no attention or linear-attention block recognised (module groups in the layer: {names:?}); a new block or a different tensor layout is needed"),
        ));
    }
    if ffns.is_empty() {
        issues.push(issue("layer", "layer.ffn: no dense FFN (gate_proj / up_proj / down_proj) or MoE (experts.{e}.*) block recognised in the layer"));
    }
    if ffns.len() > 1 {
        issues.push(issue(
            "layer",
            "layer: both a dense FFN and an MoE block in one layer; a layer can use only one FFN block",
        ));
        ffns.truncate(1);
    }
    let mut blocks = mixers;
    blocks.extend(ffns);

    // pre-block norms: the layer-level norm modules
    let norms: BTreeSet<&str> = mods
        .keys()
        .filter(|m| !m.contains('.') && (m.contains("norm") || m.starts_with("ln_")))
        .map(|m| m.as_str())
        .collect();
    let pick = |cands: &[&str]| cands.iter().find(|c| norms.contains(**c)).map(|c| c.to_string());
    let (mixer_norm, ffn_norm) = (pick(PRE_MIXER), pick(PRE_FFN));
    let mut pre = BTreeMap::new();
    for b in &blocks {
        let key = format!("pre_{}", b.name());
        let n = if matches!(b, Blk::Attn(_) | Blk::Gdn(_)) { mixer_norm.clone() } else { ffn_norm.clone() };
        match n {
            Some(n) => {
                claimed.insert(n.clone());
                if mods.get(&n).is_some_and(|s| s.contains("bias")) {
                    issues.push(issue("norms", format!("layer.norms.{key}: {n} has a bias (LayerNorm); only RMSNorm is implemented")));
                }
                pre.insert(key, n);
            }
            None => issues.push(issue(
                "norms",
                format!("layer.norms.{key}: no layer-level norm found among {:?} (norm modules in the layer: {:?}); name the norm tensor", if matches!(b, Blk::Attn(_) | Blk::Gdn(_)) { PRE_MIXER } else { PRE_FFN }, norms),
            )),
        }
    }

    let mut unclaimed = Vec::new();
    for (m, params) in mods {
        if claimed.contains(m) {
            continue;
        }
        for p in params {
            let name = if p.is_empty() { m.clone() } else { format!("{m}.{p}") };
            if !name.ends_with("inv_freq") {
                unclaimed.push(name);
            }
        }
    }
    LayerDet { blocks, pre, issues, unclaimed }
}

/// One layer type (`full_attention`, ...) or the single untyped layer kind.
struct TypeDet {
    name: String,
    first: usize,
    pattern: Vec<&'static str>,
}

/// Everything inferred from the layer tensors.
struct Det {
    found: bool,
    group: LayerGroup,
    types: Vec<TypeDet>,
    attn: Option<(AttnT, usize)>,
    gdn: Option<(GdnT, usize)>,
    ffn: Option<(FfnT, usize)>,
    moe: Option<(MoeT, usize)>,
    pre: BTreeMap<String, (String, usize)>,
    issues: Vec<Issue>,
    unclaimed: BTreeSet<String>,
}

fn rels_of_layer(inv: &Inventory, group: &LayerGroup, layer: usize) -> Vec<String> {
    let pre = format!("{}.{layer}.", group.prefix);
    inv.names.range(pre.clone()..).take_while(|n| n.starts_with(&pre)).map(|n| n[pre.len()..].to_string()).collect()
}

fn detect(inv: &Inventory, group: Option<LayerGroup>, layer_types: &[String]) -> Det {
    let found = group.is_some();
    let group = group.unwrap_or(LayerGroup { prefix: "model.layers".into(), indices: BTreeSet::new() });
    let mut det = Det {
        found,
        group: group.clone(),
        types: Vec::new(),
        attn: None,
        gdn: None,
        ffn: None,
        moe: None,
        pre: BTreeMap::new(),
        issues: Vec::new(),
        unclaimed: BTreeSet::new(),
    };
    if !found {
        det.issues.push(issue(
            "layer",
            "layer: no numbered layer stack (<prefix>.layers.<i>. / .h. / .blocks.) found in the checkpoint tensors; name the layer tensors by hand",
        ));
        det.types.push(TypeDet { name: String::new(), first: 0, pattern: vec!["attn", "ffn"] });
        return det;
    }
    let (min, max) = (*group.indices.first().unwrap(), *group.indices.last().unwrap());
    // (type name, first layer, last layer) of each distinct layer type present in the checkpoint
    let mut samples: Vec<(String, usize, usize)> = Vec::new();
    if layer_types.is_empty() {
        samples.push((String::new(), min, max));
    } else {
        for (i, t) in layer_types.iter().enumerate() {
            if !group.indices.contains(&i) {
                continue;
            }
            match samples.iter_mut().find(|s| s.0 == *t) {
                Some(s) => s.2 = i,
                None => samples.push((t.clone(), i, i)),
            }
        }
        if samples.is_empty() {
            samples.push((String::new(), min, max));
        }
    }
    for (name, first, last) in samples {
        let rels = rels_of_layer(inv, &group, first);
        let view = LayerView::new(&rels);
        let d = detect_layer(&view);
        if last != first {
            let lview = LayerView::new(&rels_of_layer(inv, &group, last));
            if lview.modules.keys().ne(view.modules.keys()) {
                let a: Vec<&String> = view.modules.keys().filter(|k| !lview.modules.contains_key(*k)).take(3).collect();
                let b: Vec<&String> = lview.modules.keys().filter(|k| !view.modules.contains_key(*k)).take(3).collect();
                det.issues.push(issue(
                    "layer",
                    format!(
                        "layer: layers {first} and {last}{} have different tensors (only in {first}: {a:?}; only in {last}: {b:?}); one pattern cannot describe both (dense-then-MoE or similar), split them by layer type",
                        if name.is_empty() { String::new() } else { format!(" of type '{name}'") }
                    ),
                ));
            }
        }
        let tname = if name.is_empty() { format!("layer {first}") } else { format!("type '{name}'") };
        let mut pattern = Vec::new();
        for b in &d.blocks {
            pattern.push(b.name());
            match b.clone() {
                Blk::Attn(a) => merge(&mut det.attn, a, first, &tname, "attn", &mut det.issues),
                Blk::Gdn(g) => merge(&mut det.gdn, g, first, &tname, "gdn", &mut det.issues),
                Blk::Ffn(f) => merge(&mut det.ffn, f, first, &tname, "ffn", &mut det.issues),
                Blk::Moe(m) => merge(&mut det.moe, m, first, &tname, "moe", &mut det.issues),
            }
        }
        for (k, v) in d.pre {
            match det.pre.get(&k) {
                Some((old, _)) if *old != v => det.issues.push(issue(
                    "norms",
                    format!("layer.norms.{k}: {tname} uses norm '{v}' but another layer type uses '{old}'; one recipe cannot express both"),
                )),
                Some(_) => {}
                None => {
                    det.pre.insert(k, (v, first));
                }
            }
        }
        det.issues.extend(d.issues);
        det.unclaimed.extend(d.unclaimed);
        det.types.push(TypeDet { name, first, pattern });
    }
    det
}

fn merge<T: PartialEq + std::fmt::Debug>(
    slot: &mut Option<(T, usize)>,
    new: T,
    layer: usize,
    tname: &str,
    blk: &str,
    issues: &mut Vec<Issue>,
) {
    match slot {
        None => *slot = Some((new, layer)),
        Some((old, _)) if *old != new => issues.push(issue(
            "layer",
            format!("layer.{blk}: {tname} names the {blk} tensors differently from an earlier layer type ({old:?} vs {new:?}); one table cannot express both"),
        )),
        Some(_) => {}
    }
}

// ---------------------------------------------------------------------------------------------------------------
// quantisation

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Cls {
    Float(String),
    Fp8,
    Nvfp4,
    Mxfp4,
    Exl3,
    Unknown(String),
}

fn float_label(dtype: &str) -> Option<&'static str> {
    match dtype {
        "BF16" => Some("bf16"),
        "F16" => Some("f16"),
        "F32" => Some("f32"),
        _ => None,
    }
}

/// How the linear layer `module` is stored, judged like `quant::detect` does (tensor names, plus dtypes when known).
fn classify(inv: &Inventory, module: &str, cfg_float: &str) -> Option<Cls> {
    let w = format!("{module}.weight");
    if inv.has(&format!("{module}.trellis")) {
        return Some(Cls::Exl3);
    }
    if !inv.has(&w) {
        return inv.has(&format!("{module}.qweight")).then(|| Cls::Unknown("packed qweight (GPTQ / AWQ style)".into()));
    }
    let dt = inv.dtype(&w);
    Some(if inv.has(&format!("{module}.weight_scale_2")) {
        Cls::Nvfp4
    } else if inv.has(&format!("{module}.weight_scale_inv")) {
        Cls::Unknown("block-wise FP8 (weight_scale_inv)".into())
    } else if inv.has(&format!("{module}.weight_scale")) {
        match dt {
            // u8 e2m1 pairs next to u8 e8m0 group scales: OCP MXFP4 (Quark)
            Some("U8") if inv.dtype(&format!("{module}.weight_scale")) == Some("U8") => Cls::Mxfp4,
            None | Some("F8_E4M3") => Cls::Fp8,
            Some(other) => Cls::Unknown(format!("weight_scale next to a {other} weight")),
        }
    } else {
        match dt {
            Some(d) => match float_label(d) {
                Some(l) => Cls::Float(l.into()),
                None if d == "F8_E4M3" => Cls::Unknown("F8_E4M3 weight without weight_scale".into()),
                None => Cls::Unknown(format!("{d} weight without scales")),
            },
            None => Cls::Float(cfg_float.to_string()),
        }
    })
}

struct QuantInfo {
    label: String,
    problems: Vec<String>,
    notes: Vec<String>,
}

fn read_json(path: &Path) -> Result<Value> {
    let t = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&t).with_context(|| format!("parsing {}", path.display()))
}

/// `modules`: checkpoint linear-layer names to classify; the flag marks routed experts.
fn analyse_quant(model: &Path, cfg: &Value, tc: &Value, inv: &Inventory, modules: &[(String, bool)]) -> QuantInfo {
    let mut problems: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    // declared metadata
    let mut doc: Option<(Value, &str)> = None;
    let hf = model.join("hf_quant_config.json");
    if hf.is_file() {
        match read_json(&hf) {
            Ok(v) => doc = Some((v, "hf_quant_config.json")),
            Err(e) => problems
                .push(format!("quant: hf_quant_config.json is unreadable ({e:#}); the declared scheme is unknown")),
        }
    }
    let qc = tc.get("quantization_config").or_else(|| cfg.get("quantization_config")).filter(|v| v.is_object());
    let method = qc.and_then(|q| q.get("quant_method")).and_then(Value::as_str).map(|s| s.to_ascii_lowercase());
    match (qc, method.as_deref()) {
        (None, _) => {}
        (Some(_), Some("modelopt")) => {}
        (Some(_), Some("quark")) => {} // parsed below (QuantScheme::parse_quark)
        (Some(_), Some("exl3")) => {} // parsed below (QuantScheme::parse_exl3)
        (Some(q), None) if q.get("quant_algo").is_some() || q.get("quantized_layers").is_some() => {}
        (Some(q), Some("fp8")) => {
            if q.get("weight_block_size").is_some_and(|v| !v.is_null()) {
                problems.push(format!(
                    "quant: config quantization_config.quant_method = \"fp8\" with weight_block_size {} (block-wise FP8); the engine implements per-tensor FP8 only",
                    q["weight_block_size"]
                ));
            }
        }
        (Some(_), Some(other)) => problems.push(format!(
            "quant: config quantization_config.quant_method = \"{other}\" is not implemented (supported: ModelOpt FP8 and W4A16_NVFP4, per-tensor FP8, Quark MXFP4, ExLlamaV3 EXL3)"
        )),
        (Some(_), None) => problems.push("quant: config quantization_config has neither quant_method nor quant_algo; the scheme is unknown".into()),
    }
    if doc.is_none() {
        if let Some(q) = qc.filter(|q| q.get("quant_algo").is_some() || q.get("quantized_layers").is_some()) {
            doc = Some((q.clone(), "config.json quantization_config"));
        }
    }
    let mut scheme: Option<QuantScheme> = None;
    if let (None, Some(q), Some("quark")) = (&doc, qc, method.as_deref()) {
        match QuantScheme::parse_quark(q) {
            Ok(s) => {
                let a = s.quant_algo.clone().unwrap_or_default();
                match Algo::parse(&a) {
                    Algo::Mxfp4W4A4 => notes.push(
                        "config.json quantization_config (quark) declares MXFP4 weights with dynamic MXFP4 activations (W4A4); \
                         the engine runs that by default (--mxfp4-act none opts into W4A16)"
                            .into(),
                    ),
                    Algo::Mxfp4W4A16 => {}
                    _ => problems.push(format!(
                        "quant: Quark scheme \"{a}\" declared in config.json is not implemented (supported: OCP MXFP4 weights, group 32, e8m0, with dynamic \"even\" fp4 or 16-bit activations)"
                    )),
                }
                scheme = Some(s);
            }
            Err(e) => problems.push(format!("quant: config.json quantization_config (quark): {e}")),
        }
    }
    if let Some((d, src)) = &doc {
        match QuantScheme::parse(d) {
            Ok(s) => scheme = Some(s),
            Err(e) => problems.push(format!("quant: {src}: {e}")),
        }
        let q = d.get("quantization").unwrap_or(d);
        let mut algos: BTreeSet<String> = BTreeSet::new();
        if let Some(a) = q.get("quant_algo").and_then(Value::as_str) {
            if !a.eq_ignore_ascii_case("MIXED_PRECISION") {
                algos.insert(a.to_string());
            }
        }
        if let Some(l) = q.get("quantized_layers").and_then(Value::as_object) {
            algos.extend(l.values().filter_map(|v| v.get("quant_algo").and_then(Value::as_str)).map(String::from));
        }
        for a in &algos {
            match Algo::parse(a) {
                Algo::Fp8W8A8 | Algo::Nvfp4W4A16 | Algo::Mxfp4W4A4 | Algo::Mxfp4W4A16 | Algo::Exl3 => {}
                Algo::Nvfp4W4A4 => problems.push(format!(
                    "quant: quant_algo \"{a}\" (NVFP4 weights AND 4-bit activations) declared in {src}; the engine implements only weight-only W4A16_NVFP4"
                )),
                Algo::Other => problems.push(format!("quant: quant_algo \"{a}\" declared in {src} is not implemented (supported: FP8, W4A16_NVFP4)")),
            }
        }
        if let Some(kv) = q
            .get("kv_cache_quant_algo")
            .and_then(Value::as_str)
            .filter(|a| !a.is_empty() && !a.eq_ignore_ascii_case("none"))
        {
            notes.push(format!(
                "{src} declares kv_cache_quant_algo {kv}; the KV cache runs as bf16 here (its k/v scales are not used)"
            ));
        }
    }

    // stored format, per linear layer
    let cfg_float = ["dtype", "torch_dtype"]
        .iter()
        .find_map(|k| tc.get(*k).or_else(|| cfg.get(*k)).and_then(Value::as_str))
        .map(|d| match d {
            "float16" => "f16",
            "float32" => "f32",
            _ => "bf16",
        })
        .unwrap_or("bf16");
    let mut classes: BTreeSet<Cls> = BTreeSet::new();
    let mut seen_msgs: BTreeSet<String> = BTreeSet::new();
    for (m, is_expert) in modules {
        let Some(c) = classify(inv, m, cfg_float) else { continue };
        if *is_expert && !matches!(c, Cls::Nvfp4) {
            let msg = format!("quant: routed experts ({m}) are stored as {c:?}; the moe block loads per-expert NVFP4 (W4A16_NVFP4) only");
            if seen_msgs.insert(msg.clone()) {
                problems.push(msg);
            }
        }
        let id = match &c {
            Cls::Float(_) => Some(BF16),
            Cls::Fp8 => Some(FP8_E4M3),
            Cls::Nvfp4 => Some(NVFP4),
            Cls::Mxfp4 => Some(MXFP4),
            Cls::Exl3 => Some(EXL3),
            Cls::Unknown(d) => {
                let msg = format!("quant: tensor layout of {m} not recognised: {d}");
                if seen_msgs.insert(msg.clone()) {
                    problems.push(msg);
                }
                None
            }
        };
        if let (Some(id), Some(s)) = (id, &scheme) {
            if let Err(e) = s.check_linear(m, id) {
                let msg = format!("quant: {e}");
                if seen_msgs.insert(msg.clone()) {
                    problems.push(msg);
                }
            }
        }
        classes.insert(c);
    }

    let mut parts: Vec<String> = Vec::new();
    let has_nvfp4 = classes.contains(&Cls::Nvfp4);
    if has_nvfp4 {
        parts.push("nvfp4".into());
    }
    if classes.contains(&Cls::Mxfp4) {
        parts.push("mxfp4".into());
    }
    if classes.contains(&Cls::Exl3) {
        parts.push("exl3".into());
    }
    if classes.contains(&Cls::Fp8) {
        parts.push(if has_nvfp4 { "fp8".into() } else { "fp8_e4m3".into() });
    }
    if parts.is_empty() {
        for c in &classes {
            if let Cls::Float(l) = c {
                parts.push(l.clone());
            }
        }
    }
    for c in &classes {
        if let Cls::Unknown(d) = c {
            parts.push(format!("unknown({d})"));
        }
    }
    let label = if parts.is_empty() { cfg_float.to_string() } else { parts.join("+") };
    // a handful of messages is enough
    if problems.len() > 6 {
        let n = problems.len() - 5;
        problems.truncate(5);
        problems.push(format!("quant: ... and {n} more quantisation problem(s)"));
    }
    QuantInfo { label, problems, notes }
}

// ---------------------------------------------------------------------------------------------------------------
// coverage of an existing recipe

#[derive(Default)]
struct Coverage {
    resolved: usize,
    total: usize,
    missing: Vec<String>,
    problems: Vec<String>,
}

impl Coverage {
    fn clean(&self) -> bool {
        self.missing.is_empty() && self.problems.is_empty()
    }
    fn ratio(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.resolved as f64 / self.total as f64
        }
    }
}

fn recipe_blocks(r: &Recipe) -> BTreeSet<String> {
    r.layer.pattern.iter().chain(r.layer.by_type.values().flatten()).cloned().collect()
}

fn num(tc: &Value, k: &str) -> Option<usize> {
    tc.get(k).and_then(Value::as_u64).map(|v| v as usize)
}

/// Expand every tensor template of `r` (for the first and last layer that uses each block) and look it up.
fn coverage(
    r: &Recipe,
    cfg: &Value,
    tc: &Value,
    n_layers: usize,
    layer_types: &[String],
    inv: &Inventory,
    tie: bool,
) -> Coverage {
    let mut cov = Coverage::default();
    let mut names: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut add = |n: String| {
        if seen.insert(n.clone()) {
            names.push(n);
        }
    };
    let mut span: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for li in 0..n_layers {
        match r.layer.pattern_for(li, layer_types) {
            Ok(p) => {
                for b in p {
                    span.entry(b.clone()).and_modify(|s| s.1 = li).or_insert((li, li));
                }
            }
            Err(e) => {
                cov.problems.push(format!("layer {li}: {e}"));
                break;
            }
        }
    }
    let n_experts = num(tc, "num_experts").or_else(|| num(tc, "num_local_experts")).unwrap_or(1);
    let shared = num(tc, "shared_expert_intermediate_size").unwrap_or(0) > 0;
    for (b, (first, last)) in &span {
        let mut ls = vec![*first];
        if last != first {
            ls.push(*last);
        }
        for li in ls {
            let e = |t: &str| Recipe::expand(t, li);
            match b.as_str() {
                "attn" => {
                    if let Some(a) = &r.layer.attn {
                        for t in [&a.q, &a.k, &a.v, &a.o] {
                            add(e(t));
                        }
                        for t in [&a.q_norm, &a.k_norm].into_iter().flatten() {
                            add(e(t));
                        }
                    }
                }
                "gdn" => {
                    if let Some(g) = &r.layer.gdn {
                        for t in [&g.in_qkv, &g.in_z, &g.in_b, &g.in_a, &g.conv, &g.a_log, &g.dt_bias, &g.norm, &g.out]
                        {
                            add(e(t));
                        }
                    }
                }
                "ffn" => {
                    if let Some(f) = &r.layer.ffn {
                        for t in [&f.gate, &f.up, &f.down] {
                            add(e(t));
                        }
                    }
                }
                "moe" => {
                    if let Some(p) = r.layer.other.get("moe").and_then(|m| m.get("prefix")).and_then(Value::as_str) {
                        let p = e(p);
                        add(format!("{p}.gate"));
                        let mut ex = vec![0];
                        if n_experts > 1 {
                            ex.push(n_experts - 1);
                        }
                        for x in ex {
                            for proj in ["gate_proj", "up_proj", "down_proj"] {
                                add(format!("{p}.experts.{x}.{proj}"));
                            }
                        }
                        if shared {
                            for proj in ["gate_proj", "up_proj", "down_proj"] {
                                add(format!("{p}.shared_expert.{proj}"));
                            }
                            add(format!("{p}.shared_expert_gate"));
                        }
                    } else {
                        cov.problems.push("recipe has a moe block but no [layer.moe] prefix".into());
                    }
                }
                "gemma" => {
                    if let Some(p) = r.layer.other.get("gemma").and_then(|m| m.get("prefix")).and_then(Value::as_str) {
                        let p = e(p);
                        for t in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "router.proj"] {
                            add(format!("{p}.{t}"));
                        }
                        let mut ex = vec![0];
                        if n_experts > 1 {
                            ex.push(n_experts - 1);
                        }
                        for x in ex {
                            for proj in ["gate_proj", "up_proj", "down_proj"] {
                                add(format!("{p}.experts.{x}.{proj}"));
                            }
                        }
                        for t in ["layer_scalar", "router.scale", "router.per_expert_scale", "post_attention_layernorm", "pre_feedforward_layernorm_2", "post_feedforward_layernorm_2"] {
                            add(format!("{p}.{t}"));
                        }
                    } else {
                        cov.problems.push("recipe has a gemma block but no [layer.gemma] prefix".into());
                    }
                }
                other => cov.problems.push(format!("block '{other}' is not implemented in this build")),
            }
            match r.layer.norms.get(&format!("pre_{b}")) {
                Some(n) => add(e(n)),
                None => cov.problems.push(format!("recipe layer.norms.pre_{b} missing")),
            }
        }
    }
    add(r.top.embed.clone());
    add(r.top.final_norm.clone());
    for n in names {
        cov.total += 1;
        if inv.resolve(&n).is_some() {
            cov.resolved += 1;
        } else {
            cov.missing.push(n);
        }
    }
    // lm_head: its own tensor, or tied to the embedding table when the config says so
    let lh = &r.top.lm_head;
    cov.total += 1;
    if inv.resolve(&lh.tensor).is_some() || (tie && lh.tied_to.as_deref().is_some_and(|t| inv.resolve(t).is_some())) {
        cov.resolved += 1;
    } else {
        cov.missing.push(format!("{} (lm_head; tie_word_embeddings = {tie})", lh.tensor));
    }
    // config
    match ModelDims::from_hf_config(cfg, &r.config) {
        Err(e) => cov.problems.push(format!("config: {e}")),
        Ok(d) => {
            if span.contains_key("gdn") && d.linear.is_none() {
                cov.problems.push("recipe has a gdn block but config.json has no linear attention dimensions".into());
            }
            if span.contains_key("moe") && d.moe.is_none() {
                cov.problems.push("recipe has a moe block but config.json has no MoE dimensions".into());
            }
            if let (Some(a), Some((first, _))) = (&r.layer.attn, span.get("attn")) {
                let rows = d.n_heads * d.head_dim * (1 + usize::from(a.gated));
                let q = Recipe::expand(&a.q, *first);
                let shape = inv.resolve(&q).and_then(|n| inv.shape(&n).map(|s| s.to_vec()));
                match shape {
                    Some(s) if !s.is_empty() && s[0] != rows => cov.problems.push(format!(
                        "attn.gated = {} but {q} has {} rows (expected {rows} = heads x head_dim x {})",
                        a.gated,
                        s[0],
                        1 + usize::from(a.gated)
                    )),
                    None => {
                        if let Some(g) = tc.get("attn_output_gate").and_then(Value::as_bool) {
                            if g != a.gated {
                                cov.problems
                                    .push(format!("attn.gated = {} but config attn_output_gate = {g}", a.gated));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    cov
}

// ---------------------------------------------------------------------------------------------------------------
// TOML emission

fn s(x: &str) -> String {
    serde_json::to_string(x).expect("string serialises")
}

fn key(k: &str) -> String {
    if !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        k.to_string()
    } else {
        s(k)
    }
}

fn str_array(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|x| s(x)).collect::<Vec<_>>().join(", "))
}

#[derive(Default)]
struct Out {
    text: String,
    todos: Vec<String>,
}

impl Out {
    fn raw(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
    }
    fn blank(&mut self) {
        self.text.push('\n');
    }
    fn comment(&mut self, c: &str) {
        self.raw(&format!("# {c}"));
    }
    /// `key = value  # source`: every field goes through here, so none is written without its source.
    fn field(&mut self, k: &str, value: &str, src: &str) {
        let l = format!("{k} = {value}");
        self.raw(&format!("{l:<46}  # {src}"));
    }
    fn todo(&mut self, text: &str) {
        let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
        self.raw(&format!("# TODO(recipe-new): {t}"));
        self.todos.push(t);
    }
    fn issues(&mut self, issues: &[Issue], at: &str) {
        for i in issues.iter().filter(|i| i.at == at) {
            self.todo(&i.text);
        }
    }
}

/// How the checkpoint's config deviates from what the engine implements, one entry per key (unlike
/// `check_supported`, which stops at the first).
struct ConfigIssues {
    act: Option<String>,
    attention_bias: bool,
    sliding: Vec<String>,
    /// (rope type, where it was found)
    rope: Option<(String, String)>,
}

fn config_issues(tc: &Value) -> ConfigIssues {
    // Gemma 4 spells the key `hidden_activation`
    let act = tc.get("hidden_act").or_else(|| tc.get("hidden_activation")).and_then(Value::as_str).filter(|a| *a != "silu").map(String::from);
    let attention_bias = tc.get("attention_bias").and_then(Value::as_bool) == Some(true);
    let mut sliding = Vec::new();
    let use_sw = tc.get("use_sliding_window").and_then(Value::as_bool);
    let sw = tc.get("sliding_window").filter(|v| v.is_number());
    if use_sw == Some(true) {
        sliding.push("text_config.use_sliding_window = true".to_string());
    } else if use_sw.is_none() && sw.is_some() {
        sliding
            .push(format!("text_config.sliding_window = {} (use_sliding_window absent, so it applies)", sw.unwrap()));
    }
    if tc
        .get("layer_types")
        .and_then(Value::as_array)
        .is_some_and(|a| a.iter().any(|t| t.as_str() == Some("sliding_attention")))
    {
        sliding.push("text_config.layer_types contains sliding_attention".to_string());
    }
    let mut rope = None;
    for what in ["rope_parameters", "rope_scaling"] {
        if let Some(v) = tc.get(what).filter(|v| !v.is_null()) {
            let ty = v.get("rope_type").or_else(|| v.get("type")).and_then(Value::as_str);
            // llama3 and yarn scaling is implemented (read from config.json at load); the rest is not
            if let Some(t) = ty.filter(|t| !matches!(*t, "default" | "llama3" | "yarn")) {
                rope = Some((t.to_string(), format!("text_config.{what}.rope_type = \"{t}\"")));
                break;
            }
        }
    }
    ConfigIssues { act, attention_bias, sliding, rope }
}

fn family_of(cfg: &Value, tc: &Value, archs: &[String]) -> (String, String) {
    let clean = |s: &str| {
        s.to_ascii_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect::<String>()
    };
    if let Some(mt) = cfg.get("model_type").and_then(Value::as_str) {
        return (clean(mt), "config.json model_type".into());
    }
    if let Some(mt) = tc.get("model_type").and_then(Value::as_str) {
        return (clean(mt), "text_config.model_type".into());
    }
    if let Some(a) = archs.first() {
        let f = a.trim_end_matches("ForConditionalGeneration").trim_end_matches("ForCausalLM");
        return (clean(f), "config.json architectures[0] (no model_type)".into());
    }
    ("model".into(), "default (no model_type or architectures in config.json)".into())
}

/// Do the model family and a recipe's family name the same thing (`qwen3_5_moe_text` vs `qwen3_5_moe`)?
fn same_family(model: &str, recipe: &str) -> bool {
    let n = |s: &str| s.trim_end_matches("_text").to_string();
    n(model) == n(recipe)
}

// ---------------------------------------------------------------------------------------------------------------
// analyse

struct Cand {
    src: String,
    recipe: Recipe,
    arch_match: bool,
    cov: Coverage,
}

/// Infer a recipe for the model in `model` and decide whether an existing one (from `recipe_dirs`, then the
/// built-ins) already covers it. Pure: reads the directory, prints and writes nothing.
pub fn analyse(model: &Path, recipe_dirs: &[PathBuf]) -> Result<Analysis> {
    let cfg = read_json(&model.join("config.json"))?;
    let tc = text_config(&cfg);
    let inv = Inventory::open(model)?;
    let mut notes: Vec<String> = Vec::new();

    let archs: Vec<String> = cfg
        .get("architectures")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let (family, family_src) = family_of(&cfg, tc, &archs);
    notes.push(format!(
        "config.json: architectures {archs:?}, family \"{family}\" ({family_src}){}",
        if std::ptr::eq(tc, &cfg) {
            String::new()
        } else {
            "; text model settings merged from text_config".to_string()
        }
    ));
    notes.push(format!("tensors: {}", inv.source));

    // layer stack
    let n_layers_cfg = num(tc, "num_hidden_layers");
    let layer_types: Vec<String> = tc
        .get("layer_types")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let mut groups: BTreeMap<String, (BTreeSet<usize>, usize)> = BTreeMap::new();
    for n in &inv.names {
        if let Some((p, i, _)) = layer_split(n) {
            let g = groups.entry(p).or_default();
            g.0.insert(i);
            g.1 += 1;
        }
    }
    let chosen = groups
        .iter()
        .max_by_key(|(p, (idx, count))| (Some(idx.len()) == n_layers_cfg, !is_non_text(p), *count, idx.len()))
        .map(|(p, (idx, _))| LayerGroup { prefix: p.clone(), indices: idx.clone() });
    if let Some(g) = &chosen {
        notes.push(format!(
            "layer prefix {}.{{i}} ({} layers{}){}",
            g.prefix,
            g.indices.len(),
            match n_layers_cfg {
                Some(n) if n == g.indices.len() => ", matches num_hidden_layers".to_string(),
                Some(n) => format!(", config num_hidden_layers = {n}"),
                None => String::new(),
            },
            if groups.len() > 1 {
                format!(
                    "; other numbered stacks ignored: {}",
                    groups
                        .iter()
                        .filter(|(p, _)| **p != g.prefix)
                        .map(|(p, (i, _))| format!("{p} ({})", i.len()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                String::new()
            }
        ));
    }
    let det = detect(&inv, chosen, &layer_types);
    let n_layers = n_layers_cfg.or_else(|| det.group.indices.last().map(|l| l + 1)).unwrap_or(0);
    let tie = tc
        .get("tie_word_embeddings")
        .or_else(|| cfg.get("tie_word_embeddings"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // quantisation
    let mut lin: Vec<(String, bool)> = Vec::new();
    if det.found {
        let g = &det.group;
        let mut add = |layer: usize, rel: &str, expert: bool| lin.push((g.concrete(layer, rel), expert));
        if let Some((a, l)) = &det.attn {
            for m in [&a.q, &a.k, &a.v, &a.o] {
                add(*l, m, false);
            }
        }
        if let Some((x, l)) = &det.gdn {
            for m in [&x.in_qkv, &x.in_z, &x.in_b, &x.in_a, &x.out] {
                add(*l, m, false);
            }
        }
        if let Some((f, l)) = &det.ffn {
            for m in [&f.gate, &f.up, &f.down] {
                add(*l, m, false);
            }
        }
        if let Some((m, l)) = &det.moe {
            let p = &m.parent;
            add(*l, &format!("{p}.gate"), false);
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                add(*l, &format!("{p}.shared_expert.{proj}"), false);
                add(*l, &format!("{p}.experts.0.{proj}"), true);
            }
            add(*l, &format!("{p}.shared_expert_gate"), false);
        }
    }
    // the output head, wherever it lives
    let head = ["lm_head".to_string()]
        .into_iter()
        .chain(
            inv.names
                .iter()
                .filter(|n| n.ends_with(".lm_head.weight") && !is_non_text(n))
                .map(|n| n.trim_end_matches(".weight").to_string()),
        )
        .find(|m| inv.resolve(m).is_some());
    if let Some(h) = &head {
        lin.push((h.clone(), false));
    }
    let quant = analyse_quant(model, &cfg, tc, &inv, &lin);
    notes.push(format!(
        "quantisation: {}{}",
        quant.label,
        if quant.problems.is_empty() { "" } else { " (not implemented in full, see TODOs)" }
    ));
    notes.extend(quant.notes.iter().cloned());

    // existing recipes
    let mut cands: Vec<Cand> = Vec::new();
    for (src, recipe) in all_recipes(recipe_dirs) {
        let arch_match = recipe.model.matcher.architectures.iter().any(|a| archs.contains(a));
        let cov = coverage(&recipe, &cfg, tc, n_layers, &layer_types, &inv, tie);
        cands.push(Cand { src, recipe, arch_match, cov });
    }
    let inferred: BTreeSet<String> = det.types.iter().flat_map(|t| t.pattern.iter().map(|b| b.to_string())).collect();
    // a recipe may accept a W4A4-declared NVFP4 checkpoint running weight-only (`[defaults] w4a4_as_w4a16`)
    let only_w4a4 = !quant.problems.is_empty() && quant.problems.iter().all(|p| p.contains("NVFP4 weights AND 4-bit activations"));
    let covering = cands
        .iter()
        .position(|c| c.arch_match && c.cov.clean() && (quant.problems.is_empty() || (only_w4a4 && c.recipe.defaults.w4a4_as_w4a16)));
    let closest = covering.or_else(|| {
        let score = |c: &Cand| {
            let rb = recipe_blocks(&c.recipe);
            let inter = rb.intersection(&inferred).count() as f64;
            let union = rb.union(&inferred).count().max(1) as f64;
            c.cov.ratio() * 2.0 + inter / union + if c.arch_match { 0.5 } else { 0.0 }
        };
        let mut best: Option<(usize, f64)> = None;
        for (i, c) in cands.iter().enumerate() {
            let sc = score(c);
            if best.is_none_or(|(_, b)| sc > b) {
                best = Some((i, sc));
            }
        }
        best.map(|(i, _)| i)
    });
    let covered_by = covering.map(|i| cands[i].src.clone());
    let mut resolved = (0, 0);
    match closest.map(|i| &cands[i]) {
        Some(c) => {
            resolved = (c.cov.resolved, c.cov.total);
            let why = if covering.is_some() {
                "covers this model".to_string()
            } else {
                let mut r = Vec::new();
                if c.arch_match {
                    r.push("architecture matches but the recipe cannot be used as is".to_string());
                } else {
                    r.push("architecture differs".to_string());
                }
                if !c.cov.missing.is_empty() {
                    let m: Vec<&String> = c.cov.missing.iter().take(4).collect();
                    r.push(format!("{} tensor template(s) missing in the checkpoint, e.g. {m:?}", c.cov.missing.len()));
                }
                r.extend(c.cov.problems.iter().take(3).cloned());
                if !quant.problems.is_empty() {
                    r.push("quantisation not implemented".into());
                }
                r.join("; ")
            };
            notes.push(format!(
                "closest existing recipe: {} (blocks {:?} vs inferred {:?}; {}/{} tensor templates resolve; {why})",
                c.src,
                recipe_blocks(&c.recipe),
                inferred,
                c.cov.resolved,
                c.cov.total
            ));
        }
        None => notes.push("no existing recipe to compare with".into()),
    }

    let (toml, todos, emit_notes) = emit(&EmitCtx {
        model,
        cfg: &cfg,
        tc,
        inv: &inv,
        det: &det,
        archs: &archs,
        family: &family,
        family_src: &family_src,
        quant: &quant,
        closest: closest.map(|i| &cands[i]),
        // model-level settings (parsers, norm_offset) follow the model family, not the tensor layout: a VL wrapper of a
        // known family (tensors under model.language_model.*) is closest by templates to another family's recipe
        family_recipe: cands.iter().find(|c| same_family(&family, &c.recipe.model.family)),
        covered: covering.is_some(),
        head: head.as_deref(),
        tie,
    });
    notes.extend(emit_notes);
    Ok(Analysis { covered_by, family, todos, toml, quant: quant.label, tensor_templates_resolved: resolved, notes })
}

struct EmitCtx<'a> {
    model: &'a Path,
    cfg: &'a Value,
    tc: &'a Value,
    inv: &'a Inventory,
    det: &'a Det,
    archs: &'a [String],
    family: &'a str,
    family_src: &'a str,
    quant: &'a QuantInfo,
    closest: Option<&'a Cand>,
    /// An existing recipe of the same family (may differ from `closest`).
    family_recipe: Option<&'a Cand>,
    covered: bool,
    head: Option<&'a str>,
    tie: bool,
}

fn emit(cx: &EmitCtx) -> (String, Vec<String>, Vec<String>) {
    let (model, cfg, tc, inv, det, archs) = (cx.model, cx.cfg, cx.tc, cx.inv, cx.det, cx.archs);
    let (family, family_src, quant, closest, head, tie) =
        (cx.family, cx.family_src, cx.quant, cx.closest, cx.head, cx.tie);
    let mut notes: Vec<String> = Vec::new();
    let mut o = Out::default();
    let g = &det.group;
    let ci = config_issues(tc);
    let cr = closest.map(|c| &c.recipe);
    let csrc = closest.map(|c| c.src.as_str()).unwrap_or("");

    // config errors that the specific checks above did not explain
    let dims = ModelDims::from_hf_config(cfg, &ConfigKeys::default());
    let mut config_todos: Vec<String> = Vec::new();
    let explained = ci.act.is_some() || ci.attention_bias || !ci.sliding.is_empty() || ci.rope.is_some();
    if let Err(e) = &dims {
        if !(matches!(e, CoreError::Unsupported(_)) && explained) {
            config_todos.push(format!(
                "config.json: {e}; the engine reads the dimensions with the Hugging Face key names, map different names in a [config] table (docs/adding-a-model.md)"
            ));
        }
    }

    o.comment(&format!("Recipe scaffold for {family} generated by `intarsia recipe new --model {}`.", model.display()));
    o.comment("Every inferred field carries the source it came from; `TODO(recipe-new)` marks what could not be mapped and needs a decision.");
    if let Some(c) = closest {
        o.comment(&format!(
            "Closest existing recipe: {}{}.",
            c.src,
            if cx.covered { " (already covers this model)" } else { "" }
        ));
    }

    // [model]
    o.raw("[model]");
    if archs.is_empty() {
        o.todo("model.match.architectures: config.json has no \"architectures\"; list the architecture name(s) this recipe handles");
        o.field("match.architectures", "[\"TODO\"]", "placeholder");
    } else {
        o.field("match.architectures", &str_array(archs), "from config.json architectures");
    }
    o.field("family", &s(family), &format!("from {family_src}"));
    o.field(
        "chat_template",
        &s("hf"),
        "default: the model's own jinja template (chat_template.jinja or tokenizer_config.json)",
    );
    let closest_same = cr.is_some_and(|r| same_family(family, &r.model.family));
    // the recipe model-level settings come from: the closest recipe when it is of this family, else another recipe of it
    let (fr, fsrc, fam_match) = match (closest_same, cx.family_recipe) {
        (true, _) => (cr, format!("closest recipe {csrc}"), true),
        (false, Some(c)) => (Some(&c.recipe), format!("recipe {}", c.src), true),
        (false, None) => (cr, format!("closest recipe {csrc}"), false),
    };
    if let (Some(r), true) = (fr, fam_match) {
        for (k, v) in [("reasoning_parser", &r.model.reasoning_parser), ("tool_parser", &r.model.tool_parser)] {
            if let Some(v) = v {
                o.field(k, &s(v), &format!("from {fsrc} (same family)"));
            }
        }
    } else {
        o.comment("reasoning_parser / tool_parser: optional, not inferable from the checkpoint (see an existing recipe of the same lineage)");
    }
    let hint = ["gemma", "qwen3_5", "qwen3_next"].iter().any(|h| family.contains(h));
    match (fr, fam_match) {
        (Some(r), true) if r.model.norm_offset != 0.0 => o.field(
            "norm_offset",
            &format!("{:?}", r.model.norm_offset),
            &format!("from {fsrc} (same family): zero-centred RMSNorm, x * (1 + w)"),
        ),
        (Some(r), false) if r.model.norm_offset != 0.0 => {
            o.todo(&format!(
                "model.norm_offset: closest recipe {csrc} uses norm_offset = {:?} (zero-centred RMSNorm) but this is a different family; the checkpoint does not say, check the model's RMSNorm in transformers (x * w vs x * (1 + w))",
                r.model.norm_offset
            ));
            o.field(
                "norm_offset",
                &format!("{:?}", r.model.norm_offset),
                "placeholder copied from the closest recipe, verify",
            );
        }
        _ if hint => {
            o.todo(&format!(
                "model.norm_offset: family \"{family}\" is of a lineage that uses zero-centred RMSNorm (x * (1 + w)) but no existing recipe of it was found; check the model's RMSNorm in transformers and set norm_offset = 1.0 if so"
            ));
            o.field("norm_offset", "0.0", "placeholder, verify");
        }
        _ => o.comment(
            "norm_offset: standard RMSNorm (x * w) assumed; zero-centred (x * (1 + w)) models set norm_offset = 1.0",
        ),
    }
    o.blank();

    // [layer]
    let tables: BTreeSet<&str> = ["attn", "gdn", "ffn", "moe", "norms"]
        .into_iter()
        .filter(|t| match *t {
            "attn" => det.attn.is_some() || !det.found,
            "gdn" => det.gdn.is_some(),
            "ffn" => det.ffn.is_some() || !det.found,
            "moe" => det.moe.is_some(),
            _ => true,
        })
        .collect();
    o.raw("[layer]");
    for t in &config_todos {
        o.todo(t);
    }
    for sw in &ci.sliding {
        o.todo(&format!(
            "layer: sliding-window attention ({sw}) is not implemented, a new attention block kind is needed"
        ));
    }
    for i in det.issues.iter().filter(|i| i.at == "layer" || !tables.contains(i.at)) {
        o.todo(&i.text);
    }
    let hybrid = det.types.len() > 1 && !layer_types_of(tc).is_empty();
    let pattern_of = |t: &TypeDet| t.pattern.iter().map(|b| b.to_string()).collect::<Vec<_>>();
    let default_ty = det.types.iter().find(|t| t.name == "full_attention").or_else(|| det.types.first());
    if let Some(t) = default_ty {
        let src = if !det.found {
            "placeholder".to_string()
        } else if hybrid {
            format!("from tensors of layer {} (type {}); used only for a layer without a type", t.first, t.name)
        } else {
            format!(
                "from tensors: blocks found in layer {} ({})",
                t.first,
                if t.pattern.is_empty() { "none".into() } else { t.pattern.join(" + ") }
            )
        };
        o.field("pattern", &str_array(&pattern_of(t)), &src);
    }
    if hybrid {
        o.field("pattern_by_layer", &s("layer_types"), "from text_config.layer_types");
        o.blank();
        o.raw("[layer.by_type]");
        for t in &det.types {
            o.field(
                &key(&t.name),
                &str_array(&pattern_of(t)),
                &format!("from tensors of layer {} (first layer of type {})", t.first, t.name),
            );
        }
    }
    o.blank();

    let tsrc = |layer: usize, rel: &str| -> String {
        match inv.resolve(&g.concrete(layer, rel)) {
            Some(n) => format!("from tensors: {n}"),
            None => format!("from tensors: {}", g.concrete(layer, rel)),
        }
    };

    // [layer.attn]
    if let Some((a, l)) = &det.attn {
        let l = *l;
        o.raw("[layer.attn]");
        o.issues(&det.issues, "attn");
        if ci.attention_bias && !det.issues.iter().any(|i| i.at == "attn" && i.text.contains("bias")) {
            o.todo("layer.attn: text_config.attention_bias = true; the output projection bias is not implemented (q / k / v biases are mapped with q_bias / k_bias / v_bias)");
        }
        o.field("kind", &s("gqa"), &format!("from tensors: q/k/v/o_proj under {}", g.concrete(l, &a.parent)));
        for (k, m) in [("q", &a.q), ("k", &a.k), ("v", &a.v), ("o", &a.o)] {
            o.field(k, &s(&g.template(m)), &tsrc(l, &format!("{m}.weight")));
        }
        for (key, m) in [("q_bias", &a.q_bias), ("k_bias", &a.k_bias), ("v_bias", &a.v_bias)] {
            if let Some(m) = m {
                o.field(key, &s(&format!("{}.bias", g.template(m))), &tsrc(l, &format!("{m}.bias")));
            }
        }
        match (&a.q_norm, &a.k_norm) {
            (Some(q), Some(k)) => {
                o.field("q_norm", &s(&g.template(q)), &tsrc(l, &format!("{q}.weight")));
                o.field("k_norm", &s(&g.template(k)), &tsrc(l, &format!("{k}.weight")));
            }
            (None, None) => o.comment(&format!(
                "no q_norm / k_norm tensors under {} in layer {l}: qk-norm is not applied",
                g.concrete(l, &a.parent)
            )),
            _ => {
                o.todo(&format!(
                    "layer.attn: only one of q_norm / k_norm exists under {}; the recipe needs both or neither",
                    g.concrete(l, &a.parent)
                ));
            }
        }
        match &ci.rope {
            Some((ty, from)) => {
                o.todo(&format!("layer.attn.rope: {from}; only the default rotary embedding is implemented (yarn / linear / dynamic / llama3 / longrope would silently change every position)"));
                o.field("rope", &s(ty), "placeholder, the engine refuses it at load");
            }
            None => {
                let rt = ["rope_parameters", "rope_scaling"]
                    .iter()
                    .find_map(|w| tc.get(*w).filter(|v| v.is_object()).map(|v| (format!("text_config.{w}"), v)));
                let scaled = rt.as_ref().and_then(|(_, v)| {
                    v.get("rope_type").or_else(|| v.get("type")).and_then(Value::as_str).filter(|t| matches!(*t, "llama3" | "yarn"))
                });
                o.field(
                    "rope",
                    &s("default"),
                    &match (rt.as_ref(), scaled) {
                        (Some((w, _)), Some(t)) => {
                            format!("from {w}: rope_type \"{t}\", the frequency scaling is applied from config.json at load")
                        }
                        (Some((w, _)), None) => format!("from {w} (rope_type absent or \"default\")"),
                        (None, _) => "from config.json (no rope scaling key)".to_string(),
                    },
                );
            }
        }
        if let Some(pf) = dims.as_ref().ok().map(|d| d.partial_rotary_factor).filter(|f| (*f - 1.0).abs() > 1e-6) {
            o.comment(&format!("partial rotary embeddings: {pf} of head_dim rotate (from rope_parameters.partial_rotary_factor), handled by the builder"));
        }
        if tc.get("rope_parameters").and_then(|r| r.get("mrope_section")).is_some() {
            o.comment("rope_parameters.mrope_section is set (multimodal rope): text-only input uses the same position on all axes, i.e. plain rope");
        }
        // gated query projection
        let hd = num(tc, "head_dim").or_else(|| Some(num(tc, "hidden_size")? / num(tc, "num_attention_heads")?));
        let heads = num(tc, "num_attention_heads");
        let qn = inv.resolve(&g.concrete(l, &a.q));
        let qshape = qn.as_deref().and_then(|n| inv.shape(n).map(|s| (n.to_string(), s.to_vec())));
        let mut gated: Option<(bool, String)> = None;
        if let (Some((n, sh)), Some(h), Some(d)) = (&qshape, heads, hd) {
            if !sh.is_empty() && sh[0] == 2 * h * d {
                gated = Some((
                    true,
                    format!(
                        "from tensors: {n} has {} rows = 2 x {h} heads x head_dim {d} (query + output gate)",
                        sh[0]
                    ),
                ));
            } else if !sh.is_empty() && sh[0] == h * d {
                gated = Some((false, format!("from tensors: {n} has {} rows = {h} heads x head_dim {d}", sh[0])));
            }
        }
        if gated.is_none() {
            if let Some(b) = tc.get("attn_output_gate").and_then(Value::as_bool) {
                gated = Some((b, "from text_config.attn_output_gate".into()));
            }
        }
        if gated.is_none() {
            if let Some(b) = cr.and_then(|r| r.layer.attn.as_ref()).map(|a| a.gated) {
                gated = Some((b, format!("from closest recipe {csrc}; no q_proj shape (no shard headers) and no attn_output_gate in the config, verify")));
            }
        }
        match gated {
            Some((true, src)) => o.field("gated", "true", &src),
            Some((false, src)) => o.comment(&format!("gated = false ({src})")),
            None => o.comment("gated = false (default): no q_proj shape and no attn_output_gate in the config"),
        }
        o.blank();
    } else if !det.found {
        o.raw("[layer.attn]");
        o.field("kind", &s("gqa"), "placeholder");
        for k in ["q", "k", "v", "o"] {
            o.field(k, &s(&format!("model.layers.{{i}}.self_attn.{k}_proj")), "placeholder");
        }
        o.blank();
    }

    // [layer.gdn]
    if let Some((x, l)) = &det.gdn {
        let l = *l;
        o.raw("[layer.gdn]");
        o.issues(&det.issues, "gdn");
        o.field(
            "kind",
            &s("gated_delta_net"),
            &format!(
                "from tensors: in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / conv1d / A_log / dt_bias under {}",
                g.concrete(l, &x.parent)
            ),
        );
        for (k, m) in [
            ("in_qkv", &x.in_qkv),
            ("in_z", &x.in_z),
            ("in_b", &x.in_b),
            ("in_a", &x.in_a),
            ("conv", &x.conv),
            ("a_log", &x.a_log),
            ("dt_bias", &x.dt_bias),
            ("norm", &x.norm),
            ("out", &x.out),
        ] {
            let bare = matches!(k, "a_log" | "dt_bias");
            o.field(k, &s(&g.template(m)), &tsrc(l, &if bare { m.clone() } else { format!("{m}.weight") }));
        }
        o.blank();
    }

    // [layer.ffn]
    if let Some((f, l)) = &det.ffn {
        let l = *l;
        o.raw("[layer.ffn]");
        o.issues(&det.issues, "ffn");
        o.field(
            "kind",
            &s("dense"),
            &format!("from tensors: gate_proj / up_proj / down_proj under {}", g.concrete(l, &f.parent)),
        );
        match &ci.act {
            Some(a) => {
                o.todo(&format!("layer.ffn.act: text_config.hidden_act = \"{a}\"; only silu (silu_glu) is implemented, a new activation is needed in ops.rs and the ffn block"));
                o.field("act", &s(&format!("{a}_glu")), "placeholder, the engine refuses it at load");
            }
            None => {
                let src = if tc.get("hidden_act").is_some() {
                    "from text_config.hidden_act = \"silu\""
                } else {
                    "default: config has no hidden_act, silu assumed"
                };
                o.field("act", &s("silu_glu"), src);
            }
        }
        for (k, m) in [("gate", &f.gate), ("up", &f.up), ("down", &f.down)] {
            o.field(k, &s(&g.template(m)), &tsrc(l, &format!("{m}.weight")));
        }
        o.blank();
    } else if !det.found {
        o.raw("[layer.ffn]");
        o.field("kind", &s("dense"), "placeholder");
        o.field("act", &s("silu_glu"), "placeholder");
        for k in ["gate", "up", "down"] {
            o.field(k, &s(&format!("model.layers.{{i}}.mlp.{k}_proj")), "placeholder");
        }
        o.blank();
    }

    // [layer.moe]
    if let Some((m, l)) = &det.moe {
        let l = *l;
        o.raw("[layer.moe]");
        o.issues(&det.issues, "moe");
        if let (Some(a), true) = (&ci.act, det.ffn.is_none()) {
            o.todo(&format!("layer.moe: text_config.hidden_act = \"{a}\"; the moe block applies silu only"));
        }
        let shared = inv.has(&g.concrete(l, &format!("{}.shared_expert.gate_proj.weight", m.parent)));
        o.field(
            "kind",
            &s("qwen3_5_moe"),
            &format!(
                "from tensors: {}.experts.<e>.* (routed experts){}",
                g.concrete(l, &m.parent),
                if shared { " + shared_expert + shared_expert_gate" } else { "" }
            ),
        );
        o.field(
            "prefix",
            &s(&g.template(&m.parent)),
            &format!("from tensors: {}.gate.weight (router)", g.concrete(l, &m.parent)),
        );
        if !shared {
            o.comment("no shared_expert tensors: the block runs without a shared expert only if the config has no shared_expert_intermediate_size");
        }
        o.blank();
    }

    // [layer.norms]
    o.raw("[layer.norms]");
    o.issues(&det.issues, "norms");
    let mut keys: Vec<String> = det.types.iter().flat_map(|t| t.pattern.iter().map(|b| format!("pre_{b}"))).collect();
    keys.sort();
    keys.dedup();
    for k in keys {
        match det.pre.get(&k) {
            Some((m, l)) => o.field(&k, &s(&g.template(m)), &tsrc(*l, &format!("{m}.weight"))),
            None => {
                if det.found
                    && !det.issues.iter().any(|i| i.at == "norms" && i.text.contains(&format!("layer.norms.{k}:")))
                {
                    o.todo(&format!("layer.norms.{k}: no norm tensor found for this block"));
                }
                if !det.found {
                    o.field(&k, &s(&format!("model.layers.{{i}}.{k}")), "placeholder");
                } else {
                    o.field(&k, &s(&format!("{}.{{i}}.TODO_{k}", g.prefix)), "placeholder");
                }
            }
        }
    }
    o.blank();

    // [top]
    o.raw("[top]");
    let base = g.base().to_string();
    let find = |cands: Vec<String>, suffixes: &[&str]| -> Option<String> {
        cands.into_iter().find(|c| inv.resolve(c).is_some()).or_else(|| {
            inv.names
                .iter()
                .filter(|n| !is_non_text(n) && layer_split(n).is_none())
                .find(|n| suffixes.iter().any(|s| n.ends_with(s)))
                .map(|n| n.trim_end_matches(".weight").to_string())
        })
    };
    let pre = |x: &str| if base.is_empty() { x.to_string() } else { format!("{base}.{x}") };
    let embed = find(
        vec![pre("embed_tokens"), pre("wte"), pre("tok_embeddings"), pre("embed_in"), "model.embed_tokens".into()],
        &["embed_tokens.weight", ".wte.weight", "tok_embeddings.weight"],
    );
    let fnorm = find(
        vec![pre("norm"), pre("final_layernorm"), pre("ln_f"), pre("norm_f"), pre("final_norm"), "model.norm".into()],
        &[".final_layernorm.weight", ".ln_f.weight", ".norm_f.weight"],
    );
    let mut embed_name = pre("embed_tokens");
    match &embed {
        Some(e) => {
            embed_name = e.clone();
            o.field("embed", &s(e), &format!("from tensors: {}", inv.resolve(e).unwrap_or_else(|| e.clone())));
        }
        None => {
            o.todo(&format!("top.embed: no embedding tensor ({}.weight or similar) in the checkpoint; name the token embedding tensor", embed_name));
            o.field("embed", &s(&embed_name), "placeholder, tensor missing from the checkpoint");
        }
    }
    match &fnorm {
        Some(n) => {
            o.field("final_norm", &s(n), &format!("from tensors: {}", inv.resolve(n).unwrap_or_else(|| n.clone())))
        }
        None => {
            o.todo(&format!(
                "top.final_norm: no final norm tensor ({} or similar) in the checkpoint; name it",
                pre("norm.weight")
            ));
            o.field("final_norm", &s(&pre("norm")), "placeholder, tensor missing from the checkpoint");
        }
    }
    let lm_src = match head {
        Some(h) => format!("from tensors: {}; tied_to is used when the tensor is absent or the config ties the embeddings", inv.resolve(h).unwrap_or_else(|| h.to_string())),
        None if tie => "lm_head is absent from the checkpoint and text_config.tie_word_embeddings = true, so it is tied to the embedding table".to_string(),
        None => "placeholder, lm_head tensor missing from the checkpoint".to_string(),
    };
    if head.is_none() && !tie {
        o.todo("top.lm_head: no lm_head tensor in the checkpoint and tie_word_embeddings is false or absent in the config; the loader will refuse it. Name the head tensor or set tie_word_embeddings");
    }
    o.field(
        "lm_head",
        &format!("{{ tensor = {}, tied_to = {} }}", s(head.unwrap_or("lm_head")), s(&embed_name)),
        &lm_src,
    );
    // tensors outside the layer stack that nothing claims
    let mut claimed_top: BTreeSet<String> = BTreeSet::new();
    for m in [Some(embed_name.clone()), fnorm.clone(), head.map(String::from)].into_iter().flatten() {
        claimed_top.insert(m);
    }
    let mut stray: Vec<String> = Vec::new();
    let mut ignored: BTreeMap<String, usize> = BTreeMap::new();
    for n in &inv.names {
        if det.found && n.starts_with(&format!("{}.", g.prefix)) {
            continue;
        }
        if n.ends_with("inv_freq") {
            continue;
        }
        let (m, _) = split_param(n);
        if claimed_top.contains(m) {
            continue;
        }
        if is_non_text(n) {
            let k = n.split('.').take(2).collect::<Vec<_>>().join(".");
            *ignored.entry(k).or_default() += 1;
            continue;
        }
        stray.push(n.clone());
    }
    if !ignored.is_empty() {
        let list = ignored.iter().map(|(k, c)| format!("{k}.* ({c})")).collect::<Vec<_>>().join(", ");
        o.comment(&format!("ignored (not part of the text decoder): {list}"));
        notes.push(format!("ignored non-text tensor groups: {list}"));
    }
    if !stray.is_empty() {
        let shown: Vec<&String> = stray.iter().take(5).collect();
        o.todo(&format!(
            "top: {} checkpoint tensor(s) outside the layer stack are claimed by no block, e.g. {shown:?}{}; decide whether the model needs them",
            stray.len(),
            if stray.len() > 5 { " ..." } else { "" }
        ));
    }
    if !det.unclaimed.is_empty() {
        let shown: Vec<&String> = det.unclaimed.iter().take(6).collect();
        o.todo(&format!(
            "layer: {} per-layer tensor(s) are claimed by no block, e.g. {shown:?}{} (relative to {}.<i>.); extra norms, biases or unknown modules need a block or an explicit decision to ignore them",
            det.unclaimed.len(),
            if det.unclaimed.len() > 6 { " ..." } else { "" },
            g.prefix
        ));
    }
    o.blank();

    // [defaults]
    o.raw("[defaults]");
    for p in &quant.problems {
        o.todo(p);
    }
    o.field("kv_dtype", &s("bf16"), "default: bf16 KV cache (fp8_e4m3 KV is accepted by the cpu oracle only)");
    let same_blocks = cr.is_some_and(|r| {
        let inferred: BTreeSet<String> =
            det.types.iter().flat_map(|t| t.pattern.iter().map(|b| b.to_string())).collect();
        recipe_blocks(r) == inferred
    });
    match (cr, same_blocks) {
        (Some(r), true) => o.field(
            "max_batch",
            &r.defaults.max_batch.to_string(),
            &format!("from closest recipe {csrc} (same block set)"),
        ),
        _ => o.field("max_batch", "1", "default; raise after measuring"),
    }
    (o.text, o.todos, notes)
}

fn layer_types_of(tc: &Value) -> Vec<String> {
    tc.get("layer_types")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use intarsia_models::recipe::{parse_recipe, select_recipe};
    use serde_json::json;

    struct T {
        name: String,
        dtype: &'static str,
        shape: Vec<usize>,
    }

    fn t(name: impl Into<String>, dtype: &'static str, shape: &[usize]) -> T {
        T { name: name.into(), dtype, shape: shape.to_vec() }
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("intarsia-recipe-new-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn dsz(d: &str) -> usize {
        match d {
            "BF16" | "F16" => 2,
            "F32" => 4,
            _ => 1,
        }
    }

    /// A real (tiny, zero-filled) safetensors file.
    fn write_st(path: &Path, ts: &[T]) {
        let mut header = serde_json::Map::new();
        let mut off = 0;
        for x in ts {
            let n = x.shape.iter().product::<usize>() * dsz(x.dtype);
            header.insert(x.name.clone(), json!({"dtype": x.dtype, "shape": x.shape, "data_offsets": [off, off + n]}));
            off += n;
        }
        let h = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut f = Vec::new();
        f.extend((h.len() as u64).to_le_bytes());
        f.extend(h);
        f.resize(f.len() + off, 0);
        std::fs::write(path, f).unwrap();
    }

    /// `model.safetensors.index.json` only (no shards).
    fn write_index(dir: &Path, ts: &[T]) {
        let wm: serde_json::Map<String, Value> =
            ts.iter().map(|x| (x.name.clone(), json!("model-00001-of-00001.safetensors"))).collect();
        std::fs::write(dir.join("model.safetensors.index.json"), json!({"metadata": {}, "weight_map": wm}).to_string())
            .unwrap();
    }

    fn write_json(dir: &Path, name: &str, v: &Value) {
        std::fs::write(dir.join(name), serde_json::to_string_pretty(v).unwrap()).unwrap();
    }

    // dims used by every fixture: hidden 8, 2 heads x head_dim 4, 1 kv head, ffn 16, vocab 10
    fn qwen3_layer(p: &str, l: usize, out: &mut Vec<T>) {
        let b = format!("{p}.{l}");
        for (n, sh) in [
            ("self_attn.q_proj", [8, 8]),
            ("self_attn.k_proj", [4, 8]),
            ("self_attn.v_proj", [4, 8]),
            ("self_attn.o_proj", [8, 8]),
            ("mlp.gate_proj", [16, 8]),
            ("mlp.up_proj", [16, 8]),
            ("mlp.down_proj", [8, 16]),
        ] {
            out.push(t(format!("{b}.{n}.weight"), "BF16", &sh));
        }
        for n in ["self_attn.q_norm", "self_attn.k_norm"] {
            out.push(t(format!("{b}.{n}.weight"), "BF16", &[4]));
        }
        for n in ["input_layernorm", "post_attention_layernorm"] {
            out.push(t(format!("{b}.{n}.weight"), "BF16", &[8]));
        }
    }

    fn qwen3_tensors(n_layers: usize) -> Vec<T> {
        let mut v = vec![t("model.embed_tokens.weight", "BF16", &[10, 8]), t("model.norm.weight", "BF16", &[8])];
        for l in 0..n_layers {
            qwen3_layer("model.layers", l, &mut v);
        }
        v
    }

    fn qwen3_config(arch: &str, model_type: &str) -> Value {
        json!({
            "architectures": [arch], "model_type": model_type, "hidden_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16,
            "vocab_size": 10, "rms_norm_eps": 1e-6, "max_position_embeddings": 128, "rope_theta": 1000000,
            "hidden_act": "silu", "attention_bias": false, "use_sliding_window": false, "sliding_window": null,
            "rope_scaling": null, "tie_word_embeddings": true, "torch_dtype": "bfloat16"
        })
    }

    fn same_structure(a: &Recipe, b: &Recipe) {
        assert_eq!(format!("{:?}", a.layer), format!("{:?}", b.layer));
        assert_eq!(format!("{:?}", a.top), format!("{:?}", b.top));
        assert_eq!(a.model.norm_offset, b.model.norm_offset);
        assert_eq!(a.model.family, b.model.family);
        assert_eq!(a.model.reasoning_parser, b.model.reasoning_parser);
        assert_eq!(a.model.tool_parser, b.model.tool_parser);
        assert_eq!(a.model.chat_template, b.model.chat_template);
        assert_eq!(format!("{:?}", a.defaults), format!("{:?}", b.defaults));
    }

    fn builtin(name: &str) -> Recipe {
        parse_recipe(intarsia_models::recipe::BUILTIN_RECIPES.iter().find(|(n, _)| *n == name).unwrap().1).unwrap()
    }

    /// Writes the scaffold into a fresh recipe dir and checks that `select_recipe` really picks that file.
    fn reload_picks(an: &Analysis, cfg: &Value, name: &str) -> (Recipe, PathBuf) {
        let rd = tmpdir(name);
        let path = rd.join(format!("{}.toml", an.family));
        std::fs::write(&path, &an.toml).unwrap();
        let (src, r) = select_recipe(cfg, &[rd.clone()]).unwrap();
        assert_eq!(src, path.display().to_string(), "the scaffold must win over the built-ins");
        (r, rd)
    }

    #[test]
    fn dense_qwen3_dir_is_covered_and_scaffold_equals_the_shipped_recipe() {
        let d = tmpdir("dense");
        let cfg = qwen3_config("Qwen3ForCausalLM", "qwen3");
        write_json(&d, "config.json", &cfg);
        write_st(&d.join("model.safetensors"), &qwen3_tensors(2));
        let an = analyse(&d, &[]).unwrap();
        assert_eq!(an.covered_by.as_deref(), Some("builtin:qwen3"));
        assert_eq!(an.quant, "bf16");
        assert_eq!(an.family, "qwen3");
        assert_eq!(an.tensor_templates_resolved.0, an.tensor_templates_resolved.1);
        assert_eq!(an.tensor_templates_resolved, (25, 25));
        assert!(an.todos.is_empty(), "{:?}", an.todos);
        // the scaffold (built although nothing is needed) says the same as the shipped recipe
        same_structure(&parse_recipe(&an.toml).unwrap(), &builtin("qwen3"));
        let (r, _) = reload_picks(&an, &cfg, "dense-reload");
        same_structure(&r, &builtin("qwen3"));
        // every key = value line carries a source comment
        for l in an.toml.lines().filter(|l| l.contains(" = ") && !l.trim_start().starts_with('#')) {
            assert!(l.contains("  # "), "field without a source comment: {l}");
        }
    }

    #[test]
    fn covered_without_out_writes_nothing_and_out_is_not_overwritten_without_force() {
        let d = tmpdir("covered-run");
        write_json(&d, "config.json", &qwen3_config("Qwen3ForCausalLM", "qwen3"));
        write_st(&d.join("model.safetensors"), &qwen3_tensors(2));
        let args = |out: Option<PathBuf>, force| NewArgs { model: d.clone(), out, recipe_dir: vec![], force };
        let before = Path::new("recipes/qwen3.toml").exists();
        new_cmd(args(None, false)).unwrap();
        assert_eq!(Path::new("recipes/qwen3.toml").exists(), before, "nothing may be written without --out");
        let out = d.join("out").join("q.toml");
        new_cmd(args(Some(out.clone()), false)).unwrap();
        assert!(out.is_file());
        let e = new_cmd(args(Some(out.clone()), false)).unwrap_err().to_string();
        assert!(e.contains("--force"), "{e}");
        new_cmd(args(Some(out), true)).unwrap();
    }

    #[test]
    fn unknown_architecture_with_qwen3_layout_gets_a_clean_scaffold() {
        let d = tmpdir("foo");
        let cfg = qwen3_config("FooForCausalLM", "foo");
        write_json(&d, "config.json", &cfg);
        write_st(&d.join("model-00001-of-00002.safetensors"), &qwen3_tensors(1)[..4]);
        let rest: Vec<T> = qwen3_tensors(2).into_iter().skip(4).collect();
        write_st(&d.join("model-00002-of-00002.safetensors"), &rest);
        let an = analyse(&d, &[]).unwrap();
        assert_eq!(an.covered_by, None);
        assert_eq!(an.family, "foo");
        assert!(an.todos.is_empty(), "{:?}", an.todos);
        assert!(
            an.notes
                .iter()
                .any(|n| n.contains("closest existing recipe: builtin:qwen3") && n.contains("architecture differs")),
            "{:?}",
            an.notes
        );
        let (r, rd) = reload_picks(&an, &cfg, "foo-reload");
        assert_eq!(r.model.family, "foo");
        assert_eq!(r.model.matcher.architectures, vec!["FooForCausalLM"]);
        // a family-less copy of the qwen3 structure; the qwen3 parsers are not copied to a different family
        assert_eq!(format!("{:?}", r.layer), format!("{:?}", builtin("qwen3").layer));
        assert_eq!(r.model.reasoning_parser, None);
        // a second analysis with the scaffold in --recipe-dir finds it covering the model
        let an2 = analyse(&d, &[rd.clone()]).unwrap();
        assert_eq!(an2.covered_by.as_deref(), Some(rd.join("foo.toml").display().to_string().as_str()));
    }

    #[test]
    fn llama_like_model_with_gelu_dynamic_rope_bias_and_unknown_quant_lists_every_todo() {
        let d = tmpdir("llama");
        let mut cfg = qwen3_config("LlamaForCausalLM", "llama");
        cfg["hidden_act"] = json!("gelu");
        cfg["attention_bias"] = json!(true);
        cfg["rope_scaling"] = json!({"rope_type": "dynamic", "factor": 4.0});
        cfg["quantization_config"] = json!({"quant_method": "gptq", "bits": 4});
        write_json(&d, "config.json", &cfg);
        let mut ts = qwen3_tensors(2);
        ts.retain(|x| !x.name.contains("q_norm") && !x.name.contains("k_norm"));
        for l in 0..2 {
            ts.push(t(format!("model.layers.{l}.self_attn.q_proj.bias"), "BF16", &[8]));
            ts.push(t(format!("model.layers.{l}.pre_feedforward_layernorm.weight"), "BF16", &[8]));
        }
        ts.push(t("model.rotary_emb.extra", "F32", &[2]));
        write_st(&d.join("model.safetensors"), &ts);
        let an = analyse(&d, &[]).unwrap();
        assert_eq!(an.covered_by, None);
        let all = an.todos.join("\n");
        for needle in [
            "hidden_act = \"gelu\"",
            "rope_type = \"dynamic\"",
            "bias",
            "quant_method = \"gptq\"",
            "post_attention_layernorm",
            "model.rotary_emb.extra",
        ] {
            assert!(all.contains(needle), "missing TODO about {needle}:\n{all}");
        }
        for todo in &an.todos {
            assert!(!todo.contains('\n'));
            assert!(an.toml.contains(&format!("# TODO(recipe-new): {todo}")), "marker missing for: {todo}");
        }
        assert_eq!(an.toml.lines().filter(|l| l.starts_with("# TODO(recipe-new): ")).count(), an.todos.len());
        // placeholders keep the file loadable by the parser
        parse_recipe(&an.toml).unwrap();
        // and `run` writes it, lists the TODOs and fails
        let out = d.join("llama.toml");
        let e = new_cmd(NewArgs { model: d.clone(), out: Some(out.clone()), recipe_dir: vec![], force: false })
            .unwrap_err()
            .to_string();
        assert_eq!(
            e,
            format!(
                "{} item(s) need manual work, see the TODO(recipe-new) markers in {}",
                an.todos.len(),
                out.display()
            )
        );
        assert_eq!(std::fs::read_to_string(&out).unwrap(), an.toml);
    }

    /// Qwen2-style attention biases (q / k / v only, no config key) and llama3 rope scaling scaffold without a TODO.
    #[test]
    fn qkv_biases_and_llama3_rope_scaffold_cleanly() {
        let d = tmpdir("qwen2like");
        let mut cfg = qwen3_config("Qwen2ForCausalLM", "qwen2");
        cfg.as_object_mut().unwrap().remove("attention_bias");
        cfg["rope_scaling"] = json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64});
        write_json(&d, "config.json", &cfg);
        let mut ts = qwen3_tensors(2);
        ts.retain(|x| !x.name.contains("q_norm") && !x.name.contains("k_norm"));
        for l in 0..2 {
            ts.push(t(format!("model.layers.{l}.self_attn.q_proj.bias"), "BF16", &[8]));
            ts.push(t(format!("model.layers.{l}.self_attn.k_proj.bias"), "BF16", &[4]));
            ts.push(t(format!("model.layers.{l}.self_attn.v_proj.bias"), "BF16", &[4]));
        }
        write_st(&d.join("model.safetensors"), &ts);
        let an = analyse(&d, &[]).unwrap();
        assert!(an.todos.is_empty(), "{:?}", an.todos);
        for needle in [
            "q_bias = \"model.layers.{i}.self_attn.q_proj.bias\"",
            "k_bias = \"model.layers.{i}.self_attn.k_proj.bias\"",
            "v_bias = \"model.layers.{i}.self_attn.v_proj.bias\"",
        ] {
            assert!(an.toml.contains(needle), "missing {needle}:\n{}", an.toml);
        }
        assert!(an.toml.contains("rope_type \"llama3\""), "{}", an.toml);
        let r = parse_recipe(&an.toml).unwrap();
        let a = r.layer.attn.as_ref().unwrap();
        assert_eq!(a.q_bias.as_deref(), Some("model.layers.{i}.self_attn.q_proj.bias"));
        assert!(a.q_norm.is_none() && a.k_norm.is_none());
    }

    #[test]
    fn a_layer_that_differs_from_the_last_one_is_flagged() {
        let d = tmpdir("ragged");
        write_json(&d, "config.json", &qwen3_config("Qwen3ForCausalLM", "qwen3"));
        let mut ts = qwen3_tensors(2);
        ts.retain(|x| x.name != "model.layers.1.self_attn.k_norm.weight");
        write_st(&d.join("model.safetensors"), &ts);
        let an = analyse(&d, &[]).unwrap();
        assert_eq!(an.covered_by, None, "a missing tensor template must stop the existing recipe from covering");
        assert!(an.notes.iter().any(|n| n.contains("1 tensor template(s) missing")), "{:?}", an.notes);
        assert!(an.todos.iter().any(|x| x.contains("layers 0 and 1") && x.contains("k_norm")), "{:?}", an.todos);
    }

    fn hybrid_layer(l: usize, full: bool, out: &mut Vec<T>) {
        let b = format!("model.layers.{l}");
        if full {
            for (n, sh) in [("q_proj", [16, 8]), ("k_proj", [4, 8]), ("v_proj", [4, 8]), ("o_proj", [8, 8])] {
                out.push(t(format!("{b}.self_attn.{n}.weight"), "BF16", &sh));
            }
            for n in ["q_norm", "k_norm"] {
                out.push(t(format!("{b}.self_attn.{n}.weight"), "BF16", &[4]));
            }
        } else {
            for (n, sh) in [
                ("in_proj_qkv", vec![16, 8]),
                ("in_proj_z", vec![8, 8]),
                ("in_proj_b", vec![2, 8]),
                ("in_proj_a", vec![2, 8]),
                ("conv1d", vec![16, 1, 4]),
                ("out_proj", vec![8, 8]),
            ] {
                out.push(t(format!("{b}.linear_attn.{n}.weight"), "BF16", &sh));
            }
            out.push(t(format!("{b}.linear_attn.A_log"), "F32", &[2]));
            out.push(t(format!("{b}.linear_attn.dt_bias"), "BF16", &[2]));
            out.push(t(format!("{b}.linear_attn.norm.weight"), "F32", &[4]));
        }
        for (n, sh) in [("gate_proj", [16, 8]), ("up_proj", [16, 8]), ("down_proj", [8, 16])] {
            out.push(t(format!("{b}.mlp.{n}.weight"), "BF16", &sh));
        }
        for n in ["input_layernorm", "post_attention_layernorm"] {
            out.push(t(format!("{b}.{n}.weight"), "BF16", &[8]));
        }
    }

    #[test]
    fn hybrid_dense_dir_is_covered_by_qwen3_5() {
        let d = tmpdir("hybrid");
        let types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"];
        let cfg = json!({
            "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5", "hidden_size": 8, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16, "vocab_size": 10,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 128, "hidden_act": "silu", "tie_word_embeddings": true,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default", "partial_rotary_factor": 0.5},
            "layer_types": types, "linear_num_key_heads": 1, "linear_num_value_heads": 2, "linear_key_head_dim": 4,
            "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4, "dtype": "bfloat16"
        });
        write_json(&d, "config.json", &cfg);
        let mut ts = vec![t("model.embed_tokens.weight", "BF16", &[10, 8]), t("model.norm.weight", "BF16", &[8])];
        for (l, ty) in types.iter().enumerate() {
            hybrid_layer(l, *ty == "full_attention", &mut ts);
        }
        write_st(&d.join("model.safetensors"), &ts);
        let an = analyse(&d, &[]).unwrap();
        assert_eq!(an.covered_by.as_deref(), Some("builtin:qwen3_5"), "{:?}", an.notes);
        assert!(an.todos.is_empty(), "{:?}", an.todos);
        // the gated q_proj (2 x heads x head_dim rows) is read from the tensor shape
        assert!(an.toml.contains("2 x 2 heads x head_dim 4"), "{}", an.toml);
        same_structure(&parse_recipe(&an.toml).unwrap(), &builtin("qwen3_5"));
        reload_picks(&an, &cfg, "hybrid-reload");
        // a q_proj of the plain width contradicts a gated recipe
        let mut ts2 = ts;
        for x in ts2.iter_mut().filter(|x| x.name == "model.layers.3.self_attn.q_proj.weight") {
            x.shape = vec![8, 8];
        }
        write_st(&d.join("model.safetensors"), &ts2);
        let an2 = analyse(&d, &[]).unwrap();
        assert_eq!(an2.covered_by, None);
        assert!(an2.notes.iter().any(|n| n.contains("attn.gated = true")), "{:?}", an2.notes);
    }

    fn moe_fixture(name: &str, shared_algo: &str) -> (PathBuf, Value) {
        let d = tmpdir(name);
        let types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"];
        let tc = json!({
            "model_type": "qwen3_5_moe_text", "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 10, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128, "hidden_act": "silu", "attn_output_gate": true,
            "rope_parameters": {"rope_theta": 1e7, "rope_type": "default", "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]},
            "layer_types": types, "linear_num_key_heads": 1, "linear_num_value_heads": 2, "linear_key_head_dim": 4,
            "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4, "num_experts": 2, "num_experts_per_tok": 1,
            "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16, "dtype": "bfloat16"
        });
        let cfg = json!({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
            "tie_word_embeddings": false, "text_config": tc});
        write_json(&d, "config.json", &cfg);
        let p = "model.language_model.layers";
        let mut ts: Vec<T> = Vec::new();
        let fp8 = |ts: &mut Vec<T>, m: String, sh: &[usize]| {
            ts.push(t(format!("{m}.weight"), "F8_E4M3", sh));
            ts.push(t(format!("{m}.weight_scale"), "F32", &[]));
            ts.push(t(format!("{m}.input_scale"), "F32", &[]));
        };
        let nv = |ts: &mut Vec<T>, m: String, n: usize, k: usize| {
            ts.push(t(format!("{m}.weight"), "U8", &[n, k / 2]));
            ts.push(t(format!("{m}.weight_scale"), "F8_E4M3", &[n, k / 16]));
            ts.push(t(format!("{m}.weight_scale_2"), "F32", &[]));
        };
        for (l, ty) in types.iter().enumerate() {
            let b = format!("{p}.{l}");
            if *ty == "full_attention" {
                for (n, sh) in [("q_proj", [16, 16]), ("k_proj", [4, 16]), ("v_proj", [4, 16]), ("o_proj", [16, 8])] {
                    fp8(&mut ts, format!("{b}.self_attn.{n}"), &sh);
                }
                for n in ["q_norm", "k_norm"] {
                    ts.push(t(format!("{b}.self_attn.{n}.weight"), "BF16", &[4]));
                }
            } else {
                for (n, sh) in [("in_proj_qkv", [16, 16]), ("in_proj_z", [8, 16]), ("out_proj", [16, 8])] {
                    fp8(&mut ts, format!("{b}.linear_attn.{n}"), &sh);
                }
                for (n, sh) in [("in_proj_b", vec![2, 16]), ("in_proj_a", vec![2, 16]), ("conv1d", vec![16, 1, 4])] {
                    ts.push(t(format!("{b}.linear_attn.{n}.weight"), "BF16", &sh));
                }
                ts.push(t(format!("{b}.linear_attn.A_log"), "F32", &[2]));
                ts.push(t(format!("{b}.linear_attn.dt_bias"), "BF16", &[2]));
                ts.push(t(format!("{b}.linear_attn.norm.weight"), "F32", &[4]));
            }
            ts.push(t(format!("{b}.mlp.gate.weight"), "BF16", &[2, 16]));
            ts.push(t(format!("{b}.mlp.shared_expert_gate.weight"), "BF16", &[1, 16]));
            for x in 0..2 {
                for pr in ["gate_proj", "up_proj", "down_proj"] {
                    nv(&mut ts, format!("{b}.mlp.experts.{x}.{pr}"), 16, 16);
                }
            }
            for pr in ["gate_proj", "up_proj", "down_proj"] {
                nv(&mut ts, format!("{b}.mlp.shared_expert.{pr}"), 16, 16);
            }
            for n in ["input_layernorm", "post_attention_layernorm"] {
                ts.push(t(format!("{b}.{n}.weight"), "BF16", &[16]));
            }
        }
        ts.push(t("model.language_model.embed_tokens.weight", "BF16", &[10, 16]));
        ts.push(t("model.language_model.norm.weight", "BF16", &[16]));
        nv(&mut ts, "lm_head".into(), 10, 16);
        ts.push(t("model.visual.blocks.0.attn.qkv.weight", "BF16", &[4, 4]));
        ts.push(t("model.visual.merger.linear.weight", "BF16", &[4, 4]));
        ts.push(t("mtp.layers.0.self_attn.q_proj.weight", "BF16", &[4, 4]));
        ts.push(t("mtp.fc.weight", "BF16", &[4, 4]));
        write_index(&d, &ts);
        write_json(
            &d,
            "hf_quant_config.json",
            &json!({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8",
                "exclude_modules": ["mtp*"],
                "quantized_layers": {
                    format!("{p}.0.linear_attn.out_proj"): {"quant_algo": "FP8"},
                    format!("{p}.0.mlp.shared_expert.gate_proj"): {"quant_algo": shared_algo, "group_size": 16},
                    "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}}}),
        );
        (d, cfg)
    }

    #[test]
    fn mixed_fp8_nvfp4_moe_from_an_index_alone_is_covered_by_qwen3_5_moe() {
        let (d, cfg) = moe_fixture("moe", "W4A16_NVFP4");
        let an = analyse(&d, &[]).unwrap();
        assert_eq!(an.covered_by.as_deref(), Some("builtin:qwen3_5_moe"), "{:?}", an.notes);
        assert_eq!(an.quant, "nvfp4+fp8");
        assert_eq!(an.family, "qwen3_5_moe");
        assert_eq!(an.tensor_templates_resolved.0, an.tensor_templates_resolved.1);
        // the kv_cache_quant_algo is a note, vision tower and MTP head are noted as ignored, neither is a TODO
        assert!(an.notes.iter().any(|n| n.contains("kv_cache_quant_algo FP8")), "{:?}", an.notes);
        assert!(an.notes.iter().any(|n| n.contains("model.visual.*") && n.contains("mtp.layers.*")), "{:?}", an.notes);
        assert!(
            an.notes.iter().any(|n| n.contains("model.language_model.layers.{i}") && n.contains("mtp.layers")),
            "{:?}",
            an.notes
        );
        assert!(an.todos.is_empty(), "{:?}", an.todos);
        let r = parse_recipe(&an.toml).unwrap();
        let shipped = builtin("qwen3_5_moe");
        same_structure(&r, &shipped);
        assert_eq!(r.model.matcher.architectures, vec!["Qwen3_5MoeForConditionalGeneration"]);
        reload_picks(&an, &cfg, "moe-reload");
    }

    #[test]
    fn declared_w4a4_nvfp4_is_not_covered() {
        let (d, _) = moe_fixture("moe-w4a4", "NVFP4");
        let an = analyse(&d, &[]).unwrap();
        assert_eq!(an.covered_by, None);
        assert!(an.todos.iter().any(|x| x.contains("NVFP4") && x.contains("W4A16_NVFP4")), "{:?}", an.todos);
        parse_recipe(&an.toml).unwrap();
    }

    #[test]
    fn header_reader_and_layer_split() {
        let d = tmpdir("hdr");
        write_st(&d.join("a.safetensors"), &[t("x.weight", "BF16", &[2, 3]), t("y", "F32", &[])]);
        let mut h = read_header(&d.join("a.safetensors")).unwrap();
        h.sort();
        assert_eq!(h, vec![("x.weight".into(), "BF16".into(), vec![2, 3]), ("y".into(), "F32".into(), vec![])]);
        std::fs::write(d.join("bad.safetensors"), [255u8; 16]).unwrap();
        assert!(read_header(&d.join("bad.safetensors")).is_err());
        assert_eq!(
            layer_split("model.layers.7.self_attn.q_proj.weight"),
            Some(("model.layers".into(), 7, "self_attn.q_proj.weight".into()))
        );
        assert_eq!(layer_split("model.language_model.layers.0.mlp.experts.5.down_proj.weight").unwrap().1, 0);
        assert_eq!(layer_split("model.embed_tokens.weight"), None);
        assert_eq!(split_param("mlp.gate_proj.weight_scale_2"), ("mlp.gate_proj", "weight_scale_2"));
        assert_eq!(split_param("linear_attn.A_log"), ("linear_attn.A_log", ""));
    }

    #[test]
    fn missing_config_or_tensors_is_an_error_not_a_scaffold() {
        let d = tmpdir("empty");
        assert!(analyse(&d, &[]).is_err());
        write_json(&d, "config.json", &qwen3_config("Qwen3ForCausalLM", "qwen3"));
        let e = analyse(&d, &[]).unwrap_err().to_string();
        assert!(e.contains("safetensors"), "{e}");
    }
}

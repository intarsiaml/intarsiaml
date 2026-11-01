//! DeepSeek-V4.1 checkpoints (lane 1a skeleton): tensor shapes from the safetensors headers alone, and the `plan show`
//! entry point. Reading only headers means a header-only copy of a checkpoint (each shard truncated after its JSON
//! header: 20 MB for the 196 GiB Mia EXL3 checkpoint) plans exactly like the real one.

use intarsia_core::dsv41::config::Dsv41Config;
use intarsia_core::dsv41::graph::WeightShapes;
use intarsia_core::{Backend, DType, Error, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// Prefill rows `plan show` records besides the decode step.
pub const PLAN_PREFILL_ROWS: usize = 300;

/// `name -> (dtype, shape)` of every tensor of a checkpoint directory, from the headers of `model.safetensors` or the
/// shards `model.safetensors.index.json` lists. Dtypes the engine has no `DType` for map to their storage width
/// (`F8_E8M0` scales -> u8).
pub struct HeaderShapes(pub BTreeMap<String, (DType, Vec<usize>)>);

fn dtype_of(s: &str) -> Result<DType> {
    match s {
        "F8_E8M0" => Ok(DType::U8),
        other => DType::from_safetensors(other),
    }
}

/// The JSON header of one safetensors file (only the header is read).
pub fn read_header(path: &Path) -> Result<serde_json::Map<String, Value>> {
    let mut f = std::fs::File::open(path).map_err(|e| Error::model(format!("{}: {e}", path.display())))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|e| Error::model(format!("{}: {e}", path.display())))?;
    let n = u64::from_le_bytes(len) as usize;
    if n > 256 << 20 {
        return Err(Error::model(format!("{}: safetensors header of {n} bytes", path.display())));
    }
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).map_err(|e| Error::model(format!("{}: header: {e}", path.display())))?;
    let v: Value = serde_json::from_slice(&buf).map_err(|e| Error::model(format!("{}: header: {e}", path.display())))?;
    v.as_object().cloned().ok_or_else(|| Error::model(format!("{}: header is not an object", path.display())))
}

impl HeaderShapes {
    pub fn open(dir: &Path) -> Result<HeaderShapes> {
        let index = dir.join("model.safetensors.index.json");
        let files: Vec<String> = if index.exists() {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&index).map_err(|e| Error::model(format!("{}: {e}", index.display())))?)
                .map_err(|e| Error::model(format!("{}: {e}", index.display())))?;
            let mut f: Vec<String> = v["weight_map"].as_object().map(|m| m.values().filter_map(Value::as_str).map(String::from).collect()).unwrap_or_default();
            f.sort();
            f.dedup();
            f
        } else {
            vec!["model.safetensors".into()]
        };
        let mut out = BTreeMap::new();
        for f in files {
            for (name, t) in read_header(&dir.join(&f))? {
                if name == "__metadata__" {
                    continue;
                }
                let dt = dtype_of(t["dtype"].as_str().unwrap_or("?")).map_err(|e| Error::model(format!("{f}: {name}: {e}")))?;
                let shape: Vec<usize> = t["shape"].as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect()).unwrap_or_default();
                out.insert(name, (dt, shape));
            }
        }
        Ok(HeaderShapes(out))
    }
}

impl WeightShapes for HeaderShapes {
    fn get(&self, name: &str) -> Option<(DType, Vec<usize>)> {
        self.0.get(name).cloned()
    }
}

/// The DeepSeek-V4.1 config of a model directory, or `None` when it is another model.
pub fn detect(dir: &Path) -> Result<Option<Dsv41Config>> {
    let p = dir.join("config.json");
    let Ok(text) = std::fs::read_to_string(&p) else { return Ok(None) };
    let v: Value = serde_json::from_str(&text).map_err(|e| Error::model(format!("{}: {e}", p.display())))?;
    if !Dsv41Config::matches(&v) {
        return Ok(None);
    }
    Ok(Some(Dsv41Config::from_json(&v)?))
}

/// `intarsia plan show` for a DeepSeek-V4.1 directory: the plan JSON of [`intarsia_core::dsv41::plan::plan_json`].
pub fn plan(dir: &Path, cfg: &Dsv41Config, backend: &dyn Backend, cap: usize) -> Result<Value> {
    let shapes = HeaderShapes::open(dir)?;
    intarsia_core::dsv41::plan::plan_json(&dir.display().to_string(), cfg, &shapes, backend, cap, PLAN_PREFILL_ROWS)
}

/// The refusal `load_engine` gives for a DeepSeek-V4.1 checkpoint until the executing lanes land.
pub fn not_executable() -> Error {
    Error::Unsupported(
        "DeepSeek-V4.1 (recipe deepseek_v41) does not execute yet: the graph and op set are in place (lane 1a), the kernels \
         are lanes 1b / 2; `intarsia plan show --model <dir>` lists its op sequence"
            .into(),
    )
}

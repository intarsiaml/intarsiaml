//! Per-rank weight cache for DeepSeek-V4.1 Flash tensor parallelism (Mia EXL3 checkpoint).
//!
//! At TP=2 each GB10 holds about 99 GiB of weights in a 121 GB unified-memory pool, so rank 1 must not re-read 196 GB over
//! NFS at every start and the page cache must never hold the weights (on GB10 the page cache is the same memory the GPU
//! needs). The cache is a rank-local, already-sharded checkpoint on local NVMe: [`build_rank_cache`] writes it (pread of exact
//! byte ranges, `posix_fadvise(DONTNEED)` after every range, one dirty part at a time), [`load_rank_cache`] streams it to the
//! device with `O_DIRECT` (reader thread + 3 aligned 64 MiB buffers, uploads overlap disk reads).
//!
//! ```text
//! manifest.json                  written LAST (atomically), so a half-built cache is never valid
//! model.safetensors.index.json   {"metadata":{..},"weight_map":{name: part}}  (HeaderShapes::open reads rank-local shapes)
//! rank-00001-of-0000K.safetensors  <= ~4 GiB each (a single bigger tensor sits alone)
//! config.json, tokenizer.json, ... copies; dsv41_token_map.json  Engram compressed token map
//! ```
//!
//! Tensor rules (`rank r` of `world w`): drop `mtp.*`, `ffn.gate.bias_vl` and layers `>= --layers`; column-parallel EXL3
//! (`attn.wq_b`, `ffn.shared_experts.w1/w3`, `head`: split N, `ShardAxis::Rows`) and row-parallel EXL3 (`attn.wo_b`,
//! `ffn.shared_experts.w2`: split K, `ShardAxis::Cols`) go through `exl3::shard_ranges` / `shard_trellis` / `scale_bytes`;
//! `attn.attn_sink` keeps `[r H/w, (r+1) H/w)`; `attn.wo_a.slice.{g}` keeps the slices of this rank and renames them to the
//! local index; the routed experts of a layer are stored as the four tensors the graph reads
//! (`ffn.experts.{arena,scales,meta,codebook}`, from `exl3_moe::load` on the CPU backend); every EXL3 linear written also gets
//! `{p}.codebook` (i32 [1]: 2 mul1, 1 mcg, 0 3inst); everything else is copied unchanged.

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::exl3;
use intarsia_core::quant::{exl3_moe, WeightMeta};
use intarsia_core::weights::{Shard, ShardAxis, WeightInfo, WeightSource};
use intarsia_core::{DType, Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const FORMAT: &str = "intarsia-dsv41-rank-cache/1";
pub const MANIFEST: &str = "manifest.json";
pub const INDEX: &str = "model.safetensors.index.json";
pub const TOKEN_MAP: &str = "dsv41_token_map.json";
/// Files copied next to the tensors (whichever exist in the source).
pub const AUX_FILES: [&str; 6] = ["config.json", "quantization_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"];
/// Default part size cap (a single bigger tensor sits alone in its part).
pub const PART_BYTES: u64 = 4 << 30;
/// Header region reserved at the start of every part (the JSON header is space-padded to it; 8 + 1 MiB - 8).
const HDR_RESERVE: usize = 1 << 20;
const GROUP_SUFFIXES: [&str; 8] = ["trellis", "suh", "svh", "su", "sv", "mcg", "mul1", "bias"];
const SYNC_EVERY: u64 = 512 << 20;
const CHUNK: usize = 64 << 20;
const NBUF: usize = 3;
const ALIGN: usize = 4096;

macro_rules! logln {
    ($($a:tt)*) => { eprintln!("[dsv41-cache] {}", format!($($a)*)) };
}

fn err(s: impl Into<String>) -> Error {
    Error::model(s.into())
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PartInfo {
    pub file: String,
    pub bytes: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Manifest {
    pub format: String,
    pub source: String,
    pub source_fingerprint: String,
    pub rank: usize,
    pub world: usize,
    pub layers: Option<usize>,
    pub tensors: usize,
    pub bytes: u64,
    pub built_unix: u64,
    /// Part files with their sizes (lets `cache_matches` and a copy check detect a truncated cache).
    #[serde(default)]
    pub parts: Vec<PartInfo>,
    /// `Some("mtp")`: only the DSpark stages (`mtp.*`), the sub-cache `<rank cache>/mtp` ([`MTP_DIR`]); `None`: the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// The sub-directory of a rank cache that holds its DSpark stages ([`BuildOpts::mtp_only`]), so adding them does not
/// rewrite the model's parts.
pub const MTP_DIR: &str = "mtp";

// ---------------------------------------------------------------------------------------------------------------------
// page-cache control

#[cfg(target_os = "linux")]
fn advise_dontneed(f: &File, off: u64, len: u64) {
    use std::os::fd::AsRawFd;
    let (a, b) = if len == 0 { (0, 0) } else { (off & !4095, (off + len + 4095) & !4095) };
    // SAFETY: plain syscall on a valid fd; only a hint.
    unsafe {
        libc::posix_fadvise(f.as_raw_fd(), a as libc::off_t, (b - a) as libc::off_t, libc::POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_dontneed(_f: &File, _off: u64, _len: u64) {}

// ---------------------------------------------------------------------------------------------------------------------
// source checkpoint (headers parsed here, data read with pread on exact ranges)

/// `(data base offset, header map)` of one safetensors file.
fn read_file_header(path: &Path) -> Result<(u64, serde_json::Map<String, Value>)> {
    let mut f = File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let n = u64::from_le_bytes(len);
    if n > 256 << 20 {
        return Err(err(format!("{}: safetensors header of {n} bytes", path.display())));
    }
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf).map_err(|e| err(format!("{}: header: {e}", path.display())))?;
    let v: Value = serde_json::from_slice(&buf).map_err(|e| err(format!("{}: header: {e}", path.display())))?;
    let m = v.as_object().cloned().ok_or_else(|| err(format!("{}: header is not an object", path.display())))?;
    Ok((8 + n, m))
}

fn dtype_of(s: &str) -> Result<DType> {
    match s {
        "F8_E8M0" => Ok(DType::U8),
        other => DType::from_safetensors(other),
    }
}

struct SrcEntry {
    file: usize,
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

/// A sharded safetensors checkpoint: header index only; bytes are read with `pread` per tensor.
pub struct Source {
    files: Vec<File>,
    entries: BTreeMap<String, SrcEntry>,
}

impl Source {
    pub fn open(dir: &Path) -> Result<Source> {
        let index = dir.join(INDEX);
        let mut names: Vec<String> = Vec::new();
        if index.exists() {
            let v: Value = serde_json::from_slice(&std::fs::read(&index)?).map_err(|e| err(format!("{}: {e}", index.display())))?;
            let wm = v.get("weight_map").and_then(Value::as_object).ok_or_else(|| err("index.json has no weight_map"))?;
            names = wm.values().filter_map(|f| f.as_str().map(String::from)).collect();
            names.sort();
            names.dedup();
        } else if dir.join("model.safetensors").exists() {
            names.push("model.safetensors".into());
        }
        if names.is_empty() {
            return Err(err(format!("no safetensors files in {}", dir.display())));
        }
        let mut files = Vec::new();
        let mut entries = BTreeMap::new();
        for (fi, n) in names.iter().enumerate() {
            let p = dir.join(n);
            let (base, hdr) = read_file_header(&p)?;
            for (name, v) in &hdr {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = v.get("dtype").and_then(Value::as_str).ok_or_else(|| err(format!("{}: {name}: no dtype", p.display())))?.to_string();
                let shape: Vec<usize> = v
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or_else(|| err(format!("{}: {name}: no shape", p.display())))?
                    .iter()
                    .filter_map(|x| x.as_u64().map(|x| x as usize))
                    .collect();
                let off = v.get("data_offsets").and_then(Value::as_array).ok_or_else(|| err(format!("{}: {name}: no data_offsets", p.display())))?;
                let (b, e) = (off.first().and_then(Value::as_u64), off.get(1).and_then(Value::as_u64));
                let (Some(b), Some(e)) = (b, e) else { return Err(err(format!("{}: {name}: bad data_offsets", p.display()))) };
                entries.insert(name.clone(), SrcEntry { file: fi, dtype, shape, begin: base + b, end: base + e });
            }
            files.push(File::open(&p)?);
        }
        Ok(Source { files, entries })
    }

    fn entry(&self, name: &str) -> Result<&SrcEntry> {
        self.entries.get(name).ok_or_else(|| err(format!("tensor '{name}' not found in the source checkpoint")))
    }

    fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Bytes of one tensor (pread of the exact range, then `DONTNEED` over it).
    fn read(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        let mut buf = vec![0u8; (e.end - e.begin) as usize];
        let f = &self.files[e.file];
        f.read_exact_at(&mut buf, e.begin).map_err(|e2| err(format!("read '{name}': {e2}")))?;
        advise_dontneed(f, e.begin, e.end - e.begin);
        Ok(buf)
    }
}

/// sha256 hex over the bytes of `config.json` + `model.safetensors.index.json` + `quantization_config.json` (those that exist).
pub fn source_fingerprint(src_dir: &Path) -> Result<String> {
    let mut h = Sha256::new();
    h.update(std::fs::read(src_dir.join("config.json")).map_err(|e| err(format!("{}/config.json: {e}", src_dir.display())))?);
    for f in [INDEX, "quantization_config.json"] {
        if let Ok(b) = std::fs::read(src_dir.join(f)) {
            h.update(b);
        }
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// A handful of EXL3 tensors held in memory as a [`WeightSource`] (the exl3 helpers read through one).
#[derive(Default)]
struct MemSrc {
    t: HashMap<String, (DType, Vec<usize>, Vec<u8>)>,
}

impl WeightSource for MemSrc {
    fn names(&self) -> Vec<String> {
        self.t.keys().cloned().collect()
    }
    fn info(&self, name: &str) -> Option<WeightInfo> {
        self.t.get(name).map(|(d, s, _)| WeightInfo { dtype: *d, shape: s.clone() })
    }
    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.t.get(name).map(|(_, _, b)| b.as_slice()).ok_or_else(|| err(format!("tensor '{name}' not found")))
    }
}

/// Read the members of the EXL3 linear `p` that exist in the source: `(full name, dtype, shape, bytes)`.
fn read_group(src: &Source, p: &str) -> Result<Vec<(String, DType, Vec<usize>, Vec<u8>)>> {
    let mut out = Vec::new();
    for s in GROUP_SUFFIXES {
        let n = format!("{p}.{s}");
        if let Some(e) = src.entries.get(&n) {
            out.push((n.clone(), dtype_of(&e.dtype)?, e.shape.clone(), src.read(&n)?));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------
// part writer

struct Cur {
    tmp: PathBuf,
    w: BufWriter<File>,
    hdr: serde_json::Map<String, Value>,
    names: Vec<String>,
    off: u64,
    since_sync: u64,
    open_tensor: Option<(String, u64, u64)>, // name, expected bytes, written bytes
}

struct Done {
    tmp: PathBuf,
    names: Vec<String>,
    bytes: u64,
}

struct PartWriter {
    dir: PathBuf,
    cap: u64,
    cur: Option<Cur>,
    done: Vec<Done>,
    seen: BTreeSet<String>,
    tensors: usize,
    bytes: u64,
}

impl PartWriter {
    fn new(dir: &Path, cap: u64) -> PartWriter {
        PartWriter { dir: dir.to_path_buf(), cap, cur: None, done: Vec::new(), seen: BTreeSet::new(), tensors: 0, bytes: 0 }
    }

    fn open_part(&mut self) -> Result<()> {
        let tmp = self.dir.join(format!("rank-{:05}.part", self.done.len() + 1));
        let f = File::create(&tmp).map_err(|e| err(format!("create {}: {e}", tmp.display())))?;
        let mut w = BufWriter::with_capacity(8 << 20, f);
        w.write_all(&vec![0u8; HDR_RESERVE])?;
        self.cur = Some(Cur { tmp, w, hdr: serde_json::Map::new(), names: Vec::new(), off: 0, since_sync: 0, open_tensor: None });
        Ok(())
    }

    fn begin(&mut self, name: &str, dtype: &str, shape: &[usize], nbytes: u64) -> Result<()> {
        if !self.seen.insert(name.to_string()) {
            return Err(err(format!("cache: tensor '{name}' written twice")));
        }
        let full = match &self.cur {
            Some(c) => !c.names.is_empty() && (c.off + nbytes > self.cap || c.names.len() >= 3000),
            None => false,
        };
        if full {
            self.close_part()?;
        }
        if self.cur.is_none() {
            self.open_part()?;
        }
        let c = self.cur.as_mut().unwrap();
        let pad = (16 - c.off % 16) % 16;
        if pad > 0 {
            c.w.write_all(&vec![0u8; pad as usize])?;
            c.off += pad;
        }
        c.hdr.insert(name.to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [c.off, c.off + nbytes]}));
        c.names.push(name.to_string());
        c.open_tensor = Some((name.to_string(), nbytes, 0));
        Ok(())
    }

    fn write(&mut self, data: &[u8]) -> Result<()> {
        let c = self.cur.as_mut().ok_or_else(|| err("cache writer: write without begin"))?;
        let t = c.open_tensor.as_mut().ok_or_else(|| err("cache writer: write without begin"))?;
        t.2 += data.len() as u64;
        if t.2 > t.1 {
            return Err(err(format!("cache writer: '{}' got more than {} bytes", t.0, t.1)));
        }
        c.w.write_all(data)?;
        c.off += data.len() as u64;
        c.since_sync += data.len() as u64;
        if c.since_sync >= SYNC_EVERY {
            c.w.flush()?;
            c.w.get_ref().sync_data()?;
            advise_dontneed(c.w.get_ref(), 0, 0);
            c.since_sync = 0;
        }
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        let c = self.cur.as_mut().ok_or_else(|| err("cache writer: end without begin"))?;
        let (name, want, got) = c.open_tensor.take().ok_or_else(|| err("cache writer: end without begin"))?;
        if want != got {
            return Err(err(format!("cache writer: '{name}' wrote {got} of {want} bytes")));
        }
        self.tensors += 1;
        self.bytes += want;
        Ok(())
    }

    fn add(&mut self, name: &str, dtype: &str, shape: &[usize], data: &[u8]) -> Result<()> {
        self.begin(name, dtype, shape, data.len() as u64)?;
        self.write(data)?;
        self.end()
    }

    fn close_part(&mut self) -> Result<()> {
        let Some(mut c) = self.cur.take() else { return Ok(()) };
        c.w.flush()?;
        let mut hdr = serde_json::Map::new();
        hdr.insert("__metadata__".into(), serde_json::json!({"format": "intarsia-dsv41-rank-cache"}));
        hdr.extend(c.hdr);
        let mut text = serde_json::to_vec(&Value::Object(hdr)).map_err(|e| err(e.to_string()))?;
        let room = HDR_RESERVE - 8;
        if text.len() > room {
            return Err(err(format!("cache writer: header of {} bytes exceeds the {room} reserved", text.len())));
        }
        text.resize(room, b' ');
        let f = c.w.get_ref();
        f.write_all_at(&(room as u64).to_le_bytes(), 0)?;
        f.write_all_at(&text, 8)?;
        f.sync_all()?;
        advise_dontneed(f, 0, 0);
        self.done.push(Done { tmp: c.tmp, names: c.names, bytes: HDR_RESERVE as u64 + c.off });
        Ok(())
    }

    /// Close the last part, rename the parts to `rank-0000i-of-0000K.safetensors`, return `(file, bytes, names)`.
    fn finish(mut self) -> Result<Vec<(String, u64, Vec<String>)>> {
        self.close_part()?;
        let k = self.done.len();
        let mut out = Vec::new();
        for (i, d) in self.done.into_iter().enumerate() {
            let file = format!("rank-{:05}-of-{:05}.safetensors", i + 1, k);
            std::fs::rename(&d.tmp, self.dir.join(&file))?;
            out.push((file, d.bytes, d.names));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// builder

#[derive(Clone, Debug)]
pub struct BuildOpts {
    pub rank: usize,
    pub world: usize,
    pub layers: Option<usize>,
    /// Write into a non-empty directory that has no manifest (and replace a stale cache).
    pub force: bool,
    pub part_bytes: u64,
    /// Only the DSpark stages (`mtp.*`, sharded with the layers' rules), for `<rank cache>/mtp`.
    pub mtp_only: bool,
}

impl BuildOpts {
    pub fn new(rank: usize, world: usize, layers: Option<usize>) -> BuildOpts {
        BuildOpts { rank, world, layers, force: false, part_bytes: PART_BYTES, mtp_only: false }
    }
}

/// Numbered scope of `name` under `root` (`layers.` or `mtp.`).
fn scope_of(name: &str, root: &str) -> Option<usize> {
    name.strip_prefix(root)?.split('.').next()?.parse().ok()
}

/// Numeric id after `prefix` in `s` (`"attn.wo_a.slice.3"` -> 3).
fn trailing_index(s: &str, prefix: &str) -> Option<usize> {
    s.strip_prefix(prefix)?.parse().ok()
}

/// `(expert, matrix)` of `ffn.experts.{e}.{w1,w2,w3}.{suffix}`.
fn parse_expert(local: &str) -> Option<(usize, &str)> {
    let r = local.strip_prefix("ffn.experts.")?;
    let (e, rest) = r.split_once('.')?;
    let (w, _) = rest.split_once('.')?;
    Some((e.parse().ok()?, w))
}

/// Replicated tensors the checkpoint is known to hold (a name outside this list is logged as unclassified).
fn known_replicated(local: &str) -> bool {
    const EXACT: [&str; 6] = ["embed.weight", "norm.weight", "attn_norm.weight", "ffn_norm.weight", "attn.kv_norm.weight", "attn.q_norm.weight"];
    EXACT.contains(&local)
        || local.starts_with("hc_")
        || local.starts_with("attn.compressor.")
        || local.starts_with("attn.indexer.")
        || local.starts_with("ffn.gate.")
        || local.starts_with("engram.")
        || local == "main_norm.weight"
        || local.starts_with("markov_head.")
        || local.starts_with("confidence_head.")
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Shard(ShardAxis),
    Whole,
}

fn codebook_value(src: &Source, p: &str) -> i32 {
    if src.has(&format!("{p}.mul1")) {
        2
    } else if src.has(&format!("{p}.mcg")) {
        1
    } else {
        0
    }
}

struct Builder<'a> {
    src: &'a Source,
    w: PartWriter,
    rank: usize,
    world: usize,
    swiglu_limit: f32,
    unclassified: Vec<String>,
    /// `layers.` (the model's layers) or `mtp.` (the DSpark stages): the prefix of a numbered scope.
    root: &'static str,
}

impl Builder<'_> {
    /// Copy the members of the EXL3 linear `in_p` unchanged as `out_p`, plus `{out_p}.codebook`.
    fn emit_whole(&mut self, in_p: &str, out_p: &str) -> Result<()> {
        for s in GROUP_SUFFIXES {
            let n = format!("{in_p}.{s}");
            if let Some(e) = self.src.entries.get(&n) {
                let data = self.src.read(&n)?;
                self.w.add(&format!("{out_p}.{s}"), &e.dtype, &e.shape.clone(), &data)?;
            }
        }
        let cb = codebook_value(self.src, in_p);
        self.w.add(&format!("{out_p}.codebook"), "I32", &[1], &cb.to_le_bytes())
    }

    /// The rank's slice of the EXL3 linear `in_p` along `axis`, written as `out_p`.
    fn emit_sharded(&mut self, in_p: &str, out_p: &str, axis: ShardAxis) -> Result<()> {
        let mut mem = MemSrc::default();
        for (n, d, s, b) in read_group(self.src, in_p)? {
            mem.t.insert(n, (d, s, b));
        }
        let meta = WeightMeta::new(&mem, in_p);
        let lm = exl3::layer_meta(&meta)?;
        let sh = Shard { axis, rank: self.rank, world: self.world };
        let ((k0, kl), (n0, nl)) = exl3::shard_ranges(lm.k, lm.n, sh, in_p)?;
        let tr = exl3::shard_trellis(mem.bytes(&format!("{in_p}.trellis"))?, lm.k, lm.n, lm.k2, (k0, kl), (n0, nl))?;
        let suh = exl3::scale_bytes(&meta, "suh", "su", lm.k, k0, kl)?;
        let svh = exl3::scale_bytes(&meta, "svh", "sv", lm.n, n0, nl)?;
        self.w.add(&format!("{out_p}.trellis"), "I16", &[kl / exl3::TILE, nl / exl3::TILE, 8 * lm.k2], &tr)?;
        self.w.add(&format!("{out_p}.suh"), "F16", &[kl], &suh)?;
        self.w.add(&format!("{out_p}.svh"), "F16", &[nl], &svh)?;
        for s in ["mcg", "mul1"] {
            let n = format!("{in_p}.{s}");
            if let Some((_, shape, b)) = mem.t.get(&n) {
                let e = self.src.entry(&n)?;
                self.w.add(&format!("{out_p}.{s}"), &e.dtype, &shape.clone(), b)?;
            }
        }
        let cb = codebook_value(self.src, in_p);
        self.w.add(&format!("{out_p}.codebook"), "I32", &[1], &cb.to_le_bytes())
    }

    /// One layer's routed experts as arena / scales / meta / codebook.
    fn emit_experts(&mut self, pre: &str, experts: &BTreeMap<usize, BTreeSet<String>>) -> Result<()> {
        let e_count = experts.keys().max().map_or(0, |m| m + 1);
        if experts.len() != e_count {
            return Err(err(format!("{pre}ffn.experts: indices are not contiguous 0..{e_count}")));
        }
        let names: Vec<[String; 3]> = (0..e_count).map(|e| ["w1", "w3", "w2"].map(|w| format!("{pre}ffn.experts.{e}.{w}"))).collect();
        for n in &names {
            for p in n {
                if !self.src.has(&format!("{p}.trellis")) {
                    return Err(err(format!("expert matrix '{p}' has no trellis")));
                }
            }
        }
        // H and I from the shapes of expert 0 (the whole width of the checkpoint)
        let t0 = self.src.entry(&format!("{}.trellis", names[0][0]))?;
        let (hidden, inter) = (16 * t0.shape[0], 16 * t0.shape[1]);
        // planned arena size from the headers
        let mut planned = 0usize;
        for n in &names {
            for (j, p) in n.iter().enumerate() {
                let e = self.src.entry(&format!("{p}.trellis"))?;
                let (k, nn, k2) = (16 * e.shape[0], 16 * e.shape[1], e.shape[2] / 8);
                let axis = if j == 2 { ShardAxis::Cols } else { ShardAxis::Rows };
                let ((_, kl), (_, nl)) = exl3::shard_ranges(k, nn, Shard { axis, rank: self.rank, world: self.world }, p)?;
                planned += (kl / exl3::TILE) * (nl / exl3::TILE) * exl3::tile_bytes(k2);
            }
        }
        // read every expert tensor of the layer (several threads), one layer in memory
        let flat: Vec<&String> = names.iter().flatten().collect();
        let nthreads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
        let per = flat.len().div_ceil(nthreads);
        let src = self.src;
        let groups: Vec<Result<Vec<(String, DType, Vec<usize>, Vec<u8>)>>> = std::thread::scope(|s| {
            let hs: Vec<_> = flat
                .chunks(per.max(1))
                .map(|chunk| {
                    s.spawn(move || {
                        let mut v = Vec::new();
                        for p in chunk {
                            v.extend(read_group(src, p)?);
                        }
                        Ok(v)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(err("expert reader thread panicked")))).collect()
        });
        let mut mem = MemSrc::default();
        for g in groups {
            for (n, d, s, b) in g? {
                mem.t.insert(n, (d, s, b));
            }
        }
        let cpu = intarsia_backend_cpu::CpuBackend::new();
        let ex = exl3_moe::load(&cpu, &mem, &names, hidden, inter, self.rank, self.world, self.swiglu_limit)?;
        drop(mem);
        let arena_bytes = ex.gate_up_w.nbytes();
        if arena_bytes != planned {
            return Err(err(format!("{pre}ffn.experts: arena is {arena_bytes} bytes, the headers plan {planned}")));
        }
        self.w.begin(&format!("{pre}ffn.experts.arena"), "I16", &ex.gate_up_w.shape, arena_bytes as u64)?;
        let mut chunk = vec![0u8; 256 << 20];
        let mut off = 0usize;
        while off < arena_bytes {
            let n = chunk.len().min(arena_bytes - off);
            cpu.download(&ex.gate_up_w.buf, ex.gate_up_w.offset + off, &mut chunk[..n], 0)?;
            self.w.write(&chunk[..n])?;
            off += n;
        }
        self.w.end()?;
        drop(chunk);
        let small = |cpu: &intarsia_backend_cpu::CpuBackend, t: &TensorRef| cpu.download_row_major(t);
        for (suffix, dt, t) in [("scales", "F16", &ex.gate_up_s), ("meta", "I32", &ex.down_w), ("codebook", "I32", &ex.down_s)] {
            let b = small(&cpu, t)?;
            self.w.add(&format!("{pre}ffn.experts.{suffix}"), dt, &t.shape, &b)?;
        }
        Ok(())
    }

    /// One scope: the global tensors (`layer` None) or one layer.
    fn scope(&mut self, layer: Option<usize>, names: &[&String]) -> Result<()> {
        let pre = layer.map_or(String::new(), |l| format!("{}{l}.", self.root));
        let local = |n: &str| n.strip_prefix(pre.as_str()).unwrap_or(n).to_string();
        let mut consumed: BTreeSet<String> = BTreeSet::new();
        // routed experts
        let mut experts: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
        for n in names {
            let l = local(n);
            if l.starts_with("ffn.experts.") {
                let (e, w) = parse_expert(&l).ok_or_else(|| err(format!("unexpected expert tensor '{n}'")))?;
                if !["w1", "w2", "w3"].contains(&w) {
                    return Err(err(format!("unexpected expert tensor '{n}'")));
                }
                experts.entry(e).or_default().insert(w.to_string());
                consumed.insert((*n).clone());
            }
        }
        // wo_a slices present in this scope
        let slices = names.iter().filter_map(|n| trailing_index(local(n).split(".trellis").next().unwrap_or(""), "attn.wo_a.slice.")).max().map_or(0, |m| m + 1);
        if slices > 0 && slices % self.world != 0 {
            return Err(err(format!("{pre}attn.wo_a: {slices} slices cannot be split over {} ranks", self.world)));
        }
        let per = slices.checked_div(self.world).unwrap_or(0);
        // EXL3 linears
        for n in names {
            let Some(p) = n.strip_suffix(".trellis") else { continue };
            if consumed.contains(*n) {
                continue;
            }
            let lp = local(p);
            let (out_p, kind) = if lp == "head" || lp == "attn.wq_b" || lp == "ffn.shared_experts.w1" || lp == "ffn.shared_experts.w3" {
                (p.to_string(), Kind::Shard(ShardAxis::Rows))
            } else if lp == "attn.wo_b" || lp == "ffn.shared_experts.w2" {
                (p.to_string(), Kind::Shard(ShardAxis::Cols))
            } else if let Some(g) = trailing_index(&lp, "attn.wo_a.slice.") {
                if g < self.rank * per || g >= (self.rank + 1) * per {
                    for s in GROUP_SUFFIXES {
                        consumed.insert(format!("{p}.{s}"));
                    }
                    continue;
                }
                (format!("{pre}attn.wo_a.slice.{}", g - self.rank * per), Kind::Whole)
            } else {
                (p.to_string(), Kind::Whole)
            };
            match kind {
                Kind::Shard(axis) => self.emit_sharded(p, &out_p, axis)?,
                Kind::Whole => self.emit_whole(p, &out_p)?,
            }
            for s in GROUP_SUFFIXES {
                consumed.insert(format!("{p}.{s}"));
            }
        }
        // everything else
        for n in names {
            if consumed.contains(*n) {
                continue;
            }
            let l = local(n);
            if l == "ffn.gate.bias_vl" {
                continue;
            }
            let e = self.src.entry(n)?;
            if l == "attn.attn_sink" {
                let h = e.shape.first().copied().unwrap_or(0);
                if e.shape.len() != 1 || h % self.world != 0 || e.dtype != "F32" {
                    return Err(err(format!("'{n}' is {} {:?}: cannot split over {} ranks", e.dtype, e.shape, self.world)));
                }
                let data = self.src.read(n)?;
                let lh = h / self.world;
                self.w.add(n, "F32", &[lh], &data[4 * self.rank * lh..4 * (self.rank + 1) * lh])?;
                continue;
            }
            if !known_replicated(&l) {
                self.unclassified.push((*n).clone());
            }
            let data = self.src.read(n)?;
            let (dtype, shape) = (e.dtype.clone(), e.shape.clone());
            self.w.add(n, &dtype, &shape, &data)?;
        }
        if !experts.is_empty() {
            self.emit_experts(&pre, &experts)?;
        }
        Ok(())
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn is_cache_file(name: &str) -> bool {
    name == MANIFEST || name == "manifest.json.tmp" || name == INDEX || name == TOKEN_MAP || AUX_FILES.contains(&name) || (name.starts_with("rank-") && (name.ends_with(".safetensors") || name.ends_with(".part")))
}

/// Build the cache of rank `o.rank` of `o.world` from the checkpoint at `src_dir` into `out`.
pub fn build_rank_cache(src_dir: &Path, out: &Path, o: &BuildOpts) -> Result<Manifest> {
    if o.world == 0 || o.rank >= o.world {
        return Err(err(format!("rank {} of world {} is invalid", o.rank, o.world)));
    }
    let t0 = Instant::now();
    let fingerprint = source_fingerprint(src_dir)?;
    let src = Source::open(src_dir)?;
    // output directory: refuse a foreign non-empty directory, replace a stale cache
    std::fs::create_dir_all(out)?;
    let existing: Vec<String> = std::fs::read_dir(out)?.flatten().filter_map(|e| e.file_name().to_str().map(String::from)).collect();
    if !existing.is_empty() && !existing.iter().any(|n| n == MANIFEST) && !o.force {
        return Err(err(format!("{} is not empty and has no {MANIFEST}: pass --force to write into it", out.display())));
    }
    let _ = std::fs::remove_file(out.join(MANIFEST));
    for n in &existing {
        if is_cache_file(n) {
            let _ = std::fs::remove_file(out.join(n));
        }
    }
    let cfg: Value = serde_json::from_slice(&std::fs::read(src_dir.join("config.json"))?).map_err(|e| err(format!("config.json: {e}")))?;
    let tcfg = cfg.get("text_config").unwrap_or(&cfg);
    let swiglu_limit = tcfg.get("swiglu_limit").and_then(Value::as_f64).map_or(f32::INFINITY, |v| v as f32);

    // aux files + Engram token map (the model's cache has them; a DSpark sub-cache only needs its tensors)
    for f in if o.mtp_only { &[][..] } else { &AUX_FILES[..] } {
        let p = src_dir.join(f);
        if p.exists() {
            std::fs::copy(&p, out.join(f))?;
        }
    }
    let tok = src_dir.join("tokenizer.json");
    if !o.mtp_only && tok.exists() {
        let tk = tokenizers::Tokenizer::from_file(&tok).map_err(|e| err(format!("tokenizer.json: {e}")))?;
        let tm = intarsia_engram::token_map::build(&tk).map_err(|e| err(format!("token map: {e}")))?;
        if tm.map.len() != 129_280 || tm.compressed_vocab != 99_092 {
            logln!("WARNING token map has {} entries / compressed vocab {} (expected 129280 / 99092)", tm.map.len(), tm.compressed_vocab);
        }
        std::fs::write(out.join(TOKEN_MAP), serde_json::to_vec(&tm.map).map_err(|e| err(e.to_string()))?)?;
        logln!("token map: {} entries, compressed vocab {}", tm.map.len(), tm.compressed_vocab);
    } else if !o.mtp_only {
        logln!("WARNING no tokenizer.json in the source: {TOKEN_MAP} not written");
    }

    // scopes: global tensors, then layers in numeric order
    let mut scopes: BTreeMap<Option<usize>, Vec<&String>> = BTreeMap::new();
    let mut dropped = 0usize;
    let root = if o.mtp_only { "mtp." } else { "layers." };
    for n in src.entries.keys() {
        if n.starts_with("mtp.") != o.mtp_only {
            dropped += 1;
            continue;
        }
        let layer = if n.starts_with(root) { Some(scope_of(n, root).ok_or_else(|| err(format!("cannot read the scope of '{n}'")))?) } else { None };
        if let (Some(l), Some(limit), false) = (layer, o.layers, o.mtp_only) {
            if l >= limit {
                dropped += 1;
                continue;
            }
        }
        scopes.entry(layer).or_default().push(n);
    }
    let n_scopes = scopes.len();
    logln!(
        "building rank {} of {} (layers {:?}) from {}: {} source tensors, {} dropped (mtp / beyond the layer limit), {} scopes",
        o.rank,
        o.world,
        o.layers,
        src_dir.display(),
        src.entries.len(),
        dropped,
        n_scopes
    );
    let mut b = Builder { src: &src, w: PartWriter::new(out, o.part_bytes), rank: o.rank, world: o.world, swiglu_limit, unclassified: Vec::new(), root };
    for (i, (layer, names)) in scopes.iter().enumerate() {
        let (t, by) = (b.w.tensors, b.w.bytes);
        b.scope(*layer, names)?;
        logln!(
            "{} ({}/{}): {} tensors, {:.2} GiB; total {} tensors, {:.2} GiB, {:.0} s",
            layer.map_or("globals".to_string(), |l| format!("{root}{l}")),
            i + 1,
            n_scopes,
            b.w.tensors - t,
            gib(b.w.bytes - by),
            b.w.tensors,
            gib(b.w.bytes),
            t0.elapsed().as_secs_f64()
        );
    }
    if !b.unclassified.is_empty() {
        logln!("WARNING {} tensors matched no known rule and were copied unchanged, e.g. {:?}", b.unclassified.len(), &b.unclassified[..b.unclassified.len().min(8)]);
    }
    let (tensors, bytes) = (b.w.tensors, b.w.bytes);
    let parts = b.w.finish()?;
    // index
    let mut weight_map = serde_json::Map::new();
    let mut wm: BTreeMap<&str, &str> = BTreeMap::new();
    for (file, _, names) in &parts {
        for n in names {
            wm.insert(n, file);
        }
    }
    for (n, f) in wm {
        weight_map.insert(n.to_string(), Value::String(f.to_string()));
    }
    let index = serde_json::json!({"metadata": {"total_size": bytes, "rank": o.rank, "world": o.world}, "weight_map": weight_map});
    std::fs::write(out.join(INDEX), serde_json::to_vec_pretty(&index).map_err(|e| err(e.to_string()))?)?;
    verify_cache(&src, out, o)?;
    let manifest = Manifest {
        format: FORMAT.into(),
        source: src_dir.display().to_string(),
        source_fingerprint: fingerprint,
        rank: o.rank,
        world: o.world,
        layers: o.layers,
        tensors,
        bytes,
        built_unix: now_unix(),
        parts: parts.iter().map(|(f, b, _)| PartInfo { file: f.clone(), bytes: *b }).collect(),
        scope: o.mtp_only.then(|| "mtp".to_string()),
    };
    let tmp = out.join("manifest.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&manifest).map_err(|e| err(e.to_string()))?)?;
    std::fs::rename(&tmp, out.join(MANIFEST))?;
    logln!("done: {} tensors, {:.2} GiB in {} parts, {:.0} s", tensors, gib(bytes), manifest.parts.len(), t0.elapsed().as_secs_f64());
    Ok(manifest)
}

/// Re-open the output with `HeaderShapes` and check the rank-local shapes against the source headers.
fn verify_cache(src: &Source, out: &Path, o: &BuildOpts) -> Result<()> {
    let shapes = crate::dsv41::HeaderShapes::open(out)?.0;
    let w = o.world;
    let bad = |n: &str, want: &str, got: Option<&(DType, Vec<usize>)>| err(format!("cache check: '{n}' is {got:?}, expected {want}"));
    let need = |n: &str, want: &[usize]| -> Result<()> {
        match shapes.get(n) {
            Some((_, s)) if s == want => Ok(()),
            other => Err(bad(n, &format!("{want:?}"), other)),
        }
    };
    let root = if o.mtp_only { "mtp." } else { "layers." };
    if let Some(n) = shapes.keys().find(|n| n.starts_with("mtp.") != o.mtp_only || n.ends_with("bias_vl") || n.contains(".experts.0.")) {
        return Err(err(format!("cache check: '{n}' must not be in the cache")));
    }
    let mut kept_groups = 0usize;
    let src_shape = |n: &str| -> Result<Vec<usize>> { Ok(src.entry(n)?.shape.clone()) };
    // linear p: expected (trellis, suh, svh) shapes for Rows / Cols splits
    let linear = |p: &str, axis: Option<ShardAxis>| -> Result<()> {
        let t = src_shape(&format!("{p}.trellis"))?;
        let (k, n) = (16 * t[0], 16 * t[1]);
        let (kl, nl) = match axis {
            Some(ShardAxis::Rows) => (k, n / w),
            Some(ShardAxis::Cols) => (k / w, n),
            _ => (k, n),
        };
        need(&format!("{p}.trellis"), &[kl / 16, nl / 16, t[2]])?;
        need(&format!("{p}.suh"), &[kl])?;
        need(&format!("{p}.svh"), &[nl])?;
        need(&format!("{p}.codebook"), &[1])
    };
    let layers: BTreeSet<usize> = src.entries.keys().filter_map(|n| scope_of(n, root)).filter(|l| o.mtp_only || o.layers.is_none_or(|lim| *l < lim)).collect();
    if src.has("head.trellis") && !o.mtp_only {
        linear("head", Some(ShardAxis::Rows))?;
        kept_groups += 1;
    }
    for l in &layers {
        let p = format!("{root}{l}.");
        for (m, ax) in [("attn.wq_b", ShardAxis::Rows), ("ffn.shared_experts.w1", ShardAxis::Rows), ("ffn.shared_experts.w3", ShardAxis::Rows), ("attn.wo_b", ShardAxis::Cols), ("ffn.shared_experts.w2", ShardAxis::Cols)] {
            if src.has(&format!("{p}{m}.trellis")) {
                linear(&format!("{p}{m}"), Some(ax))?;
                kept_groups += 1;
            }
        }
        let sink = format!("{p}attn.attn_sink");
        if src.has(&sink) {
            need(&sink, &[src_shape(&sink)?[0] / w])?;
        }
        // wo_a slices: local index i is source slice rank * per + i
        let g = (0..).take_while(|g| src.has(&format!("{p}attn.wo_a.slice.{g}.trellis"))).count();
        if g > 0 {
            let per = g / w;
            for i in 0..per {
                let (a, b) = (format!("{p}attn.wo_a.slice.{}", o.rank * per + i), format!("{p}attn.wo_a.slice.{i}"));
                let t = src_shape(&format!("{a}.trellis"))?;
                need(&format!("{b}.trellis"), &t)?;
                for sfx in ["suh", "svh", "su", "sv"] {
                    if src.has(&format!("{a}.{sfx}")) {
                        need(&format!("{b}.{sfx}"), &src_shape(&format!("{a}.{sfx}"))?)?;
                    }
                }
                need(&format!("{b}.codebook"), &[1])?;
                kept_groups += 1;
            }
            if shapes.contains_key(&format!("{p}attn.wo_a.slice.{per}.trellis")) {
                return Err(err(format!("cache check: {p}attn.wo_a has more than {per} slices")));
            }
        }
        // routed experts
        let e = (0..).take_while(|e| src.has(&format!("{p}ffn.experts.{e}.w1.trellis"))).count();
        if e > 0 {
            let t = src_shape(&format!("{p}ffn.experts.0.w1.trellis"))?;
            let (h, i) = (16 * t[0], 16 * t[1]);
            need(&format!("{p}ffn.experts.scales"), &[e, 3 * h + 3 * (i / w)])?;
            need(&format!("{p}ffn.experts.meta"), &[e, exl3_moe::META_COLS])?;
            need(&format!("{p}ffn.experts.codebook"), &[1])?;
            if !shapes.contains_key(&format!("{p}ffn.experts.arena")) {
                return Err(err(format!("cache check: {p}ffn.experts.arena missing")));
            }
        }
    }
    // every EXL3 linear in the cache has its codebook
    if let Some(p) = shapes.keys().filter_map(|n| n.strip_suffix(".trellis")).find(|p| !shapes.contains_key(&format!("{p}.codebook"))) {
        return Err(err(format!("cache check: EXL3 linear '{p}' has no .codebook")));
    }
    logln!("check: {} rank-local shapes verified ({} sharded / renamed linears checked, {} layers)", shapes.len(), kept_groups, layers.len());
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------
// manifest / matching

pub fn open_manifest(dir: &Path) -> Result<Manifest> {
    let p = dir.join(MANIFEST);
    let b = std::fs::read(&p).map_err(|e| err(format!("{}: {e} (no complete rank cache here)", p.display())))?;
    let m: Manifest = serde_json::from_slice(&b).map_err(|e| err(format!("{}: {e}", p.display())))?;
    if m.format != FORMAT {
        return Err(err(format!("{}: format '{}', expected '{FORMAT}'", p.display(), m.format)));
    }
    Ok(m)
}

/// Whether `dir` holds a complete cache of this rank / world / layer limit built from the checkpoint at `src_dir`.
pub fn cache_matches(dir: &Path, src_dir: &Path, rank: usize, world: usize, layers: Option<usize>) -> Result<bool> {
    cache_matches_scope(dir, src_dir, rank, world, layers, false)
}

/// [`cache_matches`] for the model (`mtp_only` false) or its DSpark sub-cache.
pub fn cache_matches_scope(dir: &Path, src_dir: &Path, rank: usize, world: usize, layers: Option<usize>, mtp_only: bool) -> Result<bool> {
    let Ok(m) = open_manifest(dir) else { return Ok(false) };
    let layers = if mtp_only { None } else { layers };
    if m.scope.as_deref() != mtp_only.then_some("mtp") || m.rank != rank || m.world != world || m.layers != layers || m.source_fingerprint != source_fingerprint(src_dir)? {
        return Ok(false);
    }
    Ok(m.parts.iter().all(|p| std::fs::metadata(dir.join(&p.file)).is_ok_and(|md| md.len() == p.bytes)))
}

// ---------------------------------------------------------------------------------------------------------------------
// loader

pub struct LoadedWeights {
    pub tensors: HashMap<String, TensorRef>,
    pub bytes: u64,
    pub secs: f64,
}

/// 4096-aligned host buffer (O_DIRECT).
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: exclusively owned heap block.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn new(len: usize) -> AlignedBuf {
        let layout = std::alloc::Layout::from_size_align(len, ALIGN).expect("layout");
        // SAFETY: non-zero size, valid layout.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory for a {len} byte read buffer");
        AlignedBuf { ptr, len }
    }
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: ptr/len describe the owned allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    fn as_ref(&self) -> &[u8] {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: same layout as the allocation.
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.len, ALIGN).expect("layout")) }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Direct,
    Buffered,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Direct => "O_DIRECT",
            Mode::Buffered => "buffered + DONTNEED/NOCACHE",
        }
    }
}

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path)
}

#[cfg(not(target_os = "linux"))]
fn open_direct(path: &Path) -> std::io::Result<File> {
    use std::os::fd::AsRawFd;
    let f = File::open(path)?;
    // SAFETY: valid fd; F_NOCACHE only changes caching.
    #[cfg(target_os = "macos")]
    unsafe {
        libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = f.as_raw_fd();
    Ok(f)
}

/// Open for streaming: `O_DIRECT` (probe-read to catch filesystems that refuse it) or a plain open.
fn open_stream(path: &Path) -> Result<(File, Mode)> {
    if cfg!(target_os = "linux") {
        match open_direct(path) {
            Ok(f) => {
                let mut probe = AlignedBuf::new(ALIGN);
                match f.read_at(probe.as_mut(), 0) {
                    Ok(_) => return Ok((f, Mode::Direct)),
                    Err(e) => logln!("O_DIRECT probe read of {} failed ({e}): falling back to buffered reads", path.display()),
                }
            }
            Err(e) => logln!("O_DIRECT open of {} failed ({e}): falling back to buffered reads", path.display()),
        }
        return Ok((File::open(path)?, Mode::Buffered));
    }
    // macOS: F_NOCACHE
    Ok((open_direct(path)?, Mode::Buffered))
}

struct PartTensor {
    name: String,
    begin: u64,
    end: u64,
    t: TensorRef,
}

fn load_part(path: &Path, backend: &dyn Backend, out: &mut HashMap<String, TensorRef>) -> Result<(u64, Mode)> {
    let (data_base, hdr) = read_file_header(path)?;
    let mut ts: Vec<PartTensor> = Vec::new();
    for (name, v) in &hdr {
        if name == "__metadata__" {
            continue;
        }
        let dtype = dtype_of(v.get("dtype").and_then(Value::as_str).ok_or_else(|| err(format!("{name}: no dtype")))?)?;
        let shape: Vec<usize> = v.get("shape").and_then(Value::as_array).ok_or_else(|| err(format!("{name}: no shape")))?.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect();
        let off = v.get("data_offsets").and_then(Value::as_array).ok_or_else(|| err(format!("{name}: no data_offsets")))?;
        let (b, e) = (off.first().and_then(Value::as_u64).unwrap_or(0), off.get(1).and_then(Value::as_u64).unwrap_or(0));
        let numel: usize = shape.iter().product();
        if (e - b) as usize != numel * dtype.size_bytes() {
            return Err(err(format!("{}: '{name}' has {} bytes for {dtype} {shape:?}", path.display(), e - b)));
        }
        let buf = backend.alloc(((e - b) as usize).max(16), MemKind::Device)?;
        ts.push(PartTensor { name: name.clone(), begin: data_base + b, end: data_base + e, t: TensorRef::new(buf, dtype, &shape) });
    }
    ts.sort_by_key(|t| t.begin);
    let total: u64 = ts.iter().map(|t| t.end - t.begin).sum();
    let (file, mode) = open_stream(path)?;
    let flen = file.metadata()?.len();
    if ts.last().is_some_and(|t| t.end > flen) {
        return Err(err(format!("{}: truncated ({flen} bytes, tensors reach {})", path.display(), ts.last().map_or(0, |t| t.end))));
    }
    let start = data_base & !(ALIGN as u64 - 1);
    let (free_tx, free_rx) = std::sync::mpsc::channel::<AlignedBuf>();
    let (full_tx, full_rx) = std::sync::mpsc::sync_channel::<Result<(AlignedBuf, u64, usize)>>(NBUF);
    for _ in 0..NBUF {
        let _ = free_tx.send(AlignedBuf::new(CHUNK));
    }
    let end_needed = ts.last().map_or(0, |t| t.end);
    let res: Result<()> = std::thread::scope(|s| {
        let file = &file;
        let reader = s.spawn(move || {
            let mut off = start;
            while off < end_needed {
                let Ok(mut buf) = free_rx.recv() else { return };
                let want = (CHUNK as u64).min(flen - off) as usize;
                let rl = want.next_multiple_of(ALIGN).min(CHUNK);
                let mut got = 0usize;
                let mut failed = None;
                while got < want {
                    match file.read_at(&mut buf.as_mut()[got..rl], off + got as u64) {
                        Ok(0) => break,
                        Ok(n) => got += n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(e) => {
                            failed = Some(e);
                            break;
                        }
                    }
                }
                if let Some(e) = failed {
                    let _ = full_tx.send(Err(err(format!("read {}: {e}", path.display()))));
                    return;
                }
                if got < want {
                    let _ = full_tx.send(Err(err(format!("{}: short read at {off}: {got} of {want}", path.display()))));
                    return;
                }
                if mode == Mode::Buffered {
                    advise_dontneed(file, off, got as u64);
                }
                if full_tx.send(Ok((buf, off, got))).is_err() {
                    return;
                }
                off += got as u64;
            }
        });
        let mut ti = 0usize;
        let mut res: Result<()> = Ok(());
        'recv: for msg in full_rx.iter() {
            let (buf, off, valid) = match msg {
                Ok(m) => m,
                Err(e) => {
                    res = Err(e);
                    break;
                }
            };
            let end = off + valid as u64;
            while ti < ts.len() && ts[ti].begin < end {
                let t = &ts[ti];
                let (a, b) = (t.begin.max(off), t.end.min(end));
                if b > a {
                    if let Err(e) = backend.upload(&t.t.buf, (a - t.begin) as usize, &buf.as_ref()[(a - off) as usize..(b - off) as usize], 0) {
                        res = Err(e);
                        break 'recv;
                    }
                }
                if t.end <= end {
                    ti += 1;
                } else {
                    break;
                }
            }
            // the reader may already have finished (and dropped its end) while chunks are still queued: not an error
            let _ = free_tx.send(buf);
        }
        if res.is_ok() && ti < ts.len() {
            res = Err(err(format!("{}: stream ended before '{}'", path.display(), ts[ti].name)));
        }
        drop(full_rx);
        drop(free_tx);
        let _ = reader.join();
        res
    });
    res?;
    if mode == Mode::Buffered {
        advise_dontneed(&file, 0, 0);
    }
    for t in ts {
        out.insert(t.name, t.t);
    }
    Ok((total, mode))
}

/// Stream the rank cache at `dir` onto `backend`: one buffer per tensor, parts read sequentially in 64 MiB chunks with
/// `O_DIRECT` (a reader thread keeps up to 3 chunks ahead of the uploads), the page cache never holds the weights.
pub fn load_rank_cache(dir: &Path, backend: &dyn Backend) -> Result<LoadedWeights> {
    let t0 = Instant::now();
    let m = open_manifest(dir)?;
    let mut tensors = HashMap::with_capacity(m.tensors);
    let mut bytes = 0u64;
    let mut mode = Mode::Direct;
    for p in &m.parts {
        let (b, md) = load_part(&dir.join(&p.file), backend, &mut tensors)?;
        bytes += b;
        if md == Mode::Buffered {
            mode = Mode::Buffered;
        }
    }
    backend.synchronize(0)?;
    let secs = t0.elapsed().as_secs_f64();
    if tensors.len() != m.tensors {
        return Err(err(format!("{}: loaded {} tensors, the manifest lists {}", dir.display(), tensors.len(), m.tensors)));
    }
    logln!("loaded rank {} of {} cache {}: {} tensors, {:.2} GiB in {:.1} s = {:.2} GB/s ({})", m.rank, m.world, dir.display(), tensors.len(), gib(bytes), secs, bytes as f64 / 1e9 / secs.max(1e-9), mode.name());
    Ok(LoadedWeights { tensors, bytes, secs })
}

//! Safetensors checkpoints: mmap, header parsing, sharded `model.safetensors.index.json`.
//! Tensors are exposed as raw little-endian bytes borrowed from the mapping.

use intarsia_core::backend::HostMemory;
use intarsia_core::weights::{WeightInfo, WeightSource};
use intarsia_core::{DType, Error, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

struct MmapHost(Mmap);

impl HostMemory for MmapHost {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

struct Entry {
    file: usize,
    dtype: DType,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

pub struct SafeTensors {
    files: Vec<Arc<MmapHost>>,
    entries: BTreeMap<String, Entry>,
}

impl SafeTensors {
    /// Open a model directory: sharded (`model.safetensors.index.json`), single
    /// (`model.safetensors`) or any `*.safetensors` files present.
    pub fn open_dir(dir: &Path) -> Result<SafeTensors> {
        let mut paths: Vec<std::path::PathBuf> = Vec::new();
        let index = dir.join("model.safetensors.index.json");
        if index.exists() {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)
                .map_err(|e| Error::model(format!("{}: {e}", index.display())))?;
            let wm = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| Error::model("index.json has no weight_map"))?;
            let mut names: Vec<String> = wm.values().filter_map(|f| f.as_str().map(String::from)).collect();
            names.sort();
            names.dedup();
            paths.extend(names.into_iter().map(|n| dir.join(n)));
        } else {
            let single = dir.join("model.safetensors");
            if single.exists() {
                paths.push(single);
            } else {
                for e in std::fs::read_dir(dir)?.flatten() {
                    if e.path().extension().and_then(|x| x.to_str()) == Some("safetensors") {
                        paths.push(e.path());
                    }
                }
                paths.sort();
            }
        }
        if paths.is_empty() {
            return Err(Error::model(format!("no safetensors files in {}", dir.display())));
        }
        let mut files = Vec::new();
        let mut entries = BTreeMap::new();
        for (fi, p) in paths.iter().enumerate() {
            let f = std::fs::File::open(p).map_err(|e| Error::model(format!("open {}: {e}", p.display())))?;
            // SAFETY: the file is mapped read-only and not modified while mapped.
            let mm = unsafe { Mmap::map(&f) }.map_err(|e| Error::model(format!("mmap {}: {e}", p.display())))?;
            parse_header(&mm, fi, &mut entries).map_err(|e| Error::model(format!("{}: {e}", p.display())))?;
            files.push(Arc::new(MmapHost(mm)));
        }
        Ok(SafeTensors { files, entries })
    }

    pub fn total_bytes(&self) -> u64 {
        self.entries.values().map(|e| (e.end - e.begin) as u64).sum()
    }
}

fn parse_header(data: &[u8], file: usize, out: &mut BTreeMap<String, Entry>) -> std::result::Result<(), String> {
    if data.len() < 8 {
        return Err("file too small".into());
    }
    // every size below comes from the file: a truncated or corrupt one must be an error, not an overflow or an index panic
    let hlen = u64::from_le_bytes(data[..8].try_into().unwrap());
    let base = usize::try_from(hlen).ok().and_then(|h| h.checked_add(8)).filter(|&b| b <= data.len()).ok_or("header length exceeds file")?;
    let header: serde_json::Value = serde_json::from_slice(&data[8..base]).map_err(|e| e.to_string())?;
    let obj = header.as_object().ok_or("header is not an object")?;
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dt = v.get("dtype").and_then(|d| d.as_str()).ok_or("missing dtype")?;
        // MLX packs quantised weights into `U32` words. Same 4 bytes as `I32`, and only `quant::mlx_affine` reads those
        // bits (it requires sibling `.scales` / `.biases`), so they are carried as `I32` rather than a new ABI dtype.
        let mut shape: Vec<usize> = v
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or("missing shape")?
            .iter()
            .map(|x| x.as_u64().and_then(|x| usize::try_from(x).ok()))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("tensor {name}: bad shape"))?;
        let elems = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or_else(|| format!("tensor {name}: shape overflows"))?;
        // I64 has no ABI dtype. The only I64 tensors in the checkpoints we load are scalar format versions (DSpine's `*_version`
        // markers), so they are carried as their raw bytes, `U8 [8 * n]`; a loader that wants one reads the bytes.
        let dtype = if dt == "I64" {
            shape = vec![elems.checked_mul(8).ok_or_else(|| format!("tensor {name}: shape overflows"))?];
            DType::U8
        } else {
            DType::from_safetensors(if dt == "U32" { "I32" } else { dt }).map_err(|e| format!("{name}: {e}"))?
        };
        let off = v.get("data_offsets").and_then(|o| o.as_array()).ok_or("missing offsets")?;
        let at = |i: usize| off.get(i).and_then(|x| x.as_u64()).and_then(|x| usize::try_from(x).ok()).ok_or("bad offset");
        let (b, e) = (at(0)?, at(1)?);
        if base.checked_add(e).is_none_or(|end| end > data.len()) || b > e {
            return Err(format!("tensor {name}: offsets out of range"));
        }
        // `shape` is the stored one here except for I64, whose bytes are its one dimension
        let bytes = if dt == "I64" { Some(shape[0]) } else { elems.checked_mul(dtype.size_bytes()) };
        if Some(e - b) != bytes {
            return Err(format!("tensor {name}: byte size does not match shape"));
        }
        out.insert(name.clone(), Entry { file, dtype, shape, begin: base + b, end: base + e });
    }
    Ok(())
}

impl WeightSource for SafeTensors {
    fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    fn info(&self, name: &str) -> Option<WeightInfo> {
        self.entries.get(name).map(|e| WeightInfo { dtype: e.dtype, shape: e.shape.clone() })
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self.entries.get(name).ok_or_else(|| Error::model(format!("tensor '{name}' not found")))?;
        Ok(&self.files[e.file].0[e.begin..e.end])
    }

    fn host_memory(&self, name: &str) -> Option<(Arc<dyn HostMemory>, usize, usize)> {
        let e = self.entries.get(name)?;
        Some((self.files[e.file].clone() as Arc<dyn HostMemory>, e.begin, e.end - e.begin))
    }
}

/// Write a minimal safetensors file (tests and tools).
pub fn write_safetensors(path: &Path, tensors: &[(&str, &intarsia_core::HostTensor)]) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, t) in tensors {
        let st = match t.dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::Bf16 => "BF16",
            DType::Fp8E4m3 => "F8_E4M3",
            DType::Fp8E5m2 => "F8_E5M2",
            DType::I32 => "I32",
            DType::I8 => "I8",
            DType::U8 => "U8",
            DType::I16 => "I16",
        };
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": st, "shape": t.shape, "data_offsets": [off, off + t.nbytes()]}),
        );
        off += t.nbytes();
    }
    let mut h = serde_json::to_vec(&header).map_err(|e| Error::Other(e.to_string()))?;
    while (8 + h.len()) % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for (_, t) in tensors {
        out.extend_from_slice(&t.data);
    }
    std::fs::write(path, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use intarsia_core::HostTensor;

    #[test]
    fn roundtrip_single_and_sharded() {
        let dir = std::env::temp_dir().join(format!("intarsia-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = HostTensor::from_f32_as(DType::Bf16, &[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = HostTensor::from_f32(&[2], &[7., 8.]);
        write_safetensors(&dir.join("model-00001-of-00002.safetensors"), &[("a.weight", &a)]).unwrap();
        write_safetensors(&dir.join("model-00002-of-00002.safetensors"), &[("b.weight", &b)]).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let st = SafeTensors::open_dir(&dir).unwrap();
        assert_eq!(st.names(), vec!["a.weight", "b.weight"]);
        let i = st.info("a.weight").unwrap();
        assert_eq!((i.dtype, i.shape.clone()), (DType::Bf16, vec![2, 3]));
        assert_eq!(st.bytes("a.weight").unwrap(), &a.data[..]);
        assert_eq!(st.bytes("b.weight").unwrap(), &b.data[..]);
        let (mem, off, len) = st.host_memory("b.weight").unwrap();
        assert_eq!(&mem.as_slice()[off..off + len], &b.data[..]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file of `header` (with the length field `hlen`, by default the header's) and `data` bytes after it.
    fn file(header: &str, hlen: Option<u64>, data: usize) -> Vec<u8> {
        let mut f = hlen.unwrap_or(header.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(header.as_bytes());
        f.resize(f.len() + data, 0);
        f
    }

    #[test]
    fn corrupt_headers_are_errors_not_panics() {
        let parse = |f: &[u8]| parse_header(f, 0, &mut BTreeMap::new());
        let ok = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        assert!(parse(&file(ok, None, 8)).is_ok());
        // a header length that wraps `8 + hlen`, or just runs past the file
        for hlen in [u64::MAX, u64::MAX - 7, 1 << 40] {
            assert_eq!(parse(&file(ok, Some(hlen), 8)).unwrap_err(), "header length exceeds file");
        }
        for (bad, why) in [
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0]}}"#, "bad offset"),
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[]}}"#, "bad offset"),
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,18446744073709551615]}}"#, "offsets out of range"),
            (r#"{"a":{"dtype":"F32","shape":[2,-1],"data_offsets":[0,8]}}"#, "bad shape"),
            (r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#, "shape overflows"),
            (r#"{"a":{"dtype":"F32","shape":[4611686018427387904],"data_offsets":[0,0]}}"#, "byte size does not match shape"),
            (r#"{"a":{"dtype":"I64","shape":[4611686018427387904],"data_offsets":[0,0]}}"#, "shape overflows"),
        ] {
            let err = parse(&file(bad, None, 8)).unwrap_err();
            assert!(err.contains(why), "{bad}: {err}");
        }
        // I64 scalars are still carried as their bytes
        let mut out = BTreeMap::new();
        parse_header(&file(r#"{"v":{"dtype":"I64","shape":[],"data_offsets":[0,8]}}"#, None, 8), 0, &mut out).unwrap();
        assert_eq!((out["v"].dtype, out["v"].shape.clone()), (DType::U8, vec![8]));
    }
}

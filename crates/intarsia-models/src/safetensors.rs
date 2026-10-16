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
    let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    if 8 + hlen > data.len() {
        return Err("header length exceeds file".into());
    }
    let header: serde_json::Value = serde_json::from_slice(&data[8..8 + hlen]).map_err(|e| e.to_string())?;
    let obj = header.as_object().ok_or("header is not an object")?;
    let base = 8 + hlen;
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = DType::from_safetensors(v.get("dtype").and_then(|d| d.as_str()).ok_or("missing dtype")?)
            .map_err(|e| format!("{name}: {e}"))?;
        let shape: Vec<usize> = v
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or("missing shape")?
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as usize))
            .collect();
        let off = v.get("data_offsets").and_then(|o| o.as_array()).ok_or("missing offsets")?;
        let (b, e) = (off[0].as_u64().ok_or("bad offset")? as usize, off[1].as_u64().ok_or("bad offset")? as usize);
        if base + e > data.len() || b > e {
            return Err(format!("tensor {name}: offsets out of range"));
        }
        if e - b != shape.iter().product::<usize>() * dtype.size_bytes() {
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
}

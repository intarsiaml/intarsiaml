//! On-disk table layouts and the row -> file span mapping of one table shard (one Engram layer, one rank's rows).
//!
//! Supported (all read in place; nothing is converted at load):
//!
//! 1. **Official safetensors** (DeepSeek checkpoint shards `model-00047/48-of-00048.safetensors`; MiaAI /
//!    TensorFold read these directly, spark2 `~/.cache/dsv41-tensorfold/engram-1bae0e10`). Tensors
//!    `layers.<L>.engram.embed.weight` F8_E4M3 `[rows, 256]` and `.scale` F8_E8M0 `[rows, 8]`, ~98 GB apart.
//!    The rank's row range is a parameter. Scales either resident (`ScaleMode::Resident`, 8 B/row in RAM) or read
//!    per row (`ScaleMode::Read`, a second IO). Weight rows start at `8 + header + offset + row * 256`; the data
//!    region starts at 664 / 672 (mod 512 = 152 / 160), so half the rows straddle a 512 B sector.
//! 2. **Packed DSV41EN1** (the vLLM DSv4.1 kit's `pack_engram.py` output, `engram-l<L>-r<rank>of<world>.bin`;
//!    spark1 rank 0, spark2 rank 1 under `~/dsv41-engram`). A 4096 B header of little-endian u64
//!    `[magic 0x31344E4531565344 ("DSV1EN41"), layer_id, lo, hi, total_rows, record_bytes = 264]`, then
//!    `hi - lo` records of 264 B (256 B FP8 weights, then 8 B E8M0 scales) for global rows `lo..hi`. Records are
//!    unaligned: about half straddle a 512 B sector, 6.4% a 4 KiB page. `[lo, hi)` is the bucket-boundary split
//!    ([`crate::HashLayout::rank_rows`]). Format facts only; none of that kit's (AGPL) code is used.
//! 3. **Paged IENGRPG1** (this crate's repack, [`write_paged`]): a 4096 B header of little-endian u64
//!    `[magic "IENGRPG1", version 1, layer_id, lo, hi, total_rows, record_bytes 264, rows_per_page 15,
//!    page_bytes 4096]`, then pages of 15 records (3960 B + 136 B zero pad). A record never crosses a 4 KiB
//!    page, no resident scales; 2.9% larger than packed.
//!
//! A row read is the smallest 512 B-aligned span covering its record (512 or 1024 B), or the whole 4 KiB page
//! with [`ReadGranule::Page`].

use crate::io::{read_range, DirectFile};
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};

pub const WEIGHT_BYTES: usize = 256;
pub const SCALE_BYTES: usize = 8;
/// A row as this crate hands it out: 256 B FP8 E4M3 weights followed by 8 B E8M0 scales (one per 32 values).
pub const RECORD_BYTES: usize = WEIGHT_BYTES + SCALE_BYTES;
pub const HEADER_BYTES: u64 = 4096;
pub const PACKED_MAGIC: u64 = 0x3134_4E45_3156_5344; // b"DSV1EN41" little-endian
pub const PAGED_MAGIC: u64 = u64::from_le_bytes(*b"IENGRPG1");
pub const PAGE_BYTES: u64 = 4096;
pub const ROWS_PER_PAGE: u64 = PAGE_BYTES / RECORD_BYTES as u64; // 15

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutKind {
    Safetensors,
    Packed,
    Paged,
}

/// Scales of the safetensors layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    /// Load the shard's scale rows into RAM at open (8 B/row; 1.54 GB per layer per rank at TP=2).
    Resident,
    /// Read each row's scales from disk (a second IO per row).
    Read,
}

/// How much to read per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadGranule {
    /// The 512 B sectors covering the record (512 or 1024 B).
    Sector,
    /// The 4 KiB pages covering the record (4096, or 8192 for an unaligned record crossing a page).
    Page,
}

impl ReadGranule {
    pub fn bytes(self) -> u64 {
        match self {
            Self::Sector => 512,
            Self::Page => 4096,
        }
    }
}

/// One aligned read covering a byte range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// Aligned file offset.
    pub offset: u64,
    /// Aligned length.
    pub len: u32,
    /// Where the wanted bytes start inside the read.
    pub skew: u32,
}

impl Span {
    pub fn cover(at: u64, bytes: u64, align: u64) -> Self {
        let lo = at / align * align;
        let hi = (at + bytes).div_ceil(align) * align;
        Self { offset: lo, len: (hi - lo) as u32, skew: (at - lo) as u32 }
    }
}

/// The reads one row needs.
#[derive(Clone, Copy, Debug)]
pub struct RowReads {
    /// Covers the 256 B weights (and, for packed/paged, the 8 B scales right after).
    pub main: Span,
    /// Safetensors with `ScaleMode::Read`: the 8 B scale row.
    pub scale: Option<Span>,
}

/// One Engram layer's table rows owned by this rank, open for reading.
pub struct TableShard {
    pub kind: LayoutKind,
    pub layer_id: u32,
    /// Global rows this shard holds.
    pub rows: Range<u64>,
    /// Rows in the whole (unsharded) table.
    pub total_rows: u64,
    pub path: PathBuf,
    file: DirectFile,
    /// File offset of global row `rows.start`'s weights (safetensors: of row 0's weights).
    base: u64,
    /// Safetensors: file offset of row 0's scales.
    scale_base: u64,
    scale_mode: ScaleMode,
    /// Safetensors + Resident: scales of `rows`, 8 B each.
    scales: Option<Vec<u8>>,
}

fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap())
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

impl TableShard {
    /// Open a packed (DSV41EN1) or paged (IENGRPG1) shard file; the header says which rows it holds.
    pub fn open_shard_file(path: &Path, direct: bool) -> io::Result<Self> {
        let file = DirectFile::open(path, direct)?;
        let mut h = vec![0u8; HEADER_BYTES as usize];
        read_range(&file, 0, &mut h, HEADER_BYTES as usize)?;
        let len = file.file.metadata()?.len();
        let magic = u64_at(&h, 0);
        let (kind, layer_id, lo, hi, total) = if magic == PACKED_MAGIC {
            let (layer, lo, hi, total, rb) = (u64_at(&h, 1), u64_at(&h, 2), u64_at(&h, 3), u64_at(&h, 4), u64_at(&h, 5));
            if rb != RECORD_BYTES as u64 {
                return Err(bad(format!("{}: record bytes {rb}, want 264", path.display())));
            }
            if len < HEADER_BYTES + (hi - lo) * rb {
                return Err(bad(format!("{}: truncated ({len} B)", path.display())));
            }
            (LayoutKind::Packed, layer, lo, hi, total)
        } else if magic == PAGED_MAGIC {
            let f: Vec<u64> = (1..9).map(|i| u64_at(&h, i)).collect();
            let [ver, layer, lo, hi, total, rb, rpp, pb] = f[..] else { unreachable!() };
            if ver != 1 || rb != RECORD_BYTES as u64 || rpp != ROWS_PER_PAGE || pb != PAGE_BYTES {
                return Err(bad(format!("{}: unsupported paged header {f:?}", path.display())));
            }
            if len < HEADER_BYTES + (hi - lo).div_ceil(rpp) * pb {
                return Err(bad(format!("{}: truncated ({len} B)", path.display())));
            }
            (LayoutKind::Paged, layer, lo, hi, total)
        } else {
            return Err(bad(format!("{}: not an Engram shard (magic {magic:#x})", path.display())));
        };
        if lo >= hi || hi > total {
            return Err(bad(format!("{}: bad row range {lo}..{hi} of {total}", path.display())));
        }
        Ok(Self {
            kind,
            layer_id: layer_id as u32,
            rows: lo..hi,
            total_rows: total,
            path: path.to_path_buf(),
            file,
            base: HEADER_BYTES,
            scale_base: 0,
            scale_mode: ScaleMode::Read,
            scales: None,
        })
    }

    /// Open layer `layer_id`'s table inside an official safetensors file, holding global rows `rows`
    /// (`None`: all rows).
    pub fn open_safetensors(
        path: &Path,
        layer_id: u32,
        rows: Option<Range<u64>>,
        scale_mode: ScaleMode,
        direct: bool,
    ) -> io::Result<Self> {
        let file = DirectFile::open(path, direct)?;
        let mut n8 = [0u8; 8];
        read_range(&file, 0, &mut n8, 4096)?;
        let n = u64::from_le_bytes(n8);
        if n > 100 << 20 {
            return Err(bad(format!("{}: safetensors header of {n} B", path.display())));
        }
        let mut json = vec![0u8; n as usize];
        read_range(&file, 8, &mut json, 1 << 20)?;
        let h: serde_json::Value = serde_json::from_slice(&json).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let ent = |suffix: &str| -> io::Result<(u64, u64, u64, String)> {
            let name = format!("layers.{layer_id}.engram.embed.{suffix}");
            let v = h
                .get(&name)
                .or_else(|| h.get(format!("model.{name}")))
                .ok_or_else(|| bad(format!("{}: no {name}", path.display())))?;
            let shape: Vec<u64> = v["shape"].as_array().into_iter().flatten().filter_map(|x| x.as_u64()).collect();
            let off = v["data_offsets"][0].as_u64().ok_or_else(|| bad("data_offsets"))?;
            let dt = v["dtype"].as_str().unwrap_or("").to_string();
            if shape.len() != 2 {
                return Err(bad(format!("{name}: shape {shape:?}")));
            }
            Ok((shape[0], shape[1], 8 + n + off, dt))
        };
        let (wrows, wcols, base, wdt) = ent("weight")?;
        let (srows, scols, scale_base, sdt) = ent("scale")?;
        if wcols != WEIGHT_BYTES as u64 || scols != SCALE_BYTES as u64 || wrows != srows {
            return Err(bad(format!("{}: engram table shapes [{wrows},{wcols}] / [{srows},{scols}]", path.display())));
        }
        if wdt != "F8_E4M3" || sdt != "F8_E8M0" {
            return Err(bad(format!("{}: engram dtypes {wdt} / {sdt}", path.display())));
        }
        let rows = rows.unwrap_or(0..wrows);
        if rows.start >= rows.end || rows.end > wrows {
            return Err(bad(format!("rows {rows:?} outside table of {wrows}")));
        }
        let scales = if scale_mode == ScaleMode::Resident {
            let mut s = vec![0u8; ((rows.end - rows.start) as usize) * SCALE_BYTES];
            read_range(&file, scale_base + rows.start * SCALE_BYTES as u64, &mut s, 8 << 20)?;
            Some(s)
        } else {
            None
        };
        Ok(Self {
            kind: LayoutKind::Safetensors,
            layer_id,
            rows,
            total_rows: wrows,
            path: path.to_path_buf(),
            file,
            base,
            scale_base,
            scale_mode,
            scales,
        })
    }

    /// Find layer `layer_id`'s rows for `rank` of `world` in a directory: a paged shard
    /// (`engram-l<L>-r<rank>of<world>.paged`), else a packed shard (`engram-l<L>-r<rank>of<world>.bin`), else the
    /// official safetensors (the one whose header names the layer's table) restricted to `rows`.
    pub fn open_dir(
        dir: &Path,
        layer_id: u32,
        rank: usize,
        world: usize,
        rows: Range<u64>,
        scale_mode: ScaleMode,
        direct: bool,
    ) -> io::Result<Self> {
        for ext in ["paged", "bin"] {
            let p = dir.join(format!("engram-l{layer_id}-r{rank}of{world}.{ext}"));
            if p.exists() {
                let s = Self::open_shard_file(&p, direct)?;
                if s.layer_id != layer_id || s.rows != rows {
                    return Err(bad(format!("{}: holds layer {} rows {:?}, want layer {layer_id} rows {rows:?}", p.display(), s.layer_id, s.rows)));
                }
                return Ok(s);
            }
        }
        let mut names: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        names.sort();
        for p in names {
            match Self::open_safetensors(&p, layer_id, Some(rows.clone()), scale_mode, direct) {
                Ok(s) => return Ok(s),
                Err(e) if e.kind() == io::ErrorKind::InvalidData && e.to_string().contains(": no layers.") => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::new(io::ErrorKind::NotFound, format!("no Engram table for layer {layer_id} in {}", dir.display())))
    }

    pub fn owns(&self, row: u64) -> bool {
        self.rows.contains(&row)
    }

    pub fn fd(&self) -> std::os::fd::RawFd {
        self.file.fd()
    }

    /// Whether reads bypass the page cache (O_DIRECT / F_NOCACHE).
    pub fn direct(&self) -> bool {
        self.file.direct
    }

    pub fn file(&self) -> &DirectFile {
        &self.file
    }

    /// Resident scales of this shard's rows (safetensors + `ScaleMode::Resident` only), 8 B per row from
    /// `rows.start`.
    pub fn scales(&self) -> Option<&[u8]> {
        self.scales.as_deref()
    }

    /// Bytes of RAM the shard holds (resident scales).
    pub fn resident_bytes(&self) -> usize {
        self.scales.as_ref().map_or(0, |s| s.len())
    }

    /// Byte offset of a global row's record (or weights, for safetensors) in the file.
    pub fn record_offset(&self, row: u64) -> u64 {
        match self.kind {
            LayoutKind::Safetensors => self.base + row * WEIGHT_BYTES as u64,
            LayoutKind::Packed => self.base + (row - self.rows.start) * RECORD_BYTES as u64,
            LayoutKind::Paged => {
                let i = row - self.rows.start;
                self.base + i / ROWS_PER_PAGE * PAGE_BYTES + i % ROWS_PER_PAGE * RECORD_BYTES as u64
            }
        }
    }

    /// The aligned reads a global row (must be owned) needs.
    pub fn row_reads(&self, row: u64, granule: ReadGranule) -> RowReads {
        debug_assert!(self.owns(row));
        let a = granule.bytes();
        let off = self.record_offset(row);
        match self.kind {
            LayoutKind::Safetensors => RowReads {
                main: Span::cover(off, WEIGHT_BYTES as u64, a),
                scale: (self.scale_mode == ScaleMode::Read)
                    .then(|| Span::cover(self.scale_base + row * SCALE_BYTES as u64, SCALE_BYTES as u64, a)),
            },
            LayoutKind::Packed | LayoutKind::Paged => {
                RowReads { main: Span::cover(off, RECORD_BYTES as u64, a), scale: None }
            }
        }
    }

    /// Assemble a row's 264 B record from its completed reads (`main` / `scale` are the read buffers).
    pub fn assemble(&self, row: u64, reads: &RowReads, main: &[u8], scale: Option<&[u8]>, out: &mut [u8]) {
        let m = reads.main.skew as usize;
        match self.kind {
            LayoutKind::Safetensors => {
                out[..WEIGHT_BYTES].copy_from_slice(&main[m..m + WEIGHT_BYTES]);
                if let Some(s) = &self.scales {
                    let i = (row - self.rows.start) as usize * SCALE_BYTES;
                    out[WEIGHT_BYTES..RECORD_BYTES].copy_from_slice(&s[i..i + SCALE_BYTES]);
                } else {
                    let sk = reads.scale.expect("scale read").skew as usize;
                    out[WEIGHT_BYTES..RECORD_BYTES].copy_from_slice(&scale.expect("scale buffer")[sk..sk + SCALE_BYTES]);
                }
            }
            LayoutKind::Packed | LayoutKind::Paged => out[..RECORD_BYTES].copy_from_slice(&main[m..m + RECORD_BYTES]),
        }
    }

    /// Read one row synchronously (tests, verification, hot-tier preload of a few rows).
    pub fn read_row_sync(&self, row: u64, out: &mut [u8]) -> io::Result<()> {
        if !self.owns(row) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("row {row} not in {:?}", self.rows)));
        }
        let off = self.record_offset(row);
        match self.kind {
            LayoutKind::Safetensors => {
                read_range(&self.file, off, &mut out[..WEIGHT_BYTES], 4096)?;
                if let Some(s) = &self.scales {
                    let i = (row - self.rows.start) as usize * SCALE_BYTES;
                    out[WEIGHT_BYTES..RECORD_BYTES].copy_from_slice(&s[i..i + SCALE_BYTES]);
                } else {
                    read_range(&self.file, self.scale_base + row * SCALE_BYTES as u64, &mut out[WEIGHT_BYTES..RECORD_BYTES], 4096)?;
                }
            }
            LayoutKind::Packed | LayoutKind::Paged => read_range(&self.file, off, &mut out[..RECORD_BYTES], 4096)?,
        }
        Ok(())
    }
}

/// Write a paged (IENGRPG1) shard of global rows `rows` from any open source shard that holds them. Reads the
/// source sequentially in `chunk_rows` blocks and writes whole aligned pages, both bypassing the page cache
/// (O_DIRECT on Linux, F_NOCACHE on macOS), so a 100 GB repack does not fill the unified memory pool. Writes
/// `<out>.partial`, fsyncs, renames. Returns bytes written.
pub fn write_paged(src: &TableShard, rows: Range<u64>, out: &Path, chunk_rows: u64, mut progress: impl FnMut(u64, u64)) -> io::Result<u64> {
    if !(src.rows.start <= rows.start && rows.end <= src.rows.end && rows.start < rows.end) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("source holds {:?}, not {rows:?}", src.rows)));
    }
    let chunk_rows = chunk_rows.max(ROWS_PER_PAGE) / ROWS_PER_PAGE * ROWS_PER_PAGE;
    let partial = out.with_extension("partial");
    let f = open_direct_write(&partial)?;
    let mut buf = crate::io::AlignedBuf::new((chunk_rows / ROWS_PER_PAGE * PAGE_BYTES) as usize);
    {
        let header = &mut buf.as_mut_slice()[..HEADER_BYTES as usize];
        header.fill(0);
        let fields = [PAGED_MAGIC, 1, src.layer_id as u64, rows.start, rows.end, src.total_rows, RECORD_BYTES as u64, ROWS_PER_PAGE, PAGE_BYTES];
        for (i, v) in fields.iter().enumerate() {
            header[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    pwrite_full(&f, &buf.as_slice()[..HEADER_BYTES as usize], 0)?;
    let mut written = HEADER_BYTES;
    let mut weights = Vec::new();
    let mut scales = Vec::new();
    let mut at = rows.start;
    while at < rows.end {
        let n = chunk_rows.min(rows.end - at);
        read_rows_bulk(src, at, n, &mut weights, &mut scales)?;
        let pages = n.div_ceil(ROWS_PER_PAGE);
        let out_bytes = (pages * PAGE_BYTES) as usize;
        let b = &mut buf.as_mut_slice()[..out_bytes];
        b.fill(0);
        for i in 0..n as usize {
            let o = i / ROWS_PER_PAGE as usize * PAGE_BYTES as usize + i % ROWS_PER_PAGE as usize * RECORD_BYTES;
            b[o..o + WEIGHT_BYTES].copy_from_slice(&weights[i * WEIGHT_BYTES..(i + 1) * WEIGHT_BYTES]);
            b[o + WEIGHT_BYTES..o + RECORD_BYTES].copy_from_slice(&scales[i * SCALE_BYTES..(i + 1) * SCALE_BYTES]);
        }
        pwrite_full(&f, b, written)?;
        written += out_bytes as u64;
        at += n;
        progress(at - rows.start, rows.end - rows.start);
    }
    f.sync_all()?;
    drop(f);
    std::fs::rename(&partial, out)?;
    Ok(written)
}

fn open_direct_write(path: &Path) -> io::Result<std::fs::File> {
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        if let Ok(f) = o.clone().custom_flags(libc::O_DIRECT).open(path) {
            return Ok(f);
        }
    }
    let f = o.open(path)?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: valid fd.
        unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
    }
    Ok(f)
}

fn pwrite_full(f: &std::fs::File, mut b: &[u8], mut at: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !b.is_empty() {
        let n = f.write_at(b, at)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "short write"));
        }
        b = &b[n..];
        at += n as u64;
    }
    Ok(())
}

/// Rows `at..at+n` of a shard as separate weight and scale arrays (sequential reads).
fn read_rows_bulk(src: &TableShard, at: u64, n: u64, weights: &mut Vec<u8>, scales: &mut Vec<u8>) -> io::Result<()> {
    weights.resize(n as usize * WEIGHT_BYTES, 0);
    scales.resize(n as usize * SCALE_BYTES, 0);
    match src.kind {
        LayoutKind::Safetensors => {
            read_range(&src.file, src.base + at * WEIGHT_BYTES as u64, weights, 8 << 20)?;
            if let Some(s) = &src.scales {
                let i = (at - src.rows.start) as usize * SCALE_BYTES;
                scales.copy_from_slice(&s[i..i + n as usize * SCALE_BYTES]);
            } else {
                read_range(&src.file, src.scale_base + at * SCALE_BYTES as u64, scales, 8 << 20)?;
            }
        }
        LayoutKind::Packed => {
            let mut recs = vec![0u8; n as usize * RECORD_BYTES];
            read_range(&src.file, src.record_offset(at), &mut recs, 8 << 20)?;
            for i in 0..n as usize {
                weights[i * WEIGHT_BYTES..(i + 1) * WEIGHT_BYTES].copy_from_slice(&recs[i * RECORD_BYTES..i * RECORD_BYTES + WEIGHT_BYTES]);
                scales[i * SCALE_BYTES..(i + 1) * SCALE_BYTES].copy_from_slice(&recs[i * RECORD_BYTES + WEIGHT_BYTES..(i + 1) * RECORD_BYTES]);
            }
        }
        LayoutKind::Paged => {
            let mut rec = vec![0u8; RECORD_BYTES];
            for i in 0..n {
                src.read_row_sync(at + i, &mut rec)?;
                let i = i as usize;
                weights[i * WEIGHT_BYTES..(i + 1) * WEIGHT_BYTES].copy_from_slice(&rec[..WEIGHT_BYTES]);
                scales[i * SCALE_BYTES..(i + 1) * SCALE_BYTES].copy_from_slice(&rec[WEIGHT_BYTES..]);
            }
        }
    }
    Ok(())
}

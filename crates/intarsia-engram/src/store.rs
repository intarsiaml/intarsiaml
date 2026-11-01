//! The Engram row store: one rank's table shards, a read backend, the hot tier and a small staging cache.
//!
//! Use from the decode loop:
//! 1. as soon as a token id reaches the host, hash it ([`crate::NgramState::push`]) and call [`EngramStore::fetch`]
//!    once per Engram layer with that layer's row ids (global ids; rows other ranks own come back as zeros, like
//!    the reference's masked lookup). `fetch` never blocks: hot-tier and staged rows are copied at once, the rest
//!    are submitted to the disk.
//! 2. run the layers before the Engram layer; [`EngramStore::poll`] says whether the rows are in without blocking
//!    (use it to keep IO waits out of captured CUDA graphs).
//! 3. [`EngramStore::wait`] (the caller's thread reaps completions itself, busy-polling up to `spin`, then
//!    blocking) copies the rows out: `[n, 256]` FP8 E4M3 weights and `[n, 8]` E8M0 scales.
//!
//! [`EngramStore::prefetch`] issues speculative reads (e.g. the rows of the top-k candidate next tokens before
//! sampling): they land in the staging cache, and a later `fetch` of the same row joins the read in flight or
//! copies it from staging.

use crate::hot::{key, FxBuild, HotStats, HotTier, HotTierConfig};
use crate::io::{make_backend, AlignedBuf, Backend, BackendKind, Completion, ReadOp};
use crate::layout::{ReadGranule, RowReads, TableShard, RECORD_BYTES, SCALE_BYTES, WEIGHT_BYTES};
use std::collections::{HashMap, VecDeque};
use std::io;
use std::time::{Duration, Instant};

/// Bytes per bounce slot: the largest single read (an unaligned record across a 4 KiB page, page granule).
pub const SLOT_BYTES: usize = 8192;
const NO_SLOT: u32 = u32::MAX;
const TOUCH_BIT: u64 = 1 << 63;

#[derive(Clone, Copy, Debug)]
pub struct StoreConfig {
    pub backend: BackendKind,
    /// Most reads in flight at once (queue depth).
    pub depth: usize,
    pub granule: ReadGranule,
    /// How long `wait` busy-polls before blocking in the kernel.
    pub spin: Duration,
    pub hot: HotTierConfig,
    /// Rows kept from recent reads (speculative and demand), FIFO. 0 disables it.
    pub staging_rows: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::Auto,
            depth: 256,
            granule: ReadGranule::Sector,
            spin: Duration::from_millis(5),
            hot: HotTierConfig::default(),
            staging_rows: 16384,
        }
    }
}

/// A pending fetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ticket(u64);

impl Ticket {
    /// No fetch: `poll(Ticket::NONE)` just reaps completions and submits queued reads.
    pub const NONE: Ticket = Ticket(0);
    pub fn id(self) -> u64 {
        self.0
    }
    pub fn from_id(id: u64) -> Self {
        Ticket(id)
    }
}

/// What a fetch did.
#[derive(Clone, Copy, Debug, Default)]
pub struct FetchInfo {
    pub rows: usize,
    pub not_owned: usize,
    pub hot_hits: usize,
    pub staged_hits: usize,
    /// Rows that joined a read already in flight (a duplicate, or a prefetch).
    pub joined: usize,
    /// Row reads this fetch issued.
    pub reads: usize,
    /// From `fetch` to the last row landing (not to when `wait` was called).
    pub latency: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StoreStats {
    pub fetches: u64,
    pub rows: u64,
    pub not_owned: u64,
    pub hot_hits: u64,
    pub staged_hits: u64,
    pub joined: u64,
    pub row_reads: u64,
    pub io_ops: u64,
    pub io_bytes: u64,
    pub failed_reads: u64,
    pub spec_reads: u64,
    pub touches: u64,
}

struct RowRead {
    key: u64,
    layer: usize,
    row: u64,
    reads: RowReads,
    slots: [u32; 2],
    pending: u8,
    failed: bool,
    spec: bool,
    waiters: Vec<(u64, u32)>,
}

struct TicketState {
    buf: Vec<u8>,
    remaining: usize,
    failed: bool,
    info: FetchInfo,
    t0: Instant,
}

struct Staging {
    cap: usize,
    data: Vec<u8>,
    keys: Vec<u64>,
    index: HashMap<u64, u32, FxBuild>,
    head: usize,
}

impl Staging {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            data: vec![0u8; cap * RECORD_BYTES],
            keys: vec![u64::MAX; cap],
            index: HashMap::with_capacity_and_hasher(cap, FxBuild::default()),
            head: 0,
        }
    }
    fn get(&self, k: u64) -> Option<&[u8]> {
        let s = *self.index.get(&k)? as usize;
        Some(&self.data[s * RECORD_BYTES..(s + 1) * RECORD_BYTES])
    }
    fn insert(&mut self, k: u64, rec: &[u8]) {
        if self.cap == 0 || self.index.contains_key(&k) {
            return;
        }
        let s = self.head % self.cap;
        self.head += 1;
        let old = self.keys[s];
        if self.index.get(&old) == Some(&(s as u32)) {
            self.index.remove(&old);
        }
        self.keys[s] = k;
        self.data[s * RECORD_BYTES..(s + 1) * RECORD_BYTES].copy_from_slice(&rec[..RECORD_BYTES]);
        self.index.insert(k, s as u32);
    }
}

pub struct EngramStore {
    shards: Vec<TableShard>,
    cfg: StoreConfig,
    backend: Box<dyn Backend>,
    slot_mem: AlignedBuf,
    free_slots: Vec<u32>,
    reads: Vec<Option<RowRead>>,
    free_reads: Vec<u32>,
    inflight: HashMap<u64, u32, FxBuild>,
    queue: VecDeque<u32>,
    tickets: HashMap<u64, TicketState, FxBuild>,
    next_ticket: u64,
    hot: HotTier,
    staging: Staging,
    stats: StoreStats,
    comps: Vec<Completion>,
    ops: Vec<ReadOp>,
}

impl EngramStore {
    /// `shards[i]` serves Engram layer index `i` (DSv4.1: 0 = layer 1, 1 = layer 14).
    pub fn new(shards: Vec<TableShard>, cfg: StoreConfig) -> io::Result<Self> {
        let depth = cfg.depth.max(2);
        let backend = make_backend(cfg.backend, depth)?;
        let nslots = depth.min(backend.depth()).max(2);
        Ok(Self {
            shards,
            backend,
            slot_mem: AlignedBuf::new(nslots * SLOT_BYTES),
            free_slots: (0..nslots as u32).rev().collect(),
            reads: Vec::new(),
            free_reads: Vec::new(),
            inflight: HashMap::default(),
            queue: VecDeque::new(),
            tickets: HashMap::default(),
            next_ticket: 1,
            hot: HotTier::new(cfg.hot),
            staging: Staging::new(cfg.staging_rows),
            stats: StoreStats::default(),
            comps: Vec::new(),
            ops: Vec::new(),
            cfg: StoreConfig { depth, ..cfg },
        })
    }

    pub fn shards(&self) -> &[TableShard] {
        &self.shards
    }
    pub fn stats(&self) -> StoreStats {
        self.stats
    }
    pub fn hot_stats(&self) -> HotStats {
        self.hot.stats
    }
    pub fn hot(&self) -> &HotTier {
        &self.hot
    }
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }
    pub fn config(&self) -> &StoreConfig {
        &self.cfg
    }
    /// RAM held: hot tier, staging, bounce slots, resident scales.
    pub fn resident_bytes(&self) -> usize {
        self.hot.stats.bytes as usize
            + self.staging.data.len()
            + self.slot_mem.len()
            + self.shards.iter().map(|s| s.resident_bytes()).sum::<usize>()
    }

    /// Start fetching `rows` (global row ids) of Engram layer index `layer`. Never blocks.
    pub fn fetch(&mut self, layer: usize, rows: &[u64]) -> io::Result<Ticket> {
        self.fetch_inner(layer, rows, true)
    }

    fn fetch_inner(&mut self, layer: usize, rows: &[u64], use_cache: bool) -> io::Result<Ticket> {
        if layer >= self.shards.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("no Engram layer index {layer}")));
        }
        let tid = self.next_ticket;
        self.next_ticket += 1;
        let mut t = TicketState {
            buf: vec![0u8; rows.len() * RECORD_BYTES],
            remaining: 0,
            failed: false,
            info: FetchInfo { rows: rows.len(), ..Default::default() },
            t0: Instant::now(),
        };
        for (pos, &row) in rows.iter().enumerate() {
            let dst = pos * RECORD_BYTES..(pos + 1) * RECORD_BYTES;
            if !self.shards[layer].owns(row) {
                t.info.not_owned += 1;
                continue;
            }
            let k = key(layer, row);
            if use_cache {
                if let Some(rec) = self.hot.lookup(k) {
                    t.buf[dst].copy_from_slice(rec);
                    t.info.hot_hits += 1;
                    continue;
                }
                if let Some(rec) = self.staging.get(k) {
                    t.buf[dst].copy_from_slice(rec);
                    t.info.staged_hits += 1;
                    continue;
                }
            }
            t.remaining += 1;
            if let Some(&ri) = self.inflight.get(&k) {
                let r = self.reads[ri as usize].as_mut().unwrap();
                r.waiters.push((tid, pos as u32));
                r.spec &= !use_cache;
                t.info.joined += 1;
                continue;
            }
            // preload reads (no cache) are not offered to the dynamic hot region: they are about to be pinned
            let ri = self.new_read(layer, row, !use_cache);
            self.reads[ri as usize].as_mut().unwrap().waiters.push((tid, pos as u32));
            t.info.reads += 1;
        }
        let s = &mut self.stats;
        s.fetches += 1;
        s.rows += rows.len() as u64;
        s.not_owned += t.info.not_owned as u64;
        s.hot_hits += t.info.hot_hits as u64;
        s.staged_hits += t.info.staged_hits as u64;
        s.joined += t.info.joined as u64;
        s.row_reads += t.info.reads as u64;
        if t.remaining == 0 {
            t.info.latency = t.t0.elapsed();
        }
        self.tickets.insert(tid, t);
        self.pump()?;
        Ok(Ticket(tid))
    }

    /// Speculative reads (e.g. rows of the top-k candidate next tokens, before sampling). Rows already hot,
    /// staged or in flight are skipped; the rest land in the staging cache. Never blocks.
    pub fn prefetch(&mut self, layer: usize, rows: &[u64]) -> io::Result<usize> {
        let mut n = 0;
        for &row in rows {
            if layer >= self.shards.len() || !self.shards[layer].owns(row) {
                continue;
            }
            let k = key(layer, row);
            if self.hot.contains(k) || self.staging.get(k).is_some() || self.inflight.contains_key(&k) {
                continue;
            }
            self.new_read(layer, row, true);
            n += 1;
        }
        self.stats.spec_reads += n as u64;
        self.pump()?;
        Ok(n)
    }

    /// One small read to wake the drive from a low-power state at the start of a step, ahead of the step's rows
    /// (APST is on by default on the Sparks). The result is discarded.
    pub fn touch(&mut self) -> io::Result<()> {
        let Some(slot) = self.free_slots.pop() else { return Ok(()) };
        if self.backend.in_flight() >= self.backend.depth() {
            self.free_slots.push(slot);
            return Ok(());
        }
        let sh = &self.shards[0];
        let r = sh.row_reads(sh.rows.start, ReadGranule::Sector);
        // SAFETY: slot memory is reserved for this op until its completion is reaped.
        let buf = unsafe { self.slot_mem.as_ptr().add(slot as usize * SLOT_BYTES) };
        let op = ReadOp { fd: sh.fd(), offset: r.main.offset, len: 512, buf, tag: TOUCH_BIT | slot as u64 };
        self.backend.submit(&[op])?;
        self.stats.touches += 1;
        Ok(())
    }

    fn new_read(&mut self, layer: usize, row: u64, spec: bool) -> u32 {
        let reads = self.shards[layer].row_reads(row, self.cfg.granule);
        let r = RowRead {
            key: key(layer, row),
            layer,
            row,
            reads,
            slots: [NO_SLOT; 2],
            pending: 1 + reads.scale.is_some() as u8,
            failed: false,
            spec,
            waiters: Vec::new(),
        };
        let ri = if let Some(i) = self.free_reads.pop() {
            self.reads[i as usize] = Some(r);
            i
        } else {
            self.reads.push(Some(r));
            (self.reads.len() - 1) as u32
        };
        self.inflight.insert(key(layer, row), ri);
        self.queue.push_back(ri);
        ri
    }

    /// Submit queued reads while slots and queue depth allow.
    fn pump(&mut self) -> io::Result<()> {
        self.ops.clear();
        let base = self.slot_mem.as_ptr();
        while let Some(&ri) = self.queue.front() {
            let r = self.reads[ri as usize].as_mut().unwrap();
            let need = r.pending as usize;
            if self.free_slots.len() < need || self.backend.in_flight() + self.ops.len() + need > self.backend.depth() {
                break;
            }
            self.queue.pop_front();
            let fd = self.shards[r.layer].fd();
            let spans = [Some(r.reads.main), r.reads.scale];
            for (part, sp) in spans.iter().enumerate() {
                let Some(sp) = sp else { continue };
                let slot = self.free_slots.pop().unwrap();
                r.slots[part] = slot;
                // SAFETY: the slot is reserved for this op until its completion is reaped.
                let buf = unsafe { base.add(slot as usize * SLOT_BYTES) };
                self.ops.push(ReadOp { fd, offset: sp.offset, len: sp.len, buf, tag: ((ri as u64) << 1) | part as u64 });
                self.stats.io_ops += 1;
                self.stats.io_bytes += sp.len as u64;
            }
        }
        if !self.ops.is_empty() {
            let ops = std::mem::take(&mut self.ops);
            let res = self.backend.submit(&ops);
            self.ops = ops;
            res?;
        }
        Ok(())
    }

    fn slot(&self, s: u32) -> &[u8] {
        &self.slot_mem.as_slice()[s as usize * SLOT_BYTES..(s as usize + 1) * SLOT_BYTES]
    }

    fn on_completion(&mut self, c: Completion) {
        if c.tag & TOUCH_BIT != 0 {
            self.free_slots.push((c.tag & !TOUCH_BIT) as u32);
            return;
        }
        let ri = (c.tag >> 1) as usize;
        let part = (c.tag & 1) as usize;
        let r = self.reads[ri].as_mut().unwrap();
        let need = if part == 0 {
            let body = match self.shards[r.layer].kind {
                crate::layout::LayoutKind::Safetensors => WEIGHT_BYTES,
                _ => RECORD_BYTES,
            };
            r.reads.main.skew as usize + body
        } else {
            r.reads.scale.unwrap().skew as usize + SCALE_BYTES
        };
        if c.res < need as i64 {
            r.failed = true;
        }
        r.pending -= 1;
        if r.pending == 0 {
            self.finish(ri);
        }
    }

    fn finish(&mut self, ri: usize) {
        let r = self.reads[ri].take().unwrap();
        self.free_reads.push(ri as u32);
        if self.inflight.get(&r.key) == Some(&(ri as u32)) {
            self.inflight.remove(&r.key);
        }
        let mut rec = [0u8; RECORD_BYTES];
        if !r.failed {
            let main = self.slot(r.slots[0]);
            let scale = (r.slots[1] != NO_SLOT).then(|| self.slot(r.slots[1]));
            self.shards[r.layer].assemble(r.row, &r.reads, main, scale, &mut rec);
        } else {
            self.stats.failed_reads += 1;
        }
        for s in r.slots {
            if s != NO_SLOT {
                self.free_slots.push(s);
            }
        }
        for &(tid, pos) in &r.waiters {
            if let Some(t) = self.tickets.get_mut(&tid) {
                if r.failed {
                    t.failed = true;
                } else {
                    t.buf[pos as usize * RECORD_BYTES..(pos as usize + 1) * RECORD_BYTES].copy_from_slice(&rec);
                }
                t.remaining -= 1;
                if t.remaining == 0 {
                    t.info.latency = t.t0.elapsed();
                }
            }
        }
        if !r.failed {
            self.staging.insert(r.key, &rec);
            if !r.spec {
                self.hot.offer(r.key, &rec);
            }
        }
    }

    /// Reap completions and submit queued reads. `wait`: block (after spinning) until at least one completes.
    fn drive(&mut self, wait: bool) -> io::Result<()> {
        self.pump()?;
        let mut comps = std::mem::take(&mut self.comps);
        comps.clear();
        let res = self.backend.reap(&mut comps, wait, self.cfg.spin);
        for &c in &comps {
            self.on_completion(c);
        }
        self.comps = comps;
        res?;
        self.pump()
    }

    /// Whether a fetch's rows are all in (never blocks; reaps what has completed).
    pub fn poll(&mut self, t: Ticket) -> io::Result<bool> {
        self.drive(false)?;
        Ok(self.tickets.get(&t.0).is_none_or(|s| s.remaining == 0))
    }

    fn wait_inner(&mut self, t: Ticket) -> io::Result<TicketState> {
        loop {
            let st = self.tickets.get(&t.0).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "unknown or already waited ticket"))?;
            if st.remaining == 0 {
                break;
            }
            if self.backend.in_flight() == 0 && self.queue.is_empty() {
                return Err(io::Error::other("Engram store: ticket has rows pending but no reads are queued"));
            }
            self.drive(true)?;
        }
        let st = self.tickets.remove(&t.0).unwrap();
        if st.failed {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Engram row read failed"));
        }
        Ok(st)
    }

    /// Block until a fetch's rows are in; write `[n, 256]` weights and `[n, 8]` scales (`n` = rows fetched).
    pub fn wait(&mut self, t: Ticket, out_w: &mut [u8], out_s: &mut [u8]) -> io::Result<FetchInfo> {
        let st = self.wait_inner(t)?;
        let n = st.info.rows;
        if out_w.len() < n * WEIGHT_BYTES || out_s.len() < n * SCALE_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "output buffers too small"));
        }
        for i in 0..n {
            let rec = &st.buf[i * RECORD_BYTES..(i + 1) * RECORD_BYTES];
            out_w[i * WEIGHT_BYTES..(i + 1) * WEIGHT_BYTES].copy_from_slice(&rec[..WEIGHT_BYTES]);
            out_s[i * SCALE_BYTES..(i + 1) * SCALE_BYTES].copy_from_slice(&rec[WEIGHT_BYTES..]);
        }
        Ok(st.info)
    }

    /// Like [`wait`](Self::wait), but write 264 B records (weights then scales) into `out`.
    pub fn wait_records(&mut self, t: Ticket, out: &mut [u8]) -> io::Result<FetchInfo> {
        let st = self.wait_inner(t)?;
        out[..st.buf.len()].copy_from_slice(&st.buf);
        Ok(st.info)
    }

    /// Give up on a fetch (its reads still complete and go to the staging cache).
    pub fn cancel(&mut self, t: Ticket) {
        self.tickets.remove(&t.0);
    }

    /// Wait for every read in flight (speculative ones included).
    pub fn drain(&mut self) -> io::Result<()> {
        while self.backend.in_flight() > 0 || !self.queue.is_empty() {
            self.drive(true)?;
        }
        Ok(())
    }

    /// Pin rows into the hot tier (in the given order, until its pinned region is full). Reads them in sorted
    /// batches through the normal path. Returns rows pinned.
    pub fn preload(&mut self, rows: &[(usize, u64)]) -> io::Result<usize> {
        let mut pinned = 0;
        let room = self.hot.pinned_capacity();
        let mut want: Vec<(usize, u64)> = rows
            .iter()
            .copied()
            .filter(|&(l, r)| l < self.shards.len() && self.shards[l].owns(r))
            .take(room)
            .collect();
        want.sort_unstable();
        want.dedup();
        let mut rec = vec![0u8; 4096 * RECORD_BYTES];
        for layer in 0..self.shards.len() {
            let rows: Vec<u64> = want.iter().filter(|x| x.0 == layer).map(|x| x.1).collect();
            for chunk in rows.chunks(4096) {
                let t = self.fetch_inner(layer, chunk, false)?;
                self.wait_records(t, &mut rec)?;
                for (i, &row) in chunk.iter().enumerate() {
                    if self.hot.pin(key(layer, row), &rec[i * RECORD_BYTES..(i + 1) * RECORD_BYTES]) {
                        pinned += 1;
                    }
                }
            }
        }
        Ok(pinned)
    }
}

impl Drop for EngramStore {
    fn drop(&mut self) {
        // the kernel must not write into slot memory after it is freed
        let _ = self.drain();
    }
}

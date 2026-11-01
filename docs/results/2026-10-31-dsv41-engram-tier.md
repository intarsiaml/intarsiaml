# DeepSeek V4.1 Engram table tier: crate, layouts, hash parity, NVMe measurements (spark2)

Lane `dsv41/engram`, 2026-10-31. Crate `crates/intarsia-engram`. Raw outputs in
`2026-10-31-dsv41-engram-tier/raw/`. All reads below are on spark2's local NVMe (Samsung MZALC4T0HBL1, 4 TB,
logical block 512 B, ext4, kernel 6.17.0-1032-nvidia), against tables already on that disk, read-only.

## Headline

- **Hash: one implementation**, intarsia-core's `NgramHasher` (skeleton lane); this crate wraps it (O(1) decode
  state, rank split). It is **bit-exact** with DeepSeek's official `engram.py` on 3,000 generated sequences (97K positions, dead
  spans, pad tokens, prefill/decode split, batch > 1) and on real text through the full raw-id pipeline. The
  token map built in Rust with HF `tokenizers` equals the official Python one and MiaAI's
  `dsv41_token_map.json` (sha256 of the u32 LE map `c60a8632...6481`, 129,280 ids -> 99,092).
- **Layouts: the two existing copies are read in place** (official safetensors, MiaAI/TensorFold's copy; packed
  `DSV41EN1` rank shards, the vLLM kit's copy). 40,006 random rows compared between them: identical. A paged
  layout (15 x 264 B records per 4 KiB page, zero resident scales) and its repack tool exist; **no 100 GB repack
  was written** (waiting for the coordinator's go-ahead).
- **Per-token fetch latency, one TP=2 rank (24 rows/token, uniform random rows)**: back to back p50 0.14 ms, p99
  0.41 ms. But **after the 16-75 ms idle gap of a real decode step the drive is asleep**: p50 0.59 ms, p99 0.89 ms,
  p99.9 1.29 ms at 75 ms, and ~9.6 ms when a step needed no reads and the idle stretch got longer. **A one-sector
  "touch" read every step, ~1 ms before the token id is known, removes both**: p99 0.24-0.41 ms, p99.9 0.33-0.53 ms
  (real text, with and without hot tier). The target (p99 < 1 ms) holds with room; the coordinator's ~0.4 ms
  layer-1 budget holds at p99; at p99.9 it needs the top-k prefetch (p99.9 0.10 ms when the token is in the top 4).
- **Hot tier**: 1 GiB (4M rows/rank, 3M pinned by frequency + TinyLFU dynamic) serves **70 % of rank 1's
  lookups** on held-out text (rank 0: 85 %; rank 1 owns the 3/4-gram columns and is the binding rank).
  It cuts disk reads ~3x and makes most steps IO-free (p50 3-6 us), but p99 is still one IO round (~0.3 ms).
- **io_uring, reaped by the waiting thread, is required**: a 16-thread pread pool is 4-5x slower per burst
  (0.53-1.3 ms p50) because idle workers must be woken.

## What was built

`crates/intarsia-engram` (workspace member). Outside it, only `intarsia-core`'s `EngramTier` trait changed: three
provided methods (`owned_rows`, `touch`, `prefetch`), additive, defaults keep `MemEngramTier` as it was. Public API:

| Item | What |
|---|---|
| `NvmeEngramTier::new(store, &hash, rank, world)` | implements `intarsia_core::dsv41::engram::EngramTier` (`issue`, `wait`, `owned_rows`, `touch`, `prefetch`) |
| `HashLayout::dsv41()` | core `NgramHasher` + DSv4.1 multipliers, per-column offsets, `rank_rows(layer, rank, world)` (column split) |
| `NgramState::push(&hash, Option<u32>, &mut [u64])` | one token -> 48 global row ids `[layer][column]` via the core hasher on a 4-id window, O(1); `peek` for candidate tokens; `None` = dead/image token |
| `TokenMap::load_json` / `token_map::build(&Tokenizer)` (feature `tokenizer`) | raw -> compressed ids |
| `TableShard::open_dir(dir, layer_id, rank, world, rows, ScaleMode, direct)` | finds `engram-l<L>-r<r>of<w>.paged`, `.bin`, or the safetensors holding the layer; `open_shard_file`, `open_safetensors`; `scales()` (resident E8M0, safetensors only) |
| `EngramStore::new(shards, StoreConfig)` | backend (io_uring / pread pool), depth, read granule, spin, hot tier, staging |
| `fetch(layer, &rows) -> Ticket` | never blocks; hot/staged rows copied, the rest submitted |
| `poll(Ticket) -> bool` | never blocks (keep IO out of captured graphs) |
| `wait(Ticket, &mut w, &mut s) -> FetchInfo` | `[n,256]` FP8 E4M3 + `[n,8]` E8M0; rows other ranks own are all-zero bytes, which dequantize to exactly 0 (no mask needed before the all-reduce) |
| `prefetch(layer, &rows)` | speculative (top-k candidates); a later `fetch` joins the read or hits staging |
| `touch()` | one 512 B read to wake the drive |
| `preload(&[(layer,row)])` | pin a frequency profile into the hot tier (3M rows in 4.5 s) |
| `stats()`, `hot_stats()`, `resident_bytes()` | counters |
| `fp8::dequant_row_bf16` | CPU reference dequant (reference semantics) |

`engram-bench` (feature `bench`): `raw`, `token`, `hit`, `profile`, `trace` (feature `tokenizer`), `verify`,
`repack`.

Tests: `cargo test -p intarsia-engram` (hash parity, every layout x backend x granule returns the same rows incl.
queueing beyond the depth, foreign rows, duplicates, prefetch, poll, hot-tier preload/admission); passes on macOS
(pread backend) and on spark2 (io_uring + O_DIRECT). Gated: `INTARSIA_DSV41_TOKEN_MAP=...json` and, with
`--features tokenizer`, `INTARSIA_DSV41_TOKENIZER=.../tokenizer.json` (both pass).

## Semantics confirmed from the official code

- Engram layers **1 and 14** (`engram_layer_ids`); the Engram output is added before the layer's attention, so
  layer 1's rows are needed after layer 0 only.
- Per position and layer: 3 n-gram sizes (2-4) x 8 heads = **24 rows**; all heads of an n-gram share one XOR key
  and differ in their prime modulus; column `c` owns rows `[offset_c, offset_c + prime_c)` (primes from
  16,000,057 up, never reused).
- **Rank split: columns, i.e. a row split at a column boundary** (decided). Each column owns a contiguous row range,
  so "split the 24 columns" (Mia) and "split rows" (model.py) differ only in where the boundary falls. The existing
  packed shards split at a column boundary (rank 0: columns 0-11 = 2-gram heads 0-7
  and 3-gram heads 0-3; rank 1: columns 12-23), e.g. layer 1 rank 0 = rows `0..192,001,740` = sum of the first 12
  primes. model.py's `ceil(rows/world)` split differs by 1,344 / 5,325 rows but the output is the same (masked
  rows add zeros). Why columns: (1) the IO per rank is the same either way (12 reads per layer per token; the
  measured burst latency depends on the burst size, which is equal), but the column split makes it exactly 12 on
  every token, where the ceil split moves the first 1,344 rows of column 12 to rank 0; (2) the copies on both
  Sparks are column-split, so with the ceil split rank 0 would need rows that exist only on spark2. TP=1 reads 48.
  The trait's `owned_rows` reports it; `EngramShard::range` (ceil split) stays the default for `MemEngramTier`.

## On-disk layouts supported (`src/layout.rs` documents the byte layout)

| Layout | Where it is | Per row | Resident | Notes |
|---|---|---|---|---|
| Official safetensors `model-00047/48-of-00048` | spark2 `~/.cache/dsv41-tensorfold/engram-1bae0e10` (190 GiB, both full tables) | 256 B weight read (+ 8 B scale read unless resident) | 0, or 3.07 GB/rank scales | data starts at 664/672 B: half the rows straddle a 512 B sector |
| Packed `DSV41EN1` (vLLM kit `pack_engram.py`; format facts only, kit is AGPL) | spark1 `~/dsv41-engram` r0of2, spark2 `~/dsv41-engram` r1of2 (2 x 50.7 GB each) | one 264 B record (weights+scales) | 0 | 4 KiB header `<6Q magic,layer,lo,hi,rows,264>`; 6.4 % of records cross a 4 KiB page |
| Paged `IENGRPG1` (this crate) | not written yet | one 264 B record inside one 4 KiB page | 0 | 15 records/page, 52.4 GB per layer per rank (+2.9 %) |

`engram-bench verify` read 20,003 random rows per layer (rank 1) through the safetensors and the packed copy:
0 differ.

Tables come from the official base checkpoint (shards 47-48 @1bae0e10) or these copies; the Mia EXL3 checkpoint
has none. **The spark1/spark2 packed copies are the target layout** (264 B records, scales interleaved, zero
resident scales: what the plan review asked for), so the tier reads them directly.

**Repack, not run.** `engram-bench repack --src <dir> --layer {1,14} --rank R --world 2 --out
<dir>/engram-l<L>-r<R>of2.paged --yes` writes one layer's shard (52.4 GB, O_DIRECT reads and writes, no page
cache). Both layers for one rank = 104.8 GB; spark2 has ~615 GB free. Note the packed shard already has zero
resident scales and one read per row; the paged layout only removes the 6.4 % of rows whose 264 B record
crosses a 4 KiB page (one 8 KiB read instead of 4 KiB, or 1 KiB instead of 512 B at sector granule). The
measurements below do not show that mattering; I would not write the 100 GB file unless the GPU path needs
page-aligned rows for another reason.

## Device behaviour (raw, O_DIRECT, io_uring, `engram-bench raw`)

Sustained random reads on the 50.7 GB packed file, per-IO latency (us):

| bs | QD | IOPS | p50 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 3.9K | 201 | 534 | 620 |
| 512 | 24 | 194K | 113 | 332 | 415 |
| 512 | 48 | 469K | 85 | 300 | 428 |
| 4096 | 1 | 3.5K | 300 | 486 | 570 |
| 4096 | 8 | 83K | 78 | 237 | 418 |
| 4096 | 24 | 294K | 78 | 138 | 178 |
| 4096 | 48 | 550K | 82 | 175 | 248 |
| 4096 | 64 | 655K | 91 | 191 | 306 |
| 1024 | 24 | 293K | 78 | 138 | 200 |

Token-shaped bursts (N reads issued at once, latency until all are in, us), 4 KiB reads:

| burst | gap 0 p50 / p99 / p99.9 | gap 16 ms | gap 75 ms |
|---:|---|---|---|
| 12 (one layer, TP=2) | 200 / 591 / 779 | 312 / 436 / 645 | 672 / 792 / 1110 |
| 24 (one token, TP=2) | 126 / 189 / 402 | 173 / 231 / 273 | 572 / 663 / 666 |
| 48 (one token, TP=1) | 163 / 243 / 455 | 216 / 308 / 502 | 619 / 890 / 1286 |
| 144 (DSpark step of 6) | 264 / 534 / 591 | 356 / 451 / 647 | 728 / 1007 / 1025 |

Observations:
- **Idle penalty.** After 75 ms without IO the first burst costs ~+450 us. APST is enabled
  (`nvme_core.default_ps_max_latency_us=100000`); ASPM link states may add to it. A decode step at 13 tok/s
  leaves exactly this gap.
- **Low-QD latencies are quantized at ~100 us** (QD1 p50 201/300 us; a 12-read burst 200 us vs 126 us for 24):
  completions look interrupt-coalesced. Polled queues (`nvme.poll_queues`, then `IORING_SETUP_IOPOLL`) or
  turning coalescing off would lower the floor; all three knobs (APST, poll queues, coalescing) are root-only
  system settings, so they are not changed here: the user's call.
- 4 KiB reads are as fast as 512 B ones at decode depths and faster at QD 24 sustained (p99.9 178 vs 415 us),
  so the bench uses page granule and `StoreConfig::default()` does too (`ReadGranule::Page`). The device has headroom: 650K IOPS at QD 64.
- pread pool (16 threads) at 512 B, burst 24: 664 / 827 / 1025 us at gap 0, 1302 / 2030 / 2428 at 75 ms.

## Store end to end (`engram-bench token`, rank 1 of 2, both layers, 24 owned rows per token)

Latency from `fetch` (token id known) to the rows in host memory. "layer 1" is what layer 1 waits for.

Final run, quiet disk (10 s before: 0 MiB read or written; at most 9 MiB written during any configuration), io_uring,
4 KiB reads, 1,500 steps per point (800 for 6-token steps), latency in us:

| configuration | gap ms | layer 1: p50 / p99 / p99.9 / max | step: p50 / p99 / p99.9 | hot hits | disk reads / token |
|---|---:|---|---|---:|---:|
| packed, uniform random ids | 0 | 136 / 410 / 461 / 469 | 140 / 412 / 461 | 0 % | 24.0 |
| packed, uniform random ids | 16 | 172 / 468 / 573 / 594 | 174 / 470 / 594 | 0 % | 24.0 |
| packed, uniform random ids | 75 | 594 / 887 / 1286 / 1398 | 595 / 887 / 1286 | 0 % | 24.0 |
| + touch 1 ms ahead | 16 | 195 / 368 / 449 / 519 | 195 / 389 / 449 | 0 % | 24.0 |
| + touch 1 ms ahead | 75 | 194 / 407 / 529 / 974 | 195 / 416 / 697 | 0 % | 24.0 |
| + touch + prefetch top-4 0.5 ms ahead (assumes 100 % top-4 recall) | 75 | 21 / 39 / 104 / 557 | 21 / 39 / 104 | 0 % | 0.0 |
| packed, real text, no hot tier | 0 | 130 / 321 / 434 / 712 | 134 / 362 / 468 | 0 % | 20.0 |
| packed, real text, no hot tier | 75 | 583 / 9642 / 10108 / 10505 | 584 / 9642 / 10108 | 0 % | 20.9 |
| + touch 1 ms ahead | 75 | 123 / 236 / 336 / 456 | 138 / 263 / 456 | 0 % | 20.0 |
| real text, 1 GiB hot tier | 0 | 3 / 265 / 354 / 391 | 3 / 278 / 437 | 71 % | 7.1 |
| real text, 1 GiB hot tier | 16 | 5 / 9605 / 9639 / 9662 | 5 / 9610 / 9662 | 61 % | 9.4 |
| real text, 1 GiB hot tier | 75 | 5 / 9680 / 10516 / 10537 | 5 / 9680 / 10516 | 70 % | 7.3 |
| + touch 1 ms ahead | 16 | 6 / 293 / 330 / 334 | 6 / 299 / 408 | 71 % | 7.1 |
| + touch 1 ms ahead | 75 | 6 / 299 / 332 / 404 | 6 / 303 / 395 | 61 % | 9.4 |
| real text, hot, touch, DSpark step (6 tokens) | 0 | 140 / 402 / 1584 / 1584 | 179 / 489 / 1584 | 67 % | 7.9 |
| real text, hot, touch, DSpark step (6 tokens) | 75 | 21 / 452 / 652 / 652 | 21 / 515 / 988 | 75 % | 5.9 |
| safetensors, scales read per row (2 IOs), touch | 0 | 252 / 553 / 643 / 697 | 268 / 556 / 643 | 0 % | 24.0 |
| safetensors, scales read per row (2 IOs), touch | 75 | 209 / 569 / 698 / 703 | 211 / 569 / 698 | 0 % | 24.0 |
| safetensors, resident scales (3.07 GB), touch | 75 | 151 / 315 / 486 / 528 | 154 / 375 / 486 | 0 % | 24.0 |

Findings:
- **Touch every step.** Without it, layer 1 waits ~0.6 ms after a 75 ms gap (p99.9 1.3 ms), and when a step needs no
  disk reads at all (hot tier or staging served everything) the following idle stretch is longer and the next read
  costs **~9.6 ms**: the idle-gap probe (`raw/raw-idle.json`, single 4 KiB read after an idle gap) gives p50 0.50-0.63 ms
  after 50-100 ms idle and **9.67 ms after 150 ms or more**, so the drive enters a deep power state between 100 and
  150 ms of idle (APST: `default_ps_max_latency_us=100000` permits it; turning APST down is a root-only system setting) (the p99 of 9.6 ms in the trace runs without touch; constant
  across runs, gone with touch). One 512 B read per step, about 1 ms before the ids are known, keeps p99 at
  0.24-0.41 ms and p99.9 at 0.33-0.53 ms at 16-75 ms gaps. It is cheap (one IO per step) and must be issued even
  when the step has no misses.
- **Hot tier**: 61-75 % of lookups hit; p50 drops from ~130 us to 3-6 us (most steps need no IO, 7-9 reads per
  token instead of 20-24), p99 stays one IO round (~0.3 ms).
- **Top-k prefetch** 0.5 ms before the id is known (with the true token among the 4): p50 21 us, p99 39 us, p99.9
  104 us. That meets the coordinator's ~0.4 ms layer-1 budget at p99.9; the cost is 4x the reads (the device has
  >20x headroom at decode). It needs the candidates ~0.5 ms early (top-k logits of the previous step, or DSpark
  drafts) and helps only when the sampled token is among them.
- **Safetensors with per-row scale reads** (2 IOs per row) is ~1.5-2x slower than the packed layout; with resident
  scales it matches packed but costs 3.07 GB/rank of RAM. Packed is the layout to use.
- DSpark 6-token steps (144 lookups, ~47 disk reads after the hot tier): p99 0.45-0.52 ms.


Hit rates in the `token` runs sample the held-out stream at 30 random places (`--trace-hop 50`); the offline
numbers below use the whole stream.

## Hot tier hit rate (`engram-bench hit`, no disk)

Corpus: 6.0M tokens of developer text (intarsiaml docs and crates, fleetq, deepeval, model-benchmarks; 1,848
files), file-level split 3.16M train / 2.85M test, real tokenizer and hash. Static = top-N rows of the train
split pinned; tier = 75 % pinned + 25 % TinyLFU dynamic, simulated on the test stream. Per rank, its own rows.

| rows (MB) | rank 1 static | rank 1 tier | rank 0 static | rank 0 tier |
|---:|---:|---:|---:|---:|
| 10K (3) | 24.2 % | 32.5 % | 39.1 % | 47.1 % |
| 100K (26) | 45.9 % | 55.0 % | 61.7 % | 69.7 % |
| 1M (264) | 62.0 % | 65.8 % | 78.2 % | 80.8 % |
| 2M (528) | 65.6 % | 68.7 % | 81.1 % | 83.3 % |
| 4M (1,056) | 68.0 % | 70.9 % | 83.5 % | 85.5 % |
| 8M (2,112) | 70.0 % | 73.4 % | 85.5 % | 87.7 % |

Rank 1 per n-gram order at 4M: 3-gram 73.7 %, 4-gram 65.1 %. These are higher than the scope doc's 56 % at 4M
(its corpus was smaller and broader); the train and test files here share repositories, so treat them as
optimistic for chat traffic. The real profile should come from representative traffic.

## Integration with the skeleton's `EngramTier`

`NvmeEngramTier` implements `intarsia_core::dsv41::engram::EngramTier` (`Send + Sync`, `&self`; the store sits
behind a mutex that `wait` releases between polls). Trait change, minimal and additive (defaults keep
`MemEngramTier` unchanged):

```rust
fn owned_rows(&self, layer: usize) -> Range<u64> { self.shard().range(self.layout().rows[layer]) } // NVMe: column split
fn touch(&self) {}                                                  // wake the drive ~1 ms before the ids are known
fn prefetch(&self, _layer: usize, _rows: &[u32]) -> Result<()> { Ok(()) } // top-k candidates / drafts
```

Why: `owned_rows` because the tables on disk are column-split (see above) and `EngramShard::range` can only express
the ceil split; `touch` because without it layer 1 waits 0.6-1.3 ms after every decode-step gap; `prefetch` because the
coordinator's ~0.4 ms layer-1 budget is under the device's p99.9.

Engine side:
1. Host, per sequence: `NgramState` (or `NgramHasher::hash` on the history) -> 24 row ids per layer per token.
2. `touch()` about 1 ms before sampling finishes (start of the last layer or lm_head), then `issue` both layers the
   moment the ids are on the host. Optionally `prefetch` the rows of the top-k candidates / DSpark drafts.
   Also `touch()` at request admission (well before prefill reaches layer 1) and keep touching every step while any
   sequence is active: after a long idle the first read costs ~9.6 ms, which a 1 ms lead cannot hide.
3. Split the captured graph at layers 1 and 14 (or poll a host flag from a device wait kernel) and `wait` between the
   segments; layer 14 has ~13 layers of slack.
4. Rows other ranks own come back as zero payload; zero e4m3 bytes dequantize to exactly 0 whatever the scale, so the
   all-reduce of the gathered rows works unchanged.
5. Prefill: issue a chunk's rows per layer in one call (reads queue beyond the depth, duplicates are read once).

## Reproduce

```bash
# hash fixture (any env with torch, numpy, sympy, tokenizers)
python crates/intarsia-engram/tests/fixtures/gen_hash_fixture.py --official <inference dir> \
  --tokenizer <tokenizer.json> --out crates/intarsia-engram/tests/fixtures/hash_dsv41.json
cargo test -p intarsia-engram
INTARSIA_DSV41_TOKEN_MAP=dsv41_token_map.json INTARSIA_DSV41_TOKENIZER=tokenizer.json \
  cargo test -p intarsia-engram --features tokenizer --test hash_parity
# corpus, profile, hit rate
engram-bench trace --tokenizer tokenizer.json --out-train train.u32 --out-test test.u32 --max-tokens 6000000 <dirs>
engram-bench hit --train train.u32 --test test.u32 --rank 1 --world 2
engram-bench profile --train train.u32 --rank 1 --world 2 --rows 3000000 --out profile-r1-3M.u64
# on spark2 (fleetq: --load disk,cpu --quiet disk)
engram-bench verify --a ~/.cache/dsv41-tensorfold/engram-1bae0e10 --b ~/dsv41-engram --rank 1 --world 2
engram-bench raw --file ~/dsv41-engram/engram-l1-r1of2.bin --bs 512,4096 --backend uring
engram-bench token --dir ~/dsv41-engram --rank 1 --world 2 --granule page --gap-ms 0,16,75 \
  --trace test.u32 --hot-mb 1024 --profile profile-r1-3M.u64 --touch --touch-lead-us 1000
```

Run history: `raw/run1` and `raw/run2` partly overlapped a staging job writing ~400 MB/s to the same disk. I first
blamed the ~10 ms stalls seen there on it, but the final run on a quiet disk (`raw/token-final.txt`, disk traffic
logged per configuration) reproduces them without touch and removes them with touch: they are the drive's deep
idle state, not the writer. The raw device run had a quiet disk (233 writes in the 5 s before it). Preloading 3M
pinned rows at 4 KiB granule reads 12 GB in 4.5 s (O_DIRECT, no page cache).

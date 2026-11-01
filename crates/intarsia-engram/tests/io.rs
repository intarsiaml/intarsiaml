//! Every layout x backend x granule returns the same rows, through the hot tier, staging, prefetch, queueing
//! beyond the queue depth and rows other ranks own.

use intarsia_engram::layout::{write_paged, PACKED_MAGIC};
use intarsia_engram::{
    BackendKind, EngramStore, HotTierConfig, LayoutKind, ReadGranule, ScaleMode, StoreConfig, TableShard, RECORD_BYTES,
};
use std::path::{Path, PathBuf};

const ROWS: u64 = 5000;
const LAYER: u32 = 14;

fn byte(row: u64, j: usize) -> u8 {
    let mut z = row.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 29;
    (z.wrapping_mul(0x94D0_49BB_1331_11EB) >> 56) as u8
}

fn record(row: u64) -> Vec<u8> {
    (0..RECORD_BYTES).map(|j| byte(row, j)).collect()
}

fn tmpdir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("engram-io-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// An official-style safetensors file: a small tensor first so the table starts unaligned, then weight, scale.
fn write_safetensors(dir: &Path) -> PathBuf {
    let pre = 100u64;
    let w = ROWS * 256;
    let s = ROWS * 8;
    let json = format!(
        "{{\"__metadata__\":{{\"format\":\"pt\"}},\"layers.{LAYER}.engram.q_weight\":{{\"dtype\":\"BF16\",\"shape\":[50],\"data_offsets\":[0,{pre}]}},\
         \"layers.{LAYER}.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{ROWS},256],\"data_offsets\":[{pre},{}]}},\
         \"layers.{LAYER}.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{ROWS},8],\"data_offsets\":[{},{}]}}}}",
        pre + w,
        pre + w,
        pre + w + s
    );
    let mut f = (json.len() as u64).to_le_bytes().to_vec();
    f.extend_from_slice(json.as_bytes());
    f.extend((0..pre).map(|i| i as u8));
    for r in 0..ROWS {
        f.extend_from_slice(&record(r)[..256]);
    }
    for r in 0..ROWS {
        f.extend_from_slice(&record(r)[256..]);
    }
    let p = dir.join("model-00048-of-00048.safetensors");
    std::fs::write(&p, f).unwrap();
    p
}

fn write_packed(dir: &Path, lo: u64, hi: u64) -> PathBuf {
    let mut f = vec![0u8; 4096];
    for (i, v) in [PACKED_MAGIC, LAYER as u64, lo, hi, ROWS, RECORD_BYTES as u64].iter().enumerate() {
        f[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    for r in lo..hi {
        f.extend_from_slice(&record(r));
    }
    let p = dir.join(format!("engram-l{LAYER}-r1of2.bin"));
    std::fs::write(&p, f).unwrap();
    p
}

fn backends() -> Vec<BackendKind> {
    let mut v = vec![BackendKind::Pool { threads: 0 }, BackendKind::Pool { threads: 4 }];
    if cfg!(target_os = "linux") {
        v.push(BackendKind::Uring { sqpoll_idle_ms: None });
    }
    v
}

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn check_store(mut store: EngramStore, owned: std::ops::Range<u64>, label: &str) {
    let mut s = 0x1234_5678u64;
    // several batches in flight at once, including duplicates and foreign rows, more rows than the depth
    for round in 0..6 {
        let n = [1usize, 24, 48, 300, 700, 24][round];
        let batches: Vec<Vec<u64>> = (0..3)
            .map(|_| {
                (0..n)
                    .map(|i| if i % 7 == 3 && i > 0 { owned.start + i as u64 % 3 } else { rng(&mut s) % ROWS })
                    .collect()
            })
            .collect();
        let tickets: Vec<_> = batches.iter().map(|b| store.fetch(0, b).unwrap()).collect();
        for (b, t) in batches.iter().zip(tickets) {
            let mut w = vec![1u8; b.len() * 256];
            let mut sc = vec![1u8; b.len() * 8];
            let info = store.wait(t, &mut w, &mut sc).unwrap();
            for (i, &row) in b.iter().enumerate() {
                let want = if owned.contains(&row) { record(row) } else { vec![0u8; RECORD_BYTES] };
                assert_eq!(&w[i * 256..(i + 1) * 256], &want[..256], "{label}: weights of row {row}");
                assert_eq!(&sc[i * 8..(i + 1) * 8], &want[256..], "{label}: scales of row {row}");
            }
            assert_eq!(info.rows, b.len());
        }
    }
    // prefetch then fetch: rows come from staging or join the reads
    let rows: Vec<u64> = (0..64).map(|i| owned.start + i * 7 % (owned.end - owned.start)).collect();
    store.prefetch(0, &rows).unwrap();
    let t = store.fetch(0, &rows).unwrap();
    let mut rec = vec![0u8; rows.len() * RECORD_BYTES];
    let info = store.wait_records(t, &mut rec).unwrap();
    assert_eq!(info.reads, 0, "{label}: prefetched rows read again");
    for (i, &row) in rows.iter().enumerate() {
        assert_eq!(&rec[i * RECORD_BYTES..(i + 1) * RECORD_BYTES], &record(row)[..], "{label}: prefetched row {row}");
    }
    // poll never blocks and eventually says ready
    let t = store.fetch(0, &[owned.end - 1]).unwrap();
    while !store.poll(t).unwrap() {}
    store.wait_records(t, &mut rec).unwrap();
    assert_eq!(&rec[..RECORD_BYTES], &record(owned.end - 1)[..]);
}

fn cfg(backend: BackendKind, granule: ReadGranule) -> StoreConfig {
    StoreConfig { backend, depth: 64, granule, staging_rows: 128, ..Default::default() }
}

#[test]
fn all_layouts_all_backends_same_rows() {
    let dir = tmpdir("layouts");
    let st = write_safetensors(&dir);
    let owned = 1700..4900;
    write_packed(&dir, owned.start, owned.end);
    // repack from the safetensors into the paged layout
    {
        let src = TableShard::open_safetensors(&st, LAYER, Some(owned.clone()), ScaleMode::Read, true).unwrap();
        let out = dir.join(format!("engram-l{LAYER}-r1of2.paged"));
        let n = write_paged(&src, owned.clone(), &out, 1000, |_, _| {}).unwrap();
        assert_eq!(n, 4096 + (3200u64).div_ceil(15) * 4096);
    }
    for backend in backends() {
        for granule in [ReadGranule::Sector, ReadGranule::Page] {
            for mode in [ScaleMode::Resident, ScaleMode::Read] {
                let sh = TableShard::open_safetensors(&st, LAYER, Some(owned.clone()), mode, true).unwrap();
                assert_eq!(sh.kind, LayoutKind::Safetensors);
                assert_eq!(sh.scales().is_some(), mode == ScaleMode::Resident);
                let label = format!("safetensors {mode:?} {backend:?} {granule:?}");
                check_store(EngramStore::new(vec![sh], cfg(backend, granule)).unwrap(), owned.clone(), &label);
            }
            let packed = dir.join(format!("engram-l{LAYER}-r1of2.bin"));
            let sh = TableShard::open_shard_file(&packed, true).unwrap();
            assert_eq!((sh.kind, sh.rows.clone(), sh.layer_id), (LayoutKind::Packed, owned.clone(), LAYER));
            check_store(EngramStore::new(vec![sh], cfg(backend, granule)).unwrap(), owned.clone(), &format!("packed {backend:?} {granule:?}"));
            // open_dir prefers the paged shard
            let sh = TableShard::open_dir(&dir, LAYER, 1, 2, owned.clone(), ScaleMode::Read, true).unwrap();
            assert_eq!(sh.kind, LayoutKind::Paged);
            check_store(EngramStore::new(vec![sh], cfg(backend, granule)).unwrap(), owned.clone(), &format!("paged {backend:?} {granule:?}"));
        }
    }
    // a whole-table safetensors shard found by open_dir in a directory without shard files
    let d2 = tmpdir("dir");
    write_safetensors(&d2);
    let sh = TableShard::open_dir(&d2, LAYER, 0, 1, 0..ROWS, ScaleMode::Read, true).unwrap();
    assert_eq!((sh.kind, sh.rows.clone()), (LayoutKind::Safetensors, 0..ROWS));
    let mut rec = vec![0u8; RECORD_BYTES];
    for row in [0, 1, 2, ROWS - 1] {
        sh.read_row_sync(row, &mut rec).unwrap();
        assert_eq!(rec, record(row));
    }
    assert!(TableShard::open_dir(&d2, 1, 0, 1, 0..ROWS, ScaleMode::Read, true).is_err());
}

#[test]
fn hot_tier_preload_and_admission() {
    let dir = tmpdir("hot");
    let owned = 1000..4000;
    let packed = write_packed(&dir, owned.start, owned.end);
    let sh = TableShard::open_shard_file(&packed, true).unwrap();
    let hot = HotTierConfig { capacity_bytes: 200 * RECORD_BYTES, dynamic_fraction: 0.5, sketch_log2: 12 };
    let mut store = EngramStore::new(vec![sh], StoreConfig { hot, staging_rows: 0, depth: 32, ..Default::default() }).unwrap();
    let profile: Vec<(usize, u64)> = (0..500).map(|i| (0, 1000 + i * 5)).collect();
    assert_eq!(store.preload(&profile).unwrap(), 100);
    // pinned rows: no reads, right bytes
    let rows: Vec<u64> = (0..100).map(|i| 1000 + i * 5).collect();
    let t = store.fetch(0, &rows).unwrap();
    let mut rec = vec![0u8; rows.len() * RECORD_BYTES];
    let info = store.wait_records(t, &mut rec).unwrap();
    assert_eq!((info.hot_hits, info.reads), (100, 0));
    for (i, &row) in rows.iter().enumerate() {
        assert_eq!(&rec[i * RECORD_BYTES..(i + 1) * RECORD_BYTES], &record(row)[..]);
    }
    // a repeatedly fetched row is admitted to the dynamic region after its first read
    for k in 0..3 {
        let t = store.fetch(0, &[3333]).unwrap();
        let info = store.wait_records(t, &mut rec).unwrap();
        assert_eq!(&rec[..RECORD_BYTES], &record(3333)[..]);
        assert_eq!(info.hot_hits, (k > 0) as usize);
    }
    let hs = store.hot_stats();
    assert_eq!(hs.pinned_rows, 100);
    assert!(hs.dynamic_rows >= 1);
}

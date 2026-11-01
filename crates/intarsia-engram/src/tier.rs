//! [`NvmeEngramTier`]: the store behind intarsia-core's `EngramTier` trait.

use crate::hash::HashLayout;
use crate::store::{EngramStore, Ticket};
use intarsia_core::dsv41::engram::{EngramLayout, EngramShard, EngramTicket, EngramTier};
use intarsia_core::error::{Error, Result};
use std::ops::Range;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// One rank's Engram tables on local NVMe (or any layout [`crate::TableShard`] reads), as an `EngramTier`.
///
/// The trait takes `&self`; the store sits behind a mutex that `wait` releases between polls, so `issue`/`touch`
/// from another thread are not held up by a wait (one decode thread is the expected user).
pub struct NvmeEngramTier {
    layout: EngramLayout,
    shard: EngramShard,
    owned: Vec<Range<u64>>,
    spin: Duration,
    store: Mutex<EngramStore>,
}

impl NvmeEngramTier {
    /// `store.shards()[i]` must hold layer `hash.layer_ids()[i]`'s rows under the bucket-boundary split
    /// ([`HashLayout::rank_rows`]) of `rank` / `world`.
    pub fn new(store: EngramStore, hash: &HashLayout, rank: usize, world: usize) -> Result<Self> {
        let owned: Vec<Range<u64>> = (0..hash.n_layers()).map(|l| hash.rank_rows(l, rank, world)).collect();
        if store.shards().len() != owned.len() {
            return Err(Error::invalid(format!("engram tier: {} shards for {} Engram layers", store.shards().len(), owned.len())));
        }
        for (i, (s, r)) in store.shards().iter().zip(&owned).enumerate() {
            if s.layer_id as usize != hash.layer_ids()[i] || s.rows != *r {
                return Err(Error::model(format!(
                    "engram tier: shard {} holds layer {} rows {:?}, want layer {} rows {r:?}",
                    s.path.display(),
                    s.layer_id,
                    s.rows,
                    hash.layer_ids()[i]
                )));
            }
        }
        // the tables must not go through the page cache (it is GPU memory on a unified-memory box)
        if cfg!(target_os = "linux") {
            if let Some(s) = store.shards().iter().find(|s| !s.direct()) {
                return Err(Error::model(format!("engram tier: {} could not be opened with O_DIRECT", s.path.display())));
            }
        }
        let spin = store.config().spin;
        Ok(Self { layout: hash.layout().clone(), shard: EngramShard { rank, world }, owned, spin, store: Mutex::new(store) })
    }

    /// The store (stats, hot tier, preload).
    pub fn store(&self) -> std::sync::MutexGuard<'_, EngramStore> {
        self.store.lock().unwrap()
    }
}

fn rows64(rows: &[u32]) -> Vec<u64> {
    rows.iter().map(|&r| r as u64).collect()
}

impl EngramTier for NvmeEngramTier {
    fn layout(&self) -> &EngramLayout {
        &self.layout
    }
    fn shard(&self) -> EngramShard {
        self.shard
    }
    fn owned_rows(&self, layer: usize) -> Range<u64> {
        self.owned[layer].clone()
    }
    fn issue(&self, layer: usize, rows: &[u32]) -> Result<EngramTicket> {
        if let Some(&r) = rows.iter().find(|&&r| layer < self.layout.rows.len() && r as u64 >= self.layout.rows[layer]) {
            return Err(Error::invalid(format!("engram tier: row {r} >= {} rows", self.layout.rows[layer])));
        }
        Ok(EngramTicket(self.store().fetch(layer, &rows64(rows))?.id()))
    }
    fn wait(&self, ticket: EngramTicket, payload: &mut [u8], scales: &mut [u8]) -> Result<()> {
        let t = Ticket::from_id(ticket.0);
        let start = Instant::now();
        loop {
            let mut s = self.store();
            if s.poll(t)? || start.elapsed() >= self.spin {
                // ready, or done spinning: the store's own wait blocks in the kernel
                s.wait(t, payload, scales)?;
                return Ok(());
            }
            drop(s);
            std::hint::spin_loop();
        }
    }
    fn touch(&self) {
        let _ = self.store().touch();
    }
    fn prefetch(&self, layer: usize, rows: &[u32]) -> Result<()> {
        self.store().prefetch(layer, &rows64(rows))?;
        Ok(())
    }
}

//! `intarsia dsv41-cache`: build the per-rank weight cache of a DeepSeek-V4.1 Flash EXL3 checkpoint
//! (see `intarsia_models::dsv41_cache`): a rank-local, already-sharded copy on local NVMe that the TP loader streams with O_DIRECT.

use anyhow::{bail, Result};
use clap::Args;
use intarsia_models::dsv41_cache::{build_rank_cache, cache_matches, BuildOpts};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct Dsv41CacheArgs {
    /// Source checkpoint directory (safetensors shards + index).
    #[arg(long)]
    pub model: PathBuf,
    /// Cache directory to write.
    #[arg(long)]
    pub out: PathBuf,
    /// This rank (0-based).
    #[arg(long)]
    pub rank: usize,
    /// Number of ranks.
    #[arg(long)]
    pub world: usize,
    /// Keep only the first N layers (cheap trial caches).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Rebuild even if a matching cache exists, and write into a non-empty directory without a manifest.
    #[arg(long)]
    pub force: bool,
}

pub fn run(args: Dsv41CacheArgs) -> Result<()> {
    if args.world == 0 || args.rank >= args.world {
        bail!("--rank {} is outside --world {}", args.rank, args.world);
    }
    if !args.force && cache_matches(&args.out, &args.model, args.rank, args.world, args.layers)? {
        println!("cache at {} already matches (rank {} of {}, layers {:?}); nothing to do (--force rebuilds)", args.out.display(), args.rank, args.world, args.layers);
        return Ok(());
    }
    let mut o = BuildOpts::new(args.rank, args.world, args.layers);
    o.force = args.force;
    let m = build_rank_cache(&args.model, &args.out, &o)?;
    println!("built {}: {} tensors, {:.2} GiB in {} parts", args.out.display(), m.tensors, m.bytes as f64 / (1u64 << 30) as f64, m.parts.len());
    Ok(())
}

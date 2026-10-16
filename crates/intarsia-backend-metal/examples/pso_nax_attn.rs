//! Prints the pipeline limits of the NAX flash attention variants (`maxTotalThreadsPerThreadgroup` shows register
//! pressure) and, for a chunk of `T` query rows, how many threadgroups the launch makes. No dispatch.
//!   cargo run --release -p intarsia-backend-metal --example pso_nax_attn [T...]
#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::flash_nax::NAX_FLASH_VARIANTS;
    use objc2_metal::MTLComputePipelineState;
    let be = intarsia_backend_metal::MetalBackend::new().unwrap();
    let ts: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (g, nkv) = (4usize, 8usize); // Qwen3-8B: 32 query heads over 8 kv heads
    for v in NAX_FLASH_VARIANTS {
        let pso = be.pso(v.name).unwrap();
        println!(
            "{}: nsg {} bk {} maxTotalThreads {} execWidth {} staticTgMem {}",
            v.name,
            v.nsg,
            v.bk,
            pso.maxTotalThreadsPerThreadgroup(),
            pso.threadExecutionWidth(),
            pso.staticThreadgroupMemoryLength()
        );
        if v.nsg % g == 0 {
            let bq = 16 * (v.nsg / g);
            for &t in &ts {
                println!("  T {t}: {} x {nkv} = {} threadgroups of {} threads", t.div_ceil(bq), t.div_ceil(bq) * nkv, v.nsg * 32);
            }
        }
    }
}
#[cfg(not(target_os = "macos"))]
fn main() {}

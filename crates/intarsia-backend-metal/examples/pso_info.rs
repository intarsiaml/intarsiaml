//! Prints pipeline limits (registers show up as maxTotalThreadsPerThreadgroup) of the multi-row shaders. No dispatch.
#[cfg(target_os = "macos")]
fn main() {
    use intarsia_backend_metal::multirow::{shader, Family, MR_PER_TG, MR_ROWS};
    use objc2_metal::MTLComputePipelineState;
    let be = intarsia_backend_metal::MetalBackend::new().unwrap();
    for f in Family::ALL {
        for &p in &MR_PER_TG {
            let mut line = format!("{f:?} P={p}:");
            for &r in &MR_ROWS {
                let pso = be.pso(shader(f, r, p).unwrap()).unwrap();
                line += &format!(" R{r}:{}/{}", pso.maxTotalThreadsPerThreadgroup(), pso.staticThreadgroupMemoryLength());
            }
            println!("{line}");
        }
    }
}
#[cfg(not(target_os = "macos"))]
fn main() {}

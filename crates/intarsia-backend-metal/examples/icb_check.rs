//! Metal spike part (d): record the Qwen3-8B decode-shaped GEMV sequence into an `MTLIndirectCommandBuffer`, replay it
//! with `executeCommandsInBuffer`, check the output is bit-identical to direct encoding, and compare CPU encode time and
//! GPU time per step.
//!
//!   cargo run --release -p intarsia-backend-metal --example icb_check
//!
//! Run it inside a fleetq slot (`fleetq run --host endor --exclusive ...`); it allocates ~15 GB of synthetic weights.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("icb_check needs macOS / Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main()
}

#[cfg(target_os = "macos")]
mod imp {
    use intarsia_backend_metal::bench::{gpu_seconds, stats, Stats};
    use intarsia_backend_metal::decode::{DecodeSeq, Icb, LAYERS};
    use intarsia_backend_metal::kernels::DEFAULT_GEMV;
    use intarsia_backend_metal::MetalBackend;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
    use std::time::Instant;

    const WARMUP: usize = 5;
    const REPS: usize = 100;

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Mode {
        Direct,
        DirectUnretained,
        IcbBarrier,
        IcbConcurrent,
    }

    pub fn main() {
        let be = MetalBackend::new().expect("metal backend");
        println!("device: {}", objc2_metal::MTLDevice::name(be.device()));
        let seq = DecodeSeq::qwen3_8b(&be, LAYERS, true).expect("allocate decode weights");
        println!(
            "{} GEMVs per step, step bytes {:.2} GB, variant {}",
            seq.ops.len(),
            seq.step_bytes() as f64 / 1e9,
            DEFAULT_GEMV.name
        );
        let pick = |_: usize, _: usize| DEFAULT_GEMV;

        let t = Instant::now();
        let icb_barrier = Icb::build(&be, &seq, &pick, true).expect("build ICB (barrier)");
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let icb_conc = Icb::build(&be, &seq, &pick, false).expect("build ICB (no barrier)");
        println!("ICB records {} compute commands; one-time record cost {build_ms:.2} ms", icb_barrier.count);

        // one step in `mode`: (cpu encode+commit seconds, gpu seconds, wall seconds incl. wait)
        let step = |mode: Mode| -> (f64, f64, f64) {
            let t0 = Instant::now();
            let cb = if mode == Mode::DirectUnretained {
                be.queue().commandBufferWithUnretainedReferences().unwrap()
            } else {
                be.queue().commandBuffer().unwrap()
            };
            let enc = cb.computeCommandEncoder().unwrap();
            match mode {
                Mode::Direct | Mode::DirectUnretained => seq.encode_direct(&be, &enc, &pick),
                Mode::IcbBarrier => icb_barrier.execute(&enc),
                Mode::IcbConcurrent => icb_conc.execute(&enc),
            }
            enc.endEncoding();
            be.commit(&cb);
            let t1 = Instant::now();
            MetalBackend::wait(&cb).unwrap();
            let t2 = Instant::now();
            ((t1 - t0).as_secs_f64(), gpu_seconds(&cb), (t2 - t0).as_secs_f64())
        };

        // bit-for-bit check against the direct run
        seq.clear_outputs();
        step(Mode::Direct);
        let reference = seq.read_outputs();
        assert!(reference.iter().any(|b| *b != 0), "direct run produced only zeros");
        let mut all_equal = true;
        for mode in [Mode::DirectUnretained, Mode::IcbBarrier, Mode::IcbConcurrent] {
            seq.clear_outputs();
            step(mode);
            let got = seq.read_outputs();
            let diff = got.iter().zip(&reference).filter(|(a, b)| a != b).count();
            println!(
                "RESULT bit_exact mode={mode:?} identical={} differing_bytes={diff} of {}",
                diff == 0,
                reference.len()
            );
            all_equal &= diff == 0;
        }
        println!("RESULT bit_exact_all={all_equal}");

        // timing, interleaved so thermal / clock drift hits every mode alike
        let modes = [Mode::Direct, Mode::DirectUnretained, Mode::IcbBarrier, Mode::IcbConcurrent];
        let mut cpu = vec![Vec::new(); modes.len()];
        let mut gpu = vec![Vec::new(); modes.len()];
        let mut wall = vec![Vec::new(); modes.len()];
        for rep in 0..WARMUP + REPS {
            for (i, m) in modes.iter().enumerate() {
                let (c, g, w) = step(*m);
                if rep >= WARMUP {
                    cpu[i].push(c);
                    gpu[i].push(g);
                    wall[i].push(w);
                }
            }
        }
        let ms = |s: Stats| format!("median {:.3} p10 {:.3} p90 {:.3}", s.median * 1e3, s.p10 * 1e3, s.p90 * 1e3);
        for (i, m) in modes.iter().enumerate() {
            println!(
                "RESULT timing mode={m:?} cpu_encode_commit_ms[{}] gpu_ms[{}] wall_ms[{}]",
                ms(stats(&mut cpu[i])),
                ms(stats(&mut gpu[i])),
                ms(stats(&mut wall[i]))
            );
        }
        let base = stats(&mut cpu[0]).median;
        let icb = stats(&mut cpu[2]).median;
        println!(
            "RESULT summary: direct encode {:.3} ms vs ICB replay {:.3} ms per step (saving {:.3} ms, {:.1}x)",
            base * 1e3,
            icb * 1e3,
            (base - icb) * 1e3,
            base / icb
        );
    }
}

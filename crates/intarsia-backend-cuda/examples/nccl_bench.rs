//! NCCL all-reduce vs all-gather latency between two processes (two Sparks), through the engine's own `Comm`
//! (the same libnccl, the same stream, the same settings as the TP=2 decode).
//!
//!   rank 1:  nccl_bench serve 0.0.0.0:29711
//!   rank 0:  nccl_bench connect HOST:29711
//!
//! Both ranks print one table (rank 0 is the one to read): for each payload (bytes of f32 one rank contributes):
//! `ar_inplace` ncclAllReduce f32 in place, `ar_into_bf16` the engine's `all_reduce_sum_into` (allreduce, then the f32
//! -> bf16 round kernel), `ag` ncclAllGather of the same payload (rank-major `[2, n]` out). Each cell is us per call,
//! two ways: `pipe` = `iters` calls queued back to back and one sync, `sync` = one call and a device sync each time
//! (the model's collectives sit between kernels of the same stream, so `pipe` is the closer to what a pass pays).
//! Env: INTARSIA_NCCL_LIB (libnccl path), NCCL_* as for the engine.
//! `NB_BYTES=a,b,c` replaces the payload list (default: the decode sizes up to 128 KiB, 1 MiB, 21176320 = a 1034-row
//! prefill all-reduce, 41943040 = a 2048-row one). Payloads of 1 MiB and more run `NB_BIG_ITERS` (default 40) calls, and
//! the line gets the all-reduce bandwidth in GB/s (bytes one rank contributes over the `pipe` time).
//! `NB_COLD=N` (needs a payload of 1 MiB or more; the last one is used) adds an idle-start test after the table: N rounds of
//! sleep `NB_COLD_MS` (default 3000, the gap between requests), then (a) the big all-reduce alone, (b) an all-reduce of
//! `NB_COLD_LEAD_BYTES` (default 16384), `NB_COLD_LEAD_MS` (default 7) later the big one, (c) the big one again at once (warm). Medians of us, per call.

use intarsia_core::{DType, MemKind, TensorRef};
use std::io::{Read, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: nccl_bench serve ADDR | connect ADDR [device] [iters]");
        std::process::exit(2);
    }
    let device: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let iters: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(300);
    let (rank, id, _stream) = if args[1] == "serve" {
        let l = std::net::TcpListener::bind(&args[2]).expect("bind");
        let (mut s, peer) = l.accept().expect("accept");
        eprintln!("rank 1: rank 0 connected from {peer}");
        let mut id = [0u8; 128];
        s.read_exact(&mut id).expect("id");
        (1usize, id, s)
    } else {
        let mut s = loop {
            match std::net::TcpStream::connect(&args[2]) {
                Ok(s) => break s,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(500)),
            }
        };
        let id = intarsia_backend_cuda::nccl::unique_id().expect("unique id");
        s.write_all(&id).expect("send id");
        (0usize, id, s)
    };
    eprintln!("rank {rank}: NCCL {}", intarsia_backend_cuda::nccl::version().unwrap_or_default());
    let (be, comm) = intarsia_backend_cuda::nccl::create(device, rank, 2, id).expect("nccl create");
    let sizes: Vec<usize> = match std::env::var("NB_BYTES") {
        Ok(v) => v.split(',').map(|x| x.trim().parse().expect("NB_BYTES: comma-separated byte counts")).collect(),
        Err(_) => vec![16384, 32768, 49152, 65536, 98304, 131072, 1 << 20, 21_176_320, 41_943_040],
    };
    let big_iters: usize = std::env::var("NB_BIG_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let max_n = sizes.iter().max().copied().unwrap_or(131072).max(16384) / 4;
    let mk = |dt: DType, n: usize| {
        let b = be.alloc((n * 4).max(256), MemKind::Device).expect("alloc");
        be.zero(&b, 0).expect("zero");
        (b, dt, n)
    };
    let (sb, _, _) = mk(DType::F32, max_n);
    let (rb, _, _) = mk(DType::F32, 2 * max_n);
    let (ob, _, _) = mk(DType::Bf16, max_n);
    comm.reserve(max_n, 0).expect("reserve");
    println!("rank {rank}  bytes/rank   ar_inplace(pipe,sync)  ar_into_bf16(pipe,sync)  ag(pipe,sync)   [us per call]   ar_inplace GB/s");
    for &bytes in &sizes {
        let n = bytes / 4;
        let (iters, warm) = if bytes >= 1 << 20 { (big_iters, 10) } else { (iters, 50) };
        let send = TensorRef::new(sb.clone(), DType::F32, &[n]);
        let recv = TensorRef::new(rb.clone(), DType::F32, &[2 * n]);
        let out = TensorRef::new(ob.clone(), DType::Bf16, &[n]);
        let mut row = Vec::new();
        let mut gbps = 0.0f64;
        for mode in 0..3 {
            let run = |c: usize, sync_each: bool| {
                let t0 = Instant::now();
                for _ in 0..c {
                    match mode {
                        0 => comm.all_reduce_sum(&send, 0).expect("ar"),
                        1 => comm.all_reduce_sum_into(&send, &out, 0).expect("ar_into"),
                        _ => comm.all_gather(&send, &recv, 0).expect("ag"),
                    }
                    if sync_each {
                        be.synchronize(0).expect("sync");
                    }
                }
                be.synchronize(0).expect("sync");
                t0.elapsed().as_secs_f64() * 1e6 / c as f64
            };
            run(warm, false); // warm
            let pipe = run(iters, false);
            let sync = run((iters / 3).max(3), true);
            row.push(format!("{pipe:7.1} {sync:7.1}"));
            if mode == 0 {
                gbps = bytes as f64 / (pipe * 1e-6) / 1e9;
            }
        }
        println!("rank {rank}  {bytes:>9}   {}   {gbps:6.2}", row.join("   "));
    }
    if let Some(rounds) = std::env::var("NB_COLD").ok().and_then(|v| v.parse::<usize>().ok()) {
        let idle = std::time::Duration::from_millis(std::env::var("NB_COLD_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000));
        let lead = std::time::Duration::from_micros((std::env::var("NB_COLD_LEAD_MS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(7.0) * 1000.0) as u64);
        let big = sizes.iter().copied().filter(|&b| b >= 1 << 20).max().expect("NB_COLD needs a payload of 1 MiB or more");
        let big_t = TensorRef::new(sb.clone(), DType::F32, &[big / 4]);
        let lead_bytes: usize = std::env::var("NB_COLD_LEAD_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(16384);
        let tiny_t = TensorRef::new(sb.clone(), DType::F32, &[lead_bytes / 4]);
        let timed = |f: &dyn Fn()| {
            let t0 = Instant::now();
            f();
            be.synchronize(0).expect("sync");
            t0.elapsed().as_secs_f64() * 1e6
        };
        let (mut cold, mut pre, mut pre_tiny, mut warm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..3 {
            comm.all_reduce_sum(&big_t, 0).expect("ar");
        }
        be.synchronize(0).expect("sync");
        for _ in 0..rounds {
            std::thread::sleep(idle);
            cold.push(timed(&|| comm.all_reduce_sum(&big_t, 0).expect("ar")));
            warm.push(timed(&|| comm.all_reduce_sum(&big_t, 0).expect("ar")));
            std::thread::sleep(idle);
            let t0 = Instant::now();
            comm.all_reduce_sum(&tiny_t, 0).expect("ar tiny");
            std::thread::sleep(lead);
            let t1 = Instant::now();
            comm.all_reduce_sum(&big_t, 0).expect("ar");
            be.synchronize(0).expect("sync");
            let t2 = Instant::now();
            pre.push((t2 - t1).as_secs_f64() * 1e6);
            pre_tiny.push((t2 - t0).as_secs_f64() * 1e6);
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        println!(
            "rank {rank}  cold test ({rounds} rounds, idle {} ms, lead {} us, {big} bytes): median us: after-idle {:.0}  warm-at-once {:.0}  after-idle-with-{lead_bytes}-byte-lead {:.0} (tiny issue to big done {:.0})",
            idle.as_millis(), lead.as_micros(), med(&mut cold), med(&mut warm), med(&mut pre), med(&mut pre_tiny)
        );
        println!("rank {rank}  cold all: {:?}", cold.iter().map(|v| *v as i64).collect::<Vec<_>>());
        println!("rank {rank}  lead all: {:?}", pre.iter().map(|v| *v as i64).collect::<Vec<_>>());
    }
}

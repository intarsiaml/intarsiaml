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
    let max_n = 32768usize; // 128 KiB of f32
    let mk = |dt: DType, n: usize| {
        let b = be.alloc((n * 4).max(256), MemKind::Device).expect("alloc");
        be.zero(&b, 0).expect("zero");
        (b, dt, n)
    };
    let (sb, _, _) = mk(DType::F32, max_n);
    let (rb, _, _) = mk(DType::F32, 2 * max_n);
    let (ob, _, _) = mk(DType::Bf16, max_n);
    comm.reserve(max_n, 0).expect("reserve");
    println!("rank {rank}  bytes/rank   ar_inplace(pipe,sync)  ar_into_bf16(pipe,sync)  ag(pipe,sync)   [us per call]");
    for bytes in [16384usize, 32768, 49152, 65536, 98304, 131072] {
        let n = bytes / 4;
        let send = TensorRef::new(sb.clone(), DType::F32, &[n]);
        let recv = TensorRef::new(rb.clone(), DType::F32, &[2 * n]);
        let out = TensorRef::new(ob.clone(), DType::Bf16, &[n]);
        let mut row = Vec::new();
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
            run(50, false); // warm
            let pipe = run(iters, false);
            let sync = run(iters / 3, true);
            row.push(format!("{pipe:7.1} {sync:7.1}"));
        }
        println!("rank {rank}  {bytes:>9}   {}", row.join("   "));
    }
}

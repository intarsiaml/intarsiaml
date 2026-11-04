//! Bounded-replay prefill graphs (`INTARSIA_DSV41_REPLAY=1`, `intarsia_core::dsv41::replay`) of the real Mia checkpoint (the
//! header-only fixture: every tensor's dtype and shape): the encoder part, the decoder part and a whole chunk under a floor
//! are recorded as checked `OpCall`s on the cpu backend and compared, step by step, with the plain graph. Nothing here runs a
//! kernel; the numerics of the replay are not tested by this file (docs/results/2026-11-03-dsv41-replay.md).

use intarsia_core::dsv41::config::Dsv41Config;
use intarsia_core::dsv41::graph::{build_pass, WeightShapes, build_replay, Buf, Dsv41Graph, Extra, GraphOpts, GraphTp, PassKind, Replay, ReplayPart, Stage, Step};
use intarsia_core::dsv41::replay::{self, Chunk};
use intarsia_core::{DType, Op};
use intarsia_models::dsv41::{detect, HeaderShapes};
use std::path::{Path, PathBuf};

fn mia() -> (Dsv41Config, HeaderShapes) {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv41/mia_headers");
    (detect(&dir).unwrap().expect("a deepseek_v41 config"), HeaderShapes::open(&dir).unwrap())
}

/// The shapes graphs are built against: the unsharded checkpoint (world 1) and, when `INTARSIA_DSV41_CACHE_R0` names a full
/// `--world 2` rank cache (as `dsv41.rs` does), rank 0's shards: the collectives only exist at world 2, so the tensor-parallel
/// structure is checked on that cache and not otherwise.
fn setups(w1: HeaderShapes) -> Vec<(usize, Box<dyn WeightShapes>)> {
    let mut v: Vec<(usize, Box<dyn WeightShapes>)> = vec![(1, Box::new(w1))];
    if let Some(d) = std::env::var_os("INTARSIA_DSV41_CACHE_R0").map(PathBuf::from) {
        let m = intarsia_models::dsv41_cache::open_manifest(&d).unwrap();
        if m.layers.is_none() && m.first_layer == 0 {
            v.push((m.world, Box::new(HeaderShapes::open(&d).unwrap())));
        }
    }
    v
}

fn opts(t: usize, pos0: usize) -> GraphOpts {
    GraphOpts { t, pos0, act: DType::Bf16, cap: 131_072 }
}

const CUT: usize = 20;
const PLAIN: PassKind = PassKind { all_rows: false, absorb: false, dpos: false };

fn tp(world: usize) -> GraphTp {
    GraphTp { world, layers: None, first: 0, rope_inplace: false }
}

/// A step as text (op, name, buffers, attributes): two steps are the same step when these are equal.
fn dbg(s: &Step) -> String {
    format!("{:?}", (s.op, &s.what, &s.inputs, &s.outputs, &s.attrs, s.pos, s.lane))
}

fn layer<'a>(g: &'a Dsv41Graph, l: usize) -> &'a Stage {
    g.stages.iter().find(|s| s.layer == Some(l)).unwrap_or_else(|| panic!("no stage for layer {l}"))
}

fn collectives(st: &[&Stage]) -> Vec<String> {
    st.iter()
        .flat_map(|s| s.extras.iter())
        .map(|(_, e)| match e {
            Extra::AllReduce { .. } => "all_reduce",
            Extra::AllReduceInto { .. } => "all_reduce_into",
            Extra::AllGather { .. } => "all_gather",
            Extra::Fill { .. } => "fill",
        })
        .map(String::from)
        .collect()
}

fn check(g: &Dsv41Graph) {
    for st in &g.stages {
        for s in &st.steps {
            s.phantom_call().unwrap_or_else(|e| panic!("{}: {e}", s.what));
            for o in &s.outputs {
                assert!(s.inputs.iter().all(|i| i.buf != o.buf), "{}: {:?} is read and written", s.what, o.buf);
            }
        }
    }
}

fn enc(c: &Dsv41Config, w: &dyn WeightShapes, t: usize, pos0: usize, floor: usize, world: usize) -> Dsv41Graph {
    build_replay(c, w, opts(t, pos0), tp(world), PLAIN, Replay { part: ReplayPart::Encoder, cut: CUT, floor, first: 0 }).unwrap()
}

fn dec(c: &Dsv41Config, w: &dyn WeightShapes, t: usize, pos0: usize, floor: usize, first: usize, world: usize, k: PassKind) -> Dsv41Graph {
    build_replay(c, w, opts(t, pos0), tp(world), k, Replay { part: ReplayPart::Decoder, cut: CUT, floor, first }).unwrap()
}

#[test]
fn the_default_replay_is_the_plain_graph() {
    let (c, w) = mia();
    for (t, pos0) in [(1, 4095), (17, 5), (512, 0), (2048, 2048)] {
        let a = build_pass(&c, &w, opts(t, pos0), tp(1), PLAIN).unwrap();
        let b = build_replay(&c, &w, opts(t, pos0), tp(1), PLAIN, Replay::default()).unwrap();
        assert_eq!(format!("{:?}", a.stages), format!("{:?}", b.stages));
        assert_eq!(format!("{:?}", a.slots), format!("{:?}", b.slots));
    }
}

#[test]
fn the_encoder_part_is_layers_0_to_19_and_the_cut_layers_kv_update() {
    let (c, w1) = mia();
    let (t, pos0, floor) = (512, 3584, 3978);
    for (world, w) in setups(w1) {
        let w = &*w;
        let full = build_pass(&c, w, opts(t, pos0), tp(world), PLAIN).unwrap();
        let e = enc(&c, w, t, pos0, floor, world);
        assert_eq!(e.stages.len(), 1 + CUT + 1, "embed, layers 0..20, the cut layer's kv update; no head");
        assert!(e.stages.iter().all(|s| s.name != "head"));
        if world == 1 {
            check(&e);
        }
        // the embedding and the 20 encoder layers are the plain graph's, step for step
        assert_eq!(format!("{:?}", e.stages[0]), format!("{:?}", full.stages[0]));
        for l in 0..CUT {
            assert_eq!(format!("{:?}", layer(&e, l)), format!("{:?}", layer(&full, l)), "layer {l} world {world}");
        }
        // the last stage: the layer's attention mixes and its kv-source update, each step the plain layer-20 step
        let last = e.stages.last().unwrap();
        assert_eq!(last.layer, Some(CUT));
        let plain20: Vec<String> = layer(&full, CUT).steps.iter().map(dbg).collect();
        for s in &last.steps {
            assert!(plain20.contains(&dbg(s)), "{} is not a step of the plain layer {CUT}", s.what);
        }
        let names: Vec<&str> = last.steps.iter().map(|s| s.what.as_str()).collect();
        assert!(names.iter().any(|n| n.ends_with("hc_attn.hc_sublayer")) && names.iter().any(|n| n.contains("compressor.wkv")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("act_quant(index_k)")) && names.iter().any(|n| n.contains("act_quant(latent)")), "{names:?}");
        for s in &last.steps {
            assert!(!matches!(s.op, Op::AttnSparse | Op::MoeExperts | Op::WindowKv | Op::IndexerTopk | Op::IndexerBlocks | Op::HcPost), "{} in an encoder-only chunk", s.what);
            assert!(!s.what.contains("wq_") && !s.what.contains("wo_") && !s.what.contains("ffn"), "{}", s.what);
        }
        assert!(last.extras.is_empty(), "the kv update has no collective");
        // the collectives are the plain graph's up to layer 19 (layer 20's are the decoder part's)
        let enc_stages: Vec<&Stage> = e.stages.iter().collect();
        let full_stages: Vec<&Stage> = full.stages.iter().filter(|s| s.layer.is_none_or(|l| l < CUT) && s.name != "head").collect();
        assert_eq!(collectives(&enc_stages), collectives(&full_stages), "world {world}");
        // 20 layers of two all-reduces each and the two Engram layers' gathers: a tensor-parallel graph has them, world 1 none
        let n_comm = collectives(&enc_stages).iter().filter(|k| *k != "fill").count();
        assert_eq!(n_comm == 0, world == 1);
        println!("encoder part: world {world}, {n_comm} collectives, the whole extras sequence equal to the plain graph's up to layer 19");
    }
}

#[test]
fn the_decoder_part_is_layers_20_to_39_less_the_kv_update_on_the_kept_rows() {
    let (c, w1) = mia();
    let (full_t, pos0, floor) = (2048usize, 2048usize, 3978usize);
    let first = floor - pos0;
    let kept = full_t - first;
    for (world, w) in setups(w1) {
        let w = &*w;
        let plain_at_kept = build_pass(&c, w, opts(kept, floor), tp(world), PLAIN).unwrap();
        let e = enc(&c, w, full_t, pos0, floor, world);
        let d = dec(&c, w, kept, floor, floor, first, world, PLAIN);
        if world == 1 {
            check(&d);
        }
        assert_eq!(d.stages.len(), (40 - CUT) + 1, "layers 20..40 and the head");
        assert_eq!(d.stages.first().unwrap().layer, Some(CUT));
        assert_eq!(d.stages.last().unwrap().name, "head");
        // it starts from the encoder part's streams, rows `first..` of the buffer the encoder part left them in
        let s0 = &d.stages[0].steps[0];
        assert_eq!(s0.inputs[0].buf, Buf::Ws(format!("x0@{first}")), "{}", s0.what);
        assert_eq!(s0.inputs[0].shape, [kept, c.hc_mult, c.dim]);
        assert_eq!(s0.inputs[4].buf, Buf::Ws(format!("hc_ffn.pre@{first}")));
        assert!(d.stages.iter().flat_map(|s| &s.steps).all(|s| s.inputs.iter().all(|a| a.buf != Buf::Host("token_ids".into()))), "no embedding");
        // the layers after the cut are the plain graph's (same ops, names, shapes) at the kept rows
        for l in CUT + 1..40 {
            let a: Vec<(Op, &str)> = layer(&d, l).steps.iter().map(|s| (s.op, s.what.as_str())).collect();
            let b: Vec<(Op, &str)> = layer(&plain_at_kept, l).steps.iter().map(|s| (s.op, s.what.as_str())).collect();
            assert_eq!(a, b, "layer {l}");
        }
        // layer 20: the plain layer's steps = the encoder part's kv update + the decoder part's remaining steps
        let mut got: Vec<String> = layer(&d, CUT).steps.iter().map(|s| s.what.clone()).collect();
        got.extend(e.stages.last().unwrap().steps.iter().filter(|s| !s.what.ends_with("hc_attn.hc_sublayer")).map(|s| s.what.clone()));
        let mut want: Vec<String> = layer(&plain_at_kept, CUT).steps.iter().map(|s| s.what.clone()).collect();
        got.sort();
        want.sort();
        assert_eq!(got, want, "world {world}");
        assert!(layer(&d, CUT).steps.iter().all(|s| !s.what.contains("compressor") && !s.what.contains("act_quant(index_k)") && !s.what.contains("act_quant(latent)") && !s.what.contains("indexer.wk")));
        // the indexer, candidate blocks and the attention of the cut layer are still the decoder part's
        assert!(layer(&d, CUT).steps.iter().any(|s| s.op == Op::IndexerBlocks) && layer(&d, CUT).steps.iter().any(|s| s.op == Op::AttnSparse));
        // the collectives: the decoder part has the plain graph's layers 20.. and head (rows differ, the sequence does not)
        let dec_stages: Vec<&Stage> = d.stages.iter().collect();
        let plain_dec: Vec<&Stage> = plain_at_kept.stages.iter().filter(|s| s.layer.is_some_and(|l| l >= CUT) || s.name == "head").collect();
        assert_eq!(collectives(&dec_stages), collectives(&plain_dec), "world {world}");
        let n_comm = collectives(&dec_stages).iter().filter(|k| *k != "fill").count();
        assert_eq!(n_comm == 0, world == 1);
        println!("decoder part: world {world}, {n_comm} collectives, the whole extras sequence equal to the plain graph's from layer 20 and its head");
        // every window test is at `pos0 - floor`: here 0 (the part starts on the floor)
        for s in d.stages.iter().flat_map(|s| &s.steps).filter(|s| s.op == Op::AttnSparse) {
            assert_eq!(s.attrs.pos0, 0, "{}", s.what);
        }
        // the x0 view is read before anything writes x0: a step that writes the base buffer comes after the last read of the view
        let steps: Vec<&Step> = d.stages.iter().flat_map(|s| s.steps.iter()).collect();
        let last_read = steps.iter().rposition(|s| s.inputs.iter().any(|a| matches!(&a.buf, Buf::Ws(n) if n.starts_with("x0@")))).unwrap();
        let first_write = steps.iter().position(|s| s.outputs.iter().any(|a| a.buf == Buf::Ws("x0".into()))).unwrap();
        assert!(last_read < first_write, "x0@ read at step {last_read}, x0 written at {first_write}");
        let last_read = steps.iter().rposition(|s| s.inputs.iter().any(|a| matches!(&a.buf, Buf::Ws(n) if n.starts_with("hc_ffn.pre@")))).unwrap();
        let first_write = steps.iter().position(|s| s.outputs.iter().any(|a| a.buf == Buf::Ws("hc_ffn.pre".into()))).unwrap();
        assert!(last_read < first_write, "hc_ffn.pre@ read at step {last_read}, written at {first_write}");
    }
}

#[test]
fn the_decoder_part_absorbs_the_dspark_taps_of_the_kept_rows() {
    let (c, w) = mia();
    let k = PassKind { all_rows: false, absorb: true, dpos: false };
    let d = dec(&c, &w, 118, 3978, 3978, 1930, 1, k);
    let ab = d.stages.iter().find(|s| s.name == "dspark absorb").expect("the decoder part holds the taps (layers 37-39)");
    assert!(ab.steps.iter().any(|s| s.attrs.pos0 == 3978), "the stage rings are written at the kept rows' positions");
    check(&d);
    // the decoder part is run with no token ids (`pass_replay`): with the absorb stage in it still nothing reads them or the
    // host position (a replay graph is a host-position graph: no step reads `host.pos`)
    for st in d.stages.iter().flat_map(|s| &s.steps) {
        assert!(!st.pos, "{}", st.what);
        assert!(st.inputs.iter().all(|a| a.buf != Buf::Host("token_ids".into())), "{} reads the token ids", st.what);
    }
    // the encoder part has none, with or without the rings in the state layout
    let e = build_replay(&c, &w, opts(512, 0), tp(1), k, Replay { part: ReplayPart::Encoder, cut: CUT, floor: 3978, first: 0 }).unwrap();
    assert!(e.stages.iter().all(|s| s.name != "dspark absorb"));
    assert_eq!(e.slots.len(), d.slots.len(), "the same state layout in every part");
}

#[test]
fn a_whole_chunk_under_a_floor_moves_only_the_decoder_layers_window_test() {
    let (c, w) = mia();
    let (t, pos0, floor) = (512, 4096, 3978);
    let full = build_pass(&c, &w, opts(t, pos0), tp(1), PLAIN).unwrap();
    let r = build_replay(&c, &w, opts(t, pos0), tp(1), PLAIN, Replay { part: ReplayPart::Whole, cut: CUT, floor, first: 0 }).unwrap();
    check(&r);
    assert_eq!(full.stages.len(), r.stages.len());
    let mut shifted = 0;
    for (a, b) in full.stages.iter().zip(&r.stages) {
        assert_eq!(a.steps.len(), b.steps.len());
        for (x, y) in a.steps.iter().zip(&b.steps) {
            if x.op == Op::AttnSparse && a.layer.is_some_and(|l| l >= CUT) {
                let mut x2 = x.clone();
                x2.attrs.pos0 = pos0 - floor;
                assert_eq!(dbg(&x2), dbg(y), "{}", x.what);
                shifted += 1;
            } else {
                assert_eq!(dbg(x), dbg(y), "{}", x.what);
            }
        }
    }
    assert_eq!(shifted, 40 - CUT);
}

#[test]
fn replay_graphs_refuse_what_they_do_not_do() {
    let (c, w) = mia();
    let part = |part, floor| Replay { part, cut: CUT, floor, first: 0 };
    // verify rows and device-position graphs
    let v = PassKind { all_rows: true, absorb: false, dpos: false };
    assert!(build_replay(&c, &w, opts(4, 5000), tp(1), v, part(ReplayPart::Whole, 3978)).is_err());
    let d = PassKind { all_rows: false, absorb: false, dpos: true };
    assert!(build_replay(&c, &w, opts(4, 5000), tp(1), d, part(ReplayPart::Whole, 3978)).is_err());
    // rows below the floor in a graph that runs decoder layers
    assert!(build_replay(&c, &w, opts(512, 3000), tp(1), PLAIN, part(ReplayPart::Whole, 3978)).is_err());
    assert!(build_replay(&c, &w, opts(118, 3000), tp(1), PLAIN, part(ReplayPart::Decoder, 3978)).is_err());
    // a truncated model, a wrong cut
    assert!(build_replay(&c, &w, opts(512, 4096), GraphTp { world: 1, layers: Some(30), first: 0, rope_inplace: false }, PLAIN, part(ReplayPart::Whole, 3978)).is_err());
    assert!(build_replay(&c, &w, opts(512, 4096), tp(1), PLAIN, Replay { part: ReplayPart::Whole, cut: 10, floor: 3978, first: 0 }).is_err());
    // a decoder half with an Engram layer in it
    let mut e = c.clone();
    e.engram.as_mut().unwrap().layer_ids = vec![1, 25];
    assert!(build_replay(&e, &w, opts(118, 3978), tp(1), PLAIN, Replay { part: ReplayPart::Decoder, cut: CUT, floor: 3978, first: 10 }).is_err());
}

/// The cut arithmetic against the graphs: the rows each chunk of a prompt sends through the decoder half are the rows the
/// decoder part's graph is built for, and the encoder-only chunks hold no head.
#[test]
fn a_prompts_chunks_map_onto_the_graphs() {
    let (c, w) = mia();
    for (total, chunk) in [(4106usize, 2048usize), (4106, 512), (4096, 512), (4097, 2048), (129, 512), (130, 16)] {
        let floor = replay::floor(total, c.window);
        let (mut pos, mut decoder_rows) = (0, 0);
        while pos < total {
            let t = (total - pos).min(chunk);
            let k = replay::classify(floor, pos, t);
            decoder_rows += replay::decoder_rows(k, t);
            match k {
                Chunk::Encoder => {
                    let e = enc(&c, &w, t, pos, floor, 1);
                    assert!(e.stages.iter().all(|s| s.name != "head"));
                }
                Chunk::Cross { first } => {
                    let d = dec(&c, &w, t - first, pos + first, floor, first, 1, PLAIN);
                    assert_eq!(d.stages.last().unwrap().name, "head");
                    assert_eq!(pos + first, floor, "the decoder part starts on the floor");
                }
                Chunk::Whole => {
                    build_replay(&c, &w, opts(t, pos), tp(1), PLAIN, Replay { part: ReplayPart::Whole, cut: CUT, floor, first: 0 }).unwrap();
                }
            }
            pos += t;
        }
        assert_eq!(decoder_rows, total.min(c.window), "total {total} chunk {chunk}");
    }
}

/// Every step of the replay graphs, at the row counts a decoder part can have (1 row up to a whole chunk), has a CUDA kernel on
/// sm_121: the registry is data, so this runs without a GPU (as `dsv41.rs: every_step_resolves_to_a_cuda_kernel`). It does not
/// say the kernel is right at that row count.
#[test]
fn every_replay_step_resolves_to_a_cuda_kernel() {
    let (c, w1) = mia();
    let info = intarsia_core::DeviceInfo {
        name: "registry only".into(),
        backend: "cuda".into(),
        arch: intarsia_backend_cuda::kernels::ARCH.into(),
        total_mem: 128 << 30,
        free_mem: 100 << 30,
        unified: true,
        bandwidth_gbs: None,
    };
    let reg = intarsia_core::registry::Registry::new("cuda", &info, intarsia_backend_cuda::kernels::all());
    let mut missing = std::collections::BTreeMap::new();
    let mut steps = 0;
    for (world, w) in setups(w1) {
        let w = &*w;
        let floor = 3978;
        let mut graphs = vec![];
        for t in [1usize, 2, 15, 16, 17, 118, 128, 129, 512] {
            graphs.push(dec(&c, w, t, floor, floor, 7, world, PLAIN));
        }
        for (t, pos0) in [(1usize, 3977usize), (16, 3000), (17, 3000), (512, 3000), (2048, 1024)] {
            graphs.push(enc(&c, w, t, pos0, floor, world));
        }
        for (t, pos0) in [(1usize, 4000usize), (16, 4000), (17, 3978), (128, 3978), (512, 4096), (2048, 4096)] {
            graphs.push(build_replay(&c, w, opts(t, pos0), tp(world), PLAIN, Replay { part: ReplayPart::Whole, cut: CUT, floor, first: 0 }).unwrap());
        }
        for g in graphs {
            for st in g.stages.iter().flat_map(|s| s.steps.iter()) {
                steps += 1;
                let call = st.phantom_call().unwrap();
                if let Err(e) = reg.select(&call) {
                    let t = st.inputs.first().and_then(|a| a.shape.first()).copied().unwrap_or(0);
                    missing.entry(format!("{} world {world} rows~{t}", st.op.as_str())).or_insert_with(|| format!("{}: {e}", st.what));
                }
            }
        }
    }
    assert!(steps > 1000, "{steps}");
    assert!(missing.is_empty(), "steps without a CUDA kernel:\n{}", missing.iter().map(|(k, v)| format!("  {k}: {v}")).collect::<Vec<_>>().join("\n"));
}

/// `scripts/dsv41-tp2/replay-test.sh` step 3 compares position dumps of these steps (`GATE_WHAT`, a substring filter of the dump
/// hook): every name must select a step of the right graph part whose output has one row per input row (the hook only dumps
/// outputs whose first dimension is the chunk's row count), and no other layer's step may match by accident. A rename in
/// `graph.rs` breaks this test, not the first GPU window.
#[test]
fn the_gate_script_names_steps_that_exist() {
    let script = include_str!("../../../scripts/dsv41-tp2/replay-test.sh");
    let line = script.lines().find(|l| l.starts_with("GATE_WHAT=")).expect("GATE_WHAT in the script");
    let what: Vec<&str> = line.trim_start_matches("GATE_WHAT=").trim_matches('\'').split(',').collect();
    assert_eq!(what.len(), 8, "{what:?}");
    let (c, w) = mia();
    let (t, pos0, floor) = (512usize, 1024usize, 1072usize);
    let enc = enc(&c, &w, t, pos0, floor, 1);
    let dec = dec(&c, &w, 118, floor, floor, 394, 1, PLAIN);
    let whole = build_pass(&c, &w, opts(t, 0), tp(1), PLAIN).unwrap();
    let hits = |g: &Dsv41Graph, pat: &str| -> Vec<(String, Vec<usize>)> {
        g.stages.iter().flat_map(|s| s.steps.iter()).filter(|s| s.what.contains(pat)).map(|s| (s.what.clone(), s.outputs.iter().map(|o| o.shape[0]).collect())).collect()
    };
    for pat in &what {
        let in_enc = hits(&enc, pat);
        let in_dec = hits(&dec, pat);
        let in_whole = hits(&whole, pat);
        if pat.starts_with("layers.39.") {
            assert!(in_enc.is_empty() && in_dec.len() == 1 && in_whole.len() == 1, "{pat}: enc {in_enc:?} dec {in_dec:?}");
            assert_eq!(in_dec[0].1, [118], "{pat}: the output must have one row per row of the decoder part");
        } else {
            assert_eq!(in_enc.len(), 1, "{pat}: {in_enc:?} (each name must select exactly one step of the encoder part)");
            assert_eq!(in_whole.len(), 1, "{pat}: {in_whole:?}");
            assert!(in_enc[0].1.iter().all(|&r| r == t), "{pat}: output rows {:?} are not the chunk's {t} rows, the dump hook would skip it", in_enc[0].1);
            assert!(in_dec.is_empty(), "{pat}: the decoder part must not repeat it: {in_dec:?}");
        }
    }
}

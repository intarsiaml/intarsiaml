//! Lane mgdnvtest: engine-level gate of `INTARSIA_METAL_GDN_VERIFY_FUSE` (lane mgdnv, #443; default-on proposed in #445). The kernel
//! test (`intarsia-backend-metal/tests/gdn_verify_snap.rs`) pins `Backend::gdn_verify_rows` bitwise at the op level; this runs the
//! whole `Model` (tiny hybrid fixture `tests/fixtures/tiny_qwen3_5`: 4 GatedDeltaNet layers + 2 attention layers, dk 32, bf16
//! activations, on the Metal backend, no checkpoint, seconds) through the same script of speculative verify passes with the fuse on
//! and off and requires every byte the script observes to be equal:
//!
//! - solo verify passes (`forward_verify_logits`) of 3, 10 and 16 rows, each followed by a PARTIAL acceptance (`SequenceState::
//!   spec_rollback` to 1, 6 and 12 accepted drafts) and some plain decode steps (which read the rolled-back state);
//! - batched verify passes (`forward_verify_batch_logits`, the ragged `gdn_block.rs` segment path) of two sequences, rows 3 + 10 and
//!   9 + 2, each sequence rolled back to its own partial acceptance, then decode steps;
//! - everything is recorded: the greedy tokens and the logits bits of every verify row and decode step, and after every rollback and
//!   at the end the bytes of every recurrent slot (conv state and delta state of each GatedDeltaNet layer).
//!
//! Why a subprocess per arm: the knob is read once per process into a `OnceLock` (`intarsia-backend-metal/src/gdn.rs`), so one test
//! process cannot hold both arms. `fuse_on_equals_off_at_engine_level` re-runs this test binary twice (`worker`, selected by
//! `MGDNV_ARM`), once per knob value, each dumping its record to a file, and compares the files byte for byte. Without `MGDNV_ARM`
//! the `worker` test is a no-op.
//!
//! Negative controls (same parent test): with the fuse ON, the worker either skips one rollback (`skip_rollback`) or changes one
//! draft token of one verify row (`perturb_row`); either record must DIFFER from the fuse-off clean one, so the comparison is shown
//! to see rollback state and row inputs. A third control checks the arms are separate processes with the knob they were given.
//!
//!   cargo test --release -p intarsia-models --test gdn_verify_metal -- --nocapture
#![cfg(target_os = "macos")]
use intarsia_core::model::{HeadMode, LoadOptions, Model, StepOut};
use intarsia_core::recipe::ModelDims;
use intarsia_core::state::{SequenceState, StateSlot};
use intarsia_core::weights::{MemWeights, WeightSource};
use intarsia_core::{DType, HostTensor, Telemetry};
use intarsia_models::SafeTensors;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

const V: u32 = 96;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5")
}

fn load() -> Model {
    let dir = fixture_dir();
    let recipe = intarsia_models::parse_recipe(include_str!("../../../recipes/qwen3_5.toml")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dims = ModelDims::from_hf_config(&cfg, &recipe.config).unwrap();
    // the fixture is f32; Metal's kernels take bf16 weights (as a real checkpoint has), so convert every tensor
    let st = SafeTensors::open_dir(&dir).unwrap();
    let mut src = MemWeights::new();
    for name in st.names() {
        let info = st.info(&name).unwrap();
        let h = HostTensor::new(info.dtype, info.shape.clone(), st.bytes(&name).unwrap().to_vec()).unwrap();
        src.insert(&name, HostTensor::from_f32_as(DType::Bf16, &info.shape, &h.to_f32()));
    }
    let backend = intarsia_backend_metal::MetalBackend::arc().unwrap();
    let opts = LoadOptions {
        max_seq: 256,
        prefill_chunk: 16,
        act_dtype: Some(DType::Bf16),
        kv_dtype: Some(DType::Bf16),
        profile_every: 0,
        ..Default::default()
    };
    let mut m = Model::load(backend, &recipe, &dims, &src, &opts, Arc::new(Telemetry::new())).unwrap();
    m.enable_verify(16).unwrap();
    m.allow_recurrent_batch().unwrap();
    m.enable_batch(4).unwrap();
    m.configure_plan(&Default::default(), None).unwrap();
    m
}

/// Everything the script observes, as bytes.
#[derive(Default)]
struct Rec(Vec<u8>);

impl Rec {
    fn tag(&mut self, s: &str) {
        self.0.extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn toks(&mut self, label: &str, t: &[u32]) {
        self.tag(label);
        for x in t {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn logits(&mut self, label: &str, rows: &[Vec<f32>]) {
        self.tag(label);
        for r in rows {
            for x in r {
                self.0.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
    }
    /// The bytes of every recurrent slot of the sequence (GatedDeltaNet conv + delta state).
    fn slots(&mut self, m: &Model, label: &str, st: &SequenceState) {
        self.tag(label);
        for s in &st.slots {
            if let StateSlot::Recurrent(r) = s {
                let mut v = vec![0u8; r.bytes];
                m.backend.download(&r.buf, 0, &mut v, 0).unwrap();
                self.0.extend_from_slice(&v);
            }
        }
    }
}

fn prompt(seed: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + seed * 11 + (i * i) % 13) % V).collect()
}

fn drafts(seed: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 53 + seed * 29 + 5) % V).collect()
}

fn decode(m: &Model, st: &mut SequenceState, rec: &mut Rec, label: &str, n: usize) {
    let mut next = (st.len as u32 * 7 + 3) % V;
    for i in 0..n {
        let StepOut::Logits(l) = m.forward(st, &[next], HeadMode::LastLogits, None).unwrap() else { panic!("logits") };
        next = intarsia_core::sampler::argmax(&l) as u32;
        rec.logits(&format!("{label} decode {i}"), &[l]);
        rec.toks(&format!("{label} decode {i} tok"), &[next]);
    }
}

/// The script. `skip_rollback`: leave the very first rollback out; `perturb_row`: change one draft token of the 10-row solo pass.
fn script(skip_rollback: bool, perturb_row: bool) -> Vec<u8> {
    let m = load();
    let be = m.backend.clone();
    let mut rec = Rec::default();
    let mut first_rollback = true;
    let mut rollback = |st: &mut SequenceState, keep: usize| {
        if skip_rollback && std::mem::take(&mut first_rollback) {
            return;
        }
        first_rollback = false;
        st.spec_rollback(keep, &*be).unwrap();
    };

    // solo: three verify passes with partial acceptance (rows 3 / 10 / 16, accepting 1 / 6 / 12 drafts), decoding between
    let mut a = m.new_state().unwrap();
    m.forward(&mut a, &prompt(1, 13), HeadMode::LastArgmax, None).unwrap();
    for (n, (rows, acc)) in [(3usize, 1usize), (10, 6), (16, 12)].into_iter().enumerate() {
        let mut ids = vec![(a.len as u32 * 5 + 1) % V];
        ids.extend(drafts(n as u32, rows - 1));
        if perturb_row && rows == 10 {
            ids[7] = (ids[7] + 1) % V;
        }
        let len0 = a.len;
        let (toks, logits) = m.forward_verify_logits(&mut a, &ids).unwrap();
        rec.toks(&format!("solo{rows} toks"), &toks);
        rec.logits(&format!("solo{rows} logits"), &logits);
        rec.slots(&m, &format!("solo{rows} slots after verify"), &a);
        rollback(&mut a, len0 + acc + 1);
        rec.slots(&m, &format!("solo{rows} slots after rollback"), &a);
        decode(&m, &mut a, &mut rec, &format!("solo{rows}"), 3);
        rec.slots(&m, &format!("solo{rows} slots after decode"), &a);
    }

    // batched: two sequences per ragged pass (segments of 3 + 10 rows, then 9 + 2), each rolled back to its own acceptance
    let mut b = [m.new_state().unwrap(), m.new_state().unwrap()];
    for (i, st) in b.iter_mut().enumerate() {
        m.forward(st, &prompt(2 + i as u32, 9 + 5 * i), HeadMode::LastArgmax, None).unwrap();
    }
    for (round, (rows, acc)) in [([3usize, 10], [2usize, 5]), ([9, 2], [4, 0])].into_iter().enumerate() {
        let ids: Vec<Vec<u32>> = (0..2)
            .map(|i| {
                let mut v = vec![(b[i].len as u32 * 3 + 2) % V];
                v.extend(drafts(10 * round as u32 + i as u32, rows[i] - 1));
                v
            })
            .collect();
        let len0 = [b[0].len, b[1].len];
        let (toks, logits) = {
            let mut refs: Vec<&mut SequenceState> = b.iter_mut().collect();
            m.forward_verify_batch_logits(&mut refs, &ids, &[true, true]).unwrap()
        };
        for i in 0..2 {
            rec.toks(&format!("batch{round} seq{i} toks"), &toks[i]);
            rec.logits(&format!("batch{round} seq{i} logits"), &logits[i]);
            rec.slots(&m, &format!("batch{round} seq{i} slots after verify"), &b[i]);
            // a one-draft-accepted-or-less pass on a 2-row segment (acc 0) still rolls back to its first row
            rollback(&mut b[i], len0[i] + acc[i] + 1);
            rec.slots(&m, &format!("batch{round} seq{i} slots after rollback"), &b[i]);
            decode(&m, &mut b[i], &mut rec, &format!("batch{round} seq{i}"), 3);
            rec.slots(&m, &format!("batch{round} seq{i} slots after decode"), &b[i]);
        }
    }
    rec.0
}

/// One arm in its own process: runs the script and writes the record to `MGDNV_OUT`. A no-op unless `MGDNV_ARM` is set.
#[test]
fn worker() {
    let Ok(arm) = std::env::var("MGDNV_ARM") else { return };
    let out = std::env::var("MGDNV_OUT").unwrap();
    let rec = script(arm == "skip_rollback", arm == "perturb_row");
    std::fs::write(out, rec).unwrap();
}

/// Run the worker in a subprocess with `INTARSIA_METAL_GDN_VERIFY_FUSE=fuse` and the given arm; its record.
fn arm(name: &str, fuse: &str, dir: &Path) -> Vec<u8> {
    let out = dir.join(format!("{name}-fuse{fuse}.bin"));
    let st = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "worker", "--test-threads=1", "--nocapture"])
        .env("INTARSIA_METAL_GDN_VERIFY_FUSE", fuse)
        .env("MGDNV_ARM", name)
        .env("MGDNV_OUT", &out)
        .status()
        .unwrap();
    assert!(st.success(), "worker {name} fuse={fuse} failed");
    std::fs::read(out).unwrap()
}

#[test]
fn fuse_on_equals_off_at_engine_level() {
    if std::env::var("MGDNV_ARM").is_ok() {
        return; // inside a worker process
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mgdnvtest");
    std::fs::create_dir_all(&dir).unwrap();
    let off = arm("clean", "0", &dir);
    let on = arm("clean", "1", &dir);
    assert!(off.len() > 100_000, "the record holds logits and slots ({} bytes)", off.len());
    assert!(off == on, "fuse on and off diverge (first differing byte {:?})", off.iter().zip(&on).position(|(a, b)| a != b));
    // fuse off is deterministic run to run (so a difference below is the control's doing)
    assert!(off == arm("clean", "0", &dir), "fuse off is not repeatable");
    // negative controls, fuse ON: a skipped rollback and a changed row input must both show
    let skipped = arm("skip_rollback", "1", &dir);
    assert!(skipped != off, "negative control: skipping a rollback went unseen");
    let perturbed = arm("perturb_row", "1", &dir);
    assert!(perturbed != off, "negative control: a changed verify row input went unseen");
    eprintln!("MGDNV-ENGINE: {} bytes equal fuse on = off; controls differ (skip_rollback, perturb_row)", off.len());
}

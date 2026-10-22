//! Cross-process tensor parallelism (`tp_remote`): `Truncate` and `Snapshot` are acknowledged by the remote rank, the
//! primary surfaces a refusal to the request that caused it and changes nothing, and ranks of different protocol versions
//! refuse to pair.
//!
//! The remote rank here is a scripted fake on a loopback socket speaking the wire protocol (frames: `u32` length, tag,
//! body), so each test injects exactly the answer it wants; the primary side is the real `RemotePeer` / `SequenceState`
//! on the cpu backend.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::backend::Backend;
use intarsia_core::state::{PeerState, SequenceState, SlotSpec};
use intarsia_core::tp_remote::{self, RemotePeer};
use intarsia_core::{DType, Error};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

const T_NEW: u8 = 1;
const T_TRUNCATE: u8 = 3;
const T_SNAPSHOT: u8 = 4;
const T_ACK: u8 = 0x81;
const T_HELLO: u8 = 0x40;
const OK: u8 = 0;
const REJECT_UNSUPPORTED: u8 = 1;
const REJECT_INVALID: u8 = 2;
const FAILED: u8 = 3;
/// `tp_remote::PROTOCOL_VERSION` (a literal here so the test compiles against code without it).
const PROTOCOL_VERSION: u32 = 2;

fn write_frame(w: &mut impl Write, tag: u8, body: &[u8]) {
    w.write_all(&((body.len() + 1) as u32).to_le_bytes()).unwrap();
    w.write_all(&[tag]).unwrap();
    w.write_all(body).unwrap();
    w.flush().unwrap();
}

fn read_frame(r: &mut impl Read) -> Option<(u8, Vec<u8>)> {
    let mut l = [0u8; 4];
    r.read_exact(&mut l).ok()?;
    let mut buf = vec![0u8; u32::from_le_bytes(l) as usize];
    r.read_exact(&mut buf).ok()?;
    let tag = buf.remove(0);
    Some((tag, buf))
}

/// What the fake rank answers to one `Truncate` / `Snapshot`: `(status, remote length, message)`.
type Answer = (u8, u64, &'static str);

/// Commands the fake rank received, as `(tag, argument)` (`Truncate`'s length, `Snapshot`'s keep).
type Log = Arc<Mutex<Vec<(u8, u64)>>>;

/// A remote rank that answers every `Truncate` / `Snapshot` with `script(tag, argument, how many of that tag came
/// before)`. Returns the primary's end of the connection and the log of what the fake received.
fn fake_rank(script: impl Fn(u8, u64, usize) -> Answer + Send + 'static) -> (TcpStream, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let log: Log = Arc::default();
    let l2 = log.clone();
    std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut rd = s.try_clone().unwrap();
        while let Some((tag, b)) = read_frame(&mut rd) {
            if tag == T_NEW {
                continue;
            }
            if tag == T_TRUNCATE || tag == T_SNAPSHOT {
                let arg = u64::from_le_bytes(b[8..16].try_into().unwrap());
                let nth = l2.lock().unwrap().iter().filter(|(t, _)| *t == tag).count();
                l2.lock().unwrap().push((tag, arg));
                let (status, len, msg) = script(tag, arg, nth);
                let mut body = vec![status];
                body.extend_from_slice(&len.to_le_bytes());
                body.extend_from_slice(msg.as_bytes());
                write_frame(&mut s, T_ACK, &body);
            }
        }
    });
    (TcpStream::connect(addr).unwrap(), log)
}

fn count(log: &Log, tag: u8) -> usize {
    log.lock().unwrap().iter().filter(|(t, _)| *t == tag).count()
}

fn specs() -> Vec<SlotSpec> {
    vec![SlotSpec::AttnKv { n_kv_heads: 1, head_dim: 4, dtype: DType::F32 }, SlotSpec::Recurrent { bytes: 32 }]
}

/// A state of 6 tokens with recurrent snapshots at 3 and 6, the remote rank (if any) attached and at 6. The remote rank
/// is attached after the snapshots: the fake sees only the commands a test sends.
fn state_with_snapshots(cpu: &Arc<dyn Backend>, remote: Option<&Arc<RemotePeer>>) -> SequenceState {
    let mut st = SequenceState::new(&**cpu, &specs(), 32).unwrap();
    st.commit(&[1, 2, 3]);
    st.snapshot(&**cpu, 4).unwrap();
    st.commit(&[4, 5, 6]);
    st.snapshot(&**cpu, 4).unwrap();
    if let Some(p) = remote {
        let mut r = p.new_state().unwrap();
        r.len = 6;
        st.remote = Some(r);
    }
    st
}

fn snap_lens(st: &SequenceState) -> Vec<usize> {
    st.snapshots.iter().map(|s| s.len).collect()
}

fn peer(stream: TcpStream) -> Arc<RemotePeer> {
    let cpu: Arc<dyn Backend> = CpuBackend::arc();
    let comms = intarsia_core::HostCommGroup::new(vec![cpu, CpuBackend::arc()]);
    RemotePeer::new(stream, comms[0].clone()).unwrap()
}

/// The remote rank refuses a rollback (it has no snapshot there): that `truncate` fails at once, as a refusal (not a
/// backend failure), every rank is as it was, and the next requests go through.
#[test]
fn remote_truncate_refusal_fails_that_request_and_changes_nothing() {
    let cpu: Arc<dyn Backend> = CpuBackend::arc();
    let (s, log) = fake_rank(|tag, arg, nth| {
        if tag == T_TRUNCATE && nth == 0 {
            (REJECT_UNSUPPORTED, 6, "no snapshot at that length")
        } else {
            (OK, arg, "")
        }
    });
    let p = peer(s);
    let mut st = state_with_snapshots(&cpu, Some(&p));
    let tokens = st.tokens.clone();

    let e = st.truncate(3, &*cpu).expect_err("the remote rank refused");
    let msg = e.to_string();
    assert!(matches!(e, Error::Unsupported(_)), "a refusal is not a backend error: {msg}");
    assert!(msg.contains("remote tensor-parallel rank") && msg.contains("truncate to 3") && msg.contains("no snapshot"), "{msg}");
    assert_eq!((st.len, &st.tokens, snap_lens(&st)), (6, &tokens, vec![3, 6]), "primary unchanged");
    assert_eq!(st.remote.as_ref().unwrap().len, 6, "remote length unchanged");

    // later requests work: the same rollback now, a snapshot, a reset
    st.truncate(3, &*cpu).unwrap();
    assert_eq!((st.len, snap_lens(&st), st.remote.as_ref().unwrap().len), (3, vec![3], 3));
    st.snapshot(&*cpu, 4).unwrap(); // at the same length: replaced
    st.reset(&*cpu).unwrap();
    assert_eq!((st.len, st.remote.as_ref().unwrap().len), (0, 0));
    assert_eq!(count(&log, T_TRUNCATE), 3);
}

/// A device error on the remote rank during `Truncate` is reported to that request as a backend error (the group is
/// unusable), and the primary has not rolled back.
#[test]
fn remote_truncate_failure_is_a_backend_error_and_leaves_the_primary_alone() {
    let cpu: Arc<dyn Backend> = CpuBackend::arc();
    let (s, _log) = fake_rank(|_, _, _| (FAILED, 6, "copy failed"));
    let p = peer(s);
    let mut st = state_with_snapshots(&cpu, Some(&p));
    let e = st.truncate(3, &*cpu).expect_err("the remote rank failed");
    assert!(matches!(e, Error::Backend(_)) && e.to_string().contains("copy failed"), "{e}");
    assert_eq!((st.len, snap_lens(&st)), (6, vec![3, 6]));
}

/// The remote rank refuses a `Snapshot` (it does not know the sequence): that request fails at once, the primary holds
/// no new snapshot, and a later snapshot works.
#[test]
fn remote_snapshot_refusal_fails_that_request_and_changes_nothing() {
    let cpu: Arc<dyn Backend> = CpuBackend::arc();
    let (s, log) = fake_rank(|tag, _, nth| {
        // the first snapshot is refused, the next passes
        if tag == T_SNAPSHOT && nth == 0 {
            (REJECT_INVALID, 6, "unknown sequence 1")
        } else {
            (OK, 6, "")
        }
    });
    let p = peer(s);
    let mut st = state_with_snapshots(&cpu, Some(&p));
    st.commit(&[7, 8]);
    let e = st.snapshot(&*cpu, 4).expect_err("the remote rank refused");
    assert!(matches!(e, Error::InvalidCall(_)) && e.to_string().contains("unknown sequence"), "{e}");
    assert_eq!(snap_lens(&st), vec![3, 6], "no snapshot at 8 on the primary when the remote has none");
    st.snapshot(&*cpu, 4).unwrap();
    assert_eq!(snap_lens(&st), vec![3, 6, 8]);
    assert_eq!(count(&log, T_SNAPSHOT), 2);
}

/// The primary cannot make the rollback (no snapshot at 4): it fails before it asks the remote rank or touches an
/// in-process peer, which would otherwise be left at a different length.
#[test]
fn truncate_checks_the_primary_before_touching_any_other_rank() {
    let cpu: Arc<dyn Backend> = CpuBackend::arc();

    // remote rank: never contacted
    let (s, log) = fake_rank(|_, arg, _| (OK, arg, ""));
    let p = peer(s);
    let mut st = state_with_snapshots(&cpu, Some(&p));
    let before = count(&log, T_TRUNCATE);
    let e = st.truncate(4, &*cpu).expect_err("no snapshot at 4");
    assert!(matches!(e, Error::Unsupported(_)), "{e}");
    assert_eq!(count(&log, T_TRUNCATE), before, "the remote rank was not asked");
    assert_eq!((st.len, snap_lens(&st)), (6, vec![3, 6]));

    // in-process peer that does hold a snapshot at 4: left at 6
    let mut st = state_with_snapshots(&cpu, None);
    let pb: Arc<dyn Backend> = CpuBackend::arc();
    let mut ps = SequenceState::new(&*pb, &specs(), 32).unwrap();
    ps.commit(&[1, 2, 3, 4]);
    ps.snapshot(&*pb, 4).unwrap();
    ps.commit(&[5, 6]);
    st.peers.push(PeerState { backend: pb, st: ps });
    let e = st.truncate(4, &*cpu).expect_err("no snapshot at 4 on the primary");
    assert!(matches!(e, Error::Unsupported(_)), "{e}");
    assert_eq!((st.len, st.peers[0].st.len, snap_lens(&st.peers[0].st)), (6, 6, vec![4]), "peer unchanged");
}

/// A raw rank of another protocol version: it sends one `Hello` and reads ours.
fn hello_from(version_text: String, cap: u64) -> TcpStream {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut body = cap.to_le_bytes().to_vec();
        body.extend_from_slice(version_text.as_bytes());
        write_frame(&mut s, T_HELLO, &body);
        let _ = read_frame(&mut s);
    });
    TcpStream::connect(addr).unwrap()
}

#[test]
fn ranks_of_different_protocol_versions_refuse_to_pair() {
    // an unversioned rank (protocol 1: the bare fingerprint, otherwise identical)
    let mut s = hello_from("fp".into(), 100);
    let e = tp_remote::handshake(&mut s, "fp", 50).expect_err("protocol 1 vs 2");
    let msg = e.to_string();
    assert!(msg.contains("different protocols") && msg.contains("version 2") && msg.contains("version 1"), "{msg}");

    // a newer rank
    let mut s = hello_from("tp-protocol=3;fp".into(), 100);
    let msg = tp_remote::handshake(&mut s, "fp", 50).expect_err("protocol 3 vs 2").to_string();
    assert!(msg.contains("different protocols") && msg.contains("version 3"), "{msg}");

    // same version, different start-up options: the fingerprint check still names both
    let mut s = hello_from(format!("tp-protocol={};other", PROTOCOL_VERSION), 100);
    let msg = tp_remote::handshake(&mut s, "fp", 50).expect_err("fingerprints differ").to_string();
    assert!(msg.contains("started differently") && msg.contains("other"), "{msg}");

    // same version and fingerprint: pairs, at the smaller capacity
    let mut s = hello_from(format!("tp-protocol={};fp", PROTOCOL_VERSION), 100);
    assert_eq!(tp_remote::handshake(&mut s, "fp", 50).unwrap(), 50);
}

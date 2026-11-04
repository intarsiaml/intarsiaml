//! Tensor parallelism across hosts (docs/design/35b-tp2.md): rank 0's [`Model`] drives a rank that lives in another
//! process, usually on another machine, over one TCP connection. ADR-0006's in-process ranks are threads holding a
//! `Model` each; a remote rank is the same `Model` shard in its own process, run by [`serve`], and the primary keeps a
//! [`RemotePeer`] handle instead of the shard. The collectives themselves go through the group's [`Comm`] (NCCL on the
//! Sparks), never through this connection: the connection only carries what to run.
//!
//! Commands, in order on the stream (every one applies to the remote copy of one sequence, `sid`):
//! `NewState`, `Forward(flags, ids)`, `Truncate(len)`, `Snapshot(keep)`, `Drop`, plus `Shutdown`. `Forward`'s flags
//! carry rank 0's per-step decisions the remote rank must follow: [`FWD_GRAPH`], rank 0 replays its decode graph in
//! this step, so the remote rank replays its own (the two graphs record the same collectives in the same order).
//! `Forward`, `Truncate` and `Snapshot` are answered, in order: `Forward` with (`ok`, the remote length, an error
//! message), `Truncate` and `Snapshot` with an acknowledgement (a status, the remote length, a message) that the
//! primary waits for, so a failure is reported by the request that caused it. An acknowledgement's status says whether
//! the remote copy is unchanged (`rejected`: no such snapshot, unknown sequence; the group stays usable) or undefined
//! (`failed`: a device error; the primary aborts its communicator, as for a failed forward). `NewState`, `Drop` and
//! `Shutdown` are fire-and-forget; a failed `NewState` fails the next answered command. The primary sends `Forward`
//! before it runs its own pass and reads the answer after it.
//! A reader thread on the primary watches the answers: an error aborts the primary's communicator so the primary is not
//! left waiting in a collective the remote rank never joins. After any failure the group is unusable (ADR-0006), except
//! after a rejected `Truncate` / `Snapshot`, which changed nothing on either rank.
//!
//! The wire protocol is versioned ([`PROTOCOL_VERSION`], exchanged in [`handshake`]): two ranks built with different
//! versions refuse to pair, with an error naming both. Version 1 (unversioned) answered only `Forward`.
//!
//! The protocol carries no model description: both processes load the same model with the same options (the launcher
//! passes the same flags to both), and [`handshake`] checks a fingerprint of them and agrees the KV capacity.

use crate::comm::Comm;
use crate::error::{Error, Result};
use crate::model::Model;
use crate::state::SequenceState;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const T_NEW: u8 = 1;
const T_FORWARD: u8 = 2;
const T_TRUNCATE: u8 = 3;
const T_SNAPSHOT: u8 = 4;
const T_DROP: u8 = 5;
const T_SHUTDOWN: u8 = 6;
const T_REPLY: u8 = 0x80;
/// Answer to `Truncate` / `Snapshot` (protocol 2): `[status u8][remote len u64][message]`.
const T_ACK: u8 = 0x81;
const T_HELLO: u8 = 0x40;
/// Wire protocol version. 1: only `Forward` answered. 2: `Truncate` and `Snapshot` answered with [`T_ACK`].
/// Bump it on any change to the commands, their bodies or their answers.
pub const PROTOCOL_VERSION: u32 = 2;
/// Marks the version at the start of the handshake fingerprint (protocol 1 sent the bare fingerprint).
const PROTO_TAG: &str = "tp-protocol=";
/// [`T_ACK`] statuses.
const ACK_OK: u8 = 0;
/// Refused, remote copy unchanged: the request is not supported (no snapshot at that length).
const ACK_REJECTED_UNSUPPORTED: u8 = 1;
/// Refused, remote copy unchanged: the request is invalid (beyond the remote length, unknown sequence).
const ACK_REJECTED_INVALID: u8 = 2;
/// Failed part-way (device error): the remote copy is undefined and the group unusable.
const ACK_FAILED: u8 = 3;

/// `Forward` flag: a single-token step that rank 0 runs as a captured decode graph (`model/graph_step.rs`).
pub const FWD_GRAPH: u64 = 1;
/// `Forward` flag: not a forward but a speculative drafting round after the sequence's last row ([`RankModel::draft_rank`];
/// the ids are the round's inputs, the sequence is unchanged). Only models whose handshake fingerprint says they draft
/// send it (DeepSeek-V4.1's names its DSpark stages), so both ranks know it.
pub const FWD_DRAFT: u64 = 2;
/// `Forward` flag: a speculative verify pass (the head's logits of every row; [`RankModel::verify_rank`]). Like
/// [`FWD_DRAFT`], only sent by models whose handshake fingerprint says they speculate.
pub const FWD_VERIFY: u64 = 4;

/// How long the primary waits for a forward's answer after its own pass finished (the remote runs the same pass at
/// the same time, so this only fires on a hung or dead rank). `INTARSIA_TP_REMOTE_TIMEOUT_S` overrides it.
fn reply_timeout() -> Duration {
    let s = std::env::var("INTARSIA_TP_REMOTE_TIMEOUT_S").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(600);
    Duration::from_secs(s.max(1))
}

fn io(e: std::io::Error) -> Error {
    Error::backend(format!("tensor-parallel connection: {e}"))
}

fn write_frame(w: &mut impl Write, tag: u8, body: &[u8]) -> Result<()> {
    let len = (body.len() + 1) as u32;
    w.write_all(&len.to_le_bytes()).map_err(io)?;
    w.write_all(&[tag]).map_err(io)?;
    w.write_all(body).map_err(io)
}

fn read_frame(r: &mut impl Read) -> Result<(u8, Vec<u8>)> {
    let mut l = [0u8; 4];
    r.read_exact(&mut l).map_err(io)?;
    let len = u32::from_le_bytes(l) as usize;
    if len == 0 || len > (1 << 28) {
        return Err(Error::backend(format!("tensor-parallel connection: bad frame length {len}")));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(io)?;
    let tag = buf[0];
    buf.remove(0);
    Ok((tag, buf))
}

fn u64_at(b: &[u8], i: usize) -> Result<u64> {
    b.get(i * 8..i * 8 + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| Error::backend("tensor-parallel connection: short frame"))
}

/// The first exchange on a fresh connection, both sides: each sends its protocol version, fingerprint (model, options,
/// rank, world) and KV capacity in tokens and reads the other's. Fails on any version or fingerprint difference;
/// returns the smaller capacity (every rank must hold the same number of tokens).
pub fn handshake(stream: &mut (impl Read + Write), fingerprint: &str, kv_cap: usize) -> Result<usize> {
    handshake_as(stream, fingerprint, kv_cap, PROTOCOL_VERSION)
}

fn handshake_as(stream: &mut (impl Read + Write), fingerprint: &str, kv_cap: usize, version: u32) -> Result<usize> {
    let mut body = (kv_cap as u64).to_le_bytes().to_vec();
    // the version rides in front of the fingerprint text, so a rank of protocol 1 (which compares the whole text)
    // refuses a rank of protocol 2 with a mismatch that shows both
    body.extend_from_slice(format!("{PROTO_TAG}{version};{fingerprint}").as_bytes());
    write_frame(stream, T_HELLO, &body)?;
    stream.flush().map_err(io)?;
    let (tag, b) = read_frame(stream)?;
    if tag != T_HELLO {
        return Err(Error::backend(format!("tensor-parallel handshake: unexpected message {tag}")));
    }
    let theirs = u64_at(&b, 0)? as usize;
    let text = String::from_utf8_lossy(&b[8..]);
    let (their_version, fp) = match text.strip_prefix(PROTO_TAG).and_then(|t| t.split_once(';')) {
        Some((v, fp)) => (v.parse::<u32>().ok(), fp),
        None => (Some(1), &*text),
    };
    if their_version != Some(version) {
        let theirs = their_version.map_or_else(|| "an unreadable version".to_string(), |v| format!("version {v}"));
        return Err(Error::invalid(format!(
            "tensor-parallel ranks speak different protocols: this rank speaks version {version}, the other rank {theirs}. \
             Run the same intarsia build on both ranks"
        )));
    }
    if fp != fingerprint {
        return Err(Error::invalid(format!(
            "tensor-parallel ranks were started differently:\n  this rank:  {fingerprint}\n  other rank: {fp}"
        )));
    }
    Ok(kv_cap.min(theirs))
}

struct Reply {
    /// `Forward` answer (`T_REPLY`) or `Truncate` / `Snapshot` acknowledgement (`T_ACK`).
    ack: bool,
    /// `Forward`: 1 ok / 0 failed. Acknowledgement: an `ACK_*` status.
    status: u8,
    len: usize,
    msg: String,
}

/// Rank 0's handle on a remote rank (see the module docs).
pub struct RemotePeer {
    w: Mutex<BufWriter<TcpStream>>,
    replies: Mutex<Receiver<Reply>>,
    next_sid: AtomicU64,
    dead: Arc<AtomicBool>,
    lost: Arc<Mutex<Lost>>,
}

/// Why the group is gone, as the reader thread saw it, and whom to tell: the engine thread may sit in a kernel that never
/// returns (the remote rank died mid-pass), so it cannot be the one to report the failure.
#[derive(Default)]
struct Lost {
    why: Option<String>,
    hook: Option<Box<dyn FnOnce(String) + Send>>,
}

impl Lost {
    fn set(this: &Mutex<Lost>, why: String) {
        let hook = {
            let mut l = this.lock().unwrap_or_else(|e| e.into_inner());
            if l.why.is_some() {
                return;
            }
            l.why = Some(why.clone());
            l.hook.take()
        };
        if let Some(h) = hook {
            h(why);
        }
    }
}

impl RemotePeer {
    /// Wrap a connection that passed [`handshake`]. `comm` is rank 0's communicator: aborted when the remote rank
    /// reports a failure or the connection drops.
    pub fn new(stream: TcpStream, comm: Arc<dyn Comm>) -> Result<Arc<RemotePeer>> {
        stream.set_nodelay(true).map_err(io)?;
        let mut rd = BufReader::new(stream.try_clone().map_err(io)?);
        let (tx, rx) = channel();
        let dead = Arc::new(AtomicBool::new(false));
        let d2 = dead.clone();
        let lost = Arc::new(Mutex::new(Lost::default()));
        let l2 = lost.clone();
        std::thread::Builder::new()
            .name("tp-remote-reader".into())
            .spawn(move || {
                loop {
                    match read_frame(&mut rd) {
                        Ok((tag @ (T_REPLY | T_ACK), b)) if b.len() >= 9 => {
                            let ack = tag == T_ACK;
                            let status = b[0];
                            let len = u64::from_le_bytes(b[1..9].try_into().unwrap()) as usize;
                            let msg = String::from_utf8_lossy(&b[9..]).into_owned();
                            // a rejected acknowledgement changed nothing on the remote rank: the group stays usable
                            let failed = if ack { status >= ACK_FAILED } else { status != 1 };
                            if failed {
                                d2.store(true, Ordering::SeqCst);
                                Lost::set(&l2, format!("the remote tensor-parallel rank reported a failure: {msg}"));
                            }
                            let sent = tx.send(Reply { ack, status, len, msg }).is_ok();
                            if failed {
                                break;
                            }
                            if !sent {
                                return;
                            }
                        }
                        _ => {
                            d2.store(true, Ordering::SeqCst);
                            Lost::set(&l2, "the connection to the remote tensor-parallel rank was lost".to_string());
                            break;
                        }
                    }
                }
                // the abort can block for good (`ncclCommAbort` with the other rank gone): closing the queue first ends a
                // wait in `next_reply`, after the answers already in it, instead of leaving it to the reply timeout
                drop(tx);
                comm.abort();
            })
            .map_err(|e| Error::backend(format!("spawning the tensor-parallel reader: {e}")))?;
        Ok(Arc::new(RemotePeer { w: Mutex::new(BufWriter::new(stream)), replies: Mutex::new(rx), next_sid: AtomicU64::new(1), dead, lost }))
    }

    /// Call `hook` with the reason once the remote rank reports a failure or the connection drops (at once if that has
    /// happened already). It runs on the reader thread, before the communicator is aborted.
    pub fn on_lost(&self, hook: impl FnOnce(String) + Send + 'static) {
        let known = {
            let mut l = self.lost.lock().unwrap_or_else(|e| e.into_inner());
            if l.why.is_none() {
                l.hook = Some(Box::new(hook));
                return;
            }
            l.why.clone().unwrap()
        };
        hook(known);
    }

    fn send(&self, tag: u8, body: &[u8]) -> Result<()> {
        if self.dead.load(Ordering::SeqCst) {
            return Err(Error::backend("the remote tensor-parallel rank is gone (or the group failed earlier)"));
        }
        let mut w = self.w.lock().unwrap();
        write_frame(&mut *w, tag, body)?;
        w.flush().map_err(io)
    }

    /// The next answer from the remote rank, in the order it was sent. A rank that does not answer is hung: the peer is
    /// marked unusable so nothing queues behind it and waits again.
    fn next_reply(&self, rx: &Receiver<Reply>) -> Result<Reply> {
        match rx.recv_timeout(reply_timeout()) {
            Ok(r) => Ok(r),
            Err(RecvTimeoutError::Timeout) => {
                self.dead.store(true, Ordering::SeqCst);
                Err(Error::backend(format!("the remote tensor-parallel rank did not answer within {:?}", reply_timeout())))
            }
            Err(RecvTimeoutError::Disconnected) => Err(Error::backend("the remote tensor-parallel rank is gone")),
        }
    }

    /// The group is unusable after a failed pass on this rank (the communicator is aborted, the remote rank may be
    /// stuck in a collective): later commands fail at once instead of waiting for answers that will not come.
    pub fn mark_unusable(&self) {
        self.dead.store(true, Ordering::SeqCst);
    }

    fn sid_body(sid: u64, extra: &[u64]) -> Vec<u8> {
        std::iter::once(sid).chain(extra.iter().copied()).flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Create the remote copy of a new sequence.
    pub fn new_state(self: &Arc<Self>) -> Result<RemoteSeq> {
        let sid = self.next_sid.fetch_add(1, Ordering::Relaxed);
        self.send(T_NEW, &Self::sid_body(sid, &[]))?;
        Ok(RemoteSeq { peer: self.clone(), sid, len: 0 })
    }

    /// Ask the remote rank to stop serving (it exits its loop).
    pub fn shutdown(&self) {
        let _ = self.send(T_SHUTDOWN, &[]);
    }
}

impl Drop for RemotePeer {
    /// Close the connection for both directions (the reader thread holds a clone of the socket, so dropping the writer
    /// alone would not): the remote rank's loop sees the end and returns, the reader thread exits.
    fn drop(&mut self) {
        let w = self.w.get_mut().unwrap_or_else(|e| e.into_inner());
        let _ = w.flush();
        let _ = w.get_ref().shutdown(std::net::Shutdown::Both);
    }
}

/// Rank 0's view of the remote rank's copy of one sequence. Dropping it frees the remote copy.
pub struct RemoteSeq {
    peer: Arc<RemotePeer>,
    sid: u64,
    /// The remote copy's length as of the last answered forward and the commands sent since.
    pub len: usize,
}

impl RemoteSeq {
    /// Send this sequence's next forward pass (the primary runs its own pass next, then calls [`RemoteSeq::finish`]).
    /// `graph`: the primary replays its decode graph in this pass (see [`FWD_GRAPH`]).
    pub fn start_forward(&self, ids: &[u32], graph: bool) -> Result<()> {
        let mut body = Vec::with_capacity(16 + ids.len() * 4);
        body.extend_from_slice(&self.sid.to_le_bytes());
        body.extend_from_slice(&(if graph { FWD_GRAPH } else { 0 }).to_le_bytes());
        body.extend(ids.iter().flat_map(|i| i.to_le_bytes()));
        self.peer.send(T_FORWARD, &body)
    }

    /// Send this sequence's next drafting round ([`FWD_DRAFT`]); answered like a forward ([`RemoteSeq::finish`]).
    pub fn start_draft(&self, ids: &[u32]) -> Result<()> {
        self.start_flagged(ids, FWD_DRAFT)
    }

    /// Send this sequence's next verify pass ([`FWD_VERIFY`]); answered like a forward ([`RemoteSeq::finish`]).
    pub fn start_verify(&self, ids: &[u32]) -> Result<()> {
        self.start_flagged(ids, FWD_VERIFY)
    }

    fn start_flagged(&self, ids: &[u32], flags: u64) -> Result<()> {
        let mut body = Vec::with_capacity(16 + ids.len() * 4);
        body.extend_from_slice(&self.sid.to_le_bytes());
        body.extend_from_slice(&flags.to_le_bytes());
        body.extend(ids.iter().flat_map(|i| i.to_le_bytes()));
        self.peer.send(T_FORWARD, &body)
    }

    /// The remote rank's answer to the forward sent by [`RemoteSeq::start_forward`].
    pub fn finish(&mut self) -> Result<()> {
        let rx = self.peer.replies.lock().unwrap();
        let r = self.peer.next_reply(&rx)?;
        if r.ack {
            self.peer.mark_unusable();
            return Err(Error::backend("tensor-parallel connection: out of step (an acknowledgement answered a forward)"));
        }
        self.len = r.len;
        if r.status == 1 {
            Ok(())
        } else {
            Err(Error::backend(format!("remote tensor-parallel rank: {}", r.msg)))
        }
    }

    /// Send `tag` and wait for the remote rank's acknowledgement. Returns the remote length; a refusal is an error that
    /// is not a backend error (the remote copy is unchanged, the group usable), a failure is a backend error.
    fn command(&self, tag: u8, body: &[u8], what: &str) -> Result<usize> {
        // hold the answer queue across send and wait: the answer to this command is the next one
        let rx = self.peer.replies.lock().unwrap();
        self.peer.send(tag, body)?;
        let r = self.peer.next_reply(&rx)?;
        if !r.ack {
            self.peer.mark_unusable();
            return Err(Error::backend(format!("tensor-parallel connection: out of step (a forward answer acknowledged {what})")));
        }
        let msg = format!("remote tensor-parallel rank, {what}: {}", r.msg);
        match r.status {
            ACK_OK => Ok(r.len),
            ACK_REJECTED_UNSUPPORTED => Err(Error::Unsupported(msg)),
            ACK_REJECTED_INVALID => Err(Error::invalid(msg)),
            _ => Err(Error::backend(msg)),
        }
    }

    /// See [`RemotePeer::mark_unusable`].
    pub fn mark_unusable(&self) {
        self.peer.mark_unusable();
    }

    /// Roll the remote copy back to `len` tokens and wait for it to confirm. Called before the primary rolls back, once
    /// the primary has checked it can: on an error, nothing has changed on either rank (unless the error is a backend
    /// error, which fails the group).
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        self.len = self.command(T_TRUNCATE, &RemotePeer::sid_body(self.sid, &[len as u64]), &format!("truncate to {len}"))?;
        Ok(())
    }

    /// Snapshot the remote copy's recurrent state and wait for it to confirm.
    pub fn snapshot(&self, keep: usize) -> Result<()> {
        self.command(T_SNAPSHOT, &RemotePeer::sid_body(self.sid, &[keep as u64]), "snapshot").map(|_| ())
    }
}

impl Drop for RemoteSeq {
    fn drop(&mut self) {
        let _ = self.peer.send(T_DROP, &RemotePeer::sid_body(self.sid, &[]));
    }
}

/// The acknowledgement status for a failed command: an error raised by a check before anything changed is a refusal
/// (the copy is as it was), anything else (a device error) leaves the copy undefined.
fn ack_status(e: &Error) -> u8 {
    match e {
        Error::Unsupported(_) => ACK_REJECTED_UNSUPPORTED,
        Error::InvalidCall(_) | Error::BadRequest(_) => ACK_REJECTED_INVALID,
        _ => ACK_FAILED,
    }
}

/// The remote rank's loop: apply the primary's commands to `model` (this process's shard) until `Shutdown` or the
/// connection closes. A forward that fails resets that sequence (as [`Model::forward`] does) and is answered with the
/// error; `Truncate` and `Snapshot` are acknowledged (see the module docs); a failed `NewState` is reported by the next
/// answered command of any sequence. Returns how many forward
/// passes ran.
/// What [`serve`] drives on the remote rank: this process's shard of a model and its per-sequence state (the ordinary
/// [`Model`], or the DeepSeek-V4.1 runtime, `crate::dsv41::runtime`).
pub trait RankModel {
    type State;
    fn new_state(&self) -> Result<Self::State>;
    /// This rank's half of the primary's forward pass (`graph`: [`FWD_GRAPH`]).
    fn forward_rank(&self, st: &mut Self::State, ids: &[u32], graph: bool) -> Result<()>;
    fn truncate(&self, st: &mut Self::State, len: usize) -> Result<()>;
    fn snapshot(&self, st: &mut Self::State, keep: usize) -> Result<()>;
    fn state_len(st: &Self::State) -> usize;
    /// This rank's half of the primary's drafting round ([`FWD_DRAFT`]); the sequence is unchanged.
    fn draft_rank(&self, _st: &mut Self::State, _ids: &[u32]) -> Result<()> {
        Err(Error::Unsupported("this model does not draft".into()))
    }
    /// This rank's half of the primary's verify pass ([`FWD_VERIFY`]): appended to the sequence like a forward.
    fn verify_rank(&self, _st: &mut Self::State, _ids: &[u32]) -> Result<()> {
        Err(Error::Unsupported("this model does not verify".into()))
    }
    /// Waits for what this rank's passes left queued on the device, for [`serve_group`] when the primary closes the connection
    /// (see [`drain_failure`]). `None`: a pass leaves nothing queued.
    fn drain(&self) -> Option<Drain> {
        None
    }
}

/// See [`RankModel::drain`].
pub type Drain = Box<dyn FnOnce() -> Result<()> + Send>;

impl RankModel for Model {
    type State = SequenceState;
    fn new_state(&self) -> Result<SequenceState> {
        Model::new_state(self)
    }
    fn forward_rank(&self, st: &mut SequenceState, ids: &[u32], graph: bool) -> Result<()> {
        Model::forward_rank(self, st, ids, graph)
    }
    fn truncate(&self, st: &mut SequenceState, len: usize) -> Result<()> {
        st.truncate(len, &*self.backend)
    }
    fn snapshot(&self, st: &mut SequenceState, keep: usize) -> Result<()> {
        st.snapshot(&*self.backend, keep)
    }
    fn state_len(st: &SequenceState) -> usize {
        st.len
    }
    fn drain(&self) -> Option<Drain> {
        Some(self.peer_drain())
    }
}

/// `INTARSIA_TP_SPIN_MS=N` (default 0 = off): after every command the remote rank's reader stays awake for `N` ms instead of
/// blocking in `read`, on one core, and a second thread keeps the core that takes the socket's receive interrupts out of idle
/// (the Sparks: 0.3 ms one way over TCP, cpuidle exit latency up to 433 us; the primary waits that long in the pass's first
/// collective). `INTARSIA_TP_SPIN_KEEP=0`: without the second thread. Two cores busy during decode; nothing else changes.
#[cfg(target_os = "linux")]
mod spin {
    use std::net::TcpStream;
    use std::os::fd::AsRawFd;
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    pub struct Spin {
        fd: i32,
        ms: u64,
        t0: Instant,
        until_ms: Arc<AtomicU64>,
        cpu: Arc<AtomicI32>,
        stop: Arc<AtomicBool>,
    }

    impl Spin {
        pub fn new(stream: &TcpStream) -> Option<Spin> {
            let ms: u64 = std::env::var("INTARSIA_TP_SPIN_MS").ok()?.parse().ok().filter(|&m| m > 0)?;
            let s = Spin {
                fd: stream.as_raw_fd(),
                ms,
                t0: Instant::now(),
                until_ms: Arc::new(AtomicU64::new(0)),
                cpu: Arc::new(AtomicI32::new(-1)),
                stop: Arc::new(AtomicBool::new(false)),
            };
            if std::env::var("INTARSIA_TP_SPIN_KEEP").map_or(true, |v| v != "0") {
                let (t0, until, cpu, stop) = (s.t0, s.until_ms.clone(), s.cpu.clone(), s.stop.clone());
                let _ = std::thread::Builder::new().name("tp-spin-keep".into()).spawn(move || {
                    let mut pinned = -1;
                    while !stop.load(Ordering::Relaxed) {
                        if t0.elapsed().as_millis() as u64 >= until.load(Ordering::Relaxed) {
                            std::thread::sleep(Duration::from_millis(20));
                            continue;
                        }
                        let c = cpu.load(Ordering::Relaxed);
                        if c >= 0 && c != pinned {
                            pin(c);
                            pinned = c;
                        }
                        std::hint::spin_loop();
                    }
                });
            }
            eprintln!("tp: remote rank reader spins {ms} ms after a command");
            Some(s)
        }

        /// A command came in: stay awake for the next one and note the core that took its interrupt.
        pub fn got_frame(&self) {
            let mut c: libc::c_int = -1;
            let mut l = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
            // SAFETY: a live socket and out-pointers of the right size.
            if unsafe { libc::getsockopt(self.fd, libc::SOL_SOCKET, libc::SO_INCOMING_CPU, (&mut c as *mut libc::c_int).cast(), &mut l) } == 0 {
                self.cpu.store(c, Ordering::Relaxed);
            }
            self.until_ms.store(self.t0.elapsed().as_millis() as u64 + self.ms, Ordering::Relaxed);
        }

        /// Poll the socket until a byte is there or the time since the last command has run out (then `read` blocks as usual).
        pub fn wait_readable(&self) {
            let mut b = 0u8;
            while (self.t0.elapsed().as_millis() as u64) < self.until_ms.load(Ordering::Relaxed) {
                // SAFETY: a live socket and a one-byte buffer; MSG_PEEK leaves the byte for `read`.
                let n = unsafe { libc::recv(self.fd, (&mut b as *mut u8).cast(), 1, libc::MSG_PEEK | libc::MSG_DONTWAIT) };
                if n >= 0 || std::io::Error::last_os_error().kind() != std::io::ErrorKind::WouldBlock {
                    return;
                }
                std::hint::spin_loop();
            }
        }
    }

    impl Drop for Spin {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
        }
    }

    fn pin(cpu: i32) {
        // SAFETY: a zeroed cpu_set_t is valid; the call affects this thread only.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu as usize, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

/// Exit code of a rank process that left a failed group (the server's `EXIT_FAILED`).
pub const EXIT_GROUP_FAILED: i32 = 87;

/// How long a rank process waits for its queued work after the primary closed the connection ([`drain_failure`]).
const DRAIN_LIMIT: Duration = Duration::from_secs(2);

/// Set by the first [`abort_group_and_exit`]: the process is on its way out with that call's code.
static LEAVING: AtomicBool = AtomicBool::new(false);

fn park_forever() -> ! {
    loop {
        std::thread::park();
    }
}

/// This rank cannot go on: the other rank is gone, or a pass failed here and the group is unusable (ADR-0006). Abort the
/// communicator so nothing keeps waiting in a collective, wait a few seconds for that (`ncclCommAbort` on a context that hit a
/// device trap may not return), then leave with `code` without running destructors, which would touch the same context. A
/// supervisor restarts both ranks.
pub fn abort_group_and_exit(comm: &Arc<dyn Comm>, code: i32, why: &str) -> ! {
    if LEAVING.swap(true, Ordering::AcqRel) {
        park_forever(); // another thread got here first and ends the process
    }
    eprintln!("tensor parallel: rank {}: {why}; aborting the communicator and exiting with code {code}", comm.rank());
    let c = comm.clone();
    let (tx, rx) = channel();
    let t0 = std::time::Instant::now();
    let _ = std::thread::Builder::new().name("tp-abort".into()).spawn(move || {
        c.abort();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(3)) {
        Ok(()) => eprintln!("tensor parallel: rank {}: communicator abort returned after {} ms", comm.rank(), t0.elapsed().as_millis()),
        Err(_) => eprintln!("tensor parallel: rank {}: communicator abort did not return within 3 s; exiting anyway", comm.rank()),
    }
    #[cfg(unix)]
    // SAFETY: _exit ends the process at once; that is the point (no destructors on a failed device context).
    unsafe {
        libc::_exit(code)
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

/// While a pass runs on this rank the serving thread cannot read the control stream, so a primary that died would go
/// unnoticed: its half of the collectives never comes (NCCL has no time limit of its own). This thread polls the stream for the
/// peer's hang-up (never reading it) and, if the peer is gone during a pass, leaves the group. A pass that only enqueues its work
/// is over long before that work is: such a hang-up reaches the serving loop as the end of the stream ([`drain_failure`]).
#[cfg(target_os = "linux")]
fn watch_peer(stream: TcpStream, in_pass: Arc<AtomicBool>, done: Arc<AtomicBool>, comm: Arc<dyn Comm>) {
    use std::os::fd::AsRawFd;
    let fd = stream.as_raw_fd();
    while !done.load(Ordering::Acquire) {
        let mut p = libc::pollfd { fd, events: libc::POLLRDHUP, revents: 0 };
        // SAFETY: one valid pollfd; the socket outlives the loop (this thread owns a clone).
        let n = unsafe { libc::poll(&mut p, 1, 200) };
        if n > 0 && p.revents & (libc::POLLRDHUP | libc::POLLHUP | libc::POLLERR) != 0 {
            if in_pass.load(Ordering::Acquire) {
                abort_group_and_exit(&comm, EXIT_GROUP_FAILED, "the primary rank disconnected during a pass");
            }
            // between passes: the serving loop sees the end of the stream itself; do not spin on the level-triggered flag
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// The primary closed the connection, which looks the same whether it exited or died. If it died, this rank's queued work can
/// be parked in a collective that will never complete, and freeing the device memory that work uses waits for it for good (`cudaFree`
/// on the sequences' buffers; docs/results/2026-11-02-dsv41-rdma.md section 6). So before anything is freed: why this rank must
/// leave the group, or `None` when the work finished, which is what a primary that exited leaves behind.
fn drain_failure<M: RankModel>(model: &M) -> Option<String> {
    let drain = model.drain()?;
    let (tx, rx) = channel();
    let spawned = std::thread::Builder::new().name("tp-drain".into()).spawn(move || {
        let _ = tx.send(drain());
    });
    if let Err(e) = spawned {
        return Some(format!("the primary rank disconnected and the queued work cannot be waited for: {e}"));
    }
    match rx.recv_timeout(DRAIN_LIMIT) {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(format!("the primary rank disconnected and the queued work failed: {e}")),
        Err(_) => Some(format!("the primary rank disconnected and the queued work did not finish within {DRAIN_LIMIT:?}")),
    }
}

pub fn serve<M: RankModel>(model: &M, stream: TcpStream) -> Result<u64> {
    serve_inner(model, stream, None)
}

/// [`serve`] for a rank process: with the group's communicator it also leaves the group (see [`abort_group_and_exit`]) when a
/// pass fails here, when the primary disconnects mid-pass or leaves work here that cannot finish, or on a protocol error, instead
/// of looping on a dead group.
pub fn serve_group<M: RankModel>(model: &M, stream: TcpStream, comm: Arc<dyn Comm>) -> Result<u64> {
    let r = serve_inner(model, stream, Some(comm.clone()));
    if let Err(e) = &r {
        abort_group_and_exit(&comm, EXIT_GROUP_FAILED, &format!("serving stopped: {e}"));
    }
    if LEAVING.load(Ordering::Acquire) {
        park_forever(); // the watch's abort let the pass end: the process's exit code is the watch's, not a normal return
    }
    r
}

fn serve_inner<M: RankModel>(model: &M, stream: TcpStream, comm: Option<Arc<dyn Comm>>) -> Result<u64> {
    stream.set_nodelay(true).map_err(io)?;
    let in_pass = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    struct Done(Arc<AtomicBool>);
    impl Drop for Done {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let _done = Done(done.clone());
    #[cfg(target_os = "linux")]
    if let Some(c) = comm.clone() {
        let (s2, ip, d2) = (stream.try_clone().map_err(io)?, in_pass.clone(), done.clone());
        std::thread::Builder::new().name("tp-peer-watch".into()).spawn(move || watch_peer(s2, ip, d2, c)).map_err(|e| Error::backend(format!("spawning the peer watch: {e}")))?;
    }
    #[cfg(target_os = "linux")]
    let spin = spin::Spin::new(&stream);
    let mut rd = BufReader::new(stream.try_clone().map_err(io)?);
    let mut w = BufWriter::new(stream);
    let mut seqs: HashMap<u64, M::State> = HashMap::new();
    let mut pending: Option<String> = None;
    let mut passes = 0u64;
    let reply = |w: &mut BufWriter<TcpStream>, ok: bool, len: usize, msg: &str| -> Result<()> {
        let mut b = vec![u8::from(ok)];
        b.extend_from_slice(&(len as u64).to_le_bytes());
        b.extend_from_slice(msg.as_bytes());
        write_frame(w, T_REPLY, &b)?;
        w.flush().map_err(io)
    };
    let ack = |w: &mut BufWriter<TcpStream>, status: u8, len: usize, msg: &str| -> Result<()> {
        let mut b = vec![status];
        b.extend_from_slice(&(len as u64).to_le_bytes());
        b.extend_from_slice(msg.as_bytes());
        write_frame(w, T_ACK, &b)?;
        w.flush().map_err(io)
    };
    loop {
        #[cfg(target_os = "linux")]
        if let Some(sp) = &spin {
            if rd.buffer().is_empty() {
                sp.wait_readable();
            }
        }
        let (tag, b) = match read_frame(&mut rd) {
            Ok(f) => f,
            Err(_) => {
                // the primary closed the connection
                if let Some(c) = &comm {
                    if let Some(why) = drain_failure(model) {
                        abort_group_and_exit(c, EXIT_GROUP_FAILED, &why);
                    }
                }
                return Ok(passes);
            }
        };
        #[cfg(target_os = "linux")]
        if let Some(sp) = &spin {
            sp.got_frame();
        }
        let sid = if tag == T_SHUTDOWN { 0 } else { u64_at(&b, 0)? };
        match tag {
            T_NEW => match model.new_state() {
                Ok(st) => {
                    seqs.insert(sid, st);
                }
                Err(e) => pending = pending.or(Some(format!("new state: {e}"))),
            },
            T_FORWARD => {
                let flags = u64_at(&b, 1)?;
                let ids: Vec<u32> =
                    b[16..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                if let Some(msg) = pending.take() {
                    // the primary is already in its own pass: leave the group (it aborts on this answer)
                    reply(&mut w, false, seqs.get(&sid).map_or(0, M::state_len), &msg)?;
                    return Err(Error::backend(msg));
                }
                let Some(st) = seqs.get_mut(&sid) else {
                    let msg = format!("unknown sequence {sid}");
                    reply(&mut w, false, 0, &msg)?;
                    return Err(Error::backend(msg));
                };
                passes += 1;
                in_pass.store(true, Ordering::Release);
                let r = if flags & FWD_DRAFT != 0 {
                    model.draft_rank(st, &ids)
                } else if flags & FWD_VERIFY != 0 {
                    model.verify_rank(st, &ids)
                } else {
                    model.forward_rank(st, &ids, flags & FWD_GRAPH != 0)
                };
                in_pass.store(false, Ordering::Release);
                match r {
                    Ok(()) => reply(&mut w, true, M::state_len(st), "")?,
                    Err(e) => {
                        // a failed pass leaves the group unusable (the primary aborts on this answer): do not wait for more work
                        reply(&mut w, false, M::state_len(st), &e.to_string())?;
                        return Err(e);
                    }
                }
            }
            T_TRUNCATE => {
                let len = u64_at(&b, 1)? as usize;
                let (status, msg) = match (pending.take(), seqs.get_mut(&sid)) {
                    (Some(msg), _) => (ACK_FAILED, msg),
                    (None, None) => (ACK_REJECTED_INVALID, format!("unknown sequence {sid}")),
                    (None, Some(st)) => match model.truncate(st, len) {
                        Ok(()) => (ACK_OK, String::new()),
                        Err(e) => (ack_status(&e), format!("truncate to {len}: {e}")),
                    },
                };
                ack(&mut w, status, seqs.get(&sid).map_or(0, M::state_len), &msg)?;
                if status >= ACK_FAILED {
                    return Err(Error::backend(msg));
                }
            }
            T_SNAPSHOT => {
                let keep = u64_at(&b, 1)? as usize;
                let (status, msg) = match (pending.take(), seqs.get_mut(&sid)) {
                    (Some(msg), _) => (ACK_FAILED, msg),
                    (None, None) => (ACK_REJECTED_INVALID, format!("unknown sequence {sid}")),
                    (None, Some(st)) => match model.snapshot(st, keep) {
                        Ok(()) => (ACK_OK, String::new()),
                        Err(e) => (ack_status(&e), format!("snapshot: {e}")),
                    },
                };
                ack(&mut w, status, seqs.get(&sid).map_or(0, M::state_len), &msg)?;
                if status >= ACK_FAILED {
                    return Err(Error::backend(msg));
                }
            }
            T_DROP => {
                seqs.remove(&sid);
            }
            T_SHUTDOWN => return Ok(passes),
            other => return Err(Error::backend(format!("tensor-parallel connection: unknown command {other}"))),
        }
    }
}

#[cfg(test)]
mod serve_tests {
    use super::*;
    use std::net::TcpListener;

    /// A rank whose forward fails on the ids it is told to fail on.
    struct Fake;
    impl RankModel for Fake {
        type State = usize;
        fn new_state(&self) -> Result<usize> {
            Ok(0)
        }
        fn forward_rank(&self, st: &mut usize, ids: &[u32], _graph: bool) -> Result<()> {
            if ids == [666] {
                return Err(Error::backend("launch failed: unspecified launch failure"));
            }
            *st += ids.len();
            Ok(())
        }
        fn truncate(&self, st: &mut usize, len: usize) -> Result<()> {
            *st = len;
            Ok(())
        }
        fn snapshot(&self, _st: &mut usize, _keep: usize) -> Result<()> {
            Ok(())
        }
        fn state_len(st: &usize) -> usize {
            *st
        }
    }

    fn forward(s: &mut TcpStream, ids: &[u32]) -> (u8, String) {
        let mut b = 1u64.to_le_bytes().to_vec();
        b.extend_from_slice(&0u64.to_le_bytes());
        for i in ids {
            b.extend_from_slice(&i.to_le_bytes());
        }
        write_frame(s, T_FORWARD, &b).unwrap();
        let (tag, r) = read_frame(s).unwrap();
        assert_eq!(tag, T_REPLY);
        (r[0], String::from_utf8_lossy(&r[9..]).into_owned())
    }

    /// A pass that fails on this rank is answered (so the primary aborts) and then the rank stops serving: it used to go on
    /// looping on a group that was already dead.
    #[test]
    fn a_failed_pass_is_answered_and_ends_serving() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = std::thread::spawn(move || serve(&Fake, l.accept().unwrap().0));
        let mut s = TcpStream::connect(addr).unwrap();
        write_frame(&mut s, T_NEW, &1u64.to_le_bytes()).unwrap();
        assert_eq!(forward(&mut s, &[1, 2, 3]), (1, String::new()), "a good pass is answered ok");
        let (ok, msg) = forward(&mut s, &[666]);
        assert_eq!(ok, 0);
        assert!(msg.contains("unspecified launch failure"), "{msg}");
        let r = t.join().unwrap();
        assert!(r.unwrap_err().to_string().contains("unspecified launch failure"), "serving ends with the pass's error");
    }

    #[test]
    fn the_primary_closing_the_stream_ends_serving_cleanly() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = std::thread::spawn(move || serve(&Fake, l.accept().unwrap().0));
        let mut s = TcpStream::connect(addr).unwrap();
        write_frame(&mut s, T_NEW, &1u64.to_le_bytes()).unwrap();
        assert_eq!(forward(&mut s, &[1]).0, 1);
        drop(s);
        assert_eq!(t.join().unwrap().unwrap(), 1);
    }

    /// A communicator that only counts aborts; the second field makes its abort never return, as `ncclCommAbort` does when
    /// the other rank is gone.
    struct AbortCount(std::sync::atomic::AtomicUsize, bool);
    impl Comm for AbortCount {
        fn name(&self) -> &str {
            "abort-count"
        }
        fn rank(&self) -> usize {
            0
        }
        fn world(&self) -> usize {
            2
        }
        fn all_reduce_sum(&self, _x: &crate::backend::TensorRef, _s: crate::backend::StreamId) -> Result<()> {
            unreachable!()
        }
        fn all_reduce_sum_into(&self, _p: &crate::backend::TensorRef, _o: &crate::backend::TensorRef, _s: crate::backend::StreamId) -> Result<()> {
            unreachable!()
        }
        fn all_gather(&self, _s: &crate::backend::TensorRef, _r: &crate::backend::TensorRef, _st: crate::backend::StreamId) -> Result<()> {
            unreachable!()
        }
        fn abort(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
            ABORTED.store(true, Ordering::SeqCst);
            if std::env::var_os(CHILD).is_some() {
                std::thread::sleep(Duration::from_millis(500)); // an abort that takes a while to return (392 ms on the Sparks)
            }
            while self.1 {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        fn stats(&self) -> crate::comm::CommStats {
            crate::comm::CommStats::default()
        }
        fn note_timed(&self, _d: Duration) {}
    }

    const CHILD: &str = "INTARSIA_TP_SERVE_TEST_CHILD";
    static ABORTED: AtomicBool = AtomicBool::new(false);
    /// The code a child process exits with when [`serve_group`] returned `Ok(1)`.
    const RETURNED: i32 = 42;

    /// What the fake rank does when the primary closes the connection, and in its pass.
    #[derive(Clone, Copy)]
    enum Rank {
        Stuck,
        Drained,
        DrainFails,
        NothingQueued,
        #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
        PassWaitsForAbort,
    }
    impl RankModel for Rank {
        type State = usize;
        fn new_state(&self) -> Result<usize> {
            Ok(0)
        }
        fn forward_rank(&self, st: &mut usize, ids: &[u32], _graph: bool) -> Result<()> {
            if matches!(self, Rank::PassWaitsForAbort) {
                while !ABORTED.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            *st += ids.len();
            Ok(())
        }
        fn truncate(&self, _st: &mut usize, _len: usize) -> Result<()> {
            Ok(())
        }
        fn snapshot(&self, _st: &mut usize, _keep: usize) -> Result<()> {
            Ok(())
        }
        fn state_len(st: &usize) -> usize {
            *st
        }
        fn drain(&self) -> Option<Drain> {
            match self {
                Rank::Stuck => Some(Box::new(|| loop {
                    std::thread::sleep(Duration::from_secs(1));
                })),
                Rank::Drained => Some(Box::new(|| Ok(()))),
                Rank::DrainFails => Some(Box::new(|| Err(Error::backend("stream failed")))),
                Rank::NothingQueued | Rank::PassWaitsForAbort => None,
            }
        }
    }

    /// The body of a child process: a primary that runs one pass and disappears. `wait`: it does not wait for the pass's answer.
    fn serve_one_pass_then_primary_gone(rank: Rank, wait: bool) -> ! {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let primary = std::thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            write_frame(&mut s, T_NEW, &1u64.to_le_bytes()).unwrap();
            if wait {
                assert_eq!(forward(&mut s, &[1]).0, 1);
            } else {
                let mut b = vec![0u8; 16];
                b[..8].copy_from_slice(&1u64.to_le_bytes());
                b.extend_from_slice(&1u32.to_le_bytes());
                write_frame(&mut s, T_FORWARD, &b).unwrap();
                std::thread::sleep(Duration::from_millis(100)); // the pass has started
            }
        });
        let comm: Arc<dyn Comm> = Arc::new(AbortCount(Default::default(), false));
        let r = serve_group(&rank, l.accept().unwrap().0, comm);
        let _ = primary.join();
        std::process::exit(if matches!(r, Ok(1)) { RETURNED } else { 1 })
    }

    /// `abort_group_and_exit` ends the process, so each case runs in a child process: this test binary again, on just `name`.
    fn child_code(name: &str) -> Option<i32> {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &format!("tp_remote::serve_tests::{name}"), "--nocapture"])
            .env(CHILD, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap()
            .code()
    }

    macro_rules! primary_gone_case {
        ($name:ident, $rank:expr, $wait:expr, $code:expr, $why:literal) => {
            #[test]
            fn $name() {
                if std::env::var_os(CHILD).is_some() {
                    serve_one_pass_then_primary_gone($rank, $wait);
                }
                assert_eq!(child_code(stringify!($name)), Some($code), $why);
            }
        };
    }
    primary_gone_case!(
        queued_work_that_cannot_finish_ends_the_rank_before_anything_is_freed,
        Rank::Stuck,
        true,
        EXIT_GROUP_FAILED,
        "the serving loop must not go on to free memory the stuck stream uses"
    );
    primary_gone_case!(queued_work_that_fails_ends_the_rank, Rank::DrainFails, true, EXIT_GROUP_FAILED, "a failed stream is a failed group");
    primary_gone_case!(a_primary_that_left_nothing_stuck_is_a_clean_end, Rank::Drained, true, RETURNED, "a primary that exited is not a failure");
    primary_gone_case!(a_rank_with_nothing_queued_does_not_wait, Rank::NothingQueued, true, RETURNED, "a model without a drain (the trait's default) has nothing to wait for");
    // the race seen with INTARSIA_TP_PEER_SYNC=1: the abort lets the pass end, the serving loop reads the end of the stream and
    // returns, and the process exited 0 while the watch was still waiting for the abort
    #[cfg(target_os = "linux")]
    primary_gone_case!(
        the_exit_code_of_a_rank_leaving_mid_pass_is_not_lost_to_a_normal_return,
        Rank::PassWaitsForAbort,
        false,
        EXIT_GROUP_FAILED,
        "the serving thread returned while the watch was leaving"
    );

    /// A primary's sequence with one forward sent, the far end of its connection, and a communicator whose abort never returns.
    fn forward_sent_and_abort_blocks() -> (RemoteSeq, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let far = std::thread::spawn(move || l.accept().unwrap().0);
        let near = TcpStream::connect(addr).unwrap();
        let far = far.join().unwrap();
        let peer = RemotePeer::new(near, Arc::new(AbortCount(Default::default(), true))).unwrap();
        let seq = peer.new_state().unwrap();
        seq.start_forward(&[1], false).unwrap();
        (seq, far)
    }

    /// `finish` on its own thread, its result well before the reply timeout or not at all.
    fn finish_within_limit(mut seq: RemoteSeq, while_waiting: impl FnOnce()) -> (RemoteSeq, Result<()>) {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let r = seq.finish();
            tx.send((seq, r)).unwrap();
        });
        std::thread::sleep(Duration::from_millis(100)); // it is waiting for the answer
        while_waiting();
        rx.recv_timeout(Duration::from_secs(5)).expect("finish waited behind a communicator abort that does not return")
    }

    fn reply(far: &mut TcpStream, status: u8, msg: &str) {
        let mut b = vec![status];
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(msg.as_bytes());
        write_frame(far, T_REPLY, &b).unwrap();
    }

    /// The 116 s hang of the 35B worker kill (docs/results/2026-11-02-dsv41-rdma.md, section 6).
    #[test]
    fn a_lost_connection_ends_a_wait_for_an_answer_even_if_the_abort_never_returns() {
        let (seq, far) = forward_sent_and_abort_blocks();
        let (seq, r) = finish_within_limit(seq, move || drop(far));
        assert!(r.unwrap_err().to_string().contains("gone"), "the wait ends with the connection");
        assert!(seq.start_forward(&[2], false).is_err(), "and nothing more is sent");
    }

    #[test]
    fn a_reported_failure_is_delivered_even_if_the_abort_never_returns() {
        let (seq, mut far) = forward_sent_and_abort_blocks();
        let (mut seq, r) = finish_within_limit(seq, || reply(&mut far, 0, "launch failed"));
        assert!(r.unwrap_err().to_string().contains("launch failed"), "the rank's own reason reaches the engine");
        assert!(seq.truncate(0).is_err(), "a command after the failure does not wait either");
    }

    #[test]
    fn an_answer_sent_before_the_connection_was_lost_is_still_delivered() {
        let (seq, mut far) = forward_sent_and_abort_blocks();
        // read what was sent: closing a socket with unread data resets it, and the reset discards the answer on the way
        assert_eq!(read_frame(&mut far).unwrap().0, T_NEW);
        assert_eq!(read_frame(&mut far).unwrap().0, T_FORWARD);
        reply(&mut far, 1, "");
        drop(far);
        std::thread::sleep(Duration::from_millis(200)); // the reader has seen both
        let (seq, r) = finish_within_limit(seq, || {});
        r.unwrap();
        assert_eq!(seq.len, 1);
    }

    /// The remote rank dying must reach the engine from the reader thread: the engine thread may be stuck in a kernel.
    #[test]
    fn a_lost_connection_reports_to_the_hook_before_the_abort_whether_installed_before_or_after() {
        for late in [false, true] {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap();
            let far = std::thread::spawn(move || l.accept().unwrap().0);
            let near = TcpStream::connect(addr).unwrap();
            let far = far.join().unwrap();
            let comm = Arc::new(AbortCount(Default::default(), false));
            let peer = RemotePeer::new(near, comm.clone()).unwrap();
            let (tx, rx) = channel();
            if !late {
                let tx = tx.clone();
                peer.on_lost(move |why| tx.send(why).unwrap());
            }
            drop(far);
            if late {
                std::thread::sleep(Duration::from_millis(300));
                peer.on_lost(move |why| tx.send(why).unwrap());
            }
            let why = rx.recv_timeout(Duration::from_secs(5)).expect("the hook runs");
            assert!(why.contains("connection"), "{why}");
            std::thread::sleep(Duration::from_millis(200));
            assert_eq!(comm.0.load(Ordering::SeqCst), 1, "the communicator is aborted once");
        }
    }
}

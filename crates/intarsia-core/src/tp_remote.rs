//! Tensor parallelism across hosts (docs/design/35b-tp2.md): rank 0's [`Model`] drives a rank that lives in another
//! process, usually on another machine, over one TCP connection. ADR-0006's in-process ranks are threads holding a
//! `Model` each; a remote rank is the same `Model` shard in its own process, run by [`serve`], and the primary keeps a
//! [`RemotePeer`] handle instead of the shard. The collectives themselves go through the group's [`Comm`] (NCCL on the
//! Sparks), never through this connection: the connection only carries what to run.
//!
//! Commands, in order on the stream (every one applies to the remote copy of one sequence, `sid`):
//! `NewState`, `Forward(flags, ids)`, `Truncate(len)`, `Snapshot(keep)`, `Drop`, plus `Shutdown`. `Forward`'s flags
//! carry rank 0's per-step decisions the remote rank must follow: [`FWD_GRAPH`], rank 0 replays its decode graph in
//! this step, so the remote rank replays its own (the two graphs record the same collectives in the same order). Only `Forward` is answered
//! (`ok`, the remote length, an error message); the others are fire-and-forget, applied in order, and a failure among
//! them fails the next forward. The primary sends `Forward` before it runs its own pass and reads the answer after it.
//! A reader thread on the primary watches the answers: an error aborts the primary's communicator so the primary is not
//! left waiting in a collective the remote rank never joins. After any failure the group is unusable (ADR-0006).
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
const T_HELLO: u8 = 0x40;

/// `Forward` flag: a single-token step that rank 0 runs as a captured decode graph (`model/graph_step.rs`).
pub const FWD_GRAPH: u64 = 1;

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

/// The first exchange on a fresh connection, both sides: each sends its fingerprint (model, options, rank, world) and
/// KV capacity in tokens and reads the other's. Fails on any fingerprint difference; returns the smaller capacity
/// (every rank must hold the same number of tokens).
pub fn handshake(stream: &mut TcpStream, fingerprint: &str, kv_cap: usize) -> Result<usize> {
    let mut body = (kv_cap as u64).to_le_bytes().to_vec();
    body.extend_from_slice(fingerprint.as_bytes());
    write_frame(stream, T_HELLO, &body)?;
    stream.flush().map_err(io)?;
    let (tag, b) = read_frame(stream)?;
    if tag != T_HELLO {
        return Err(Error::backend(format!("tensor-parallel handshake: unexpected message {tag}")));
    }
    let theirs = u64_at(&b, 0)? as usize;
    let fp = String::from_utf8_lossy(&b[8..]);
    if fp != fingerprint {
        return Err(Error::invalid(format!(
            "tensor-parallel ranks were started differently:\n  this rank:  {fingerprint}\n  other rank: {fp}"
        )));
    }
    Ok(kv_cap.min(theirs))
}

struct Reply {
    ok: bool,
    len: usize,
    msg: String,
}

/// Rank 0's handle on a remote rank (see the module docs).
pub struct RemotePeer {
    w: Mutex<BufWriter<TcpStream>>,
    replies: Mutex<Receiver<Reply>>,
    next_sid: AtomicU64,
    dead: Arc<AtomicBool>,
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
        std::thread::Builder::new()
            .name("tp-remote-reader".into())
            .spawn(move || loop {
                match read_frame(&mut rd) {
                    Ok((T_REPLY, b)) if b.len() >= 9 => {
                        let ok = b[0] == 1;
                        let len = u64::from_le_bytes(b[1..9].try_into().unwrap()) as usize;
                        let msg = String::from_utf8_lossy(&b[9..]).into_owned();
                        if !ok {
                            comm.abort();
                        }
                        if tx.send(Reply { ok, len, msg }).is_err() {
                            return;
                        }
                    }
                    _ => {
                        d2.store(true, Ordering::SeqCst);
                        comm.abort();
                        return;
                    }
                }
            })
            .map_err(|e| Error::backend(format!("spawning the tensor-parallel reader: {e}")))?;
        Ok(Arc::new(RemotePeer { w: Mutex::new(BufWriter::new(stream)), replies: Mutex::new(rx), next_sid: AtomicU64::new(1), dead }))
    }

    fn send(&self, tag: u8, body: &[u8]) -> Result<()> {
        if self.dead.load(Ordering::SeqCst) {
            return Err(Error::backend("the remote tensor-parallel rank is gone"));
        }
        let mut w = self.w.lock().unwrap();
        write_frame(&mut *w, tag, body)?;
        w.flush().map_err(io)
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

    /// The remote rank's answer to the forward sent by [`RemoteSeq::start_forward`].
    pub fn finish(&mut self) -> Result<()> {
        let rx = self.peer.replies.lock().unwrap();
        let r = match rx.recv_timeout(reply_timeout()) {
            Ok(r) => r,
            Err(RecvTimeoutError::Timeout) => {
                return Err(Error::backend(format!("the remote tensor-parallel rank did not answer within {:?}", reply_timeout())))
            }
            Err(RecvTimeoutError::Disconnected) => return Err(Error::backend("the remote tensor-parallel rank is gone")),
        };
        self.len = r.len;
        if r.ok {
            Ok(())
        } else {
            Err(Error::backend(format!("remote tensor-parallel rank: {}", r.msg)))
        }
    }

    /// Roll the remote copy back to `len` tokens (after the primary rolled back successfully).
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        self.peer.send(T_TRUNCATE, &RemotePeer::sid_body(self.sid, &[len as u64]))?;
        self.len = len;
        Ok(())
    }

    /// Snapshot the remote copy's recurrent state (after the primary did).
    pub fn snapshot(&self, keep: usize) -> Result<()> {
        self.peer.send(T_SNAPSHOT, &RemotePeer::sid_body(self.sid, &[keep as u64]))
    }
}

impl Drop for RemoteSeq {
    fn drop(&mut self) {
        let _ = self.peer.send(T_DROP, &RemotePeer::sid_body(self.sid, &[]));
    }
}

/// The remote rank's loop: apply the primary's commands to `model` (this process's shard) until `Shutdown` or the
/// connection closes. A forward that fails resets that sequence (as [`Model::forward`] does) and is answered with the
/// error; a failed fire-and-forget command is reported by the next forward of any sequence. Returns how many forward
/// passes ran.
pub fn serve(model: &Model, stream: TcpStream) -> Result<u64> {
    stream.set_nodelay(true).map_err(io)?;
    let mut rd = BufReader::new(stream.try_clone().map_err(io)?);
    let mut w = BufWriter::new(stream);
    let mut seqs: HashMap<u64, SequenceState> = HashMap::new();
    let mut pending: Option<String> = None;
    let mut passes = 0u64;
    let reply = |w: &mut BufWriter<TcpStream>, ok: bool, len: usize, msg: &str| -> Result<()> {
        let mut b = vec![u8::from(ok)];
        b.extend_from_slice(&(len as u64).to_le_bytes());
        b.extend_from_slice(msg.as_bytes());
        write_frame(w, T_REPLY, &b)?;
        w.flush().map_err(io)
    };
    loop {
        let (tag, b) = match read_frame(&mut rd) {
            Ok(f) => f,
            Err(_) => return Ok(passes), // the primary closed the connection
        };
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
                    reply(&mut w, false, seqs.get(&sid).map_or(0, |s| s.len), &msg)?;
                    continue;
                }
                let Some(st) = seqs.get_mut(&sid) else {
                    reply(&mut w, false, 0, &format!("unknown sequence {sid}"))?;
                    continue;
                };
                passes += 1;
                match model.forward_rank(st, &ids, flags & FWD_GRAPH != 0) {
                    Ok(()) => reply(&mut w, true, st.len, "")?,
                    Err(e) => reply(&mut w, false, st.len, &e.to_string())?,
                }
            }
            T_TRUNCATE => {
                let len = u64_at(&b, 1)? as usize;
                if let Some(st) = seqs.get_mut(&sid) {
                    if let Err(e) = st.truncate(len, &*model.backend) {
                        pending = pending.or(Some(format!("truncate to {len}: {e}")));
                    }
                }
            }
            T_SNAPSHOT => {
                let keep = u64_at(&b, 1)? as usize;
                if let Some(st) = seqs.get_mut(&sid) {
                    if let Err(e) = st.snapshot(&*model.backend, keep) {
                        pending = pending.or(Some(format!("snapshot: {e}")));
                    }
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

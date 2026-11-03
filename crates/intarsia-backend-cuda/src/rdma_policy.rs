//! `INTARSIA_DSV41_RDMA`: the setting, how the two ranks agree on it, and the lock-step setup of the RDMA ring with its
//! fallback to NCCL (lane dsv41/rdma, docs/results/2026-11-02-dsv41-rdma.md). No CUDA or verbs here: the ring itself is
//! behind [`RingOps`], so the whole flow runs in the unit tests (and on a host without a GPU) with a fake ring.
//!
//! Three states: unset = auto (on for a DeepSeek-V4.1 model, tried, and NCCL if anything about the setup fails), `1` = required
//! (an error where it cannot work), `0` = off. `auto` is also accepted as a value (any model).
//!
//! The setup runs on the control stream between the two ranks, which the tensor-parallel handshake reads next, so every phase
//! is a symmetric exchange: both ranks send their frame and read the other's, whatever happened locally, and both decide from
//! the same two frames. Rank 0 writes first and rank 1 reads first (no deadlock on a full socket buffer). A read that outlasts
//! [`SETUP_TIMEOUT`] is a hard error: the stream's state would be unknown.

use std::io::{Read, Write};
use std::time::Duration;

/// What one rank asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wish {
    Off,
    Auto,
    Required,
}

impl Wish {
    pub fn byte(self) -> u8 {
        match self {
            Wish::Off => 0,
            Wish::Auto => 1,
            Wish::Required => 2,
        }
    }
    pub fn from_byte(b: u8) -> Option<Wish> {
        match b {
            0 => Some(Wish::Off),
            1 => Some(Wish::Auto),
            2 => Some(Wish::Required),
            _ => None,
        }
    }
    /// The env value: unset = auto for a DeepSeek-V4.1 model and off for any other; `1` required; `auto` auto; anything else
    /// (`0`) off.
    pub fn parse(value: Option<&str>, dsv41: bool) -> Wish {
        match value.map(str::trim) {
            None => {
                if dsv41 {
                    Wish::Auto
                } else {
                    Wish::Off
                }
            }
            Some("1") => Wish::Required,
            Some("auto") => Wish::Auto,
            Some(_) => Wish::Off,
        }
    }
}

/// What the two wishes together mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    Off,
    /// Try it; `required`: a failure is an error instead of a fallback.
    Try { required: bool },
}

/// Both ranks call this with the two wishes and get the same answer.
///
/// | rank A | rank B | result |
/// |---|---|---|
/// | auto | auto | try, fall back on any failure |
/// | off | auto | off (NCCL) |
/// | required | auto or required | try, a failure is an error |
/// | required | off | error: the ranks disagree |
pub fn resolve(a: Wish, b: Wish) -> Result<Plan, String> {
    use Wish::*;
    match (a, b) {
        (Off, Off) | (Off, Auto) | (Auto, Off) => Ok(Plan::Off),
        (Required, Off) | (Off, Required) => Err("INTARSIA_DSV41_RDMA is 1 (required) on one rank and 0 (off) on the other: set it the same on both".to_string()),
        (Auto, Auto) => Ok(Plan::Try { required: false }),
        (Required, _) | (_, Required) => Ok(Plan::Try { required: true }),
    }
}

/// Devices and GID index of the ring, from the values of `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX` (the ones NCCL is told to
/// use, so a launch needs no new variable). Port 1 is used, so a name with `:port`, a `=` or `^` prefix is refused.
pub fn devices(hca: Option<&str>, gid: Option<&str>) -> Result<(String, i32), String> {
    let hca = hca.map(str::trim).unwrap_or("");
    if hca.is_empty() || hca.starts_with(['=', '^']) {
        return Err("NCCL_IB_HCA does not name the RDMA device(s) (e.g. rocep1s0f1,roceP2p1s0f1)".to_string());
    }
    if hca.contains(':') {
        return Err("NCCL_IB_HCA names devices with ports; the ring uses port 1 of each device name".to_string());
    }
    let gid = match gid.map(str::trim) {
        None | Some("") => 3,
        Some(v) => v.parse::<i32>().map_err(|_| format!("NCCL_IB_GID_INDEX '{v}' is not a number"))?,
    };
    Ok((hca.to_string(), gid))
}

/// Seconds a device wait may last before it traps (`INTARSIA_DSV41_RDMA_DEADLINE_S`). Above the 34 s a dead link takes to
/// complete with an error, well under the 600 s the primary waits for a rank's answer; default 180.
pub fn deadline_s(v: Option<&str>) -> f64 {
    v.and_then(|s| s.trim().parse::<f64>().ok()).filter(|d| d.is_finite() && *d >= 1.0).map_or(180.0, |d| d.min(3600.0))
}

pub const SETUP_TIMEOUT: Duration = Duration::from_secs(60);
const MSG: usize = 160;
/// Room for the ring's queue-pair info block.
pub const INFO_MAX: usize = 256;
const FRAME: usize = 1 + MSG + INFO_MAX;

fn frame(ok: bool, msg: &str, info: &[u8]) -> Vec<u8> {
    let mut f = vec![0u8; FRAME];
    f[0] = u8::from(ok);
    let m = msg.as_bytes();
    let n = m.len().min(MSG - 1);
    f[1..1 + n].copy_from_slice(&m[..n]);
    let k = info.len().min(INFO_MAX);
    f[1 + MSG..1 + MSG + k].copy_from_slice(&info[..k]);
    f
}

struct Peer {
    ok: bool,
    msg: String,
    info: Vec<u8>,
}

fn unframe(f: &[u8]) -> Peer {
    let m = &f[1..1 + MSG];
    let end = m.iter().position(|&b| b == 0).unwrap_or(MSG);
    Peer { ok: f[0] != 0, msg: String::from_utf8_lossy(&m[..end]).into_owned(), info: f[1 + MSG..].to_vec() }
}

/// One symmetric exchange of equal-sized buffers: rank 0 writes then reads, rank 1 reads then writes.
fn xchg<S: Read + Write>(s: &mut S, rank: usize, mine: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut theirs = vec![0u8; mine.len()];
    if rank == 0 {
        s.write_all(mine)?;
        s.flush()?;
        s.read_exact(&mut theirs)?;
    } else {
        s.read_exact(&mut theirs)?;
        s.write_all(mine)?;
        s.flush()?;
    }
    Ok(theirs)
}

/// The ring, as the setup needs it. Every method is local to this rank.
pub trait RingOps {
    /// Parse the settings and open the ring (devices, memory, queue pairs); `min_rails` 2 under auto, 1 when required.
    fn open(&mut self, min_rails: i32) -> Result<(), String>;
    /// The queue-pair info the peer needs (at most [`INFO_MAX`] bytes); only after a successful `open`.
    fn info(&self) -> Vec<u8>;
    fn connect(&mut self, peer_info: &[u8]) -> Result<(), String>;
    /// A host round trip on every rail, with a timeout.
    fn ping(&mut self, timeout_ms: i32) -> Result<(), String>;
    /// Free everything: the ring is not used (setup fell back, or never got that far).
    fn discard(&mut self);
}

/// How the setup ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Both ranks agree on NCCL only (a setting), nothing was opened.
    Off,
    /// Auto, and something failed here or on the other rank: the ring is freed, NCCL carries everything. The text is the one
    /// log line's reason.
    Fallback(String),
    /// Both rings are connected and pinged; the caller starts them.
    Ready,
}

/// Run the setup. `Err` is a hard error on both ranks (a disagreement, a required ring that failed, a silent peer).
pub fn setup<S: Read + Write>(ctl: &mut S, rank: usize, wish: Wish, ring: &mut dyn RingOps, ping_ms: i32) -> Result<Outcome, String> {
    let io = |what: &str, e: std::io::Error| format!("RDMA setup ({what}) with the other rank: {e}");
    let theirs = xchg(ctl, rank, &[wish.byte()]).map_err(|e| io("settings", e))?;
    let other = Wish::from_byte(theirs[0]).ok_or_else(|| "RDMA setup: the other rank sent an unknown setting (different builds?)".to_string())?;
    let plan = resolve(wish, other)?;
    let Plan::Try { required } = plan else { return Ok(Outcome::Off) };

    // each phase: (local result, exchange, peer's result); a failed phase on either side ends in the same decision
    let phase = |ctl: &mut S, what: &str, local: &Result<(), String>, info: &[u8]| -> Result<Peer, String> {
        let (ok, msg) = match local {
            Ok(()) => (true, String::new()),
            Err(e) => (false, e.clone()),
        };
        let f = xchg(ctl, rank, &frame(ok, &msg, info)).map_err(|e| io(what, e))?;
        Ok(unframe(&f))
    };
    let decide = |what: &str, local: &Result<(), String>, peer: &Peer| -> Option<String> {
        match (local, peer.ok) {
            (Ok(()), true) => None,
            (Err(e), true) => Some(format!("{what}: {e}")),
            (Ok(()), false) => Some(format!("{what} failed on the other rank: {}", peer.msg)),
            (Err(e), false) => Some(format!("{what}: {e}; other rank: {}", peer.msg)),
        }
    };
    let fail = |ring: &mut dyn RingOps, why: String| -> Result<Outcome, String> {
        ring.discard();
        if required {
            Err(format!("INTARSIA_DSV41_RDMA=1 but the RDMA ring cannot be set up: {why}"))
        } else {
            Ok(Outcome::Fallback(why))
        }
    };

    let opened = ring.open(if required { 1 } else { 2 });
    let info = if opened.is_ok() { ring.info() } else { Vec::new() };
    let p = phase(ctl, "open", &opened, &info)?;
    if let Some(why) = decide("open", &opened, &p) {
        return fail(ring, why);
    }
    let connected = ring.connect(&p.info);
    let p = phase(ctl, "connect", &connected, &[])?;
    if let Some(why) = decide("connect", &connected, &p) {
        return fail(ring, why);
    }
    let pinged = ring.ping(ping_ms);
    let p = phase(ctl, "ping", &pinged, &[])?;
    if let Some(why) = decide("ping", &pinged, &p) {
        return fail(ring, why);
    }
    Ok(Outcome::Ready)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    #[test]
    fn wish_parsing() {
        assert_eq!(Wish::parse(None, true), Wish::Auto);
        assert_eq!(Wish::parse(None, false), Wish::Off);
        assert_eq!(Wish::parse(Some("1"), false), Wish::Required);
        assert_eq!(Wish::parse(Some("0"), true), Wish::Off);
        assert_eq!(Wish::parse(Some("auto"), false), Wish::Auto);
        assert_eq!(Wish::parse(Some("yes"), true), Wish::Off);
        for w in [Wish::Off, Wish::Auto, Wish::Required] {
            assert_eq!(Wish::from_byte(w.byte()), Some(w));
        }
        assert_eq!(Wish::from_byte(9), None);
    }

    #[test]
    fn wish_matrix() {
        use Wish::*;
        let t = |a, b| resolve(a, b);
        assert_eq!(t(Auto, Auto), Ok(Plan::Try { required: false }));
        assert_eq!(t(Off, Auto), Ok(Plan::Off));
        assert_eq!(t(Auto, Off), Ok(Plan::Off));
        assert_eq!(t(Off, Off), Ok(Plan::Off));
        assert_eq!(t(Required, Auto), Ok(Plan::Try { required: true }));
        assert_eq!(t(Auto, Required), Ok(Plan::Try { required: true }));
        assert_eq!(t(Required, Required), Ok(Plan::Try { required: true }));
        assert!(t(Required, Off).is_err());
        assert!(t(Off, Required).is_err());
        // symmetric
        for a in [Off, Auto, Required] {
            for b in [Off, Auto, Required] {
                assert_eq!(t(a, b), t(b, a));
            }
        }
    }

    #[test]
    fn device_settings() {
        assert_eq!(devices(Some("rocep1s0f1,roceP2p1s0f1"), Some("3")), Ok(("rocep1s0f1,roceP2p1s0f1".to_string(), 3)));
        assert_eq!(devices(Some("rocep1s0f1"), None), Ok(("rocep1s0f1".to_string(), 3)));
        assert!(devices(Some("mlx5_0:1"), None).is_err());
        assert!(devices(Some("^mlx5_1"), None).is_err());
        assert!(devices(Some("=mlx5_1"), None).is_err());
        assert!(devices(None, None).is_err());
        assert!(devices(Some(""), None).is_err());
        assert!(devices(Some("a"), Some("x")).is_err());
    }

    #[test]
    fn deadline_setting() {
        assert_eq!(deadline_s(None), 180.0);
        assert_eq!(deadline_s(Some("20")), 20.0);
        assert_eq!(deadline_s(Some("0.2")), 180.0);
        assert_eq!(deadline_s(Some("nan")), 180.0);
        assert_eq!(deadline_s(Some("x")), 180.0);
        assert_eq!(deadline_s(Some("99999")), 3600.0);
    }

    /// A ring that fails at a chosen phase and records what was done to it.
    #[derive(Default)]
    struct Fake {
        fail_at: Option<&'static str>,
        min_rails_seen: Arc<Mutex<Vec<i32>>>,
        log: Arc<Mutex<Vec<&'static str>>>,
        rails: i32,
    }
    impl RingOps for Fake {
        fn open(&mut self, min_rails: i32) -> Result<(), String> {
            self.log.lock().unwrap().push("open");
            self.min_rails_seen.lock().unwrap().push(min_rails);
            if self.fail_at == Some("open") || self.rails < min_rails {
                return Err(format!("{} rails, {} wanted", self.rails, min_rails));
            }
            Ok(())
        }
        fn info(&self) -> Vec<u8> {
            vec![7; 88]
        }
        fn connect(&mut self, peer_info: &[u8]) -> Result<(), String> {
            self.log.lock().unwrap().push("connect");
            assert_eq!(&peer_info[..88], &[7u8; 88][..], "the peer's info arrives whole");
            if self.fail_at == Some("connect") {
                return Err("qp RTR".into());
            }
            Ok(())
        }
        fn ping(&mut self, _ms: i32) -> Result<(), String> {
            self.log.lock().unwrap().push("ping");
            if self.fail_at == Some("ping") {
                return Err("no answer".into());
            }
            Ok(())
        }
        fn discard(&mut self) {
            self.log.lock().unwrap().push("discard");
        }
    }

    /// Both ranks of `setup` over a real loopback socket; a third frame afterwards proves the stream is still in step.
    fn run(w0: Wish, w1: Wish, f0: Fake, f1: Fake) -> (Result<Outcome, String>, Result<Outcome, String>, Vec<&'static str>, Vec<&'static str>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let (log0, log1) = (f0.log.clone(), f1.log.clone());
        let t = std::thread::spawn(move || {
            let mut s = l.accept().unwrap().0;
            let mut f1 = f1;
            let r = setup(&mut s, 1, w1, &mut f1, 100);
            // the stream is in step: the next thing rank 0 sends is exactly this
            let mut b = [0u8; 4];
            s.read_exact(&mut b).unwrap();
            assert_eq!(&b, b"next");
            r
        });
        let mut s = TcpStream::connect(addr).unwrap();
        let mut f0 = f0;
        let r0 = setup(&mut s, 0, w0, &mut f0, 100);
        s.write_all(b"next").unwrap();
        let r1 = t.join().unwrap();
        let (a, b) = (log0.lock().unwrap().clone(), log1.lock().unwrap().clone());
        (r0, r1, a, b)
    }

    fn ok2() -> Fake {
        Fake { rails: 2, ..Default::default() }
    }
    fn rails1() -> Fake {
        Fake { rails: 1, ..Default::default() }
    }
    /// A ring that fails at `phase` (its own log: clones would share one).
    fn bad(phase: &'static str) -> Fake {
        Fake { fail_at: Some(phase), rails: 2, ..Default::default() }
    }

    #[test]
    fn auto_auto_with_two_rails_is_ready() {
        let (a, b, la, lb) = run(Wish::Auto, Wish::Auto, ok2(), ok2());
        assert_eq!((a, b), (Ok(Outcome::Ready), Ok(Outcome::Ready)));
        assert_eq!(la, ["open", "connect", "ping"]);
        assert_eq!(lb, ["open", "connect", "ping"]);
    }

    #[test]
    fn off_on_one_rank_means_nccl_without_opening_anything() {
        let (a, b, la, lb) = run(Wish::Off, Wish::Auto, ok2(), ok2());
        assert_eq!((a, b), (Ok(Outcome::Off), Ok(Outcome::Off)));
        assert!(la.is_empty() && lb.is_empty());
    }

    #[test]
    fn required_against_off_is_an_error_on_both() {
        let (a, b, ..) = run(Wish::Required, Wish::Off, ok2(), ok2());
        assert!(a.unwrap_err().contains("set it the same on both"));
        assert!(b.unwrap_err().contains("set it the same on both"));
    }

    #[test]
    fn auto_needs_two_rails_required_accepts_one() {
        let (a, b, la, lb) = run(Wish::Auto, Wish::Auto, rails1(), rails1());
        for r in [&a, &b] {
            match r {
                Ok(Outcome::Fallback(why)) => assert!(why.contains("2 wanted"), "{why}"),
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(la, ["open", "discard"]);
        assert_eq!(lb, ["open", "discard"]);
        let (a, b, ..) = run(Wish::Required, Wish::Required, rails1(), rails1());
        assert_eq!((a, b), (Ok(Outcome::Ready), Ok(Outcome::Ready)));
    }

    #[test]
    fn a_failure_on_one_rank_sends_both_to_nccl_at_every_phase() {
        for phase in ["open", "connect", "ping"] {
            // the failing rank is rank 1, then rank 0: both orders
            for swap in [false, true] {
                let (g0, g1) = if swap { (bad(phase), ok2()) } else { (ok2(), bad(phase)) };
                let (a, b, la, lb) = run(Wish::Auto, Wish::Auto, g0, g1);
                for r in [&a, &b] {
                    match r {
                        Ok(Outcome::Fallback(why)) => assert!(why.contains(phase), "{phase}: {why}"),
                        other => panic!("{phase} swap {swap}: {other:?}"),
                    }
                }
                assert_eq!(la.last(), Some(&"discard"), "{phase} swap {swap}: {la:?}");
                assert_eq!(lb.last(), Some(&"discard"), "{phase} swap {swap}: {lb:?}");
            }
        }
    }

    #[test]
    fn a_required_ring_that_fails_is_an_error_on_both() {
        let (a, b, ..) = run(Wish::Required, Wish::Auto, ok2(), bad("ping"));
        assert!(a.unwrap_err().contains("cannot be set up"));
        assert!(b.unwrap_err().contains("cannot be set up"));
    }

    #[test]
    fn frames_round_trip_and_truncate() {
        let f = frame(false, &"x".repeat(500), &[1, 2, 3]);
        assert_eq!(f.len(), FRAME);
        let p = unframe(&f);
        assert!(!p.ok);
        assert_eq!(p.msg.len(), MSG - 1);
        assert_eq!(&p.info[..3], &[1, 2, 3]);
    }
}

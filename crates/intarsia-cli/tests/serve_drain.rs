//! A supervised `intarsia serve` told to stop (SIGTERM to the supervisor): an idle server stops at once and the supervisor never
//! reaches for SIGKILL; a request still running at the server's drain deadline (`INTARSIA_DRAIN_S`) ends with `finish_reason` `abort`, an
//! error event and `[DONE]` instead of a cut connection, and the supervisor still has no SIGKILL to send.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The tiny checkpoint of the model tests, with a tokenizer.
fn model_dir() -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../intarsia-models/tests/fixtures/tiny_qwen3_5");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("serve_drain_model");
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["config.json", "model.safetensors"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    let tok = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"t0": 0}, "unk_token": "t0"}
    });
    std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
    dir
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

struct Server {
    child: Child,
    port: u16,
    /// What the supervisor and the server wrote to stderr, read by a thread.
    log: std::sync::mpsc::Receiver<String>,
    lines: Vec<String>,
}

impl Server {
    fn start(env: &[(&str, &str)]) -> Server {
        let port = free_port();
        let mut c = Command::new(env!("CARGO_BIN_EXE_intarsia"));
        c.args(["serve", "-m"]).arg(model_dir()).args(["--backend", "cpu", "--host", "127.0.0.1", "--port", &port.to_string(), "--no-ui"]);
        for (k, v) in env {
            c.env(k, v);
        }
        let mut child = c.stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
        let (tx, log) = std::sync::mpsc::channel();
        let err = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            for l in BufReader::new(err).lines().map_while(|l| l.ok()) {
                if tx.send(l).is_err() {
                    break;
                }
            }
        });
        let mut s = Server { child, port, log, lines: Vec::new() };
        let t0 = Instant::now();
        while !s.get("/health").starts_with("HTTP/1.1 200") {
            assert!(t0.elapsed() < Duration::from_secs(60), "no /health 200: {}", s.stderr());
            std::thread::sleep(Duration::from_millis(200));
        }
        s
    }

    fn get(&self, path: &str) -> String {
        let mut got = String::new();
        if let Ok(mut s) = TcpStream::connect(("127.0.0.1", self.port)) {
            s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let _ = s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes());
            let _ = s.read_to_string(&mut got);
        }
        got
    }

    fn stderr(&mut self) -> String {
        self.lines.extend(self.log.try_iter());
        self.lines.join("\n")
    }

    fn term(&self) {
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
    }

    /// Waits for the supervisor to exit: how long it took after now, its exit code, and everything it wrote.
    fn exit(mut self, limit: Duration) -> (Duration, Option<i32>, String) {
        let t0 = Instant::now();
        let code = loop {
            if let Some(s) = self.child.try_wait().unwrap() {
                break s.code();
            }
            if t0.elapsed() > limit {
                unsafe { libc::kill(self.child.id() as i32, libc::SIGKILL) };
                let _ = self.child.wait();
                panic!("still running {limit:?} after the stop signal: {}", self.stderr());
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let took = t0.elapsed();
        std::thread::sleep(Duration::from_millis(200)); // the reader thread's last lines
        (took, code, self.stderr())
    }
}

#[test]
fn an_idle_server_stops_at_once_without_a_sigkill() {
    let s = Server::start(&[("INTARSIA_DRAIN_S", "5")]);
    s.term();
    let (took, code, err) = s.exit(Duration::from_secs(15));
    assert_eq!(code, Some(0), "{err}");
    assert!(took < Duration::from_secs(5), "an idle server waited for the drain deadline: {took:?}\n{err}");
    assert!(!err.contains("SIGKILL"), "{err}");
    assert!(!err.contains("drain deadline"), "{err}");
}

#[test]
fn a_request_still_running_at_the_drain_deadline_ends_explicitly() {
    // the debug hook holds every request in the engine for 30 s before it generates, so it is still running at the 1 s deadline
    let mut s = Server::start(&[("INTARSIA_DRAIN_S", "1"), ("INTARSIA_DEBUG_STALL_S", "30")]);
    let body = r#"{"prompt": "t0", "stream": true, "max_tokens": 4}"#;
    let mut c = TcpStream::connect(("127.0.0.1", s.port)).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    c.write_all(
        format!("POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len())
            .as_bytes(),
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(500)); // admitted, stalled in the engine
    let t0 = Instant::now();
    s.term();
    let mut reply = String::new();
    c.read_to_string(&mut reply).unwrap_or_else(|e| panic!("the connection was cut, not ended: {e}\nso far: {reply}\n{}", s.stderr()));
    let ended = t0.elapsed();
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert!(reply.contains(r#""finish_reason":"abort""#), "{reply}");
    assert!(reply.contains(r#""code":503"#) && reply.contains("shutting down"), "{reply}");
    assert!(reply.ends_with("0\r\n\r\n") && reply.contains("data: [DONE]"), "the chunked body must end cleanly: {reply}");
    assert!(ended >= Duration::from_millis(800) && ended < Duration::from_secs(10), "ended {ended:?} after the signal");
    let (_, code, err) = s.exit(Duration::from_secs(15));
    assert_eq!(code, Some(0), "{err}");
    assert!(err.contains("drain deadline"), "{err}");
    assert!(!err.contains("SIGKILL"), "{err}");
}

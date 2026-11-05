//! A supervised `intarsia serve` whose port is not to be had: it waits for a server that is stopping, with the model loaded, and stops with
//! `EXIT_FATAL` only if that does not end, instead of reloading the model for ever.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The tiny checkpoint of the model tests, with a tokenizer.
fn model_dir() -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../intarsia-models/tests/fixtures/tiny_qwen3_5");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("serve_bind_model");
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

fn serve(host: &str, port: &str, retry_s: Option<&str>) -> Child {
    let mut c = Command::new(env!("CARGO_BIN_EXE_intarsia"));
    c.args(["serve", "-m"]).arg(model_dir()).args(["--backend", "cpu", "--host", host, "--port", port, "--no-ui"]);
    if let Some(r) = retry_s {
        c.env("INTARSIA_BIND_RETRY_S", r);
    }
    c.stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap()
}

fn stop(c: &mut Child) {
    // SIGTERM: the supervisor takes its child with it
    unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
    let _ = c.wait();
}

fn stderr_of(c: &mut Child) -> String {
    let mut err = String::new();
    c.stderr.take().unwrap().read_to_string(&mut err).unwrap();
    err
}

/// Runs the server to its exit: the exit code, its stderr, and how long it ran after it said it would retry, which leaves out the model load.
fn to_exit(mut c: Child, limit: Duration) -> (Option<i32>, String, Option<Duration>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let out = c.stderr.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut all = String::new();
        for line in std::io::BufRead::lines(std::io::BufReader::new(out)).map_while(|l| l.ok()) {
            if line.contains("retrying for up to") {
                let _ = tx.send(Instant::now());
            }
            all.push_str(&line);
            all.push('\n');
        }
        all
    });
    let t0 = Instant::now();
    let code = loop {
        if let Some(s) = c.try_wait().unwrap() {
            break s.code();
        }
        if t0.elapsed() > limit {
            stop(&mut c);
            panic!("still running after {limit:?}");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let ended = Instant::now();
    let err = reader.join().unwrap();
    (code, err, rx.try_recv().ok().map(|t| ended.duration_since(t)))
}

#[test]
fn a_port_that_stays_taken_stops_the_supervisor_after_the_retry() {
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = taken.local_addr().unwrap().port().to_string();
    let (code, err, retried) = to_exit(serve("127.0.0.1", &port, Some("2")), Duration::from_secs(60));
    assert_eq!(code, Some(intarsia_server::EXIT_FATAL), "{err}");
    assert!(err.contains(&format!("bind 127.0.0.1:{port}")), "the model must have loaded and the bind failed (a load error is 78 too): {err}");
    assert!(retried.is_some_and(|d| d >= Duration::from_millis(1500)), "{retried:?}: it did not retry for 2 s: {err}");
    assert_eq!(err.matches("[supervisor] started server").count(), 1, "{err}");
}

/// What a restart after exit 86 or 87 meets when the previous process still holds the port: the outage ends by itself, without a second load.
#[test]
fn a_port_a_stopping_server_still_holds_is_taken_over_without_a_reload() {
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = taken.local_addr().unwrap();
    let mut c = serve("127.0.0.1", &addr.port().to_string(), None);
    std::thread::sleep(Duration::from_secs(4));
    assert!(c.try_wait().unwrap().is_none(), "it gave up while the port was held");
    drop(taken);
    let t0 = Instant::now();
    let answer = loop {
        let mut got = String::new();
        if let Ok(mut s) = TcpStream::connect(addr) {
            s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let _ = s.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
            let _ = s.read_to_string(&mut got);
        }
        if got.starts_with("HTTP/1.1 200") {
            break got;
        }
        if t0.elapsed() > Duration::from_secs(30) {
            stop(&mut c);
            panic!("no /health 200 after the port came free: {:?}\n{}", got, stderr_of(&mut c));
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(answer.contains("200"));
    stop(&mut c);
    let err = stderr_of(&mut c);
    assert_eq!(err.matches("[supervisor] started server").count(), 1, "one model load: {err}");
    assert!(!err.contains("restart #"), "{err}");
}

/// An address the network has not brought up yet (192.0.2.1 is reserved for documentation, so never on this box) is retried too.
#[test]
fn an_address_not_available_is_retried_too() {
    let (code, err, retried) = to_exit(serve("192.0.2.1", "18999", Some("2")), Duration::from_secs(60));
    assert_eq!(code, Some(intarsia_server::EXIT_FATAL), "{err}");
    assert!(retried.is_some_and(|d| d >= Duration::from_millis(1500)), "{retried:?}: it did not retry for 2 s: {err}");
}

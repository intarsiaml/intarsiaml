//! A supervised `intarsia serve` whose port is taken: it must stop with `EXIT_FATAL`, not reload the model for ever.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
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

#[test]
fn a_supervised_server_stops_when_its_port_is_taken() {
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = taken.local_addr().unwrap().port().to_string();
    let mut c = Command::new(env!("CARGO_BIN_EXE_intarsia"))
        .args(["serve", "-m"])
        .arg(model_dir())
        .args(["--backend", "cpu", "--host", "127.0.0.1", "--port", &port, "--no-ui"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let t0 = Instant::now();
    let code = loop {
        if let Some(s) = c.try_wait().unwrap() {
            break s.code();
        }
        if t0.elapsed() > Duration::from_secs(60) {
            // SIGTERM: the supervisor takes its child with it
            unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
            let _ = c.wait();
            panic!("still restarting after 60 s");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut err = String::new();
    std::io::Read::read_to_string(&mut c.stderr.take().unwrap(), &mut err).unwrap();
    assert_eq!(code, Some(intarsia_server::EXIT_FATAL), "{err}");
    assert!(err.contains(&format!("bind 127.0.0.1:{port}")), "the model must have loaded and the bind failed (a load error is 78 too): {err}");
}

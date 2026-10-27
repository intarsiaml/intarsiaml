//! `scripts/deploy.sh` syncs with `rsync --delete`; run output kept in the deployed tree (`runs/`, raw logs and
//! `state.json`) must survive a second deploy (drill g4d lost it).

use std::path::Path;
use std::process::Command;

#[test]
fn deploy_rsync_keeps_runs_and_stale_files_go() {
    if Command::new("rsync").arg("--version").output().is_err() {
        panic!("rsync is needed for this test");
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/deploy.sh");
    let script = std::fs::read_to_string(root).unwrap();
    let line = script.lines().find(|l| l.starts_with("RSYNC=(")).expect("RSYNC=( line in deploy.sh");
    let t = std::env::temp_dir().join(format!("deploy-runs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&t);
    let (src, dst) = (t.join("src"), t.join("dst"));
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(dst.join("runs/onboard1/logs")).unwrap();
    std::fs::write(src.join("a.txt"), "new").unwrap();
    std::fs::write(dst.join("stale.txt"), "old").unwrap();
    std::fs::write(dst.join("runs/onboard1/logs/acc-short.out"), "raw log").unwrap();
    std::fs::write(dst.join("runs/onboard1/state.json"), "{}").unwrap();
    let cmd = format!("{line}; \"${{RSYNC[@]}}\" '{}/' '{}/'", src.display(), dst.display());
    let st = Command::new("bash").arg("-c").arg(cmd).status().unwrap();
    assert!(st.success());
    assert!(dst.join("a.txt").exists());
    assert!(!dst.join("stale.txt").exists(), "--delete still removes stale deployed files");
    assert!(dst.join("runs/onboard1/logs/acc-short.out").exists(), "raw logs were deleted by the deploy");
    assert!(dst.join("runs/onboard1/state.json").exists());
    std::fs::remove_dir_all(&t).ok();
}

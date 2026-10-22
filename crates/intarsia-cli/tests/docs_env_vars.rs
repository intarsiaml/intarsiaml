//! Drift check for the environment-variable table in `docs/tuning.md`: every `INTARSIA_*` name that appears in the
//! code or scripts needs a row there, every row must name a variable that still exists, and the file a row cites must
//! still mention it. Add a row (variable, `path:line` where it is read, default, effect, numerics) when this fails.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Directories searched for variable names (docs and results are deliberately not: they only quote variables).
const ROOTS: &[&str] = &["crates", "kernels", "scripts", "tools", "deploy", ".github"];
/// Placeholders and prefixes used in prose and example commands, never read as a variable.
const NOT_VARIABLES: &[&str] = &["INTARSIA_X"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if !matches!(name.as_str(), "target" | ".git" | "node_modules" | "__pycache__" | ".venv") {
                walk(&p, out);
            }
        } else if p.metadata().is_ok_and(|m| m.len() < 4 << 20) {
            out.push(p);
        }
    }
}

/// Variable names in `text`: `INTARSIA_` followed by upper-case letters, digits and underscores. A name ending in `_`
/// is a prefix (`INTARSIA_HIP_*`) and is skipped.
fn names_in(text: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let prefix = ["INTARSIA", "_"].concat();
    let mut rest = text;
    while let Some(at) = rest.find(&prefix) {
        let before_ok = rest[..at].chars().next_back().is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
        let tail = &rest[at + prefix.len()..];
        let len = tail.bytes().take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || *b == b'_').count();
        let name = &rest[at..at + prefix.len() + len];
        if before_ok && len > 0 && tail.as_bytes()[0].is_ascii_uppercase() && !name.ends_with('_') && !NOT_VARIABLES.contains(&name) {
            found.insert(name.to_string());
        }
        rest = &rest[at + prefix.len()..];
    }
    found
}

/// Variable name -> (cited file, row) for every table row of `docs/tuning.md`.
fn documented(root: &Path) -> BTreeMap<String, (String, String)> {
    let text = std::fs::read_to_string(root.join("docs/tuning.md")).expect("docs/tuning.md");
    let mut rows = BTreeMap::new();
    for line in text.lines().filter(|l| l.starts_with("| `INTARSIA_")) {
        let cells: Vec<&str> = line.trim_matches('|').split(" | ").map(str::trim).collect();
        assert!(cells.len() == 5, "docs/tuning.md row needs 5 cells (variable, read at, default, effect, numerics): {line}");
        let name = cells[0].trim_matches('`').to_string();
        let file = cells[1].trim_matches('`').rsplit_once(':').map_or(cells[1], |(f, _)| f).trim_matches('`').to_string();
        let numerics = cells[4].split_whitespace().next().unwrap_or("");
        assert!(
            ["yes", "no", "n/a", "unverified"].contains(&numerics),
            "{name}: numerics cell must start with yes, no, n/a or unverified, got `{}`",
            cells[4]
        );
        assert!(rows.insert(name.clone(), (file, line.to_string())).is_none(), "{name} has two rows in docs/tuning.md");
    }
    rows
}

#[test]
fn every_env_var_in_the_code_has_a_row_in_docs_tuning_md() {
    let root = repo_root();
    let mut files = Vec::new();
    for r in ROOTS {
        walk(&root.join(r), &mut files);
    }
    let this_file = Path::new(file!()).file_name().unwrap().to_owned();
    let mut in_code: BTreeMap<String, String> = BTreeMap::new();
    for f in files.iter().filter(|f| f.file_name() != Some(&this_file)) {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        for n in names_in(&text) {
            in_code.entry(n).or_insert_with(|| f.strip_prefix(&root).unwrap().display().to_string());
        }
    }
    assert!(in_code.len() > 100, "scan found only {} variables; the scan roots are wrong", in_code.len());

    let docs = documented(&root);
    let missing: Vec<String> = in_code.iter().filter(|(n, _)| !docs.contains_key(*n)).map(|(n, f)| format!("{n} (first seen in {f})")).collect();
    assert!(
        missing.is_empty(),
        "INTARSIA_* variables in the code but missing from the table in docs/tuning.md; add a row for each \
         (read-at file:line, default, effect, numerics yes/no/n/a/unverified):\n  {}",
        missing.join("\n  ")
    );
    let stale: Vec<&String> = docs.keys().filter(|n| !in_code.contains_key(*n)).collect();
    assert!(stale.is_empty(), "rows in docs/tuning.md for variables no longer in the code (delete them): {stale:?}");
    for (name, (file, row)) in &docs {
        let text = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("{name}: docs/tuning.md cites `{file}`, which cannot be read ({e}): {row}"));
        assert!(text.contains(name.as_str()), "{name}: docs/tuning.md cites `{file}`, which does not mention it: {row}");
    }
}

#[test]
fn name_scanner_skips_prefixes_and_placeholders() {
    let t = ["INTARSIA", "_"].concat();
    let text = format!("{t}FOO_BAR=1 {t}HIP_* {t}X=0 x{t}NOPE {t}lower {t}A1 {t}B_");
    let got: Vec<String> = names_in(&text).into_iter().collect();
    assert_eq!(got, vec![format!("{t}A1"), format!("{t}FOO_BAR")]);
}

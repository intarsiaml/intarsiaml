#!/usr/bin/env python3
"""memmix: shrink a run's raw files for the repo: usage: reduce.py <raw-dir> <tag>...
serve-<tag>.log -> the non-XML lines only (the malloc_info() blocks are summarised in memstat-<tag>.txt by memstat.py, one row per block); drops the keepalive
files; the per-request jsonl stays. Idempotent (skips a log that has no XML left)."""
import pathlib, subprocess, sys
raw = pathlib.Path(sys.argv[1]); here = pathlib.Path(__file__).parent
for tag in sys.argv[2:]:
    log = raw / f"serve-{tag}.log"
    if not log.exists():
        continue
    text = log.read_text(errors="replace")
    if "<malloc" not in text:
        continue
    t0 = [int(l.split("\t")[0]) for l in open(raw / f"marks-{tag}.tsv") if l.strip().endswith("\tload")][0]
    out = subprocess.run([sys.executable, str(here / "memstat.py"), str(log), str(t0)], capture_output=True, text=True, check=True).stdout
    (raw / f"memstat-{tag}.txt").write_text(out)
    keep = []
    skip = False
    for line in text.splitlines():
        if line.startswith("<malloc"):
            skip = True
        if not skip:
            keep.append(line)
        if line.startswith("</malloc>"):
            skip = False
    log.write_text("\n".join(keep) + "\n")
    print(tag, len(text), "->", sum(len(k) + 1 for k in keep))

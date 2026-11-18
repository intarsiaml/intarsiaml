#!/usr/bin/env python3
"""mgdnvsoak: writes long-context cells long16k.json / long32k.json (one user message: Rust sources of this repo concatenated, then a question) into OUT.
Sizes are in characters (CH16, CH32; about 3.3 chars per token of Rust), deterministic."""
import json, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parents[2]; OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ch = {"long16k": int(sys.argv[2]), "long32k": int(sys.argv[3])}
files = sorted((REPO / "crates/intarsia-core/src").rglob("*.rs"))
blob = ""
for f in files:
    blob += f"\n// ===== {f.relative_to(REPO)} =====\n" + f.read_text()
    if len(blob) > 140000:
        break
for name, n in ch.items():
    q = ("\n\nAbove is part of a Rust inference engine. Write a detailed technical overview of it: the main modules you saw, what each does, "
         "and how a request flows through them. Be thorough and specific, naming functions and types.")
    (OUT / f"{name}.json").write_text(json.dumps({"messages": [{"role": "user", "content": "Source code:\n```rust" + blob[:n] + "\n```" + q}], "max_tokens": 1024}))

#!/usr/bin/env python3
"""scribe: turn a raw decision note into an ADR using a fleet model.

Usage: fleetq run --uses guidance --label "intarsiaml ADR" -- tools/scribe.py "<title>" < note.txt
Writes docs/decisions/NNNN-<slug>.md. Always launch it through fleetq (above)
(--uses guidance) so nobody takes the role down mid-write. Falls back to
writing the raw note verbatim if the fleet model is unreachable.
"""
import json, os, re, subprocess, sys, urllib.request, datetime, pathlib

URL = os.environ.get("SCRIBE_URL", "http://spark-4e77:8000/v1/chat/completions")
MODEL = os.environ.get("SCRIBE_MODEL", "guidance")
ROOT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "decisions"

PROMPT = """You are the decision scribe for IntarsiaML, an LLM inference engine.
Rewrite the raw note below as a concise Architecture Decision Record in Markdown with
exactly these sections: Status, Context, Decision, Alternatives considered, Consequences,
Evidence. Keep every number, path, record id and name from the note exactly. Do not invent
facts, measurements or alternatives that are not in the note; write "none recorded" instead.
Plain, short sentences. No preamble.

Title: {title}
Date: {date}

Raw note:
{note}
"""

def call(title, note):
    body = {"model": MODEL, "temperature": 0.2, "max_tokens": 1800,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": PROMPT.format(
                title=title, date=datetime.date.today().isoformat(), note=note)}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()

def main():
    title, note = sys.argv[1], sys.stdin.read()
    ROOT.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name[:4]) for p in ROOT.glob("[0-9][0-9][0-9][0-9]-*.md")] or [0])
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    try:
        text = call(title, note)
        src = f"UNREVIEWED draft by fleet model `{MODEL}` from the raw note"
    except Exception as e:  # fleet down: keep the raw note, never lose a decision
        text, src = note, f"raw note (scribe model unavailable: {e})"
    out = ROOT / f"{n:04d}-{slug}.md"
    out.write_text(f"# ADR-{n:04d}: {title}\n\n_{src}_\n\n{text}\n\n<details><summary>Raw note</summary>\n\n{note}\n</details>\n")
    print(out)

if __name__ == "__main__":
    main()

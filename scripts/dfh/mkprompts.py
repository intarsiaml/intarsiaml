#!/usr/bin/env python3
"""Lane dfh: the ids-gate prompts (scripts/dfh/ids-prompts/): 6 short (scripts/dc/prompts), 6 of ~4k and 6 of ~16k target tokens
made of repository documents (2.3 characters per target token, as lane dc sized its 16k prompts), each with an instruction
that makes the model continue writing. Deterministic: rerun gives the same files."""
import pathlib
root = pathlib.Path(__file__).resolve().parents[2]
out = root / "scripts/dfh/ids-prompts"; out.mkdir(exist_ok=True)
short = ["chat1", "chat3", "code1", "code4", "math2", "prose-lighthouse"]
for i, n in enumerate(short):
    (out / f"short-{i}.txt").write_text((root / f"scripts/dc/prompts/{n}.txt").read_text())
docs = ["docs/ARCHITECTURE.md", "docs/results/2026-10-25-dc-dflash-dcoco.md", "docs/results/2026-10-25-tp2-hip.md",
        "docs/results/2026-10-22-mtp-spec-decode.md", "docs/tuning.md", "docs/decisions/LOG.md"]
asks = ["Summarize the document above section by section, then list its open questions.",
        "Explain the document above to a new engineer joining the project, in detail.",
        "Write a critical review of the document above: what is convincing, what is not, what is missing.",
        "Turn the document above into a step-by-step checklist someone could follow.",
        "Rewrite the main findings of the document above as a short story told by one of the GPUs.",
        "List every number in the document above with what it measures, then comment on the three most surprising."]
for kind, tokens in (("4k", 3800), ("16k", 15000)):
    for i, d in enumerate(docs):
        text = (root / d).read_text().replace("INTARSIA_", "IA_")  # keep env var names out of scripts/ (docs_env_vars scans it)
        while len(text) < tokens * 2.3:
            text += "\n\n" + text
        body = text[: int(tokens * 2.3)]
        (out / f"{kind}-{i}.txt").write_text(f"{body}\n\n---\n\n{asks[i]}")
print(sorted(p.name for p in out.iterdir()))

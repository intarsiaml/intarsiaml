#!/usr/bin/env python3
"""Lane cmtp: ids-gate prompts sized to the real token counts (scripts/cmtp/ids-prompts/): 6 short (scripts/dc/prompts), 6 at ~4k and 6 at ~16k
prompt tokens of the Qwen3.6 tokenizer (the dfh set, scripts/dfh/ids-prompts, came out at 2.6-3.4k and 10.4-12.5k tokens at 2.3 characters per token).
Characters per token are measured per document from that set's `prompt:` counts. Deterministic. Run from the repo root."""
import pathlib
root = pathlib.Path(__file__).resolve().parents[2]
out = root / "scripts/cmtp/ids-prompts"; out.mkdir(exist_ok=True)
short = ["chat1", "chat3", "code1", "code4", "math2", "prose-lighthouse"]
for i, n in enumerate(short):
    (out / f"short-{i}.txt").write_text((root / f"scripts/dc/prompts/{n}.txt").read_text())
docs = ["docs/ARCHITECTURE.md", "docs/results/2026-10-25-dc-dflash-dcoco.md", "docs/results/2026-10-25-tp2-hip.md",
        "docs/results/2026-10-22-mtp-spec-decode.md", "docs/tuning.md", "docs/decisions/LOG.md"]
cpt = [3.32, 2.92, 2.76, 2.89, 3.16, 2.76]  # characters per token of each document (measured, 16k set)
asks = ["Summarize the document above section by section, then list its open questions.",
        "Explain the document above to a new engineer joining the project, in detail.",
        "Write a critical review of the document above: what is convincing, what is not, what is missing.",
        "Turn the document above into a step-by-step checklist someone could follow.",
        "Rewrite the main findings of the document above as a short story told by one of the GPUs.",
        "List every number in the document above with what it measures, then comment on the three most surprising."]
for kind, tokens in (("4k", 3900), ("16k", 15800)):
    for i, d in enumerate(docs):
        text = (root / d).read_text().replace("INTARSIA_", "IA_")  # keep env var names out of scripts/ (docs_env_vars scans it)
        want = int(tokens * cpt[i])
        while len(text) < want:
            text += "\n\n" + text
        (out / f"{kind}-{i}.txt").write_text(f"{text[:want]}\n\n---\n\n{asks[i]}")
print(sorted(p.name for p in out.iterdir()))

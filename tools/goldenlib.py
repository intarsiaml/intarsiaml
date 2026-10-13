"""Shared pieces of the golden tools (golden.py, golden_vllm.py, golden_topk.py). Stdlib only.

Prompt sets are JSON-lines files, one object per line:

    {"name", "kind": "raw",  "text": "..."}
    {"name", "kind": "raw",  "file": "ppl.txt", "skip_tokens": 0, "max_tokens": 400}   # long prompt
    {"name", "kind": "chat", "messages": [...]}

`file` is relative to tools/prompts/. For file prompts the text is tokenised, `skip_tokens` ids are dropped
and the next `max_tokens` ids are kept; the golden stores ids, so every consumer sees identical tokens.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PROMPT_DIR = HERE / "prompts"
MIN_TOK, MAX_TOK = 8, 4096
FORMAT_FULL = "intarsia-golden-v1"
FORMAT_TOPK = "intarsia-golden-topk-v1"


def chatml(messages, add_generation_prompt=True):
    """Plain ChatML (Qwen family). Fixed text, independent of the model's own template."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def load_prompts(paths):
    """Concatenate prompt-set files (paths: list of str/Path)."""
    out = []
    for p in paths:
        for line in open(p):
            if line.strip():
                out.append(json.loads(line))
    names = [p["name"] for p in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"duplicate prompt names: {sorted(dup)}")
    return out


def prompt_text(p, template="chatml", apply_template=None):
    """Prompt text for a prompt object (file prompts return the whole file text)."""
    if p["kind"] == "raw":
        if "file" in p:
            return (PROMPT_DIR / p["file"]).read_text()
        return p["text"]
    if template == "chatml":
        return chatml(p["messages"])
    return apply_template(p["messages"])


def prompt_ids(p, tokenize, template="chatml", apply_template=None):
    """(ids, text) for a prompt; `tokenize(text) -> list[int]` (with special tokens as the engine would)."""
    text = prompt_text(p, template, apply_template)
    ids = list(tokenize(text))
    if "file" in p:
        skip = int(p.get("skip_tokens", 0))
        ids = ids[skip:skip + int(p["max_tokens"])]
    if not MIN_TOK <= len(ids) <= MAX_TOK:
        raise SystemExit(f"prompt {p['name']} has {len(ids)} tokens, want {MIN_TOK}..{MAX_TOK}")
    return ids, text

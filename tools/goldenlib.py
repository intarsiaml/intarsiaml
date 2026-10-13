"""Shared pieces of the golden tools (golden.py, golden_vllm.py, golden_topk.py). Stdlib only.

Prompt sets are JSON-lines files, one object per line:

    {"name", "kind": "raw",  "text": "..."}
    {"name", "kind": "raw",  "file": "ppl.txt", "skip_tokens": 0, "max_tokens": 400}   # long prompt
    {"name", "kind": "chat", "messages": [...]}

`file` is relative to tools/prompts/. For file prompts the text is tokenised, `skip_tokens` ids are dropped
and the next `max_tokens` ids are kept; the golden stores ids, so every consumer sees identical tokens.

Optional keys (any kind):

    "continue": N              append N tokens of the REFERENCE's own greedy continuation to the prompt, so the
                               last N positions (the teacher-forced decode window of `intarsia accuracy`) are
                               on the model's distribution, not on arbitrary prompt text
    "reuse_after_message": k   (chat) the first k messages plus the assistant header are "turn 1": the golden
                               records that token count as `reuse_split`, where `intarsia accuracy --mode reuse`
                               cuts the conversation for its prefix-reuse check

Chat messages may carry `"content_file": "corpus_docs.txt", "skip_chars": 0, "max_chars": 9000` instead of
`content` (text taken from tools/prompts/ and cut by characters).
"""
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
PROMPT_DIR = HERE / "prompts"
MIN_TOK, MAX_TOK = 8, 16384
FORMAT_FULL = "intarsia-golden-v1"
FORMAT_TOPK = "intarsia-golden-topk-v1"
# metadata text of a long prompt is cut to this many characters (the ids are what the golden is about)
META_TEXT_CHARS = 1500


def chatml(messages, add_generation_prompt=True):
    """Plain ChatML (Qwen family). Fixed text, independent of the model's own template."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def expand_messages(messages):
    """Resolve `content_file` entries to plain `content`."""
    out = []
    for m in messages:
        if "content_file" in m:
            txt = (PROMPT_DIR / m["content_file"]).read_text()
            a = int(m.get("skip_chars", 0))
            txt = txt[a:a + int(m["max_chars"])] if "max_chars" in m else txt[a:]
            m = {"role": m["role"], "content": txt}
        out.append(m)
    return out


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
    msgs = expand_messages(p["messages"])
    if template == "chatml":
        return chatml(msgs)
    return apply_template(msgs)


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


def reuse_split(p, ids, tokenize):
    """Token count of turn 1 of a two-turn chat (`reuse_after_message`), or None. Checked to be a prefix of `ids`."""
    k = p.get("reuse_after_message")
    if k is None:
        return None
    head = tokenize(chatml(expand_messages(p["messages"])[:int(k)]))
    if list(ids[:len(head)]) != list(head):
        raise SystemExit(f"prompt {p['name']}: the tokens of the first {k} messages are not a prefix of the whole prompt "
                         "(tokenisation across the turn boundary differs); pick another boundary")
    return len(head)


def snapshot_hash(path):
    """`snapshots/<hash>` component of a Hugging Face cache path, else None."""
    m = re.search(r"snapshots/([0-9a-f]{7,64})", str(path))
    return m.group(1) if m else None


def meta_prompt(p, name_text, n_total, n_prompt, split):
    text = name_text if len(name_text) <= META_TEXT_CHARS else name_text[:META_TEXT_CHARS]
    e = {"name": p["name"], "kind": p["kind"], "n_tokens": n_total, "n_prompt": n_prompt,
         "continuation": n_total - n_prompt, "text": text}
    if len(name_text) > META_TEXT_CHARS:
        e["text_truncated_from_chars"] = len(name_text)
    if split is not None:
        e["reuse_split"] = split
    return e

#!/usr/bin/env python3
"""Personal prompts for the MTP head fine-tune: the owner's own requests, from Claude Code transcripts and/or a claude.ai export.

Only the PROMPT text is kept (the target model regenerates every response). Output is the chat-message JSONL regen.py and
prompts.py take: {"id", "source", "messages": [{"role": "user", "content": ...}]}. The output holds personal data: write it only to
a git-ignored scratch dir or the training box, never commit it.

  personal_prompts.py --claude-code ~/.claude/projects --out personal-cc.jsonl
  personal_prompts.py --claude-ai ~/Downloads/data-....zip --out personal-ai.jsonl

Filters:
* Claude Code: main-thread user turns typed by the person (no sub-agent sidechains, no meta/command/notification/tool-result
  records). A prompt must stand alone: the session's first prompt, or a later one of >= 200 characters that does not open with a
  reply word ("yes", "ok", "go", "do it", ...). Pasted logs dominate some prompts: anything over --max-chars is dropped.
* claude.ai: every human message that opens a conversation, plus later human messages of >= 200 characters (same opener rule).
* Both: drop prompts that open with a tag, a code fence or a shell prompt, plugin relays (auto-handoff), and prompts whose own words
  (outside pasted blocks and code fences) are under 30 % of the text; drop prompts containing credentials (API keys, tokens, private keys, passwords), replace home paths and the account
  e-mail with placeholders, exact-dedup, and drop prompts equal to an eval prompt (--exclude, default scripts/dfh/cats/*.jsonl).
"""
import argparse, glob, hashlib, json, os, re, sys, zipfile

SECRET = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{10,}|oat01|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|xox[abpr]-|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY|(?i:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*\S{6,}"
)
REPLY_OPENER = re.compile(r"^\s*(yes|yep|yeah|no|nope|ok|okay|go|do it|sure|continue|keep going|proceed|thanks|thank you|great|perfect|"
                          r"that|this|it|those|these|same|again|now|also|and|but|then)\b", re.I)
SKIP_PREFIX = ("<", "```", "Caveat:", "[Request interrupted", "The auto-handoff", "[auto-handoff]", "[fast-", "[Image", "\U000f0142")  # tags, pastes, plugins, shell prompt glyph
PASTE = re.compile(r"<pasted_content[^>]*>.*?</pasted_content[^>]*>|```.*?```", re.S)
SHELL_PASTE = re.compile(r"^(\(\w+\) )?(\S+@\S+:\S*[$#]|PS [A-Z]:\\|[A-Z]:\\\S*>)")  # pasted terminal: user@host:~$, PowerShell, cmd


def scrub(t: str, home: str, email: str) -> str:
    t = t.replace(home, "~")
    if email:
        t = t.replace(email, "user@example.com")
    t = re.sub(r"[A-Z]:\\Users\\[A-Za-z0-9._-]+", "~", t)
    return re.sub(r"/Users/[A-Za-z0-9._-]+", "~", t)


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # tool_result blocks are not the person's words
        if any(isinstance(x, dict) and x.get("type") == "tool_result" for x in content):
            return ""
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text")
    return ""


def keep(t: str, first: bool, a) -> bool:
    s = t.strip()
    if len(s) < a.min_chars or len(s) > a.max_chars or s.startswith(SKIP_PREFIX) or SHELL_PASTE.match(s):
        return False
    own = PASTE.sub("", s).strip()  # the person's own words, without pasted logs or code blocks
    if len(own) < a.min_chars or len(own) < 0.3 * len(s):
        return False
    return first or (len(s) >= 200 and not REPLY_OPENER.match(s))


def from_claude_code(root: str, a):
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(root), "*", "*.jsonl"))):
        first = True
        for line in open(f, errors="ignore"):
            if '"user"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "user" or r.get("isSidechain") or r.get("isMeta") or r.get("toolUseResult") is not None:
                continue
            t = text_of(r.get("message", {}).get("content"))
            if not t.strip():
                continue
            if keep(t, first, a):
                yield "cc", t
            first = False


def from_claude_ai(path: str, a):
    p = os.path.expanduser(path)
    if p.endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            convs = json.loads(z.read(next(n for n in z.namelist() if n.endswith("conversations.json"))))
    else:
        convs = json.load(open(p))
    for c in convs:
        first = True
        for m in c.get("chat_messages", []):
            if m.get("sender") != "human":
                continue
            t = m.get("text") or text_of(m.get("content"))
            if keep(t or "", first, a):
                yield "ai", t
            first = False


def norm(t: str) -> str:
    return " ".join(t.split()).casefold()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude-code", help="Claude Code projects dir (~/.claude/projects)")
    ap.add_argument("--claude-ai", help="claude.ai export: the zip or its conversations.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=None, help="eval prompt JSONL files (default scripts/dfh/cats/*.jsonl)")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--email", default=os.environ.get("PERSONAL_EMAIL", ""), help="address to replace with a placeholder")
    a = ap.parse_args()
    if not (a.claude_code or a.claude_ai):
        ap.error("give --claude-code and/or --claude-ai")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    excl = a.exclude if a.exclude is not None else glob.glob(os.path.join(repo, "scripts/dfh/cats/*.jsonl"))
    banned = {norm(json.loads(l).get("input", "")) for f in excl for l in open(f) if l.strip()}
    home = os.path.expanduser("~")
    srcs = []
    if a.claude_code:
        srcs.append(from_claude_code(a.claude_code, a))
    if a.claude_ai:
        srcs.append(from_claude_ai(a.claude_ai, a))
    seen, n = set(), {"kept": 0, "secret": 0, "dup": 0, "eval": 0}
    with open(a.out, "w") as out:
        for src in srcs:
            for tag, t in src:
                if SECRET.search(t):
                    n["secret"] += 1
                    continue
                t = scrub(t, home, a.email).strip()
                k = norm(t)
                if k in banned:
                    n["eval"] += 1
                    continue
                if k in seen:
                    n["dup"] += 1
                    continue
                seen.add(k)
                pid = f"{tag}-{hashlib.sha1(k.encode()).hexdigest()[:12]}"
                out.write(json.dumps({"id": pid, "source": f"personal-{tag}", "messages": [{"role": "user", "content": t}]}) + "\n")
                n["kept"] += 1
    print(json.dumps(n), file=sys.stderr)


if __name__ == "__main__":
    main()

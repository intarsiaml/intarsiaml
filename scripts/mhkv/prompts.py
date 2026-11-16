#!/usr/bin/env python3
"""Lane mstore: chat message files for the restore measurements. usage: prompts.py OUTDIR
For CTX in 4k, 32k: P (system + a document of about CTX tokens + a first question), Q (P, a fixed assistant reply, a new
question: the next turn, which restores P from the store) and U (an unrelated short chat that replaces the engine's held
sequence, so a restore comes from the store and not from the engine's own prefix reuse)."""
import json, os, sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
docs = open(os.path.join(root, "tools/prompts/corpus_docs.txt")).read()
code = open(os.path.join(root, "tools/prompts/corpus_code.txt")).read()
ppl = open(os.path.join(root, "tools/prompts/ppl.txt")).read()
text = docs + "\n\n" + code
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
SYS = {"role": "system", "content": "You are a careful technical reader. Answer from the material the user gives you."}
A1 = ("The material describes how the project is organised and how its parts are documented: an overview of the goals, "
      "a walk through the main components and how they talk to each other, the configuration knobs with their defaults, "
      "and the procedures for building, testing and measuring a change. It closes with source code that implements part "
      "of what the documents describe. The tone is practical: each section says what to run and what to expect.")
Q2 = ("Now list eight specific details from the material above, one per line. For each, quote the relevant phrase and "
      "explain in two sentences why it matters to someone changing the code.")
for name, chars in (("4k", 15000), ("32k", 118000)):
    p = [SYS, {"role": "user", "content": text[:chars] + "\n\nSummarise the material above in one paragraph."}]
    q = p + [{"role": "assistant", "content": A1}, {"role": "user", "content": Q2}]
    json.dump(p, open(f"{out}/P-{name}.json", "w"))
    json.dump(q, open(f"{out}/Q-{name}.json", "w"))
u = [{"role": "user", "content": ppl + "\n\nWhat is this text about?"}]
json.dump(u, open(f"{out}/U.json", "w"))
print("written", sorted(os.listdir(out)))

#!/usr/bin/env python3
"""Build g3-realistic.jsonl (localeval quality --dataset format: name, category, messages, input) from
tools/prompts/corpus_*.txt, and prompts/<name>.txt (the user message) for the `generate` runs."""
import json, pathlib
R = pathlib.Path("/opt/code/intarsiaml-spec")
code = (R / "tools/prompts/corpus_code.txt").read_text().splitlines()
docs = (R / "tools/prompts/corpus_docs.txt").read_text().splitlines()
def seg(lines, a, b): return "\n".join(lines[a - 1:b])

rows = []
def add(name, cat, prompt): rows.append({"name": name, "category": cat, "input": prompt,
                                          "messages": [{"role": "user", "content": prompt}]})

# ---- grounded: edit this code (5) ----
edits = [
 ("edit-rename-fn", 600, 700, "Rename the function `common_prefix` to `shared_prefix_len` and update every call in this file. Output the complete modified code in one Rust code block and nothing else."),
 ("edit-docs-comments", 227, 330, "Add a `///` doc comment to every function, struct and impl in this file that lacks one. Output the complete modified code in one Rust code block and nothing else."),
 ("edit-field", 20, 120, "Add a field `pub top_k: Option<usize>` (default `None`, documented) to the request struct and to its Default impl. Output the complete modified code in one Rust code block and nothing else."),
 ("edit-error-msgs", 616, 720, "Change every error message string in this function so that it starts with a capital letter and ends with a period. Output the complete modified function in one Rust code block and nothing else."),
 ("edit-refactor-model", 848, 975, "Change `LoadOptions` so every field is wrapped in `Option` with a `None` default, adjusting the Default impl. Output the complete modified code in one Rust code block and nothing else."),
]
for n, a, b, ins in edits:
    add("grounded-" + n, "grounded", f"Here is a Rust source file excerpt.\n\n```rust\n{seg(code, a, b)}\n```\n\n{ins}")

# ---- grounded: summarise this doc while quoting (3) ----
summ = [("summ-overview", 49, 94), ("summ-bench", 368, 438), ("summ-health", 537, 596)]
for n, a, b in summ:
    add("grounded-" + n, "grounded", f"Document:\n\n{seg(docs, a, b)}\n\nSummarise this document section in about 200 words. Quote at least six key sentences or phrases verbatim, in quotation marks, and say what each one means.")

# ---- grounded: extract fields (2) ----
add("grounded-extract-fields", "grounded", f"Document:\n\n{seg(docs, 697, 730)}\n\nExtract every field described in this section into a JSON array of objects with the keys \"name\", \"type\" and \"description\". Copy the descriptions verbatim. Output only the JSON.")
add("grounded-extract-ops", "grounded", f"Document:\n\n{seg(docs, 1039, 1069)}\n\nExtract every kernel or operation named in this section into a JSON array of objects with the keys \"kernel\", \"purpose\" and \"notes\", copying the wording from the document. Output only the JSON.")

# ---- open-ended (10): story 4, explanation 3, brainstorm 3 ----
open_ = [
 ("story-lighthouse", "Write a short story about a lighthouse keeper who discovers that the light has been answering someone."),
 ("story-robot", "Write a short story about a retired delivery robot who takes one last route through a city that has forgotten it."),
 ("story-cartographer", "Write a short story about a cartographer who is hired to map a village that moves every night."),
 ("story-chef", "Write a short story about a chef whose recipes only work when cooked for someone she has wronged."),
 ("explain-kvcache", "Explain how the key-value cache works in transformer language model inference and why it makes decoding faster."),
 ("explain-hashmap", "Explain how a hash map works internally, including collisions and resizing, to someone who has only used arrays."),
 ("explain-tides", "Explain why there are two tides a day, and why spring tides and neap tides differ."),
 ("brainstorm-appnames", "Brainstorm twelve creative names for a note-taking app for hikers, with a one-sentence pitch for each."),
 ("brainstorm-weekend", "Brainstorm ten unusual weekend projects for someone with a soldering iron, a Raspberry Pi and a free Saturday."),
 ("brainstorm-lessons", "Brainstorm eight lesson ideas that teach fractions to ten-year-olds without worksheets."),
]
for n, p in open_:
    add("open-" + n, "open", p)

out = pathlib.Path(__file__).parent
(out / "prompts").mkdir(exist_ok=True)
with open(out / "g3-realistic.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / "prompts" / (r["name"] + ".txt")).write_text(r["input"])
        print(r["name"], len(r["input"]) // 4, "tok~")

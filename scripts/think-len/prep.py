#!/usr/bin/env python3
"""think-len: build the probe inputs (token-id prompts) from the soak's GSM8K thinking runs, using vLLM's /tokenize as the
tokenizer of record (the prompt ids were shown identical to intarsia's in crates/intarsia-models/tests/qwen36_prompt_ids.rs).
usage: prep.py <vllm base url> <raw dir> <soak raw dir>
  needs gsm8k-prompts.json in <raw dir> with `template`, `questions` (>= 200), written from localeval's pinned template."""
import json, sys, urllib.request

base, raw, soak = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"), sys.argv[3].rstrip("/")
P = json.load(open(f"{raw}/gsm8k-prompts.json"))
head, tail = P["template"].split("{question}", 1)


def post(path, body):
    r = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))


def chat_ids(q, thinking):
    msgs = [{"role": "user", "content": head + q + tail}]
    return post("/tokenize", {"model": "guidance", "messages": msgs, "add_generation_prompt": True,
                              "chat_template_kwargs": {"enable_thinking": thinking}})["tokens"]


def text_ids(text):
    return post("/tokenize", {"model": "guidance", "prompt": text, "add_special_tokens": False})["tokens"]


def replies(run):
    d = json.load(open(f"{soak}/{run}/result.json"))["benchmarks"][0]
    return {x["id"]: x for x in d["replies"]}


V = replies("20261113-100540-adhoc-soak-V-think-gsm8k")
B = replies("20261113-105601-adhoc-soak-Bc-gsm8k")
items = []
tfbc = []  # the same teacher-forced end-of-thinking probe, built from intarsiaml's (Bc) traces
for i, q in enumerate(P["questions"]):
    qid = f"gsm8k-{i:05d}"
    if i < 85:
        on = chat_ids(q, True)
        items.append({"id": qid, "kind": "on", "ids": on, "max_tokens": 96})
        tr = V[qid]["reasoning_content"]
        items.append({"id": qid, "kind": "tf", "ids": on + text_ids(tr + "\n</think>\n\n"), "max_tokens": 1})
        tfbc.append({"id": qid, "kind": "tf", "ids": on + text_ids(B[qid]["reasoning_content"] + "\n</think>\n\n"), "max_tokens": 1})
    items.append({"id": qid, "kind": "off", "ids": chat_ids(q, False), "max_tokens": 1})
json.dump(items, open(f"{raw}/probe-inputs.json", "w"))
json.dump(tfbc, open(f"{raw}/probe-inputs-tf-bc.json", "w"))
print(len(items), "items,", len(tfbc), "tf-bc items")

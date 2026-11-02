#!/usr/bin/env python3
"""Multi-turn few-shot eval: each request extends the previous one (a chat-style session), gold answers in the history so both
arms send identical prompts. usage: mt.py PORT OUT.json mmlu|gsm8k"""
import sys, json, re, urllib.request, time
from datasets import load_dataset
from deepeval.benchmarks.mmlu.template import MMLUTemplate
from deepeval.benchmarks.mmlu.task import MMLUTask
port, out, which = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{port}/v1/completions"
def call(prompt, n, stop=None):
    body = {"model": "intarsia-metal", "prompt": prompt, "max_tokens": n, "temperature": 0}
    if stop: body["stop"] = stop
    r = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f: j = json.load(f)
    u = j["usage"]
    return j["choices"][0]["text"], u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens")
res = []
if which == "mmlu":
    for task in MMLUTask:
        d = load_dataset("cais/mmlu", task.value)
        shots = list(d["dev"])[:5]; test = list(d["test"])[:25]
        prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(MMLUTemplate.format_subject(task.value))
        for s in shots: prompt += MMLUTemplate.format_question(s)
        for i, it in enumerate(test):
            prompt += MMLUTemplate.format_question(it, include_answer=False)   # ends with "Answer:"
            txt, pt, ct = call(prompt, 3)
            m = re.search(r"[ABCD]", txt); gold = "ABCD"[it["answer"]]
            res.append({"id": f"{task.value}:{i}", "pred": m.group(0) if m else None, "gold": gold, "ok": bool(m and m.group(0) == gold), "prompt_tokens": pt, "cached": ct})
            prompt += " {}\n\n".format(gold)
else:
    g = load_dataset("openai/gsm8k", "main", split="test"); g = list(g)[:200]
    def clean(a): return re.sub(r"<<.*?>>", "", a)
    num = re.compile(r"-?\d[\d,]*\.?\d*")
    def last_num(t):
        m = re.search(r"####\s*(-?[\d,\.]+)", t)
        if m: s = m.group(1)
        else:
            ms = num.findall(t); s = ms[-1] if ms else None
        return None if s is None else s.replace(",", "").rstrip(".")
    for c in range(0, 200, 20):
        prompt = ""
        for i, it in enumerate(g[c:c + 20]):
            prompt += f"Question: {it['question']}\nAnswer: <think>\n\n</think>\n\n"
            txt, pt, ct = call(prompt, 400, stop=["\n\nQuestion:"])
            gold = it["answer"].split("####")[-1].strip().replace(",", "")
            p = last_num(txt)
            try: ok = p is not None and abs(float(p) - float(gold)) < 1e-6
            except ValueError: ok = False
            res.append({"id": f"{c+i}", "pred": p, "gold": gold, "ok": ok, "prompt_tokens": pt, "cached": ct, "text": txt})
            prompt += clean(it["answer"]).replace("####", "The answer is") + "\n\n"
json.dump(res, open(out, "w"))
print(which, "n", len(res), "acc", sum(r["ok"] for r in res) / len(res))

import json, sys
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from probe_vllm import prompt, post
chat = {r["i"]: r for r in map(json.loads, open(sys.argv[1]))}
same = 0; diff = []
for i in range(200):
    tk = post("http://spark-4e77:8000/tokenize", {"model": "guidance", "messages": [{"role": "user", "content": prompt(i)}],
              "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True})["tokens"]
    c = post("http://spark-4e77:8091/v1/completions", {"model": "intarsia-35b", "prompt": tk, "max_tokens": 1024, "temperature": 0})
    t = c["choices"][0]["text"].strip()
    ok = t == chat[i]["text"] and c["usage"]["prompt_tokens"] == chat[i]["pt"]
    same += ok
    if not ok: diff.append((i, t[:40], chat[i]["text"][:40]))
print("intarsia /v1/completions on vLLM's ids == intarsia chat reply:", same, "/200", diff[:10])

import json, sys, math
sys.path.insert(0, sys.argv[0].rsplit("/",1)[0])
from probe_vllm import prompt, post
base = "http://spark-4e77:8000"
for i in map(int, sys.argv[1].split(",")):
    for extra in [{}, {"stream": False}]:
        body = {"model": "guidance", "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": 1024, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 5, "return_token_ids": True}
        c = post(base + "/v1/chat/completions", body)
        ch = c["choices"][0]
        lp = ch["logprobs"]["content"][0]
        print(i, repr(ch["message"]["content"][:40]), "pt", c["usage"]["prompt_tokens"], "first", repr(lp["token"]), " ".join(f"{t['token']!r}:{math.exp(t['logprob']):.3f}" for t in lp["top_logprobs"]))
        print("   prompt_ids tail", (c.get("prompt_token_ids") or ch.get("prompt_token_ids") or [])[-12:])
        break

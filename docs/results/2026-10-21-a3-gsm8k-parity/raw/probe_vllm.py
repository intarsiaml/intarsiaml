import json, sys, urllib.request, math
from localeval.quality import make_benchmark
from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
b = make_benchmark("gsm8k", None, None, None); G = b.load_benchmark_dataset()
def prompt(i):
    return GSM8KTemplate.generate_output(train_set=b.shots_dataset, input=G[i].input, n_shots=b.n_shots, enable_cot=b.enable_cot) + f"\n\n{b.confinement_instructions}"
def render(i):
    return f"<|im_start|>user\n{prompt(i)}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def post(url, body):
    r = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
if __name__ == "__main__":
    base = "http://spark-4e77:8000"
    for i in map(int, sys.argv[1].split(",")):
        tk = post(base + "/tokenize", {"model": "guidance", "messages": [{"role": "user", "content": prompt(i)}], "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True})
        tk2 = post(base + "/tokenize", {"model": "guidance", "prompt": render(i), "add_special_tokens": False})
        same = tk["tokens"] == tk2["tokens"]
        c = post(base + "/v1/completions", {"model": "guidance", "prompt": tk["tokens"], "max_tokens": 1, "temperature": 0, "logprobs": 8})
        top = c["choices"][0]["logprobs"]["top_logprobs"][0]
        s = sorted(top.items(), key=lambda x: -x[1])
        print(i, "render==template" if same else "RENDER DIFF", len(tk["tokens"]), " ".join(f"{k!r}:{math.exp(v):.3f}" for k, v in s))

#!/usr/bin/env python3
"""Chat-form perplexity of a reference on tools/prompts/ppl_chat.json: the number to compare with `intarsia accuracy --quick`
under `[accuracy] ppl_form = "chat"`. The conversation goes through the model's own chat template (HF tokenizer). For every
assistant message i the sequence is the prompt the model is served with (messages[:i] + generation prompt) followed by the
tokens the template renders for the turn itself (content and turn-close); only the latter are scored. This is chat_ppl_seqs
in crates/intarsia-cli/src/cmd/accuracy.rs: the turn starts where the generation prompt and the rendering of messages[:i+1]
stop agreeing (Gemma-4's generation prompt ends in an empty thought channel that its finished-turn rendering omits). Prints
the number of scored tokens, which must equal the engine's.
usage: chat_ppl.py hf MODEL_DIR [--device cuda]            (HF transformers, tools/gemma4_nvfp4_golden.py loader)
       chat_ppl.py vllm BASE_URL MODEL MODEL_DIR            (prompt_logprobs of a running vLLM server)
"""
import json, math, pathlib, sys, urllib.request
from transformers import AutoTokenizer

mode = sys.argv[1]
tokdir = sys.argv[-1] if mode == "vllm" else sys.argv[2]
tok = AutoTokenizer.from_pretrained(tokdir)
msgs = json.load(open(pathlib.Path(__file__).resolve().parent / "prompts" / "ppl_chat.json"))


def render(m, gen):
    return tok(tok.apply_chat_template(m, add_generation_prompt=gen, tokenize=False), add_special_tokens=False).input_ids


seqs = []  # (ids, index of the first scored token)
for i, m in enumerate(msgs):
    if m["role"] != "assistant":
        continue
    prompt, done = render(msgs[:i], True), render(msgs[: i + 1], False)
    n = 0
    while n < min(len(prompt), len(done)) and prompt[n] == done[n]:
        n += 1
    assert 0 < n < len(done) and n + 8 >= len(prompt), f"turn {i}: generation prompt and turn share only {n} tokens"
    seqs.append((prompt + done[n:], len(prompt)))

nll = []
if mode == "hf":
    import torch
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import gemma4_nvfp4_golden as g
    g.DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    model = g.build_model(sys.argv[2])
    for ids, first in seqs:
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids]).to(g.DEV)).logits[0].float(), -1)
        nll += [-lp[t - 1, ids[t]].item() for t in range(first, len(ids))]
else:
    base, name = sys.argv[2], sys.argv[3]
    for ids, first in seqs:
        req = urllib.request.Request(base.rsplit("/v1", 1)[0] + "/v1/completions", json.dumps({"model": name, "prompt": ids, "max_tokens": 1, "prompt_logprobs": 1, "temperature": 0}).encode(), {"Content-Type": "application/json"})
        pl = json.load(urllib.request.urlopen(req))["choices"][0]["prompt_logprobs"]
        nll += [-pl[t][str(ids[t])]["logprob"] for t in range(first, len(ids))]
print(f"{mode} chat ppl over {len(nll)} assistant tokens ({len(seqs)} turns): {math.exp(sum(nll) / len(nll)):.3f}")

import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
d = sys.argv[1]
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16)
P = json.load(open('raw/nab2/mmlu_prompts.json'))
res = []
for p in P:
    ids = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], add_generation_prompt=True, return_tensors='pt', return_dict=True)
    with torch.no_grad():
        o = m.generate(**ids, max_new_tokens=6, do_sample=False)
    res.append(tok.decode(o[0, ids['input_ids'].shape[1]:], skip_special_tokens=True))
json.dump(res, open('raw/nab2/mmlu_hf_out.json', 'w'))
print('done', len(res))

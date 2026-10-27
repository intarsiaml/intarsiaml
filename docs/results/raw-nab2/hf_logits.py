import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
d = sys.argv[1]; prompt = sys.argv[2]; n = int(sys.argv[3])
dtype = torch.float32 if len(sys.argv) > 4 and sys.argv[4] == "f32" else torch.bfloat16
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=dtype)
ids = tok(prompt, return_tensors="pt").input_ids
print("ids", ids[0].tolist())
with torch.no_grad():
    lg = m(ids).logits[0, -1].float()
v, i = lg.topk(5)
print("top5", [(int(a), round(float(b), 3)) for a, b in zip(i, v)])
out = m.generate(ids, max_new_tokens=n, do_sample=False)
print("gen", tok.decode(out[0, ids.shape[1]:]))

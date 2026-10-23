import os, sys, time, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
d = os.path.expanduser("~/g4-trunc/gemma4-2L-bf16")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(d)
model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16, output_loading_info=False)
model.eval()
print("loaded", type(model).__name__, f"{time.time()-t0:.0f}s", flush=True)
# loading info
m2, info = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16, output_loading_info=True) if os.environ.get("INFO") else (None, None)
if info: print("LOADING INFO", {k: v[:10] for k, v in info.items()})
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
for i, l in enumerate(lm.layers):
    a = l.self_attn
    print(i, "sliding" if a.is_sliding else "full", "head_dim", a.head_dim, "v_proj", a.v_proj is not None)
text = "The quick brown fox jumps over the lazy dog. Count: 1, 2, 3, 4"
ids = tok(text, return_tensors="pt").input_ids[:, :12]
print("ids", ids.shape, ids.tolist())
caps = {}
def mk(n):
    def h(mod, a, out):
        caps[n] = (out[0] if isinstance(out, tuple) else out).detach().float()
    return h
hs = [l.register_forward_hook(mk(f"layer{i}")) for i, l in enumerate(lm.layers)]
hs.append(lm.embed_tokens.register_forward_hook(mk("embed")))
hs.append(lm.norm.register_forward_hook(mk("final_norm")))
with torch.no_grad():
    lg = model(input_ids=ids, use_cache=False).logits[0].float()
print("logits", tuple(lg.shape), "max|logit|", lg.abs().max().item(), "finite", bool(torch.isfinite(lg).all()))
print("argmax last", lg[-1].argmax().item(), "top5", lg[-1].topk(5).values.tolist())
for k, v in caps.items():
    print(f"{k}: shape={tuple(v.shape)} rms={v.pow(2).mean().sqrt().item():.4f} max|x|={v.abs().max().item():.3f} finite={bool(torch.isfinite(v).all())}")
torch.save({k: v for k, v in caps.items()}, os.path.expanduser("~/g4-trunc/hf-hidden-states.pt"))

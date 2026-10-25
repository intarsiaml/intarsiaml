"""ChatML scored-row mask of the accuracy gate (crates/intarsia-cli/src/cmd/accuracy/turns.rs), in Python. MODEL_DIR holds tokenizer.json."""
import os
from tokenizers import Tokenizer
tk=Tokenizer.from_file(os.environ.get('MODEL_DIR','/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4')+'/tokenizer.json')
enc=lambda s: tk.encode(s,add_special_tokens=False).ids
op=enc('<|im_start|>')[0]; cl=enc('<|im_end|>')[0]
hid=[enc('<|im_start|>'+r)[1:] for r in ('system','user')]
def scored(ids):
    n=len(ids); s=[True]*n; t=0
    while t<n:
        if ids[t]!=op: t+=1; continue
        after=ids[t+1:]
        if not any(after[:len(h)]==h for h in hid): t+=1; continue
        try: b=t+1+ids[t+1:].index(cl)
        except ValueError: b=n
        for j in range(t,min(b,n)): s[j]=False
        t=b+1
    return s

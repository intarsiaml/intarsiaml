import json
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4/tokenizer.json")
T = json.load(open(".scratch/logs/gates/prompts-text.json"))
LEN = json.load(open(".scratch/logs/exp3/len.json")); LEN.update(json.load(open(".scratch/logs/exp4/nopeel-len.json")))
A = json.load(open(".scratch/logs/sweep/all.json")); LEN["772"] = {"solo": A["q4-solo"], "behind": A["q4-behind"]}
S = {}
for k in ["258", "260", "262", "772", "1028"]:
    p = tok.encode(T[k], add_special_tokens=False).ids; assert len(p) == int(k), k
    S[k] = p + LEN[k]["solo"]["ids"]
json.dump(S, open(".scratch/logs/oracle/hf-seqs.json", "w")); print({k: len(v) for k, v in S.items()})

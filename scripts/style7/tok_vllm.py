import json,urllib.request,sys
P=json.load(open('scratch/prompts200.json'))
def post(path, body):
    r=urllib.request.Request('http://spark-fa00:8000'+path, json.dumps(body).encode(), {'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(r))
out=[]
for p in P:
    r=post('/tokenize',{"model":"guidance","messages":[{"role":"user","content":p['prompt']}],"chat_template_kwargs":{"enable_thinking":False},"add_generation_prompt":True})
    out.append({"id":p['id'],"tokens":r['tokens']})
json.dump(out,open('docs/results/2026-11-20-style7/raw/vllm-tokenize-pinned200-think-off.json','w'))
json.dump([{"id":p['id'],"prompt":p['prompt']} for p in P],open('docs/results/2026-11-20-style7/raw/gsm8k-pinned200-prompts.json','w'))
print(len(out), sum(len(o['tokens']) for o in out)/len(out))

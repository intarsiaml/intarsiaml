"""Re-query vLLM prompt_logprobs for stored golden ids (no re-tokenisation) and save as a top-20 golden-format file.
usage: vllm_requery.py GOLDEN OUT.safetensors PROMPT_NAME [--salt S]
"""
import sys, json, time, urllib.request, argparse
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
ap = argparse.ArgumentParser()
ap.add_argument('golden'); ap.add_argument('out'); ap.add_argument('names')
ap.add_argument('--salt', default=None)
ap.add_argument('--base', default='http://spark-4e77:8000/v1')
ap.add_argument('--model', default='guidance')
a = ap.parse_args()
f = safe_open(a.golden, 'np')
md = dict(f.metadata()); prompts = json.loads(md['prompts'])
idx = {p['name']: i for i, p in enumerate(prompts)}
tensors, metas = {}, []
for j, name in enumerate(a.names.split(',')):
    i = idx[name]
    ids = f.get_tensor(f'ids_{i}').tolist()
    n = len(ids)
    body = {'model': a.model, 'prompt': ids + [ids[-1]], 'max_tokens': 1, 'temperature': 0, 'prompt_logprobs': 20}
    if a.salt:
        body['cache_salt'] = a.salt
    t0 = time.time()
    req = urllib.request.Request(a.base + '/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    resp = json.load(urllib.request.urlopen(req, timeout=1800))
    pl = resp['choices'][0]['prompt_logprobs']
    assert len(pl) == n + 1 and pl[0] is None, len(pl)
    tid = np.full((n, 20), -1, np.int32); tlp = np.full((n, 20), -1e4, np.float32)
    for t in range(n):
        items = sorted(((int(k), v['logprob'], v.get('rank', 0)) for k, v in pl[t + 1].items()), key=lambda x: -x[1])
        items = [x for x in items if x[2] <= 20][:20]
        for q, (k, lp, _) in enumerate(items):
            tid[t, q], tlp[t, q] = k, lp
    tensors[f'ids_{j}'] = np.array(ids, np.int32); tensors[f'topk_ids_{j}'] = tid; tensors[f'topk_logprobs_{j}'] = tlp
    metas.append(prompts[i])
    print(f'{name}: {n} rows in {time.time() - t0:.1f}s, usage {resp.get("usage")}', flush=True)
md['prompts'] = json.dumps(metas)
md['note_requery'] = f'M3 re-query of stored ids, salt={a.salt}, {time.strftime("%Y-%m-%d %H:%M:%S")}'
save_file(tensors, a.out, metadata=md)
print('wrote', a.out)

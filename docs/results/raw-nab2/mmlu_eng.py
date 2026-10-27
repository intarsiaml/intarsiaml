import json, urllib.request
W = '/opt/code/intarsiaml/.worktrees/nab2/raw/nab2/'
P = json.load(open(W + 'mmlu_prompts.json'))
res = []
for p in P:
    body = json.dumps({'model': 'lfm', 'messages': [{'role': 'user', 'content': p['prompt']}], 'max_tokens': 6, 'temperature': 0}).encode()
    r = urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8124/v1/chat/completions', body, {'Content-Type': 'application/json'}))
    res.append(json.load(r)['choices'][0]['message']['content'])
json.dump(res, open(W + 'mmlu_eng_out.json', 'w'))
print('done', len(res))

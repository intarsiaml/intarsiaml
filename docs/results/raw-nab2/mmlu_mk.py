import glob, json, pandas as pd
base = glob.glob('/Users/josef/.cache/huggingface/hub/datasets--cais--mmlu/snapshots/*')[0]
out = []
for d in sorted(glob.glob(base + '/*/')):
    f = glob.glob(d + 'test-*.parquet')
    if not f:
        continue
    df = pd.read_parquet(f[0]).head(2)
    for _, r in df.iterrows():
        ch = list(r['choices'])
        q = r['question'] + '\n' + '\n'.join(f'{l}. {c}' for l, c in zip('ABCD', ch)) + '\nAnswer with only the letter of the correct option.'
        out.append({'prompt': q, 'gold': 'ABCD'[int(r['answer'])]})
json.dump(out, open('raw/nab2/mmlu_prompts.json', 'w'))
print(len(out))

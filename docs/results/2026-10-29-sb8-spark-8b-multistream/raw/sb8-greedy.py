import json, sys, threading
sys.path.insert(0, '/opt/code/intarsiaml/.worktrees/sb8/scripts/htk')
import equal
url, ns, reps = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], int(sys.argv[3])
solo = json.load(open('/opt/code/intarsiaml/.worktrees/sb8/docs/results/2026-10-29-sb8-spark-8b-multistream/raw/ident/solo.json'))['greedy/n16']
for n in ns:
    for r in range(reps):
        got = [None]*n
        def one(i):
            body = {"model": "intarsia", "prompt": equal.PROMPTS[i % 16], "max_tokens": 64, "ignore_eos": True, "temperature": 0.0, "top_p": 1.0, "seed": 100+i, "return_token_ids": True}
            got[i] = equal.post(url, body)["choices"][0]["token_ids"]
        ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        [t.start() for t in ths]; [t.join() for t in ths]
        bad = []
        for i in range(n):
            if got[i] != solo[i]['ids']:
                d = next(j for j in range(64) if got[i][j] != solo[i]['ids'][j]); bad.append((i, d))
        print(f"n={n} rep={r}: {'IDENTICAL' if not bad else 'DIFFER (stream, first diff pos) ' + str(bad)}", flush=True)

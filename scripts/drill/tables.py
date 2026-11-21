#!/usr/bin/env python3
"""drill: turn raw/ into the doc's tables. usage: tables.py RAW_DIR  (stdout: markdown)"""
import json, pathlib, re, sys, glob, datetime, statistics
R = pathlib.Path(sys.argv[1])
def kv(line):
    return dict(re.findall(r'(\w+)=(\S+)', line))
def stages():
    return sorted({p.name[4:-4] for p in R.glob('ctl-S*.log')}, key=lambda s: int(s[1:]))
def offsets():
    d = {}
    for l in (R / 'clock-offset.txt').read_text().splitlines():
        m = re.match(r'(\S+) offset_spark2_minus_mac_s (\S+)', l)
        if m: d[m.group(1)] = float(m.group(2))
    return d
OFF = offsets()
def iso(s): return datetime.datetime.fromisoformat(s).timestamp()
base = {}
for l in (R / 'ctl-snap.log').read_text().splitlines():
    if l.startswith('SNAP label=before-all'): base = kv(l)
def snaps(tag):
    out = {}
    for l in (R / f'ctl-{tag}.log').read_text().splitlines():
        if l.startswith('SNAP'):
            d = kv(l); out[d['label']] = d
    return out
rows_start, rows_kill = [], []
for tag in stages():
    L = (R / f'ctl-{tag}.log').read_text().splitlines()
    sn = snaps(tag)
    st = next((kv(l) for l in L if l.startswith(f'RESULT start {tag} ')), None)
    ft = None
    if st:
        m = re.search(r'first_token_json=(\{.*\})', next(l for l in L if l.startswith(f'RESULT start {tag} ')))
        ft = json.loads(m.group(1)) if m else None
    rows_start.append((tag, st, ft, sn.get(f'pre-{tag}'), sn.get(f'up-{tag}')))
    rows_kill.append((tag, L, sn))
print('### Start (launch to healthy), per stage\n')
print('| stage | launch to listening s | launch to /health 200 s | /health to first token (client of the controller) s | first content token s | used MB before launch | used MB up |')
print('|---|---|---|---|---|---|---|')
for tag, st, ft, pre, up in rows_start:
    if not st: print(f'| {tag} | (restarted by the supervisor, see kill table) | | | | | |'); continue
    print(f"| {tag} | {st['launch_to_listen_s']} | {st['launch_to_health_s']} | {st['health_to_first_token_s']} | {ft and ft['first_content_s']} | {pre and pre['used_mb']} | {up and up['used_mb']} |")
print('\n### Kill (signal to gone), per stage; memory against the box before any launch (`before-all`: used %s MB, available %s MB)\n' % (base.get('used_mb'), base.get('avail_mb')))
print('| stage | signal | supervisor gone after s | child gone after s | port 8191 free after s | used MB +0 / +3 s | available MB +3 s | GPU apps left (+3 s) | our processes left |')
print('|---|---|---|---|---|---|---|---|---|')
for tag, L, sn in rows_kill:
    sig = next((l for l in L if l.startswith('SIGNAL')), '')
    sup = next((kv(l) for l in L if l.startswith(f'RESULT') and ' pid=' in l and 'exit_after_s' in l), {})
    ex = [kv(l) for l in L if l.startswith('RESULT') and 'exit_after_s' in l]
    spid = re.search(r'spid=(\d+) cpid=(\d+)', sig)
    byp = {e['pid']: e['exit_after_s'] for e in ex}
    port = next((kv(l).get('port_freed_after_s') for l in L if 'port_freed_after_s' in l), '')
    s0, s3 = sn.get(f't0+0-{tag}', {}), sn.get(f't0+3-{tag}', {})
    left = next((re.search(r'ours=\[(.*)\]', l).group(1).strip() for l in L if 'leftover ours' in l), '')
    tgt = re.search(r'to \[(.*?)\]', sig).group(1) if sig else ''
    print(f"| {tag} | {sig.split(' ')[1] if sig else ''} to [{tgt}] | {byp.get(spid.group(1)) if spid else ''} | {byp.get(spid.group(2)) if spid else ''} | {port} | {s0.get('used_mb')} / {s3.get('used_mb')} | {s3.get('avail_mb')} | {s3.get('gpu_apps')} | {left or 'none'} |")
print('\n### Supervisor restart of a killed child (stage S5)\n')
for tag, L, sn in rows_kill:
    for l in L:
        if 'child restarted' in l or 'kill_to_health_s' in l or 'NOT restarted' in l: print('-', l)
print('\n### Clients: load requests and the consumer around each kill (spark2 times converted with the clock offset, error 0.2 s)\n')
print('| stage | signal | load requests in flight at the signal: outcome | load requests started after it | consumer: failed calls | consumer attempts per failed call / median duration of a failed call s | consumer first success after signal s | consumer calls in flight at the signal |')
print('|---|---|---|---|---|---|---|---|')
for tag, L, sn in rows_kill:
    sig = next((l for l in L if l.startswith('SIGNAL')), None)
    if not sig: continue
    T = float(re.search(r' t=([\d.]+)', sig).group(1)) - OFF.get('before-kill-' + tag, 0.15)
    rows = []
    for f in glob.glob(str(R / 'load' / f'L{tag}-*.jsonl')):
        for l in open(f): rows.append(json.loads(l))
    infl, after = [], 0
    for r in rows:
        s = iso(r['t_start']); e = s + (r['end_s'] - r['start_s'])
        if s <= T < e: infl.append(f"{'ok ' + str(r['finish_reason']) if r['ok'] else 'ERR ' + str(r['error'])[:40]} ({r['completion_tokens']} tok, {round(T-s,1)} s in)")
        elif s >= T: after += 1
    cons = [json.loads(l) for l in open(R / f'consumer-{tag}.jsonl')] if (R / f'consumer-{tag}.jsonl').exists() else []
    calls = [c for c in cons if c['kind'] in ('short', 'long')]
    failed = [c for c in calls if c['error'] and c['t1'] >= T]
    firstok = min((c['t1'] for c in calls if not c['error'] and c['t0'] >= T), default=None)
    longif = [f"{c['kind']}: {'ok ' + str(c['finish']) if not c['error'] else 'ERR ' + c['error'][:30]} ({c['chunks']} chunks, {c['attempts']} attempts)" for c in calls if c['t0'] <= T < c['t1']]
    fdur = round(statistics.median([c['t1'] - c['t0'] for c in failed if c['t0'] >= T]), 2) if any(c['t0'] >= T for c in failed) else None
    print(f"| {tag} | {sig.split(' ')[1]} | {'; '.join(infl) or 'none'} | {after} | {len(failed)} | {sorted(set(c['attempts'] for c in failed))} / {fdur} | {None if firstok is None else round(firstok - T, 1)} | {'; '.join(longif) or 'none'} |")
print('\n### Byte identity of the 20 greedy replies (cmp.txt)\n')
print('```'); print((R / 'cmp.txt').read_text().strip()); print('```')

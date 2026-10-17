import sys,re
for f in sys.argv[1:]:
    t=open(f).read()
    def g(n): 
        m=re.search(r'^'+n+r' ([0-9.]+)$',t,re.M); return float(m.group(1)) if m else None
    tot=g('intarsia_requests_total'); fin={m.group(1):float(m.group(2)) for m in re.finditer(r'^intarsia_requests_finished_total\{reason="([^"]+)"\} ([0-9.]+)$',t,re.M)}
    print(f.split('/')[-1], 'total',tot,'finished',fin,'sum',sum(fin.values()),'running',g('intarsia_requests_running'),'waiting',g('intarsia_requests_waiting'),'rejected',g('intarsia_requests_rejected_total'),'OK' if tot==sum(fin.values()) else 'MISMATCH')

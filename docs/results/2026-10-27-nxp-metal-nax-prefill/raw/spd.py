import re,glob,statistics as st,sys
d=sys.argv[1]; res={}
for f in sorted(glob.glob(d+'/*-localeval.out')):
    tag=f.split('/')[-1].replace('-localeval.out','')
    for m in re.finditer(r'(\d+) tok rep (\d)/3: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s decode=([\d.]+)',open(f).read()):
        res.setdefault(tag,{}).setdefault(int(m[1]),[]).append((int(m[5]),float(m[6])))
for tag,v in res.items():
    print(tag,{k:(round(st.median(a for a,b in x)),round(st.median(b for a,b in x),1)) for k,x in v.items()})

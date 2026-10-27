import sys,re,glob
# flag arms where any non-arm process exceeded 300% CPU in a 4 s sample
for f in sorted(glob.glob(sys.argv[1])):
    bad=[];n=0
    for line in open(f):
        n+=1
        t=line.split(' ',1)
        for m in re.finditer(r'(\d+) ([\d.]+) (/[^;]+)',line):
            pid,cpu,cmd=m.groups()
            if float(cpu)>300 and 'intarsia-' not in cmd: bad.append((t[0],cpu,cmd.split('/')[-1][:40]))
    print(f.split('/')[-1],n,'samples','CONTAMINATED %d: %s'%(len(bad),bad[:1]) if bad else 'clean')

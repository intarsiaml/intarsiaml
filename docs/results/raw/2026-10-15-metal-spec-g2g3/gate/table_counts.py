#!/usr/bin/env python3
import re,glob,sys,collections
cases=["open-256","grounded-256","golden-docs-256","golden-code-256","4k-256","short-256"]
cfgs=sorted({re.match(r"(\d+-.*?)-(open|grounded|golden|4k|short)",f).group(1) for f in glob.glob("*.err")},key=lambda x:int(x.split('-')[0]))
print("| cfg | "+" | ".join(cases)+" |"); print("|---|"+"---|"*len(cases))
for c in cfgs:
    row=[]
    for k in cases:
        t=open(f"{c}-{k}.err").read()
        m=re.search(r"decode \d+ ms \(([\d.]+) tok/s\)",t); sp=float(m[1])
        tr=re.findall(r"spec: token \d+ pos \d+ drafted (\d+) accepted (\d+)",t)
        p=len(tr); a=sum(int(x[1]) for x in tr)
        row.append(f"p{p} a{a}")
    print(f"| {c} | "+" | ".join(row)+" |")

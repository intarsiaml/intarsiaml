import json,re,glob,statistics as st,sys
from datetime import datetime
H=sys.argv[1]
mm=[]
for l in open(f"{H}/macmon.jsonl"):
    try:d=json.loads(l)
    except: continue
    mm.append((datetime.fromisoformat(d["timestamp"]).timestamp(),d["gpu_usage"][0],d["gpu_power"],d["gpu_usage"][1]))
mm.sort()
for f in sorted(glob.glob(f"{H}/passes/clk-*-localeval.ts")):
    lab=re.search(r"(clk-.*)-localeval",f)[1]
    if "safe" not in lab and "cand" not in lab: continue
    for line in open(f):
        ts,_,body=line.partition("\t")
        m=re.search(r"\]\s+(\d+) tok rep (\d)/3: prompt=(\d+) ttft=([\d.]+)s prefill=\S+ tok/s decode=([\d.]+) tok/s.*out=(\d+)",body)
        if not m or m[1]!="4096": continue
        tend=float(ts); ttft=float(m[4]); out=int(m[6]); dec=float(m[5])
        pb=tend-out/dec; pa=pb-ttft
        s=[x for x in mm if pa<x[0]<=pb+0.1]
        mhz=[x[1] for x in s]
        print(f"{lab:12s} rep{m[2]} prompt={m[3]} ttft={ttft:.2f} mean={st.mean(mhz):.0f} min={min(mhz)} W={st.mean(x[2] for x in s):.0f} util={st.mean(x[3] for x in s)*100:.0f}% series={[x[1] for x in s]}")

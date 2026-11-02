#!/usr/bin/env python3
"""One streamed completion; prints JSON {prompt, ttft_s, decode_s, chunks, text_head}."""
import json,sys,time,urllib.request
api,key,model,pfile,maxtok=sys.argv[1:6]
body=json.dumps({"model":model,"prompt":open(pfile).read(),"max_tokens":int(maxtok),"temperature":0,"stream":True}).encode()
req=urllib.request.Request(api+"/v1/completions",data=body,headers={"Authorization":"Bearer "+key,"content-type":"application/json"})
t0=time.time(); tf=None; tl=None; n=0; text=""
with urllib.request.urlopen(req,timeout=300) as r:
    for line in r:
        if not line.startswith(b"data:"): continue
        d=line[5:].strip()
        if d==b"[DONE]": break
        j=json.loads(d); ch=j["choices"][0].get("text","")
        if ch:
            now=time.time()
            if tf is None: tf=now
            tl=now; n+=1; text+=ch
print(json.dumps({"prompt":pfile,"ttft_s":round(tf-t0,3) if tf else None,"decode_s":round(tl-tf,3) if tf else None,"chunks":n,"chars":len(text),"head":text[:60]}))

import json, sys, urllib.request, urllib.parse
d = json.load(open(sys.argv[1]))
base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9090"
ok = bad = nodata = 0
for p in d["panels"]:
    for t in p.get("targets", []):
        e = t["expr"].replace("$__rate_interval", "1m").replace("$__range", "30m").replace("$job", ".*").replace("$instance", ".*")
        url = base + "/api/v1/query?" + urllib.parse.urlencode({"query": e})
        try:
            r = json.load(urllib.request.urlopen(url))
        except urllib.error.HTTPError as ex:
            r = json.load(ex)
        n = len(r["data"]["result"]) if r["status"] == "success" else -1
        if r["status"] != "success":
            bad += 1
            print("ERR", p["title"], r.get("error"), e[:140])
        elif n == 0:
            nodata += 1
            print("empty:", p["title"])
        else:
            ok += 1
            print("data :", p["title"], n, "series")
print(f"ok={ok} empty={nodata} error={bad}")

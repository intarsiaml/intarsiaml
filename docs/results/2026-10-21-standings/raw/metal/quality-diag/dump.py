import sys, json, os
from localeval import model as M
from localeval.cli import main
orig = M.TimedLocalLLM.call_messages
OUT = os.environ["DUMP"]
def patched(self, messages, schema=None, temperature=0.0):
    r = orig(self, messages, schema=schema, temperature=temperature)
    with open(OUT, "a") as f:
        f.write(json.dumps({"schema": getattr(schema, "__name__", None), "prompt_tail": messages[-1]["content"][-400:],
            "ok": r.ok, "error": r.error, "finish": r.finish_reason, "text": r.text[:1500], "reasoning": (r.reasoning or "")[:300],
            "dropped": list(self.client.dropped)}) + "\n")
    return r
M.TimedLocalLLM.call_messages = patched
sys.exit(main(sys.argv[1:]))

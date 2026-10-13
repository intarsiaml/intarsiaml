#!/usr/bin/env python3
"""Build the wrapper-experiment golden of section 9.5 from the long vLLM golden: the user document of chat_reuse_4k as
(A) raw tokens, (B) system + open user turn (the first 4516 ids of the chat prompt), (C) user turn closed with
<|im_end|>. Reference rows of A and C are copies of the chat prompt's rows and mean nothing: only the self-golden
comparisons (flash vs SIMT, chunk vs chunk) are used for those two.
usage: wrapgold.py <vllm long golden> <out.safetensors>"""
import json, struct, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
f = open(src, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
def get(k, dt):
    t = h[k]; f.seek(base + t['data_offsets'][0])
    return np.frombuffer(f.read(t['data_offsets'][1] - t['data_offsets'][0]), dtype=dt).reshape(t['shape'])
ids = get('ids_2', np.int32); ti = get('topk_ids_2', np.int32); tl = get('topk_logprobs_2', np.float32)
assert ids[4516] == 248046 and ids[25] == 248045 and ids[26] == 846   # <|im_end|> closing the document, <|im_start|> user
doc = [int(x) for x in ids[29:4516]]                                   # after "<|im_start|>user\n" (25, 26, 27) and the first doc token
variants = [('A_raw_doc', doc), ('B_system_user_doc_open', [int(x) for x in ids[:4516]]), ('C_user_doc_closed', [248045, 846, 198] + doc + [248046, 198])]
meta = dict(h['__metadata__']); prompts = []; hdr = {}; blob = b''
def add(k, dt, arr):
    global blob
    b = np.ascontiguousarray(arr).tobytes(); hdr[k] = {'dtype': dt, 'shape': list(arr.shape), 'data_offsets': [len(blob), len(blob) + len(b)]}; blob += b
for j, (nm, v) in enumerate(variants):
    L = len(v)
    prompts.append({'name': f'{nm}@{L}', 'kind': 'raw', 'n_tokens': L, 'n_prompt': L, 'continuation': 0})
    add(f'ids_{j}', 'I32', np.array(v, dtype=np.int32)); add(f'topk_ids_{j}', 'I32', ti[:L]); add(f'topk_logprobs_{j}', 'F32', tl[:L])
meta['prompts'] = json.dumps(prompts); hdr['__metadata__'] = meta
hb = json.dumps(hdr).encode(); hb += b' ' * ((8 - len(hb) % 8) % 8)
open(out, 'wb').write(struct.pack('<Q', len(hb)) + hb + blob)

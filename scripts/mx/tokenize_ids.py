#!/usr/bin/env python3
"""Lane mx: tokenize a text file with a model's tokenizer.json -> int32 .npy (input of make_tails.py).
    tokenize_ids.py tokenizer.json text.txt out.npy"""
import sys
import numpy as np
from tokenizers import Tokenizer

t = Tokenizer.from_file(sys.argv[1])
ids = t.encode(open(sys.argv[2]).read(), add_special_tokens=False).ids
np.save(sys.argv[3], np.array(ids, dtype=np.int32))
print(f"{sys.argv[3]}: {len(ids)} ids")

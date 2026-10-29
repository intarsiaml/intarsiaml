#!/usr/bin/env python3
"""Lane PF27c (from ARX)2: `localeval` with a seeded request nonce (LEVAL_SEED, default 1), so two servers of an A/B get the same prompts in the same order
(MTP's acceptance, hence its decode speed, depends on the text; the uuid nonce of every request otherwise changes it).
  cd /opt/code/deepeval && uv run python leval.py speed <target> ...
"""
import os
import random
import sys
import uuid

_rng = random.Random(int(os.environ.get("LEVAL_SEED", "1")))
uuid.uuid4 = lambda: uuid.UUID(int=_rng.getrandbits(128), version=4)

from localeval.cli import main  # noqa: E402

main(sys.argv[1:])

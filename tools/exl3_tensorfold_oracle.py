"""Independent oracle for the EXL3 decoder: random EXL3 tensors decoded by TensorFold's MIT numpy reference.

Usage: <python with numpy + safetensors> -I tools/exl3_tensorfold_oracle.py <TensorFold checkout> <out.safetensors>
       EXL3_TF_ORACLE=<out.safetensors> cargo test -p intarsia-backend-cpu --test quant_exl3 -- --ignored tensorfold

TensorFold's `src/tensorfold/cuda/exl3/format.py` (MIT) is loaded by path and used as is: `unpack` (W_q in the rotated
basis, fp16), `dequantize` (W [K, N], float64) and `forward` (x @ W, float64, the kernels' order). For every width and
codebook exllamav3 writes (1..8 bits x 3INST/MCG/MUL1, 1.5/2.5/3.5 bits MUL1) one K=256, N=128 tensor with random
trellis words and random fp16 scales is written with those three results. Nothing here shares code with
crates/intarsia-core/src/quant/exl3.rs.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

tf = Path(sys.argv[1]) / "src/tensorfold/cuda/exl3/format.py"
spec = importlib.util.spec_from_file_location("tf_exl3_format", tf)
fmt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fmt  # dataclasses look their module up
spec.loader.exec_module(fmt)

rng = np.random.default_rng(20261021)
K, N, M = 256, 128, 3
out, cases = {}, []
for bits in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8):
    for cb in fmt.CODEBOOKS:
        if not float(bits).is_integer() and cb != "mul1":
            continue
        name = f"c{len(cases)}"
        tr = rng.integers(-32768, 32768, size=(K // 16, N // 16, int(16 * bits)), dtype=np.int16)
        suh = (rng.uniform(0.5, 1.5, K) * rng.choice([-1, 1], K) / np.sqrt(K)).astype(np.float16)
        svh = (rng.uniform(0.5, 1.5, N) * rng.choice([-1, 1], N)).astype(np.float16)
        x = rng.standard_normal((M, K)).astype(np.float32)
        out[f"{name}.trellis"] = tr
        out[f"{name}.suh"] = suh
        out[f"{name}.svh"] = svh
        out[f"{name}.x"] = x
        out[f"{name}.wq"] = fmt.unpack(tr, bits, cb)
        out[f"{name}.w"] = fmt.dequantize(tr, suh, svh, bits, cb).astype(np.float32)
        out[f"{name}.y"] = fmt.forward(x, tr, suh, svh, bits, cb).astype(np.float32)
        cases.append({"name": name, "bits": bits, "codebook": cb, "K": K, "N": N})
save_file(out, sys.argv[2], metadata={"cases": json.dumps(cases), "source": str(tf)})
print(f"{len(cases)} cases -> {sys.argv[2]}")

"""Real-tensor fixture for the MXFP4 unit tests, with expected values computed independently of the Rust code.

Usage: tools/.venv/bin/python -I tools/gen_mxfp4_fixture.py <quark-mxfp4-checkpoint-dir> [<quark-site-packages-dir>]

Input : a Quark OCP MXFP4 checkpoint (amd/Qwen3.8-27B-Quark-AWQ-MXFP4: weight u8 [N,K/2], weight_scale u8 [N,K/32]).
Output: crates/intarsia-backend-cpu/tests/fixtures/qwen38_mxfp4_slices.safetensors   (row slices of real tensors)
        crates/intarsia-backend-cpu/tests/fixtures/qwen38_mxfp4_expected.safetensors (dequant, QDQ, matmul results)

Independence from crates/intarsia-core/src/quant/mxfp4.rs:
* e2m1 decode by building the f32 bit pattern (Quark's Pack_mxfp4.unpack trick), element 2i in the low nibble (Quark's
  `unpacked[:, ::2] = element_part & 0x0F`), e8m0 as torch.exp2(s - 127);
* the activation scale is Quark's own `even_round` (quark/torch/quantization/utils.py), extracted from the installed
  amd-quark wheel's source with `ast` and executed as is (importing quark.torch needs scipy and more, which we do not
  install); pass the wheel's site-packages dir as the second argument (default: the M1 drill venv);
* e2m1 rounding is a nearest search over the value grid with ties to the even code, saturating at +-6.
Matmuls are float64.
"""
import ast, json, struct, sys
from pathlib import Path

import torch
from safetensors.torch import save_file

model = Path(sys.argv[1])
quark_site = Path(sys.argv[2]) if len(sys.argv) > 2 else None
fx = Path(__file__).resolve().parent.parent / "crates/intarsia-backend-cpu/tests/fixtures"

# ---- tensors (stdlib header parse, row slices) -------------------------------------------------------------------
idx = model / "model.safetensors.index.json"
files = {}
if idx.exists():
    wm = json.load(open(idx))["weight_map"]
else:
    wm = None


def header(f):
    if f not in files:
        with open(model / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            files[f] = (json.loads(fh.read(n)), 8 + n)
    return files[f]


def rows(name, r0, r1):
    f = wm[name] if wm else "model.safetensors"
    h, base = header(f)
    t = h[name]
    assert t["dtype"] == "U8", (name, t["dtype"])
    C = t["shape"][1]
    b0 = t["data_offsets"][0]
    with open(model / f, "rb") as fh:
        fh.seek(base + b0 + r0 * C)
        data = fh.read((r1 - r0) * C)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(r1 - r0, C).clone()


L = "model.language_model.layers."
SLICES = {  # key: (module, rows)
    "gate": (L + "0.mlp.gate_proj", (0, 16)),  # K = 5120
    "down": (L + "0.mlp.down_proj", (100, 108)),  # K = 17408
    "in_a": (L + "0.linear_attn.in_proj_a", (0, 48)),  # all 48 rows, K = 5120
    "q": (L + "3.self_attn.q_proj", (256, 264)),  # gated attention q_proj, K = 5120
    "o": (L + "63.self_attn.o_proj", (5000, 5008)),  # last layer, K = 6144
}
sl = {}
for key, (mod, (r0, r1)) in SLICES.items():
    sl[f"{key}.weight"] = rows(mod + ".weight", r0, r1)
    sl[f"{key}.weight_scale"] = rows(mod + ".weight_scale", r0, r1)

# ---- independent math ------------------------------------------------------------------------------------------------
def dequant(w, s):
    lo, hi = (w & 0x0F).to(torch.int32), ((w >> 4) & 0x0F).to(torch.int32)
    codes = torch.stack([lo, hi], -1).reshape(w.shape[0], -1)
    # e2m1 -> f32 bits: magnitude bits into exponent/mantissa positions, then rescale by 2^126 (Quark's unpack)
    bits = ((codes & 0x07) << 22) | ((codes & 0x08) << 28)
    vals = bits.view(torch.float32).to(torch.float64) * (2.0 ** 126)
    sc = torch.exp2(s.to(torch.float64) - 127.0)
    sc[s == 255] = float("nan")
    return vals * sc.repeat_interleave(32, dim=1)


def quark_even_round():
    site = quark_site or Path("/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/quarkenv/lib/python3.13/site-packages")
    src = (site / "quark/torch/quantization/utils.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "even_round")
    # Dtype only appears in the signature's annotation; get_dtype_params is Quark's fp4 row: ebits 2, mbits 1, emax 2
    ns = {"torch": torch, "Dtype": type("Dtype", (), {}), "get_dtype_params": lambda d: (2, 1, 2)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "quark_even_round", "exec"), ns)
    return ns["even_round"]


even_round = quark_even_round()
GRID = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def round_e2m1(v):
    a = v.abs().clamp(max=6.0)
    d = (a.unsqueeze(-1) - GRID).abs()
    best = d.min(-1, keepdim=True).values
    # ties go to the even code (lowest index among equal distances that is even)
    cand = (d == best)
    idx = torch.arange(8)
    even_first = torch.where(cand & (idx % 2 == 0), idx, torch.full_like(idx, 99)).min(-1).values
    any_first = torch.where(cand, idx, torch.full_like(idx, 99)).min(-1).values
    code = torch.where(even_first < 99, even_first, any_first)
    return torch.where(v < 0, -GRID[code], GRID[code])


def qdq(x):  # x [M, K] float32 -> QDQ'd float32, groups of 32 along K
    m, k = x.shape
    b = x.reshape(m, k // 32, 32)
    amax = b.abs().max(-1, keepdim=True).values
    scale = even_round(amax.clone(), "fp4").to(torch.float32)
    return (round_e2m1(b / scale) * scale).reshape(m, k)


g = torch.Generator().manual_seed(11)
exp = {}
for key in SLICES:
    w, s = sl[f"{key}.weight"], sl[f"{key}.weight_scale"]
    W = dequant(w, s)
    k = W.shape[1]
    exp[f"{key}.W"] = W.to(torch.float32)
    # activations: bf16-representable, some groups scaled up / down so the e8m0 range is exercised
    x = (torch.randn(3, k, generator=g) * torch.exp2(torch.randint(-6, 6, (3, k // 32), generator=g).float()).repeat_interleave(32, 1))
    x = x.to(torch.bfloat16).to(torch.float32)
    xq = qdq(x)
    exp[f"{key}.x"] = x.to(torch.bfloat16)
    exp[f"{key}.xq"] = xq
    exp[f"{key}.y_w4a16"] = (x.double() @ W.T).to(torch.float32)
    exp[f"{key}.y_w4a4"] = (xq.double() @ W.T).to(torch.float32)

# hand-made QDQ edge cases: ties, saturation, rounding-up of the scale, zeros, tiny and huge groups
edge = torch.zeros(8, 32)
edge[0, :4] = torch.tensor([7.0, 1.0, 0.4, 3.5])  # amax mantissa 1.75 -> scale 2
edge[1, :3] = torch.tensor([-6.9, 0.25, 0.75])  # saturate at 6, ties 0.25 -> 0 and 0.75 -> 1
edge[2, :2] = torch.tensor([1e-30, -3e-31])
edge[3, :2] = torch.tensor([3e37, -1e37])
edge[4] = torch.linspace(-5, 5, 32)
edge[5] = torch.randn(32, generator=g) * 1e-3
edge[6, :3] = torch.tensor([2.5, 1.25, 5.0])  # amax 5 = 1.25*4 -> scale 1: 2.5 -> 2, 1.25 -> 1, 5 -> 4
exp["edge.x"] = edge
exp["edge.xq"] = qdq(edge)

fx.mkdir(parents=True, exist_ok=True)
save_file(sl, str(fx / "qwen38_mxfp4_slices.safetensors"))
save_file(exp, str(fx / "qwen38_mxfp4_expected.safetensors"))
for key in SLICES:
    W = exp[f"{key}.W"]
    print(f"{key:5s} {tuple(W.shape)} std {W.std().item():.4g} max {W.abs().max().item():.4g} "
          f"scale bytes {sl[key + '.weight_scale'].min().item()}..{sl[key + '.weight_scale'].max().item()}")
print("edge xq rows 0..2:", exp["edge.xq"][:3, :4].tolist())

"""Copy an ExLlamaV3 (EXL3) checkpoint with its plain fp16 tensors (embedding, norms, an fp16 head) cast to bf16.

Usage: python3 -I tools/exl3_plain_to_bf16.py <exl3 model dir> <out dir>

Why: exllamav3 runs fp16 activations and stores the non-quantised tensors in fp16; intarsiaml's GPU backends run bf16
activations, and `Model::load` takes the activation dtype from the embedding table. The EXL3 parts (`.trellis`, `.suh`,
`.svh`, `.su`, `.sv`, `.mcg`, `.mul1`) are copied byte for byte; every other F16 tensor is rounded to bf16 (round to
nearest even), which is the only numeric change (at most 2^-8 relative per element of the embedding and norm weights).
numpy; streams one tensor at a time, so memory stays small. Non-weight files are copied.
"""
import json
import shutil
import struct
import sys
from pathlib import Path

EXL3_PARTS = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")


def f16_to_bf16(buf: bytes) -> bytes:
    """fp16 -> f32 (exact) -> bf16, round to nearest even (inf / nan kept)."""
    import numpy as np

    f = np.frombuffer(buf, dtype="<f2").astype(np.float32).view(np.uint32)
    finite = (f & 0x7F800000) != 0x7F800000
    r = np.where(finite, (f + 0x7FFF + ((f >> 16) & 1)) >> 16, f >> 16).astype("<u2")
    return r.tobytes()


def convert(src: Path, dst: Path) -> tuple[int, int]:
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr.pop("__metadata__", None)
        names = sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0])
        new, off, cast = {}, 0, 0
        for k in names:
            v = hdr[k]
            size = v["data_offsets"][1] - v["data_offsets"][0]
            dt = v["dtype"]
            if dt == "F16" and k.rsplit(".", 1)[-1] not in EXL3_PARTS:
                dt, cast = "BF16", cast + 1
            new[k] = {"dtype": dt, "shape": v["shape"], "data_offsets": [off, off + size]}
            off += size
        if meta is not None:
            new["__metadata__"] = meta
        hb = json.dumps(new).encode()
        hb += b" " * (-len(hb) % 8)
        with open(dst, "wb") as g:
            g.write(struct.pack("<Q", len(hb)))
            g.write(hb)
            for k in names:
                v = hdr[k]
                f.seek(8 + n + v["data_offsets"][0])
                left = v["data_offsets"][1] - v["data_offsets"][0]
                conv = new[k]["dtype"] != v["dtype"]
                while left:
                    chunk = f.read(min(left, 1 << 24))
                    left -= len(chunk)
                    g.write(f16_to_bf16(chunk) if conv else chunk)
    return len(names), cast


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.suffix == ".safetensors":
            total, cast = convert(p, dst / p.name)
            print(f"{p.name}: {total} tensors, {cast} plain F16 -> BF16")
        elif p.is_file():
            shutil.copy2(p, dst / p.name)


if __name__ == "__main__":
    main()

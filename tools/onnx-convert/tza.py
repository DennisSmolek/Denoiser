"""Parser for OIDN TZA weight blobs.

Direct port of packages/denoiser/src/tza.ts (parseTZA). Returns a dict of
name -> numpy.ndarray. Conv weights stay in their native OIHW layout (the JS
code transposes to HWIO for TensorFlow; ONNX Conv wants OIHW, so we keep it).
"""
from __future__ import annotations

import struct
from typing import Dict

import numpy as np

MAGIC = 0x41D7
MAJOR_VERSION = 2


def parse_tza(buf: bytes) -> Dict[str, np.ndarray]:
    """Parse a TZA blob into {tensor_name: ndarray} (OIHW for 4D weights)."""
    magic = struct.unpack_from("<H", buf, 0)[0]
    if magic != MAGIC:
        raise ValueError("Invalid or corrupted weights blob")

    major = buf[2]
    # buf[3] is the minor version, ignored
    if major != MAJOR_VERSION:
        raise ValueError(f"Unsupported weights blob version: {major}")

    table_offset = struct.unpack_from("<Q", buf, 4)[0]
    offset = table_offset

    num_tensors = struct.unpack_from("<I", buf, offset)[0]
    offset += 4

    tensors: Dict[str, np.ndarray] = {}
    for _ in range(num_tensors):
        name_len = struct.unpack_from("<H", buf, offset)[0]
        offset += 2
        name = buf[offset:offset + name_len].decode("utf-8")
        offset += name_len

        ndims = buf[offset]
        offset += 1

        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack_from("<I", buf, offset)[0])
            offset += 4

        # layout string (one char per dim), not needed for parsing
        offset += ndims

        data_type = buf[offset:offset + 1].decode("ascii")
        offset += 1

        tensor_offset = struct.unpack_from("<Q", buf, offset)[0]
        offset += 8

        num_elements = int(np.prod(shape)) if shape else 1

        if data_type == "f":
            raw = buf[tensor_offset:tensor_offset + num_elements * 4]
            arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif data_type == "h":
            raw = buf[tensor_offset:tensor_offset + num_elements * 2]
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise ValueError(f"Invalid tensor data type: {data_type!r}")

        if arr.size != num_elements:
            raise ValueError(
                f"Element count mismatch for {name}: got {arr.size}, expected {num_elements}"
            )

        tensors[name] = arr.reshape(shape) if shape else arr
    return tensors


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "rb") as f:
        data = f.read()
    parsed = parse_tza(data)
    for k in sorted(parsed):
        print(f"{k:24s} {tuple(parsed[k].shape)} {parsed[k].dtype}")
    print(f"\n{len(parsed)} tensors")

#!/usr/bin/env python3
"""Generate a metadata-only GGUF for a dense architecture (qwen2/qwen3/llama/
mistral). No tensors are written, so the file is tiny and lets you exercise the
dense dispatch + shape construction in ds4 WITHOUT downloading real weights:

    python3 tests/mkgguf_dense_meta.py /tmp/qwen2.gguf
    ./ds4 --inspect -m /tmp/qwen2.gguf

Expected: ds4 recognizes the dense architecture, prints the constructed shape,
then stops with the "dense execution graph not yet available (Fase 3)" message.
See docs/DENSE_SUPPORT_DESIGN.md.
"""
import struct
import sys

GGUF_MAGIC = 0x46554747
T_U32, T_F32, T_STR, T_U64 = 4, 6, 8, 10


def gstr(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):
    return gstr(k) + struct.pack("<I", T_U32) + struct.pack("<I", v)


def kv_f32(k, v):
    return gstr(k) + struct.pack("<I", T_F32) + struct.pack("<f", v)


def kv_str(k, v):
    return gstr(k) + struct.pack("<I", T_STR) + gstr(v)


def build(arch="qwen2"):
    # Defaults model a Qwen2-7B-class config; adjust as needed.
    kvs = [
        kv_str("general.architecture", arch),
        kv_str("general.name", arch + "-meta-test"),
        kv_u32(arch + ".block_count", 28),
        kv_u32(arch + ".embedding_length", 3584),
        kv_u32(arch + ".attention.head_count", 28),
        kv_u32(arch + ".attention.head_count_kv", 4),
        kv_u32(arch + ".feed_forward_length", 18944),
        kv_u32(arch + ".vocab_size", 152064),
        kv_u32(arch + ".context_length", 32768),
        kv_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-6),
        kv_f32(arch + ".rope.freq_base", 1000000.0),
    ]
    hdr = (
        struct.pack("<I", GGUF_MAGIC)
        + struct.pack("<I", 3)            # version
        + struct.pack("<Q", 0)            # tensor_count
        + struct.pack("<Q", len(kvs))     # metadata_kv_count
    )
    return hdr + b"".join(kvs)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "dense-meta-test.gguf"
    arch = sys.argv[2] if len(sys.argv) > 2 else "qwen2"
    with open(out, "wb") as f:
        f.write(build(arch))
    print("wrote", out, "arch=", arch)

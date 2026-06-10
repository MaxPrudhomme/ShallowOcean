"""Step 1b — GGUF surgery: write a pruned copy of diffusiongemma-26B-A4B keeping
only the selected experts per layer (route-once, arXiv:2501.02086).

Slices every `blk.N.ffn_*_exps` tensor along the expert axis (raw quantized
block copy — the expert axis is the slowest-varying, so each expert is a
contiguous slab; no requantization), slices the router `ffn_gate_inp.weight`
rows and the per-expert `ffn_down_exps.scale` to match, and sets
`expert_count` in the metadata. llama.cpp then sees an ordinary smaller-expert
model that fits in RAM.

Usage:
    python step1/prune_gguf.py --experts experts.json --out pruned.gguf [--model ...]

`experts.json` is produced by step1/route_stats.py: {"0": [ids...], ...} with
one sorted expert-id list per layer (all lists the same length).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGML_QUANT_SIZES

DEFAULT_MODEL = Path.home() / ".lmstudio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

# tensors sliced along the expert axis, and where that axis lives in the
# reader's (reversed-ne) numpy view
EXPERT_WEIGHTS = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight",
                  "ffn_gate_exps.weight", "ffn_up_exps.weight")
EXPERT_VECTORS = ("ffn_down_exps.scale",)          # F32 [n_expert]
ROUTER_ROWS    = ("ffn_gate_inp.weight",)          # F32 [n_embd, n_expert] -> rows in reader view


def copy_metadata(reader: GGUFReader, writer: GGUFWriter, n_keep: int, arch: str):
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
            "general.architecture"}  # arch is set by the GGUFWriter ctor
    expert_count_key = f"{arch}.expert_count"
    for field in reader.fields.values():
        if field.name in skip:
            continue
        val = field.contents()
        if field.name == expert_count_key:
            val = n_keep
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            writer.add_key_value(field.name, val, vtype, sub_type=field.types[-1])
        else:
            writer.add_key_value(field.name, val, vtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--experts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    keep = {int(k): np.asarray(sorted(v), dtype=np.int64)
            for k, v in json.loads(args.experts.read_text()).items()}
    n_keep = len(next(iter(keep.values())))
    assert all(len(v) == n_keep for v in keep.values()), "uneven expert sets per layer"

    t0 = time.time()
    reader = GGUFReader(args.model)
    arch = reader.fields["general.architecture"].contents()
    n_expert = reader.fields[f"{arch}.expert_count"].contents()
    print(f"arch={arch}  experts {n_expert} -> {n_keep}", file=sys.stderr)

    writer = GGUFWriter(args.out, arch)
    copy_metadata(reader, writer, n_keep, arch)

    for t in reader.tensors:
        name = t.name
        suffix = name.split(".", 2)[-1] if name.startswith("blk.") else name
        data = t.data  # mmap-backed numpy view of the (possibly quantized) data
        shape = list(t.shape)  # GGUF ne order

        if name.startswith("blk.") and suffix in EXPERT_WEIGHTS + EXPERT_VECTORS + ROUTER_ROWS:
            il = int(name.split(".")[1])
            sel = keep[il]
            if suffix in EXPERT_WEIGHTS:
                # expert axis = last ne = slowest-varying: view raw bytes as
                # [n_expert, bytes_per_expert] and gather rows (quantized blocks
                # are copied verbatim)
                raw = data.reshape(n_expert, -1)
                data = np.ascontiguousarray(raw[sel])
                shape[-1] = n_keep
            else:
                # F32 per-expert vector / router rows: expert axis is axis 0 of
                # the reader's reversed-ne view
                data = np.ascontiguousarray(data[sel])
                shape[-1] = n_keep

        # for quantized (uint8) data, GGUFWriter wants the byte shape: numpy
        # (reversed-ne) order with the row dimension in bytes
        if data.dtype == np.uint8:
            block, ts = GGML_QUANT_SIZES[t.tensor_type]
            np_shape = (*shape[::-1][:-1], shape[0] // block * ts)
        else:
            np_shape = tuple(shape[::-1])
        writer.add_tensor(name, data, raw_shape=np_shape, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {args.out} ({args.out.stat().st_size/1e9:.2f} GB) "
          f"in {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

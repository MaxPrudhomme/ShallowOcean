"""Instruction-Following Pruning demo (arXiv:2501.02086) on Gemma 4 E4B.

The paper's idea: instead of a per-token MoE router, decide *once at prompt
processing* which FFN parameters the request needs, load only those into
memory, and decode with the rest left on SSD.

Gemma 4 E4B is a dense model (no expert tensors in the checkpoint), so we
treat each 64-channel quantization group of every FFN as an "expert":
42 layers x 160 groups. Phase 1 routes (scores groups against the prompt),
phase 2 materializes only the selected groups from the mmap'd safetensors,
phase 3 decodes with the pruned FFNs. Unselected weight bytes are never
evaluated, so they stay on SSD.

Usage:
    python demo/ifprune_demo.py --prompt "..." [--keep 0.5] [--max-tokens 200]
"""

import argparse
import gc
import glob
import json
import re
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import stream_generate
from mlx_lm.models import gemma4
from mlx_lm.utils import load_model, load_tokenizer

GROUP = 64  # quantization group size == "expert" granularity
BITS = 8  # all FFN tensors in this checkpoint are 8-bit affine
PACK = 32 // BITS  # values per packed uint32


@partial(mx.compile, shapeless=True)
def geglu(gate, x):
    return nn.gelu_approx(gate) * x


def patch_sanitize(config):
    """Drop redundant k/v projections stored for KV-shared layers."""
    tc = config["text_config"]
    first_shared = tc["num_hidden_layers"] - tc["num_kv_shared_layers"]
    pat = re.compile(
        r"language_model\.model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm)\."
    )
    orig = gemma4.Model.sanitize

    def sanitize(self, weights):
        weights = {
            k: v
            for k, v in weights.items()
            if not ((m := pat.match(k)) and int(m.group(1)) >= first_shared)
        }
        return orig(self, weights)

    gemma4.Model.sanitize = sanitize


class RoutingMLP(nn.Module):
    """Wraps a dense FFN and accumulates per-channel activation mass."""

    def __init__(self, mlp):
        super().__init__()
        self.inner = mlp
        self.acc = None

    def __call__(self, x):
        a = geglu(self.inner.gate_proj(x), self.inner.up_proj(x))
        s = mx.abs(a).sum(axis=tuple(range(a.ndim - 1)))
        self.acc = s if self.acc is None else self.acc + s
        return self.inner.down_proj(a)


class SlicedQuantizedLinear(nn.Module):
    def __init__(self, weight, scales, biases):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases

    def __call__(self, x):
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )


class PrunedMLP(nn.Module):
    """FFN restricted to the channel groups selected at prompt time."""

    def __init__(self, lazy, prefix, groups):
        super().__init__()
        rows = mx.array((groups[:, None] * GROUP + np.arange(GROUP)).reshape(-1))
        wcols = mx.array(
            (groups[:, None] * (GROUP // PACK) + np.arange(GROUP // PACK)).reshape(-1)
        )
        gcols = mx.array(groups)

        def t(name):
            return lazy[f"{prefix}.{name}"]

        self.gate_proj = SlicedQuantizedLinear(
            t("gate_proj.weight")[rows],
            t("gate_proj.scales")[rows],
            t("gate_proj.biases")[rows],
        )
        self.up_proj = SlicedQuantizedLinear(
            t("up_proj.weight")[rows],
            t("up_proj.scales")[rows],
            t("up_proj.biases")[rows],
        )
        self.down_proj = SlicedQuantizedLinear(
            t("down_proj.weight")[:, wcols],
            t("down_proj.scales")[:, gcols],
            t("down_proj.biases")[:, gcols],
        )

    def __call__(self, x):
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


def mlp_nbytes(mlp):
    from mlx.utils import tree_flatten

    return sum(v.nbytes for _, v in tree_flatten(mlp.parameters()))


def generate_timed(model, tokenizer, prompt_tokens, max_tokens):
    text, gen_tps, peak = "", 0.0, 0.0
    for r in stream_generate(model, tokenizer, prompt_tokens, max_tokens=max_tokens):
        text += r.text
        gen_tps, peak = r.generation_tps, r.peak_memory
    return text, gen_tps, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gemma-4-e4b")
    ap.add_argument("--prompt", default="Write a Python function that checks whether a string is a palindrome, then explain its complexity.")
    ap.add_argument("--keep", type=float, default=0.7, help="fraction of FFN channel groups to keep")
    ap.add_argument("--floor", type=float, default=0.5, help="minimum per-layer keep fraction")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model)
    config = json.load(open(model_path / "config.json"))
    patch_sanitize(config)

    print(f"== Loading {model_path} (lazy, mmap) ==")
    model, _ = load_model(model_path, lazy=True)
    tokenizer = load_tokenizer(model_path)

    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )
    layers = model.language_model.model.layers
    full_ffn = sum(mlp_nbytes(l.mlp) for l in layers)

    baseline = None
    if not args.skip_baseline:
        print("\n== Baseline: full dense model ==")
        t0 = time.time()
        text, tps, peak = generate_timed(model, tokenizer, prompt_tokens, args.max_tokens)
        baseline = (text, tps, peak, time.time() - t0)
        print(text)

    # ---- Phase 1: route once at prompt processing ----
    print(f"\n== Phase 1: routing pass over {len(prompt_tokens)} prompt tokens ==")
    t0 = time.time()
    originals = [l.mlp for l in layers]
    for l in layers:
        l.mlp = RoutingMLP(l.mlp)
    mx.eval(model(mx.array(prompt_tokens)[None]))

    n_groups = config["text_config"]["intermediate_size"] // GROUP
    all_scores = []
    for l in layers:
        # score = prompt activation mass x how hard that channel hits the
        # residual stream (L2 norm of its down_proj column)
        d = l.mlp.inner.down_proj
        wd = mx.dequantize(d.weight, d.scales, d.biases, group_size=GROUP, bits=BITS)
        col_norm = mx.sqrt(wd.astype(mx.float32).square().sum(axis=0))
        scores = l.mlp.acc.astype(mx.float32) * col_norm
        all_scores.append(np.array(scores).reshape(n_groups, GROUP).sum(-1))

    # Global budget: groups compete across layers (normalized per layer so
    # scales are comparable); a floor keeps every layer minimally functional.
    budget = round(len(layers) * n_groups * args.keep)
    floor = round(n_groups * args.floor)
    norm = np.stack([s / s.sum() for s in all_scores])  # (layers, n_groups)
    order = np.argsort(norm, axis=None)[::-1]
    counts = np.full(len(layers), floor)
    chosen = np.zeros_like(norm, dtype=bool)
    for li in range(len(layers)):
        top = np.argsort(norm[li])[-floor:]
        chosen[li, top] = True
    for flat in order:
        if counts.sum() >= budget:
            break
        li, gi = divmod(flat, n_groups)
        if not chosen[li, gi]:
            chosen[li, gi] = True
            counts[li] += 1
    selections = [np.flatnonzero(chosen[li]) for li in range(len(layers))]
    keep = counts.mean()
    print(f"selected {keep:.0f}/{n_groups} channel groups per layer on average "
          f"(min {min(len(s) for s in selections)}, max {max(len(s) for s in selections)}) "
          f"({time.time() - t0:.1f}s)")

    # ---- Phase 2: materialize only the selected groups from SSD ----
    print("\n== Phase 2: loading selected experts, leaving the rest on SSD ==")
    lazy = {}
    for f in glob.glob(str(model_path / "model-*.safetensors")):
        lazy.update(mx.load(f))  # mmap'd; nothing read until evaluated
    t0 = time.time()
    for i, (l, groups) in enumerate(zip(layers, selections)):
        l.mlp = PrunedMLP(lazy, f"language_model.model.layers.{i}.mlp", groups)
        mx.eval(l.mlp.parameters())
    pruned_ffn = sum(mlp_nbytes(l.mlp) for l in layers)
    del lazy, originals
    gc.collect()
    mx.clear_cache()
    print(f"FFN resident: {pruned_ffn / 1e6:.0f} MB of {full_ffn / 1e6:.0f} MB "
          f"({pruned_ffn / full_ffn:.0%}), loaded in {time.time() - t0:.1f}s")

    # ---- Phase 3: decode with the pruned model ----
    print("\n== Phase 3: generation with pruned FFNs ==")
    t0 = time.time()
    text, tps, peak = generate_timed(model, tokenizer, prompt_tokens, args.max_tokens)
    print(text)

    print("\n== Summary ==")
    print(f"FFN weights resident:  {pruned_ffn / 1e6:8.0f} MB  (full: {full_ffn / 1e6:.0f} MB)")
    print(f"FFN weights on SSD:    {(full_ffn - pruned_ffn) / 1e6:8.0f} MB  never loaded")
    print(f"pruned decode:         {tps:8.1f} tok/s   wall {time.time() - t0:.1f}s")
    if baseline:
        btext, btps, bpeak, bwall = baseline
        print(f"full decode:           {btps:8.1f} tok/s   wall {bwall:.1f}s")
        print(f"peak mlx memory:       {peak:.1f} GB pruned vs {bpeak:.1f} GB full")


if __name__ == "__main__":
    main()

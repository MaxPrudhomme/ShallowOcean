"""Route-once MoE offload demo (idea from arXiv:2501.02086) on gpt-oss-20b.

gpt-oss-20b is a real MoE: 24 layers x 32 experts, top-4 per token, with the
router normally re-run at every decoding step. Following the paper's
"select parameters once per prompt" idea, this demo:

  1. Runs the prompt once and accumulates each layer's router probability
     mass per expert (the routing decision, made once at prompt processing).
  2. Keeps only the top experts per layer (default 50%), slicing them out of
     the mmap'd quantized safetensors. Unselected experts' bytes are never
     evaluated, so they stay on SSD.
  3. Decodes with the router restricted to the loaded experts (it still picks
     top-4 per token, but only among them).

Usage:
    python demo/moe_route_once_demo.py --prompt "..." [--keep 0.5]
"""

import argparse
import gc
import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import stream_generate
from mlx_lm.models.gpt_oss import mlx_topk
from mlx_lm.utils import load_model, load_tokenizer
from mlx.utils import tree_flatten


class RoutingMLP(nn.Module):
    """Wraps an MLPBlock and accumulates router probability mass per expert."""

    def __init__(self, mlp):
        super().__init__()
        self.inner = mlp
        self.mass = None

    def __call__(self, x):
        p = mx.softmax(self.inner.router(x), axis=-1, precise=True)
        m = p.sum(axis=tuple(range(p.ndim - 1)))
        self.mass = m if self.mass is None else self.mass + m
        return self.inner(x)


class PrunedMoE(nn.Module):
    """MLPBlock restricted to the experts selected at prompt time.

    The full (tiny) router still runs; its logits are gathered down to the
    kept experts, so top-k indices land directly in the compact expert axis.
    """

    def __init__(self, mlp, lazy, prefix, sel, top_k):
        super().__init__()
        self.router = mlp.inner.router
        self.experts = mlp.inner.experts  # reuse module: quant meta stays valid
        self.sel = mx.array(sel)
        self.top_k = top_k
        for proj in ("gate_proj", "up_proj", "down_proj"):
            sub = self.experts[proj]
            for p in ("weight", "scales", "biases", "bias"):
                t = lazy.get(f"{prefix}.experts.{proj}.{p}")
                if t is not None:
                    sub[p] = t[self.sel]

    def __call__(self, x):
        g = mx.take(self.router(x), self.sel, axis=-1)
        vals, indices = mlx_topk(g, k=self.top_k, axis=-1)
        w = mx.softmax(vals, axis=-1, precise=True)
        y = self.experts(x, indices) * mx.expand_dims(w, axis=-1)
        return y.sum(axis=-2)


def expert_nbytes(module):
    return sum(v.nbytes for k, v in tree_flatten(module.parameters()) if "experts" in k)


def generate_timed(model, tokenizer, prompt_tokens, max_tokens):
    text, gen_tps, peak = "", 0.0, 0.0
    for r in stream_generate(model, tokenizer, prompt_tokens, max_tokens=max_tokens):
        text += r.text
        gen_tps, peak = r.generation_tps, r.peak_memory
    return text, gen_tps, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gpt-oss-20b")
    ap.add_argument("--prompt", default="Write a Python function that checks whether a string is a palindrome, then explain its complexity.")
    ap.add_argument("--keep", type=float, default=0.5, help="fraction of experts to keep per layer")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model)
    config = json.load(open(model_path / "config.json"))
    n_experts = config["num_local_experts"]
    top_k = config["num_experts_per_tok"]

    print(f"== Loading {model_path} (lazy, mmap) ==")
    model, _ = load_model(model_path, lazy=True)
    tokenizer = load_tokenizer(model_path)

    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )
    layers = model.model.layers
    full_bytes = sum(expert_nbytes(l.mlp) for l in layers)

    baseline = None
    if not args.skip_baseline:
        print("\n== Baseline: full MoE, router at every token ==")
        t0 = time.time()
        text, tps, peak = generate_timed(model, tokenizer, prompt_tokens, args.max_tokens)
        baseline = (text, tps, peak, time.time() - t0)
        print(text)

    # ---- Phase 1: route once at prompt processing ----
    print(f"\n== Phase 1: routing pass over {len(prompt_tokens)} prompt tokens ==")
    t0 = time.time()
    for l in layers:
        l.mlp = RoutingMLP(l.mlp)
    mx.eval(model(mx.array(prompt_tokens)[None]))

    keep = max(top_k, round(n_experts * args.keep))
    selections = []
    for l in layers:
        mass = np.array(l.mlp.mass.astype(mx.float32))
        selections.append(np.sort(np.argsort(mass)[-keep:]))
    coverage = [
        np.array(l.mlp.mass.astype(mx.float32))[sel].sum()
        / np.array(l.mlp.mass.astype(mx.float32)).sum()
        for l, sel in zip(layers, selections)
    ]
    print(f"selected {keep}/{n_experts} experts per layer, covering "
          f"{np.mean(coverage):.0%} of prompt router mass ({time.time() - t0:.1f}s)")

    # ---- Phase 2: load selected experts, leave the rest on SSD ----
    print("\n== Phase 2: loading selected experts ==")
    lazy = {}
    for f in glob.glob(str(model_path / "model-*.safetensors")):
        lazy.update(mx.load(f))  # mmap'd; nothing read until evaluated
    t0 = time.time()
    for i, (l, sel) in enumerate(zip(layers, selections)):
        l.mlp = PrunedMoE(l.mlp, lazy, f"model.layers.{i}.mlp", sel, top_k)
        mx.eval(l.mlp.parameters())
    pruned_bytes = sum(expert_nbytes(l.mlp) for l in layers)
    del lazy
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    print(f"expert weights resident: {pruned_bytes / 1e6:.0f} MB of "
          f"{full_bytes / 1e6:.0f} MB ({pruned_bytes / full_bytes:.0%}), "
          f"loaded in {time.time() - t0:.1f}s")
    print(f"mlx active memory after pruning: {mx.get_active_memory() / 1e9:.2f} GB")

    # ---- Phase 3: decode with the restricted expert set ----
    print("\n== Phase 3: generation with route-once expert set ==")
    t0 = time.time()
    text, tps, peak = generate_timed(model, tokenizer, prompt_tokens, args.max_tokens)
    print(text)

    print("\n== Summary ==")
    print(f"expert weights resident: {pruned_bytes / 1e6:8.0f} MB  (full: {full_bytes / 1e6:.0f} MB)")
    print(f"expert weights on SSD:   {(full_bytes - pruned_bytes) / 1e6:8.0f} MB  never loaded")
    print(f"route-once decode:       {tps:8.1f} tok/s   wall {time.time() - t0:.1f}s")
    if baseline:
        btext, btps, bpeak, bwall = baseline
        print(f"full MoE decode:         {btps:8.1f} tok/s   wall {bwall:.1f}s")
        print(f"peak mlx memory:         {peak:.1f} GB pruned vs {bpeak:.1f} GB full")


if __name__ == "__main__":
    main()

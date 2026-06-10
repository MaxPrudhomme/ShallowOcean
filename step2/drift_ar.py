"""Step 2 (AR gemma-4-26B-A4B) — measure routing drift: how well does a
route-once expert selection made from *prefill* router stats cover the experts
the model actually uses during *decode*?

Patches mlx-lm's gemma4 Router to log, per layer:
  - prefill: full-softmax router mass per expert (the route-once signal)
  - decode:  full-softmax mass + per-token top-8 pick counts (ground truth)

Then reports, per keep fraction: pick coverage (fraction of decode top-8 picks
that land inside the prefill-selected set), decode mass coverage, and the size
of the union of experts actually used during decode.

Usage:
    .venv/bin/python step2/drift_ar.py [--prompt "..."] [--max-tokens 200]
"""

import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.gemma4_text import Router

DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"

stats = {}          # id(router) -> {"prefill": np[128], "dec_mass": np[128], "dec_cnt": np[128]}
router_layer = {}   # id(router) -> layer index

_orig_call = Router.__call__


def patched_call(self, x):
    h = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)
    scores = self.proj(h)
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).reshape(-1, scores.shape[-1])
    mass = np.array(probs.sum(0))
    k = self.config.top_k_experts
    top = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    cnt = np.bincount(np.array(top).reshape(-1), minlength=scores.shape[-1])

    s = stats.setdefault(id(self), {
        "prefill": np.zeros(scores.shape[-1]),
        "dec_mass": np.zeros(scores.shape[-1]),
        "dec_cnt": np.zeros(scores.shape[-1], dtype=np.int64),
    })
    if probs.shape[0] > 1:          # prefill (or chunked prompt processing)
        s["prefill"] += mass
    else:                            # decode: one token at a time
        s["dec_mass"] += mass
        s["dec_cnt"] += cnt
    return _orig_call(self, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="Write a Python function that checks if a string is a palindrome.")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    Router.__call__ = patched_call
    model, tokenizer = load(args.model)

    lm = model.language_model if hasattr(model, "language_model") else model
    for il, layer in enumerate(lm.model.layers):
        if getattr(layer, "router", None) is not None:
            router_layer[id(layer.router)] = il

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, tokenize=False)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=True)

    layers = sorted((router_layer[i], s) for i, s in stats.items() if i in router_layer)
    n_expert = len(layers[0][1]["prefill"])
    print(f"\nMoE layers: {len(layers)}, experts: {n_expert}")
    print(f"{'keep':>6} {'pick-cov':>9} {'dec-mass-cov':>13} {'prefill-mass-cov':>17}")
    for keep in (8, 16, 32, 48, 64, 96):
        pc, dm, pm = [], [], []
        for il, s in layers:
            sel = np.argsort(-s["prefill"])[:keep]
            m = np.zeros(n_expert, bool); m[sel] = True
            pc.append(s["dec_cnt"][m].sum() / s["dec_cnt"].sum())
            dm.append(s["dec_mass"][m].sum() / s["dec_mass"].sum())
            pm.append(s["prefill"][m].sum() / s["prefill"].sum())
        print(f"{keep:>4}/{n_expert} {np.mean(pc):>9.3f} {np.mean(dm):>13.3f} {np.mean(pm):>17.3f}")

    union = [int((s["dec_cnt"] > 0).sum()) for _, s in layers]
    print(f"\nexperts actually used during decode (union of top-8 over "
          f"{args.max_tokens} tokens): mean {np.mean(union):.1f}/{n_expert}, "
          f"min {min(union)}, max {max(union)}")


if __name__ == "__main__":
    main()

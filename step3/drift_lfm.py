"""Step 3 / Phase 0 (LFM2.5-8B-A1B) — measure routing drift: how well does a
route-once expert selection made from *prefill* router stats cover the experts
the model actually uses during *decode*?

Same methodology as step2/drift_ar.py, ported to mlx-lm's lfm2_moe. Patches
Lfm2MoeSparseMoeBlock to log, per MoE layer:
  - prefill: full-softmax router mass per expert (the route-once signal)
  - decode:  full-softmax mass + per-token top-k pick counts (ground truth)

Reports, per keep size: pick coverage (fraction of decode top-4 picks that
land inside the prefill-selected set), decode mass coverage, decode expert
union, plus tok/s and peak memory.

Usage:
    .venv/bin/python step3/drift_lfm.py --model <path-to-MLX-model> \
        [--prompt "..."] [--max-tokens 200] [--all-prompts] [--json out.json]

--all-prompts runs every prompt in step3/eval_prompts.json and reports
per-prompt + aggregate numbers (the findings.md table).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.lfm2_moe import Lfm2MoeSparseMoeBlock

DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/LFM2.5-8B-A1B-MLX-8bit"
KEEPS = (4, 8, 16, 32)

stats = {}        # id(block) -> {"prefill": np[E], "dec_mass": np[E], "dec_cnt": np[E]}
block_layer = {}  # id(block) -> layer index

_orig_call = Lfm2MoeSparseMoeBlock.__call__


def patched_call(self, x):
    probs = mx.softmax(self.gate(x).astype(mx.float32), axis=-1)
    probs = probs.reshape(-1, self.num_experts)
    mass = np.array(probs.sum(0))          # unbiased: sums to 1 per token
    # selection uses the bias-shifted gates, exactly like the model does
    sel_scores = probs + self.expert_bias if self.use_expert_bias else probs
    k = self.top_k
    top = mx.argpartition(sel_scores, kth=-k, axis=-1)[..., -k:]
    cnt = np.bincount(np.array(top).reshape(-1), minlength=self.num_experts)

    s = stats.setdefault(id(self), {
        "prefill": np.zeros(self.num_experts),
        "dec_mass": np.zeros(self.num_experts),
        "dec_cnt": np.zeros(self.num_experts, dtype=np.int64),
    })
    if probs.shape[0] > 1:           # prefill (or chunked prompt processing)
        s["prefill"] += mass
    else:                            # decode: one token at a time
        s["dec_mass"] += mass
        s["dec_cnt"] += cnt
    return _orig_call(self, x)


def run_prompt(model, tokenizer, user_prompt, max_tokens, verbose):
    stats.clear()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        add_generation_prompt=True, tokenize=False)
    mx.reset_peak_memory()
    t0 = time.time()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                    verbose=verbose)
    dt = time.time() - t0
    peak_gb = mx.get_peak_memory() / 1e9

    layers = sorted((block_layer[i], s) for i, s in stats.items() if i in block_layer)
    n_expert = len(layers[0][1]["prefill"])
    n_dec = int(layers[0][1]["dec_cnt"].sum() // 4) if layers else 0

    res = {"prompt": user_prompt, "n_experts": n_expert,
           "n_moe_layers": len(layers), "time_s": round(dt, 1),
           "decode_tokens": n_dec, "peak_gb": round(peak_gb, 2),
           "coverage": {}, "text": text}
    for keep in KEEPS:
        pc, dm, pm = [], [], []
        for il, s in layers:
            sel = np.argsort(-s["prefill"])[:keep]
            m = np.zeros(n_expert, bool); m[sel] = True
            pc.append(s["dec_cnt"][m].sum() / max(s["dec_cnt"].sum(), 1))
            dm.append(s["dec_mass"][m].sum() / max(s["dec_mass"].sum(), 1e-9))
            pm.append(s["prefill"][m].sum() / max(s["prefill"].sum(), 1e-9))
        res["coverage"][keep] = {
            "pick": round(float(np.mean(pc)), 3),
            "dec_mass": round(float(np.mean(dm)), 3),
            "prefill_mass": round(float(np.mean(pm)), 3),
        }
    union = [int((s["dec_cnt"] > 0).sum()) for _, s in layers]
    res["decode_union"] = {"mean": round(float(np.mean(union)), 1),
                           "min": int(min(union)), "max": int(max(union))}
    res["per_layer_union"] = union
    return res


def print_result(res):
    print(f"\nMoE layers: {res['n_moe_layers']}, experts: {res['n_experts']}, "
          f"decode tokens: {res['decode_tokens']}, peak {res['peak_gb']} GB")
    print(f"{'keep':>6} {'pick-cov':>9} {'dec-mass-cov':>13} {'prefill-mass-cov':>17}")
    for keep, c in res["coverage"].items():
        print(f"{keep:>4}/{res['n_experts']} {c['pick']:>9.3f} "
              f"{c['dec_mass']:>13.3f} {c['prefill_mass']:>17.3f}")
    u = res["decode_union"]
    print(f"decode expert union: mean {u['mean']}/{res['n_experts']}, "
          f"min {u['min']}, max {u['max']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="Write a Python function that checks if a string is a palindrome.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--all-prompts", action="store_true",
                    help="run every prompt in step3/eval_prompts.json")
    ap.add_argument("--json", type=Path, help="write full results to this file")
    args = ap.parse_args()

    Lfm2MoeSparseMoeBlock.__call__ = patched_call
    model, tokenizer = load(args.model)

    for il, layer in enumerate(model.model.layers):
        if isinstance(layer.feed_forward, Lfm2MoeSparseMoeBlock):
            block_layer[id(layer.feed_forward)] = il
    print(f"model: {args.model.name} — {len(block_layer)} MoE layers "
          f"(of {len(model.model.layers)})")

    if args.all_prompts:
        prompts = json.loads((Path(__file__).parent / "eval_prompts.json").read_text())
        prompts = [p["prompt"] for p in prompts]
    else:
        prompts = [args.prompt]

    results = []
    for p in prompts:
        print(f"\n=== {p[:80]}")
        res = run_prompt(model, tokenizer, p, args.max_tokens,
                         verbose=not args.all_prompts)
        print_result(res)
        results.append(res)

    if len(results) > 1:
        n_expert = results[0]["n_experts"]
        print(f"\n===== aggregate over {len(results)} prompts =====")
        print(f"{'keep':>6} {'pick-cov':>9} {'dec-mass-cov':>13}")
        for keep in KEEPS:
            pc = np.mean([r["coverage"][keep]["pick"] for r in results])
            dm = np.mean([r["coverage"][keep]["dec_mass"] for r in results])
            print(f"{keep:>4}/{n_expert} {pc:>9.3f} {dm:>13.3f}")
        um = np.mean([r["decode_union"]["mean"] for r in results])
        print(f"decode expert union mean: {um:.1f}/{n_expert}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

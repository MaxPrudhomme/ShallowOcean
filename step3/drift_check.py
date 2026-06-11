#!/usr/bin/env python3
"""Prefill→decode routing-drift measurement on the CUDA box.

Port of the Mac-side drift_lfm.py logic: for each eval prompt, route the
prompt (prefill), pick top-k experts per layer by prefill mass, then decode
N tokens and measure how much of the decode routing lands inside that keep
set. Reports pick coverage / mass coverage at several k plus the decode
expert-union size — the findings.md Phase 0 table, reproduced here so the
same harness can re-measure after freeze-aware training (--adapter).

Usage:
    python step3/drift_check.py [--adapter runs/k8] [--limit 20]
        [--max-new 200] [--keep 4 8 16] [--out step3/drift_results_cuda.json]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from freeze_utils import (
    RouterLogger,
    load_adapter_compat,
    load_expert_lora,
    load_quantized_model,
    load_tokenizer,
    moe_blocks,
    set_keep_masks,
)

REPO = Path(__file__).resolve().parent.parent


def normalized_mass(logits: torch.Tensor) -> torch.Tensor:
    w = logits.sigmoid()
    return w / w.sum(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="peft adapter dir to load on top of the base model")
    ap.add_argument("--prompts", default=str(REPO / "step3" / "eval_prompts.json"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--keep", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--out", default=str(REPO / "step3" / "drift_results_cuda.json"))
    args = ap.parse_args()

    tok = load_tokenizer()
    model = load_quantized_model()
    if args.adapter:
        model = load_adapter_compat(model, args.adapter)
        if load_expert_lora(model, args.adapter):
            print(f"loaded expert LoRA from {args.adapter}")
    model.eval()
    set_keep_masks(model, None)

    prompts = json.loads(Path(args.prompts).read_text())[: args.limit]
    blocks = moe_blocks(model)
    blocks_idx = [idx for idx, _ in blocks]
    bias = {idx: block.expert_bias.detach().float().cpu() for idx, block in blocks}
    num_experts = bias[blocks_idx[0]].numel()
    top_k = model.config.num_experts_per_tok

    results = []
    for i, item in enumerate(prompts):
        msgs = [{"role": "user", "content": item["prompt"]}]
        input_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False
        ).to(model.device)
        n_prompt = input_ids.shape[1]

        t0 = time.time()
        with RouterLogger(model) as log, torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        dt = time.time() - t0
        n_new = out.shape[1] - n_prompt

        per_prompt = {"category": item["category"], "n_new": n_new, "tok_s": n_new / dt}
        layer_stats = []
        for idx in blocks_idx:
            logits = torch.cat(log.logits[idx], dim=0)  # [n_prompt + n_decode, E]
            prefill, decode = logits[:n_prompt], logits[n_prompt:]
            if decode.numel() == 0:
                continue
            prefill_mass = normalized_mass(prefill).sum(dim=0)

            # Decode picks exactly as the model selects: sigmoid + expert_bias.
            scores = decode.sigmoid() + bias[idx]
            picks = torch.topk(scores, k=top_k, dim=-1).indices
            decode_mass = normalized_mass(decode).sum(dim=0)
            union = picks.flatten().unique().numel()

            stats = {"layer": idx, "union": union}
            for k in args.keep:
                keep = torch.topk(prefill_mass, k=k).indices
                keep_set = torch.zeros(num_experts, dtype=torch.bool)
                keep_set[keep] = True
                pick_cov = keep_set[picks.flatten()].float().mean().item()
                mass_cov = (decode_mass[keep].sum() / decode_mass.sum()).item()
                stats[f"pick_cov@{k}"] = pick_cov
                stats[f"mass_cov@{k}"] = mass_cov
            layer_stats.append(stats)

        per_prompt["layers"] = layer_stats
        results.append(per_prompt)

        mean_union = sum(s["union"] for s in layer_stats) / len(layer_stats)
        covs = " ".join(
            f"pick@{k}={sum(s[f'pick_cov@{k}'] for s in layer_stats) / len(layer_stats):.2f}"
            for k in args.keep
        )
        print(
            f"[{i + 1}/{len(prompts)}] {item['category']:<10} "
            f"union={mean_union:.1f}/{num_experts} {covs} ({per_prompt['tok_s']:.1f} tok/s)"
        )

    # Aggregate table, findings.md format.
    print("\n| keep (by prefill mass) | decode pick coverage | decode mass coverage |")
    print("|---|---|---|")
    all_layers = [s for r in results for s in r["layers"]]
    for k in args.keep:
        pc = sum(s[f"pick_cov@{k}"] for s in all_layers) / len(all_layers)
        mc = sum(s[f"mass_cov@{k}"] for s in all_layers) / len(all_layers)
        print(f"| {k}/{num_experts} | {pc:.2f} | {mc:.2f} |")
    mean_union = sum(s["union"] for s in all_layers) / len(all_layers)
    print(f"\ndecode expert union: mean {mean_union:.1f}/{num_experts} per layer")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

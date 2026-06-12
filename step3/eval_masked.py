#!/usr/bin/env python3
"""Phase 2 quality eval: generate under a frozen top-k expert mask.

The deployment condition Phase 1 trained for: route the prompt (unmasked
prefill), pick the top-k experts per layer by prefill mass, apply the keep
mask (gate scores AND expert_bias, via freeze_utils' patched router), then
generate the full response masked. Writes one JSON per arm for blind A/B
against the other arms / baseline_outputs.json.

The three-arm comparison (run on the CUDA box):

    # arm 1: base model, no mask — harness quality baseline
    python step3/eval_masked.py --out step3/outputs_base.json

    # arm 2: base model + k16 mask — cost of masking without training
    python step3/eval_masked.py --k 16 --out step3/outputs_base_k16.json

    # arm 3: trained adapter + k16 mask — what Phase 1 bought
    python step3/eval_masked.py --k 16 --adapter runs/k16 \
        --out step3/outputs_k16_k16.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from freeze_utils import (
    compute_keep_masks,
    load_adapter_compat,
    load_expert_lora,
    load_quantized_model,
    load_tokenizer,
    set_keep_masks,
)

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="peft adapter dir to load on top of the base model")
    ap.add_argument("--k", type=int, help="keep top-k experts per layer (omit for unmasked)")
    ap.add_argument("--prompts", default=str(REPO / "step3" / "eval_prompts.json"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = load_tokenizer()
    model = load_quantized_model()
    if args.adapter:
        model = load_adapter_compat(model, args.adapter)
        if load_expert_lora(model, args.adapter):
            print(f"loaded expert LoRA from {args.adapter}")
    model.eval()

    prompts = json.loads(Path(args.prompts).read_text())[: args.limit]
    results = []
    for i, item in enumerate(prompts):
        msgs = [{"role": "user", "content": item["prompt"]}]
        input_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False
        ).to(model.device)
        n_prompt = input_ids.shape[1]

        if args.k is not None:
            masks = compute_keep_masks(model, input_ids, k=args.k)
            set_keep_masks(model, masks)
        else:
            set_keep_masks(model, None)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        dt = time.time() - t0
        set_keep_masks(model, None)

        n_new = out.shape[1] - n_prompt
        text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
        results.append(
            {
                "category": item["category"],
                "prompt": item["prompt"],
                "output": text,
                "n_new": n_new,
                "tok_s": n_new / dt,
            }
        )
        preview = text[:80].replace("\n", " ")
        print(f"[{i + 1}/{len(prompts)}] {item['category']:<10} {n_new} tok "
              f"({n_new / dt:.1f} tok/s) {preview!r}")

    meta = {"adapter": args.adapter, "k": args.k, "max_new": args.max_new}
    Path(args.out).write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"\nwrote {args.out}")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()

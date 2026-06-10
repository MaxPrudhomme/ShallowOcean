"""Step 3 / Phase 0.3 — quality baseline: generate the fixed eval set with the
*unmodified* model. Kept for blind A/B grading in Phase 2.

Usage:
    .venv/bin/python step3/gen_baseline.py --model <path-to-MLX-model> \
        [--max-tokens 512] [--out step3/baseline_outputs.json]
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import load, generate

DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/LFM2.5-8B-A1B-MLX-8bit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "baseline_outputs.json")
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    prompts = json.loads((Path(__file__).parent / "eval_prompts.json").read_text())

    results = []
    for p in prompts:
        text = generate(
            model, tokenizer,
            prompt=tokenizer.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}],
                add_generation_prompt=True, tokenize=False),
            max_tokens=args.max_tokens)
        print(f"[{p['id']}] {len(text)} chars")
        results.append({**p, "model": str(args.model.name),
                        "max_tokens": args.max_tokens, "output": text})

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out} ({len(results)} outputs, "
          f"generated {time.strftime('%Y-%m-%d')})")


if __name__ == "__main__":
    main()

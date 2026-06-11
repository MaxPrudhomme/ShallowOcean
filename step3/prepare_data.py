#!/usr/bin/env python3
"""Download and cache the SFT mixture on local disk.

Uses HuggingFaceTB/smoltalk (general SFT mixture, plan.md Phase 1). The full
"all" config is ~1.1M conversations; we keep a length-filtered random subset
(default 200k, plan's upper bound) and save it with save_to_disk so training
never touches the network. Re-running is a no-op if the cache exists.

Usage:
    python step3/prepare_data.py [--max-examples 200000] [--seed 0]
"""

import argparse
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "smoltalk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    if DATA_DIR.exists() and not args.force:
        print(f"already cached at {DATA_DIR} — use --force to rebuild")
        return

    print("downloading HuggingFaceTB/smoltalk (config 'all', split 'train')...")
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    print(f"loaded {len(ds)} conversations")

    # Drop degenerate or very long conversations: freeze-aware training wants
    # a real prompt (for routing) and a real response, and the 3080 caps
    # sequence length anyway.
    def ok(ex):
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] not in ("user", "system"):
            return False
        total_chars = sum(len(m["content"]) for m in msgs)
        return 64 <= total_chars <= 8000

    ds = ds.filter(ok, num_proc=8)
    print(f"{len(ds)} after length filter")

    if len(ds) > args.max_examples:
        ds = ds.shuffle(seed=args.seed).select(range(args.max_examples))

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATA_DIR))
    print(f"saved {len(ds)} examples to {DATA_DIR}")


if __name__ == "__main__":
    main()

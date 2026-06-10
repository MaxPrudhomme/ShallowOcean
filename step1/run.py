"""One-shot route-once pipeline: route -> prune -> decode for a single prompt.

Usage:
    .venv/bin/python step1/run.py "Write a haiku about the ocean."
    .venv/bin/python step1/run.py --keep 0.375 --visual "..."
    .venv/bin/python step1/run.py --reuse "..."   # skip route+prune, reuse last pruned GGUF

Routing stats and the pruned GGUF are cached in /tmp/route-once/ keyed by
prompt + keep fraction, so re-running the same prompt goes straight to decode.
"""

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv/bin/python"
CLI = Path("/tmp/llamacpp-src/build/bin/llama-diffusion-cli")
WORK = Path("/tmp/route-once")
LAST = WORK / "last.gguf"  # symlink to the most recent pruned model


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--visual", action="store_true", help="live canvas animation")
    ap.add_argument("--reuse", action="store_true",
                    help="decode against the last pruned GGUF (skip route+prune)")
    args = ap.parse_args()

    if not CLI.exists():
        sys.exit(f"{CLI} not found — rebuild llama.cpp PR #24423 (see step1/README.md)")

    WORK.mkdir(exist_ok=True)
    if args.reuse:
        gguf = LAST.resolve()
        if not gguf.exists():
            sys.exit("no previous pruned GGUF to reuse — run once without --reuse")
        print(f"reusing {gguf}", file=sys.stderr)
    else:
        key = hashlib.sha1(f"{args.prompt}|{args.keep}".encode()).hexdigest()[:12]
        experts = WORK / f"experts-{key}.json"
        gguf = WORK / f"pruned-{key}.gguf"
        if not gguf.exists():
            t0 = time.time()
            run([PYTHON, REPO / "step1/route_stats.py", "--prompt", args.prompt,
                 "--keep", str(args.keep), "--out", experts])
            run([PYTHON, REPO / "step1/prune_gguf.py", "--experts", experts, "--out", gguf])
            print(f"route+prune: {time.time()-t0:.0f}s", file=sys.stderr)
        else:
            print(f"cached: {gguf}", file=sys.stderr)
        LAST.unlink(missing_ok=True)
        LAST.symlink_to(gguf)

    cmd = [CLI, "-m", gguf, "-p", args.prompt, "-ub", "512",
           "--diffusion-steps", str(args.steps), "--temp", str(args.temp)]
    if args.visual:
        cmd.append("--diffusion-visual")
    run(cmd)


if __name__ == "__main__":
    main()

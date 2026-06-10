# Instruction-Following Pruning demo on Gemma 4 E4B

A working demo of the core systems idea in [arXiv:2501.02086](../2501.02086.md)
("Instruction-Following Pruning for Large Language Models", Apple): instead of
running a MoE-style router at every token, **route once during prompt
processing**, load only the selected FFN parameters into memory, and decode
with everything else left on SSD. The selection stays fixed for the whole
response, so there is no per-token weight traffic.

## Why this isn't a literal MoE port

Two facts discovered along the way:

- **The paper is not a classic MoE.** IFPruning selects rows/columns of every
  dense FFN per prompt (a fine-grained, instruction-dependent pruning mask),
  which is exactly the "router once at prefill" behavior.
- **The checkpoint in `models/gemma-4-e4b` is the dense E4B variant** —
  `enable_moe_block: false`, `num_experts: null`, no router/expert tensors.

So the demo treats each 64-channel quantization group of every FFN as an
"expert": 42 layers × 160 groups = 6,720 routable units (~3.5 GB of the
checkpoint, all 8-bit affine quantized).

## How it works (`ifprune_demo.py`)

1. **Route (once, at prompt processing).** The prompt is run through the
   mmap'd model while each FFN records per-channel activation mass. A channel
   group's score is `prompt activation mass × L2 norm of its down_proj
   column` (how hard it hits the residual stream). Groups compete for a
   **global budget across layers** (per-layer normalized, with a per-layer
   floor), so sensitive layers automatically keep more.
2. **Load selected experts.** The chosen groups are sliced directly out of
   the quantized safetensors at group granularity — `gate/up` by rows,
   `down_proj` by packed-column blocks — and only those bytes are ever
   evaluated. Unselected weights are never materialized: they stay on SSD
   behind the mmap.
3. **Decode** with the pruned FFNs. Smaller matmuls → faster decoding.

The paper trains a small sparsity predictor jointly with the LLM; we don't
have one, so the model's own prompt activations stand in as the router. That
is the main quality limiter (see below).

## Results (M-series Mac, 150-token generations)

| keep | FFN resident | on SSD | decode speed | quality |
|------|-------------|--------|--------------|---------|
| 1.0 (full) | 3509 MB | 0 | ~48 tok/s | baseline |
| 0.75 | 2632 MB | 877 MB | ~55 tok/s | near-identical to baseline |
| 0.70 (default) | 2457 MB | 1053 MB | ~62 tok/s | fluent, minor factual drift |
| 0.625 | 2193 MB | 1316 MB | ~67 tok/s | coherent, repetition on short prompts |
| 0.50 | 1755 MB | 1755 MB | ~76 tok/s | degenerate |

The paper reaches ~33% activation with no quality loss because the predictor
and LLM are jointly fine-tuned; with a purely heuristic prompt-time router on
a quantized model, ~70% is the comfortable floor. The routing pass itself
takes <1 s and the selected experts load from SSD in ~1 s.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install mlx-lm numpy
.venv/bin/python demo/ifprune_demo.py \
    --prompt "Write a function that checks for palindromes." \
    --keep 0.7 --max-tokens 200
```

Flags: `--keep` (global fraction of channel groups kept), `--floor` (minimum
per-layer fraction, default 0.5), `--skip-baseline` (skip the full-model
comparison run), `--model`, `--max-tokens`.

Note: mlx-lm 0.31.3's gemma4 module rejects this checkpoint because it stores
redundant k/v projections for the 18 KV-shared layers; the demo patches
`sanitize` to drop them.

# Route-once MoE offload demo on gpt-oss-20b

A working demo of the core systems idea in [arXiv:2501.02086](../2501.02086.md)
("Instruction-Following Pruning for Large Language Models", Apple): instead of
running the MoE router at every decoding step, **route once during prompt
processing**, load only the experts the request needs into memory, and decode
with everything else left on SSD. The selection stays fixed for the whole
response, so there is no per-token expert traffic and the unused ~half of the
model never touches RAM.

Model: `models/gpt-oss-20b` (mlx-community MXFP4-Q8) — a real MoE with
24 layers x 32 experts, top-4 per token, ~3.6B active parameters. The stacked
expert tensors are ~10.2 GB of the ~12 GB checkpoint.

## How it works (`moe_route_once_demo.py`)

1. **Route (once, at prompt processing).** The prompt runs through the
   mmap'd model while each layer accumulates its router's softmax mass per
   expert. The top `--keep` fraction of experts per layer is selected
   (default 16/32, which covers ~70% of the prompt's router mass).
2. **Load selected experts.** Expert tensors are stacked with the expert
   axis first, so the kept experts are sliced straight out of the quantized
   safetensors. Only those bytes are ever evaluated — unselected experts stay
   on SSD behind the mmap and are never materialized.
3. **Decode.** The (tiny) router still runs per token, but its logits are
   gathered down to the loaded experts, so it picks top-4 among them.

## Results (M-series Mac, ~120-token generations, 85-token prompt)

| experts kept | resident | on SSD | router mass covered | quality |
|------|---------|--------|------|---------|
| 32/32 (full) | 10166 MB | 0 | 100% | baseline |
| 16/32 (default) | 5083 MB | 5083 MB | ~70% | coherent, on-task; occasional repetition on open-ended prose |
| 12/32 | 3812 MB | 6354 MB | ~58% | fluent but loops |
| 8/32 | 2541 MB | 7624 MB | ~43% | degenerate loops |

Decode speed is unchanged (~76 tok/s) since per-token compute is still top-4
experts either way — the win is memory residency: at the default setting half
the model (5 GB) never leaves SSD. Routing takes ~0.3 s; loading the selected
experts ~4 s. The paper closes the remaining quality gap by jointly
fine-tuning the selection with the LLM; here the pretrained router's prompt
statistics are used as-is.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install mlx-lm numpy
.venv/bin/python demo/moe_route_once_demo.py \
    --prompt "Write a function that checks for palindromes." \
    --keep 0.5 --max-tokens 200
```

Flags: `--keep` (fraction of experts kept per layer), `--skip-baseline`
(skip the full-model comparison run), `--model`, `--max-tokens`.

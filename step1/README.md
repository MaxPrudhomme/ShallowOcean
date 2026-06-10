# Route-once expert offload on diffusiongemma-26B-A4B (plan.md steps 0–1)

Target: `~/.lmstudio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`
(16.8 GB, 30 layers × 128 experts, top-8) on a 24 GB Mac.

## Prerequisite: llama.cpp with diffusion-gemma support

Upstream llama.cpp does **not** support the `diffusion-gemma` arch (only
Dream/LLaDA/RND1). Two open PRs add it; **#24423** matches this GGUF's tensor
naming exactly (#24427 expects `self_cond_norm` instead of
`self_cond_pre_norm` and doesn't load the 30 `enc_layer_output_scale`
tensors, so the file fails to load there). LM Studio 2.21.0 also fails to
load the model.

```bash
git clone https://github.com/ggml-org/llama.cpp /tmp/llamacpp-src
cd /tmp/llamacpp-src
git fetch origin pull/24423/head:pr-24423 && git checkout pr-24423
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j8 --target llama-diffusion-cli llama-tokenize
```

## Step 0 — baseline (full 128-expert model)

Prompt: "Write a Python function that checks whether a string is a
palindrome, with a short explanation."; entropy-bound denoising, 256-token
canvas, `--temp 0`, `-ub 512`.

| config | result |
|---|---|
| full GPU offload | **OOM** at denoising step 0 (Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`) |
| `-ngl 0` CPU + mmap | runs, coherent; **4210 ms/step, 130.5 s total (31 steps)**, 18.5 GB max RSS |

## Step 1 — route once, prune the GGUF, decode with stock llama.cpp

```bash
.venv/bin/python step1/route_stats.py --prompt "..." --keep 0.5 --out experts.json   # ~60 s
.venv/bin/python step1/prune_gguf.py --experts experts.json --out /tmp/dg-pruned-64.gguf  # ~22 s
/tmp/llamacpp-src/build/bin/llama-diffusion-cli -m /tmp/dg-pruned-64.gguf -p "..." -ub 512 --temp 0
```

`route_stats.py` replicates PR #24423's UNIFIED forward (prompt causal +
bidirectional 256-mask canvas, zero self-conditioning = exactly the first
denoising step) in numpy, streaming dequantized layers from the mmap'd GGUF,
and accumulates each layer's router softmax mass per expert. `prune_gguf.py`
slices the kept experts out of `ffn_gate_up_exps`/`ffn_down_exps` (raw
quantized block copy), slices `ffn_gate_inp.weight` rows and
`ffn_down_exps.scale` to match, and sets `expert_count = 64`.

### Results (64/128 experts, same prompt)

| metric | baseline (CPU mmap) | pruned 64/128 (full GPU) |
|---|---|---|
| model size | 16.8 GB | **9.2 GB** |
| time per step | 4210 ms | **764 ms** (5.5×) |
| wall time | 130.5 s | **26.0 s** |
| max RSS | 18.5 GB | 14.0 GB |
| output | correct palindrome answer + reasoning | same structure/content, coherent and on-task |

Mean canvas router-mass coverage of the kept 64 experts: **0.705** per layer
(range ~0.65–0.78) — notably *not* higher than gpt-oss-20b at 16/32, i.e.
the fine-grained 128-expert routing is still fairly load-balanced on the
first denoising step. Quality nonetheless held up at keep=0.5 on this prompt.

One-off costs per prompt/task: ~60 s routing + ~22 s GGUF rewrite; amortize
by reusing a pruned file per task family (plan.md's calibration idea).

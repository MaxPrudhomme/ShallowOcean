# Plan: route-once expert offload for diffusiongemma-26B-A4B on a 24 GB Mac

Goal: force the diffusion MoE at
`~/.lmstudio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`
to run with half its experts in memory, using the route-once idea from
arXiv:2501.02086 (already demoed on gpt-oss-20b in `demo/`).

## Why this model is a good candidate

| fact | value |
|---|---|
| architecture | `diffusion-gemma`, 30 layers, non-causal, canvas = 256 tokens |
| experts | **128 per layer, 8 active** (fine-grained) |
| expert tensor bytes | 15.1 GB of the 16.8 GB file |
| non-expert bytes (attn, embeddings, routers) | 1.66 GB |
| machine | 24 GB RAM → full model doesn't fit comfortably |
| at 64/128 experts | 7.6 + 1.7 ≈ **9.3 GB resident** → fits easily |

Fine-grained 128-expert routing is much more skewed/specialized than
gpt-oss's 32 load-balanced experts, so keeping 64/128 should cover far more
router mass than 16/32 did there. (Lesson from the gpt-oss demo: top-k per
token ≠ working set; the kept set must cover the whole response's routing.)

## Step 0 — baseline sanity test (free)

Run the model as-is with `llama-diffusion-cli` (llama.cpp) with mmap enabled.
macOS pages experts in on demand, so it may run but thrash: every denoising
step processes the full 256-token canvas and touches many experts. Record
wall time / residency as the baseline.

## Step 1 — per-prompt GGUF surgery (the main deliverable)

Force half the experts with no llama.cpp code changes:

1. **Route once in Python.** Read the GGUF with `gguf-py` (it dequantizes
   Q4_K), run the prompt through the model layer-by-layer in MLX (streaming,
   never holding more than ~a layer), accumulate each layer's router softmax
   mass per expert. Diffusion wrinkle: there is no prefill/decode split —
   collect stats on the **first denoising step** (prompt + masked canvas),
   then freeze the expert set for the remaining steps.
2. **Write a pruned GGUF.** Slice every `blk.N.ffn_*_exps` tensor to the
   selected 64 experts (raw quantized block copy, no requantization), slice
   the router `ffn_gate_inp` rows to match, set `expert_count = 64` in the
   metadata. Output: a self-consistent ~9 GB model.
3. **Decode with stock llama.cpp**, which sees an ordinary 64-expert model
   that fits in RAM.

Cost: ~30 s of GGUF rewriting per prompt. Amortize by building per-*task*
expert sets from a few calibration prompts and reusing the pruned file.

Why this split: llama.cpp's battle-tested forward pass does the actual
generation; the hand-written Python forward only produces routing
*statistics*, which tolerate small numerical differences.

## Step 2 — full MLX port (only if Step 1 falls short)

Port diffusion-gemma to MLX like the gpt-oss demo (live expert slicing, no
GGUF rewrite). Requires reimplementing the diffusion sampling loop (iterative
canvas unmasking) and arch quirks from llama.cpp source: per-layer sliding
window pattern, dual RoPE bases (1e6 full / 1e4 SWA), mixed KV head counts
(8/2), `enc_layer_output_scale`, logit softcap 30. Higher effort, silent-
garbage risk if any detail is wrong.

## Risks / open questions

- Does the installed llama.cpp build support the `diffusion-gemma` arch?
  (unsloth shipped the GGUF, so upstream support presumably exists — verify.)
- Quality at 64/128 is unmeasured; sweep keep fraction like the gpt-oss demo
  (it stayed coherent at 50% experts, degenerated at 25%).
- Router stats from denoising step 1 may shift in later steps as the canvas
  unmasks; if quality suffers, collect stats over the first few steps instead.

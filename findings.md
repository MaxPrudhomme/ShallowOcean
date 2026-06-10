# Findings: route-once expert offload on diffusiongemma-26B-A4B

Based on the Step 0/1 runs (palindrome prompt, "Write a defense of JS against
TS"), plan and scripts in `plan.md` / `step1/`.

## It works

The full pipeline (route in numpy → prune the GGUF to 64/128 experts →
decode with stock llama.cpp PR #24423) is reliable and fast:

| metric | full model (CPU mmap) | pruned 64/128 (full GPU) |
|---|---|---|
| feasibility | GPU OOMs; CPU-only | fits on GPU (9.2 GB model, ~14 GB RSS) |
| time per step | 4210 ms | **748–764 ms** (~5.5×) |
| one-off cost per prompt | — | ~60 s routing + ~21 s GGUF rewrite |

Quality at keep=0.5 stays coherent and on-task across both prompts. The JS
run shows mild degradation at the edges — occasional broken phrases ("it has
it's power", "JS de-coupled with without-scripting built-in features") that
read like routing misses, not collapse. A same-prompt full-model comparison
hasn't been run, so how much of this is pruning vs. the model itself is open.

## The big surprise: expert selection is essentially prompt-independent

Two unrelated prompts (Python palindrome code vs. a JS-vs-TS essay) produced:

- nearly identical per-layer coverage profiles (mean 0.705 vs 0.704; layer 0:
  0.757 vs 0.757, layer 20: 0.776 vs 0.770, …)
- **93.9% mean overlap** between the kept-64 expert sets (per-layer range
  86–98%).

Interpretation: on the first denoising step the canvas is 256 identical mask
tokens — ~92% of the rows the router sees carry no prompt-specific content,
only position. The selection is therefore dominated by the model's generic
"denoise a masked canvas" working set, not by the instruction. Two
consequences:

1. **Good for amortization**: one pruned GGUF is effectively a universal
   artifact, not a per-prompt one. The ~80 s route+prune cost can be paid
   once; per-prompt routing buys almost nothing over a single calibration.
2. **Bad for the paper's premise** (instruction-specific pruning): the
   instruction signal route_stats.py captures is weak. If prompt-conditioned
   selection is the goal, stats should be collected from *later* denoising
   steps (partially-unmasked canvas, where rows actually carry content) —
   plan.md's listed risk, confirmed from the other direction.

## Routing is less skewed than hoped

plan.md hypothesized fine-grained 128-expert routing would be more
specialized than gpt-oss's 32 load-balanced experts. It isn't: top-64/128
covers only ~70% of router mass (~ the same as gpt-oss at 16/32), flat across
layers (0.65–0.78). The model spreads mass widely; quality at 50% keep
survives anyway, presumably because top-8 renormalization keeps the selected
experts' relative weights intact and misses are individually small.

## Practical notes

- Generation hit the 48-step cap mid-sentence on the JS prompt (truncated at
  "Point 2"); the 256-token canvas with one block simply ran out — use
  `--diffusion-blocks N` (block-autoregressive) for longer answers.
- The model's output begins with a `<|channel>thought` reasoning section —
  budget canvas space for it.
- Upstream llama.cpp does not support `diffusion-gemma`; PR #24423 matches
  this GGUF's tensor naming (#24427 and LM Studio 2.21.0 both fail to load
  it). Build lives in `/tmp/llamacpp-src` — wiped on reboot, rebuild per
  `step1/README.md`.

## Keep-fraction sweep: the cliff is between 32 and 64, and pruning doesn't buy speed

Sweep on the palindrome prompt (one routing pass reused via the new
`route_stats.py --from-mass`; experiment of 2026-06-10):

| keep | model size | router-mass coverage | ms/step | output |
|---|---|---|---|---|
| 8/128 | 2.6 GB | 0.183 | 671 | collapse — echoes the prompt, no code |
| 32/128 | 5.4 GB | 0.464 | 719 | degenerate — broken repeating phrases |
| 64/128 | 9.2 GB | 0.705 | 748–764 | coherent (earlier runs) |

Two conclusions:

1. **"Freeze the top-8" (route-once at the active-expert count) does not
   work zero-shot.** 8 experts/layer cover 18% of router mass; the model
   collapses. The Apple paper (2501.02086) gets away with prompt-frozen 3B/8B
   because its mask predictor is *trained jointly with the model*; the
   selection problem, not the freezing, is what their training solves.
2. **Pruning experts barely changes per-step time** (671 vs 749 ms at 3.5×
   fewer resident experts). Per-token FLOPs are dominated by the top-8
   *active* experts, which stay top-8 regardless of how many are resident.
   Expert pruning is a **memory** lever, not a speed lever, once the model
   already fits on GPU.

So the ~2 min/prompt cost is not the decode (35 s for 48 steps): it is the
~60 s numpy routing + ~21 s GGUF rewrite paid per prompt. Given the 94%
prompt-overlap result above, the fix is free: treat the k64 pruned GGUF as a
universal artifact and decode every prompt against it (`run.py --reuse`) —
zero per-prompt overhead, ~0.7 s/step.

## Step-count is the real latency knob

On the cached k64 model (palindrome prompt): `--diffusion-eb-max-steps 24`
gives 18.7 s total (780 ms/step) with only mild extra garbling in the final
code block — latency scales linearly with steps, so the step cap is a clean
speed/quality dial. `--diffusion-blocks 4` changed nothing here (the
entropy-bound decoder still ran 48 steps over 1 block; blocks only matter
once generation exceeds one 256-token canvas). ms/step is pinned at
~700–780 regardless of expert count, block count, or step cap — that is the
cost of one batch-256 forward of the 26B graph on this machine, and it is
the floor of this implementation.

Pushing lower breaks: at 20 and 16 steps the tail of the answer (the second
code block) collapses into repeated fragments — **24 steps is the practical
floor** on this prompt; below it the canvas doesn't converge.

Best current recipe: `run.py --reuse` (universal k64 GGUF, no per-prompt
route/prune) + `--diffusion-eb-max-steps 24..32` → **~19–25 s per ~250-token
response (~10–13 tok/s effective)**, down from ~2 min.

## Step 2 probe: route-once does NOT transfer to the AR model (gemma-4-26B-A4B)

`step2/drift_ar.py` (new) patches mlx-lm's gemma4 Router to log prefill vs
decode routing on the MLX-4bit QAT build, palindrome prompt, 200 tokens:

| keep (by prefill mass) | decode pick coverage | decode mass coverage |
|---|---|---|
| 8/128 | 0.17 | 0.11 |
| 32/128 | 0.49 | 0.36 |
| 64/128 | 0.69 | 0.60 |
| 96/128 | 0.86 | 0.81 |

Union of experts actually used during decode: **mean 94.6/128 per layer**
(min 73, max 111) over just 200 tokens. Token-level routing genuinely spreads
across ~3/4 of all experts within one short response — there is no small
frozen working set to find, zero-shot. Route-once worked on diffusiongemma
*because* its canvas routing was static and position-driven, not despite it.

Also: the full model already runs fine as-is on this machine — 14.4 tok/s
decode, 14.5 GB peak in MLX 4-bit. There is no memory problem to solve and
pruning would only cost quality. Conclusion: for AR MoE, prompt-frozen expert
selection at low keep requires Apple-style joint training (2501.02086); the
zero-shot version is a diffusion-specific trick.

## Suggested next experiments

1. ~~Keep-fraction sweep~~ — done, see above; cliff is between 32 and 64.
2. **Probe keep=40/48/56** to find the smallest coherent set (cheap now:
   `route_stats.py --from-mass ... --keep K` + prune + decode, no routing).
3. **Full-model vs pruned A/B on the same prompt** (CPU baseline) to
   attribute the small grammar glitches.
4. **Union-of-actually-used experts**: instrument a decode to log per-token
   top-8 across all steps; the union working set may be far smaller than a
   mass-coverage threshold suggests and is the right zero-shot analogue of
   Apple's trained selection.
5. **Decode-speed levers** (the actual ms/step): fewer steps via entropy
   bound tuning, smaller canvas / `--diffusion-blocks`, and checking whether
   the MoE matmul batches well on Metal.

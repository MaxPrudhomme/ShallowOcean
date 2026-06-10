# Step 3 — freeze-aware fine-tuning of LFM2.5-8B-A1B

See `plan.md`. Model: LFM2.5-8B-A1B (8.3B total / 1.5B active, 32 experts,
top-4, first layers dense).

## Phase 0 — baselines (Mac, MLX)

Drift measurement (prefill→decode pick coverage at k=4/8/16/32, decode expert
union, tok/s, peak memory):

```sh
# single prompt, verbose generation
.venv/bin/python step3/drift_lfm.py --model ~/.lmstudio/models/<LFM2.5-8B-A1B-MLX>

# the full findings.md table: all 20 eval prompts + aggregate
.venv/bin/python step3/drift_lfm.py --model ... --all-prompts --json step3/drift_results.json
```

Quality baseline for Phase 2 blind A/B grading:

```sh
.venv/bin/python step3/gen_baseline.py --model ... --out step3/baseline_outputs.json
```

`eval_prompts.json`: 20 fixed prompts, 5 each of code / reasoning / chat /
knowledge.

## Phase 1 — freeze-aware SFT (3080 box, CUDA)

```sh
pip install torch transformers peft bitsandbytes datasets
python step3/train_freeze.py --model LiquidAI/LFM2.5-8B-A1B --load-4bit \
    --stages "16:2000,8:2000,6:2000,4:4000" --out runs/freeze8b
```

What it does per example: no-grad prompt prefill → accumulate router softmax
mass per MoE layer → top-k mask (additive −1e9 on gate logits) → teacher-force
the full example under the mask → LM loss on the response. Batch size 1 with
`--grad-accum` (default 16) because masks are per-example.

- `--stages k:steps,...` is the k anneal; a checkpoint is saved after each
  stage (`stage_k16/`, `stage_k8/`, ...).
- Router gate projections are trained **fully** (`modules_to_save=["gate"]`),
  LoRA everywhere else; the router gets its own lower LR (`--router-lr`).
- `--distill-alpha 0.5` adds a KL term against the *base* model (adapters
  disabled, unmasked) on response tokens.
- `train_log.jsonl` logs loss and `prompt_mass_cov` — the prompt router mass
  inside the kept set. **Go/no-go**: if `prompt_mass_cov` doesn't climb during
  the k=16 stage, the router isn't re-concentrating; stop and rethink.

## Phase 2 — evaluation (Mac)

Re-run `drift_lfm.py` on the fused/adapted model and compare against the
Phase 0 numbers; A/B the eval set against `baseline_outputs.json`. (Frozen-
resident decode harness to be written once Phase 1 produces a checkpoint.)

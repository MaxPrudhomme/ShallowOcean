# Step 3 — freeze-aware fine-tuning of LFM2.5-8B-A1B (CUDA box)

Phase 1 of plan.md, running on the RTX 3080 (10 GB, WSL2). Phase 0
(drift baseline, eval prompts, baseline outputs) lives on the Mac in MLX;
this directory is the CUDA training side plus a CUDA drift harness so
before/after can be measured on the same box that trains.

## Setup (already done on this box)

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install "transformers>=5.0" peft bitsandbytes datasets accelerate hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download LiquidAI/LFM2.5-8B-A1B \
    --local-dir models/LFM2.5-8B-A1B          # 16 GB
.venv/bin/python step3/prepare_data.py        # caches smoltalk to data/smoltalk
```

Everything (model, dataset) is cached on local disk; nothing touches the
network after setup.

## Files

- `freeze_utils.py` — model loading + the two tricks that make this work
  on 10 GB and on this architecture:
  - **Custom 4-bit quantization.** LFM2-MoE stores experts as fused 3D
    `nn.Parameter`s, which bitsandbytes ignores (it only converts
    `nn.Linear`) — stock QLoRA would leave 7.7B params in bf16. We split
    them into per-expert `Linear4bit` modules (`QuantExperts`) and quantize
    attention/conv/dense-MLP linears too. Router gates, norms and the tied
    embedding stay bf16. ~5 GB resident.
  - **Freeze masks.** `route_tokens_to_experts` is patched to honor a
    per-block `_freeze_keep` bool[32], masking the *post-bias* selection
    scores (`sigmoid(logits) + expert_bias`) — masking gate logits alone is
    not enough because `expert_bias` is added after the sigmoid and only
    affects selection (the trap in findings.md). Kept top-4 weights are
    renormalized by the model's own `norm_topk_prob`.
- `prepare_data.py` — caches a 200k length-filtered subset of
  HuggingFaceTB/smoltalk to `data/smoltalk` (`save_to_disk`).
- `train_freeze.py` — the Phase 1 trainer (see below).
- `drift_check.py` — prefill→decode pick/mass coverage + decode expert
  union, same metrics and table format as the Phase 0 findings; takes
  `--adapter` to evaluate a trained checkpoint.
- `eval_prompts.json` — 20 prompts (5× code/reasoning/chat/knowledge).

## Training objective

Per example (batch 1, grad accumulation — keep sets differ per example):

1. prompt-only forward, no grad, unmasked → per-layer expert mass
   (per-token normalized sigmoid, summed)
2. top-k by mass → keep masks
3. teacher-force the full conversation with selection restricted to the
   keep set; CE loss on response tokens only

Trainable: LoRA r=8 on attention q/k/v/out + all expert gate_up/down
(1,408 Linear4bit modules), and the router `gate` linears trained **fully**
(`modules_to_save`) — re-concentrating routing is most of the work.
Gradient checkpointing + paged 8-bit AdamW.

## k-anneal schedule (plan.md)

```bash
.venv/bin/python step3/train_freeze.py --k 16 --steps 2000 --out runs/k16
.venv/bin/python step3/train_freeze.py --k 8  --steps 2000 --out runs/k8 --init-adapter runs/k16
.venv/bin/python step3/train_freeze.py --k 6  --steps 2000 --out runs/k6 --init-adapter runs/k8
.venv/bin/python step3/train_freeze.py --k 4  --steps 2000 --out runs/k4 --init-adapter runs/k6
```

Go/no-go signal (plan.md risks): after the k=16 stage, re-run

```bash
.venv/bin/python step3/drift_check.py --adapter runs/k16
```

and look at pick coverage at k=16 — if it hasn't moved well above the 0.63
baseline, the router isn't re-concentrating and the run isn't worth
continuing. Phase 1 success = pick coverage at k=8 going 0.37 → ≥0.95
(then Phase 2 quality A/B happens on the Mac against
`step3/baseline_outputs.json`).

## Not implemented yet

- Distillation term against the unmasked model's logits (plan.md's optional
  quality anchor) — would need a second unmasked forward per example;
  add only if quality regresses.
- OLMoE fallback path.

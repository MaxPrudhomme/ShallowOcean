# Session state — Step 3 Phase 1 (freeze-aware QLoRA on LFM2.5-8B-A1B)

Date: 2026-06-11. Machine: RTX 3080 10GB (WSL2).

## What happened this session

1. **Diagnosed slow training** (`step3/train_freeze.py --k 16`): 59s/optimizer
   step, GPU 1-2% util. Cause: `QuantExperts` held each expert as a separate
   bnb `Linear4bit`, each PEFT-wrapped → ~1,400 tiny wrapped module calls per
   forward (22 MoE layers × 32 experts × 2 projections) + a GPU→CPU sync per
   expert. Pure kernel-launch starvation.

2. **Rewrote the expert path** (`step3/freeze_utils.py`):
   - `FusedQuantExperts` replaces the ModuleList: one fused NF4 blob per
     projection per layer, dequantized once per layer call, all experts in a
     single batched `bmm` over sorted zero-padded token groups. NF4 blocks
     (64-wide) align with row lengths (2048, 1792) → numerically identical to
     per-expert quantization.
   - Expert LoRA moved inside the module as batched 3D tensors (A kaiming,
     B zeros = identity at init). PEFT now only handles attention q/k/v/out
     LoRA + fully-trained router gates (`modules_to_save`). Expert LoRA is
     saved/loaded as `expert_lora.pt` next to each adapter
     (`save_expert_lora` / `load_expert_lora` / `expert_lora_parameters`).
   - `train_freeze.py` and `drift_check.py` updated accordingly
     (`--init-adapter` and `--adapter` load expert_lora.pt automatically).

3. **Result: ~14s/step (4.2× faster)**, k16 stage ≈ 8h instead of 33h.
   Step-1 loss matches old run (1.6547 vs 1.6656) → numerics confirmed.
   nvidia-smi shows 66-94% GPU util at ~280W/320W — now bandwidth-bound
   (dequant of 7.7B params per forward), really VRAM-capacity-bound.
   (Windows Task Manager's default GPU graph under-reports CUDA; use
   `/usr/lib/wsl/lib/nvidia-smi`.)

4. **Added exact checkpoint/resume** to `train_freeze.py`:
   - Each checkpoint (step500/1000/1500 + final) now also writes
     `train_state.pt` (optimizer moments, step, dataset position).
   - `--resume runs/k16/stepN` restores everything exactly (grads are empty
     at checkpoint boundaries; data order is deterministic from seed).
     Asserts --k/--seed/--grad-accum match.
   - Also fixed the step-10 log divisor bug (first post-step-1 print
     understated loss ~10%; old "1.4055" was really ~1.56).

5. **Run status at session end**: user's k16 run (2000 steps, grad-accum 16)
   was in progress, ~step 50, loss 1.67 → 0.83 through warmup (LR 1e-4
   reached at step 50). NOTE: if that process predates the resume change it
   has no `train_state.pt` in its checkpoints — only restartable from
   scratch or usable via `--init-adapter` (weights only, fresh optimizer).

## Next actions

- When `runs/k16/step500` exists: stop training (or wait until done — VRAM
  can't fit both) and run
  `python step3/drift_check.py --adapter runs/k16/step500`.
- **The metric**: decode pick coverage at k=16. Phase 0 baseline 0.63
  (findings.md table; union 27.1/32). Good signal at step500 ≈ 0.75-0.85+.
  Stuck at ~0.65 → router gates not moving, consider gate LR.
  Phase 1 success criterion: pick cov @ k=8 going 0.37 → ≥0.95.
- Guard against routing collapse: keep sets should still differ across
  prompts; loss should stay flat-to-falling. Loss is the quality guard
  (mask is forced during training), coverage is the success metric.
- Baseline caveat: Phase 0 table was measured on the MLX 8-bit build; run
  drift_check once without --adapter on the CUDA box to confirm ~0.63
  reproduces before crediting training.
- k-anneal chain: 16 → 8 → 6 → 4 via `--init-adapter` (each stage's
  expert_lora.pt is picked up automatically).
- Old slow-run log preserved at `runs/k16/train_log.slow-baseline.jsonl`.
- Possible future speedup: batch 2 examples per micro-step (~1.5×, VRAM
  tight at 7.4GB peak; needs per-token keep masks + padding). Better
  hardware: 24GB card (3090/4090) removes 4-bit dequant entirely → 3-5×+.

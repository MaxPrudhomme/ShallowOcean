#!/usr/bin/env python3
"""Phase 1: freeze-aware QLoRA fine-tuning of LFM2.5-8B-A1B (plan.md).

Per example, two passes:
  1. Route the prompt alone (no grad, unmasked); take top-k experts per MoE
     layer by prefill mass.
  2. Teacher-force the full conversation with the router masked to that keep
     set (selection restricted, top-4 weights renormalized within it); LM
     loss on the response tokens only.

Trainable params: LoRA (r=--lora-r) on attention q/k/v/out (via PEFT) and on
every expert gate_up/down projection (batched 3D LoRA inside
FusedQuantExperts, saved as expert_lora.pt next to the adapter), plus the
router `gate` linears trained *fully* (modules_to_save) — re-concentrating
routing is most of the work and the gates are tiny.

The k-anneal is run as separate invocations (plan.md: 16 -> 8 -> 6 -> 4),
chaining checkpoints:

    python step3/train_freeze.py --k 16 --steps 2000 --out runs/k16
    python step3/train_freeze.py --k 8  --steps 2000 --out runs/k8 \
        --init-adapter runs/k16
    ...

An interrupted run resumes exactly (optimizer moments + dataset position;
gradients are empty at checkpoint boundaries, so nothing else is needed):

    python step3/train_freeze.py --k 16 --steps 2000 --out runs/k16 \
        --resume runs/k16/step500

Batch size is 1 (per-example keep masks differ) with gradient accumulation.
Designed for the RTX 3080 (10 GB): 4-bit base via freeze_utils, gradient
checkpointing, paged 8-bit AdamW.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

from freeze_utils import (
    compute_keep_masks,
    load_adapter_compat,
    expert_lora_parameters,
    load_expert_lora,
    load_quantized_model,
    load_tokenizer,
    save_expert_lora,
    set_keep_masks,
)

REPO = Path(__file__).resolve().parent.parent

# Expert LoRA is handled inside FusedQuantExperts (PEFT can't target the
# fused 4-bit blobs); PEFT only wraps attention + the routers.
TARGET_MODULES = r".*\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$"


def build_example(tok, messages, max_len):
    """Returns (prompt_ids, full_ids, n_prompt) or None if it doesn't fit."""
    # Final assistant turn is the response; everything before it is the prompt
    # whose routing decides the keep set.
    if messages[-1]["role"] != "assistant":
        return None
    prompt_ids = tok.apply_chat_template(
        messages[:-1], add_generation_prompt=True, return_tensors="pt", return_dict=False
    )
    full_ids = tok.apply_chat_template(messages, return_tensors="pt", return_dict=False)
    n_prompt = prompt_ids.shape[1]
    if full_ids.shape[1] > max_len or full_ids.shape[1] - n_prompt < 8:
        return None
    if not torch.equal(full_ids[0, :n_prompt], prompt_ids[0]):
        return None  # template didn't compose as prefix; skip rather than mislabel
    return prompt_ids, full_ids, n_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="experts kept per layer during training")
    ap.add_argument("--steps", type=int, default=2000, help="optimizer steps")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument(
        "--bf16-base",
        action="store_true",
        help="keep base/expert weights in bf16 instead of 4-bit; needs ~24-32GB VRAM but avoids dequant overhead",
    )
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--data", default=str(REPO / "data" / "smoltalk"))
    ap.add_argument("--out", required=True, help="adapter output dir, e.g. runs/k16")
    ap.add_argument("--init-adapter", help="previous k-stage adapter to continue from")
    ap.add_argument(
        "--resume",
        help="checkpoint dir of THIS run (e.g. runs/k16/step500): restores "
        "adapter, expert LoRA, optimizer state and dataset position",
    )
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=1))

    tok = load_tokenizer()
    model = load_quantized_model(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        quantize_base=not args.bf16_base,
    )
    model.config.use_cache = False

    if args.resume:
        model = load_adapter_compat(
            model, args.resume, lora_r=args.lora_r, lora_alpha=args.lora_alpha, is_trainable=True
        )
        if not load_expert_lora(model, args.resume):
            raise SystemExit(f"no expert_lora.pt in {args.resume}")
    elif args.init_adapter:
        model = load_adapter_compat(
            model,
            args.init_adapter,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            is_trainable=True,
        )
        if not load_expert_lora(model, args.init_adapter):
            raise SystemExit(f"no expert_lora.pt in {args.init_adapter}")
        print(f"continuing from adapter {args.init_adapter}")
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=TARGET_MODULES,
            modules_to_save=["feed_forward.gate"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    # PEFT froze everything it doesn't manage, including the fused expert LoRA.
    for p in expert_lora_parameters(model):
        p.requires_grad_(True)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    import bitsandbytes as bnb

    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr)

    step = 0
    start_pos = 0
    if args.resume:
        state = torch.load(Path(args.resume) / "train_state.pt", map_location="cpu")
        for key in ("k", "seed", "grad_accum"):
            if state[key] != getattr(args, key):
                raise SystemExit(f"--resume {key} mismatch: ckpt={state[key]} args={getattr(args, key)}")
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        start_pos = state["ds_pos"]
        print(f"resumed {args.resume}: step {step}, dataset position {start_pos}")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    ds = load_from_disk(args.data).shuffle(seed=args.seed)
    print(f"{len(ds)} examples from {args.data}")

    log_path = out_dir / "train_log.jsonl"
    micro = 0
    micro_log = 0
    loss_acc, tok_acc = 0.0, 0
    t_start = time.time()
    model.train()

    def save_checkpoint(ckpt_dir, ds_pos):
        model.save_pretrained(ckpt_dir)
        save_expert_lora(model, ckpt_dir)
        torch.save(
            {
                "optimizer": opt.state_dict(),
                "step": step,
                "ds_pos": ds_pos + 1,  # next unconsumed example
                "k": args.k,
                "seed": args.seed,
                "grad_accum": args.grad_accum,
            },
            Path(ckpt_dir) / "train_state.pt",
        )

    ds_pos = start_pos - 1
    for ds_pos in range(start_pos, len(ds)):
        ex = ds[ds_pos]
        built = build_example(tok, ex["messages"], args.max_len)
        if built is None:
            continue
        prompt_ids, full_ids, n_prompt = built
        prompt_ids = prompt_ids.to(model.device)
        full_ids = full_ids.to(model.device)

        # Pass 1: prompt routing -> keep masks (the inference-time constraint).
        model.eval()
        masks = compute_keep_masks(model, prompt_ids, k=args.k)
        model.train()

        # Pass 2: masked teacher forcing, loss on response tokens.
        set_keep_masks(model, masks)
        labels = full_ids.clone()
        labels[0, :n_prompt] = -100
        out = model(input_ids=full_ids, labels=labels, use_cache=False)
        # Masks must stay set through backward: gradient checkpointing
        # re-runs the forward, and an unmasked recompute routes differently.
        (out.loss / args.grad_accum).backward()
        set_keep_masks(model, None)
        loss_acc += out.loss.item()
        tok_acc += int((labels != -100).sum())
        micro_log += 1
        micro += 1

        if micro % args.grad_accum == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0 or step == 1:
                mean_loss = loss_acc / micro_log
                rec = {
                    "step": step,
                    "loss": round(mean_loss, 4),
                    "lr": lr_at(step),
                    "resp_tok": tok_acc,
                    "elapsed_s": round(time.time() - t_start),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
                print(rec)
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                loss_acc, tok_acc, micro_log = 0.0, 0, 0

            if step % args.save_every == 0:
                save_checkpoint(out_dir / f"step{step}", ds_pos)
                print(f"saved {out_dir / f'step{step}'}")

            if step >= args.steps:
                break

    save_checkpoint(out_dir, ds_pos)
    print(f"done: {step} steps, adapter saved to {out_dir}")


if __name__ == "__main__":
    main()

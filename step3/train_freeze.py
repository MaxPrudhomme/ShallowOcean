"""Step 3 / Phase 1 — freeze-aware fine-tuning of LFM2.5-8B-A1B (CUDA box).

SFT where each example is trained under the constraint the model will face at
inference: route once on the prompt, freeze the router to the prompt's top-k
experts per layer, teacher-force the full example with the router masked to
that set.

Per training example (batch size 1 + gradient accumulation, because the mask
is per-example):
  1. no-grad prefill of the prompt only; accumulate router softmax mass per
     MoE layer (drift_lfm.py logic).
  2. top-k experts per layer -> additive logit mask (-inf outside the set).
  3. forward the full example with the mask applied; top-4 selection then
     renormalizes within the kept k automatically (softmax of -inf logits = 0).
  4. loss = LM loss on the response tokens; optional KL distillation against
     the unmasked model's logits (same weights, mask disabled, no grad).

k anneal: --stages "16:2000,8:2000,6:2000,4:4000" (k:steps). Each stage ends
with a checkpoint; the k=16 coverage curve is the go/no-go signal.

Trainable params: LoRA on attention + expert FFNs; router gate projections
trained FULLY (modules_to_save) — re-concentrating routing is most of the work.

Usage (3080, QLoRA 4-bit):
    python step3/train_freeze.py --model LiquidAI/LFM2.5-8B-A1B --load-4bit \
        --stages "16:2000,8:2000,6:2000,4:4000" --out runs/freeze8b

Requires: torch (CUDA), transformers>=4.56, peft, bitsandbytes, datasets.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- router I/O


def _gate_out_features(gate):
    # peft modules_to_save wraps the Linear; look through the wrapper
    for g in (gate, getattr(gate, "original_module", None)):
        if g is not None and hasattr(g, "out_features"):
            return g.out_features
    return None


def find_moe_blocks(model):
    """Return [(layer_idx, block)] for every sparse MoE block, by duck-typing:
    a module with a `gate` projection whose out_features == its expert count."""
    blocks = []
    for name, mod in model.named_modules():
        gate = getattr(mod, "gate", None)
        if gate is None or not isinstance(gate, torch.nn.Module):
            continue
        n_exp = getattr(mod, "num_experts", None)
        if n_exp is None and hasattr(mod, "experts"):
            try:
                n_exp = len(mod.experts)
            except TypeError:
                n_exp = None
        if n_exp and _gate_out_features(gate) == n_exp:
            # layer index from the module path, e.g. model.layers.7.feed_forward
            idx = next(int(p) for p in name.split(".") if p.isdigit())
            blocks.append((idx, mod))
    blocks.sort(key=lambda t: t[0])
    return blocks


class RouterMasker:
    """Hooks every MoE gate. Two modes:
    - collect: accumulate softmax router mass per layer (prompt prefill)
    - mask:    add -inf to gate logits outside the per-layer kept set
    """

    NEG = -1e9  # large negative, safe in fp16/bf16 softmax

    def __init__(self, model):
        self.blocks = find_moe_blocks(model)
        assert self.blocks, "no MoE blocks found — wrong model class?"
        self.n_experts = _gate_out_features(self.blocks[0][1].gate)
        self.collecting = False
        self.mass = {}    # layer_idx -> float32 tensor [E]
        self._masks = None  # layer_idx -> additive mask [E], or None = off
        # LFM's aux-free expert_bias is added AFTER the softmax for top-k
        # selection, so a -inf logit mask alone (prob 0) can still lose the
        # top-4 race to a positively-biased masked expert. Mask the bias too.
        self._orig_bias = {li: blk.expert_bias.detach().clone()
                           for li, blk in self.blocks
                           if getattr(blk, "expert_bias", None) is not None}
        for li, blk in self.blocks:
            blk.gate.register_forward_hook(self._make_hook(li))

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, masks):
        self._masks = masks
        for li, blk in self.blocks:
            if li not in self._orig_bias:
                continue
            with torch.no_grad():
                if masks is None:
                    blk.expert_bias.copy_(self._orig_bias[li])
                else:
                    blk.expert_bias.copy_(
                        self._orig_bias[li] + masks[li].to(blk.expert_bias.dtype))

    def _make_hook(self, li):
        def hook(module, inputs, output):
            if self.collecting:
                probs = F.softmax(output.detach().float(), dim=-1)
                m = probs.reshape(-1, probs.shape[-1]).sum(0)
                self.mass[li] = self.mass.get(li, 0) + m
            if self._masks is not None:
                output = output + self._masks[li].to(output.dtype)
            return output
        return hook

    def start_collect(self):
        self.mass = {}
        self.collecting = True

    def stop_collect(self):
        self.collecting = False

    def build_masks(self, k, device):
        """top-k by collected prompt mass -> additive logit masks."""
        masks = {}
        for li, _ in self.blocks:
            sel = torch.topk(self.mass[li], k).indices
            m = torch.full((self.n_experts,), self.NEG, device=device)
            m[sel] = 0.0
            masks[li] = m
        return masks


# ------------------------------------------------------------------- data


def iter_examples(tokenizer, dataset_name, split, max_len, seed):
    """Yield (input_ids, prompt_len) for single-turn-ified chat examples."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for ex in ds:
        msgs = ex.get("messages") or ex.get("conversations")
        if not msgs:
            continue
        # normalize to role/content and cut at the first assistant turn
        norm = []
        for m in msgs:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            role = {"human": "user", "gpt": "assistant"}.get(role, role)
            norm.append({"role": role, "content": content})
            if role == "assistant":
                break
        if not norm or norm[-1]["role"] != "assistant":
            continue
        prompt_ids = tokenizer.apply_chat_template(
            norm[:-1], add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(norm)
        if len(full_ids) > max_len or len(full_ids) - len(prompt_ids) < 8:
            continue
        yield full_ids, len(prompt_ids)


# ------------------------------------------------------------------- train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA via bitsandbytes")
    ap.add_argument("--stages", default="16:2000,8:2000,6:2000,4:4000",
                    help="comma list of k:steps")
    ap.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--router-lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--distill-alpha", type=float, default=0.0,
                    help=">0: add KL vs unmasked logits on response tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/freeze8b"))
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kw = dict(dtype=torch.bfloat16, device_map={"": 0})
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                        "in_proj", "w1", "w2", "w3",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["gate"],   # router projections trained fully
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    masker = RouterMasker(model)
    print(f"{len(masker.blocks)} MoE layers, {masker.n_experts} experts")

    router_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (router_params if ".gate." in n or n.endswith(".gate.weight")
         else other_params).append(p)
    try:
        from bitsandbytes.optim import PagedAdamW8bit as Opt
    except ImportError:
        Opt = torch.optim.AdamW
    opt = Opt([{"params": other_params, "lr": args.lr},
               {"params": router_params, "lr": args.router_lr}],
              weight_decay=0.0)

    stages = [(int(k), int(n)) for k, n in
              (s.split(":") for s in args.stages.split(","))]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    log_f = open(args.out / "train_log.jsonl", "a")

    data = iter_examples(tokenizer, args.dataset, args.dataset_split,
                         args.max_len, args.seed)
    gstep, t0 = 0, time.time()
    for k, n_steps in stages:
        print(f"\n===== stage k={k}, {n_steps} optimizer steps =====")
        run_loss, run_cov = [], []
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                full_ids, plen = next(data)
                ids = torch.tensor([full_ids], device=device)

                # 1-2. route once on the prompt, build the per-layer mask
                masker.masks = None
                masker.start_collect()
                with torch.no_grad():
                    model(ids[:, :plen])
                masker.stop_collect()
                # coverage diagnostic: prompt mass inside the kept set
                cov = float(torch.stack([
                    torch.topk(m, k).values.sum() / m.sum()
                    for m in masker.mass.values()]).mean())
                run_cov.append(cov)
                masks = masker.build_masks(k, device)

                # optional distillation reference: base model (adapters off),
                # unmasked, no grad — a stationary quality anchor
                ref_logits = None
                if args.distill_alpha > 0:
                    with torch.no_grad(), model.disable_adapter():
                        ref_logits = model(ids).logits[:, plen - 1:-1]

                # 3-4. teacher-force under the frozen router
                masker.masks = masks
                logits = model(ids).logits
                masker.masks = None

                labels = ids.clone()
                labels[:, :plen] = -100
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    labels[:, 1:].reshape(-1), ignore_index=-100)
                if ref_logits is not None:
                    kl = F.kl_div(
                        F.log_softmax(logits[:, plen - 1:-1].float(), dim=-1),
                        F.log_softmax(ref_logits.float(), dim=-1),
                        log_target=True, reduction="batchmean")
                    loss = loss + args.distill_alpha * kl
                (loss / args.grad_accum).backward()
                run_loss.append(loss.item())

            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            gstep += 1

            if gstep % args.log_every == 0:
                rec = {"step": gstep, "k": k,
                       "loss": round(sum(run_loss) / len(run_loss), 4),
                       "prompt_mass_cov": round(sum(run_cov) / len(run_cov), 4),
                       "elapsed_s": round(time.time() - t0)}
                print(rec)
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                run_loss, run_cov = [], []

        ckpt = args.out / f"stage_k{k}"
        model.save_pretrained(ckpt)
        print(f"saved {ckpt}")

    log_f.close()
    print("done")


if __name__ == "__main__":
    main()

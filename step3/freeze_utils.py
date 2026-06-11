"""Shared machinery for step 3 on the CUDA box (RTX 3080, 10 GB).

Three jobs:

1. `load_quantized_model()` — load LFM2.5-8B-A1B in a form that fits the
   3080. Stock bitsandbytes QLoRA only converts nn.Linear, but this model
   stores its 7.7B expert params as fused 3D nn.Parameters
   (Lfm2MoeExperts.gate_up_proj / down_proj). We quantize each projection
   as ONE fused NF4 blob per layer (`FusedQuantExperts`) and compute the
   expert forward with batched bmm over sorted token groups — a per-expert
   Linear4bit ModuleList was launch-overhead-bound (22 layers x 32 experts
   x 2 projections of tiny kernels left the GPU ~1% utilized). NF4 blocks
   are 64 wide and both row lengths (2048, 1792) divide 64, so the fused
   quantization is numerically identical to per-expert. Expert LoRA lives
   inside the module as 3D batched tensors (PEFT can't target the fused
   weights); attention linears still get standard bnb Linear4bit + PEFT.
   Router `gate` linears, conv mixers, dense MLPs, norms and the (tied)
   embedding stay bf16. Result: ~5 GB resident.

2. `RouterLogger` — capture per-token router logits from every MoE layer
   via forward hooks on the `gate` linears.

3. Freeze masks — `route_tokens_to_experts` is patched at class level to
   honor an optional per-block `_freeze_keep` bool[32]: excluded experts are
   masked out of the *post-bias selection scores* (sigmoid(logits) +
   expert_bias), which closes the findings.md trap where expert_bias alone
   can pull an excluded expert back into the top-k. The gathered top-4
   weights are the unmasked sigmoid values, renormalized by norm_topk_prob,
   so kept experts keep their relative weights — same renorm rule as
   step1/route_stats.py.
"""

import gc
import math
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeSparseMoeBlock

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models" / "LFM2.5-8B-A1B"

def _linear4bit_from_weight(weight: torch.Tensor) -> bnb.nn.Linear4bit:
    out_features, in_features = weight.shape
    lin = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    )
    # Quantization happens when the module is moved to CUDA.
    lin.weight = bnb.nn.Params4bit(
        weight.detach().clone().contiguous(),
        requires_grad=False,
        quant_type="nf4",
        module=lin,
    )
    return lin


EXPERT_LORA_NAMES = ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down")


class FusedQuantExperts(nn.Module):
    """Drop-in replacement for Lfm2MoeExperts: fused NF4 weights + batched bmm.

    One quantized blob per projection per layer; the forward dequantizes it
    once and runs every expert in a single bmm over zero-padded token groups.
    Expert LoRA (B zero-initialized, so identity at start) is applied as
    batched 3D bmm — these params are NOT managed by PEFT; use
    expert_lora_parameters / save_expert_lora / load_expert_lora.
    """

    def __init__(self, num_experts: int, lora_r: int, lora_alpha: int):
        super().__init__()
        self.num_experts = num_experts
        self.scaling = lora_alpha / lora_r

    @classmethod
    def from_dense(cls, experts, lora_r: int = 8, lora_alpha: int = 16, device="cuda"):
        new = cls(experts.num_experts, lora_r, lora_alpha)
        E = experts.num_experts
        gu = experts.gate_up_proj.data  # (E, 2I, h)
        dn = experts.down_proj.data  # (E, h, I)
        two_i, h = gu.shape[1], gu.shape[2]
        inter = dn.shape[2]
        new.gate_up_q, new.gate_up_state = bnb.functional.quantize_4bit(
            gu.reshape(E * two_i, h).contiguous().to(device), quant_type="nf4"
        )
        new.down_q, new.down_state = bnb.functional.quantize_4bit(
            dn.reshape(E * h, inter).contiguous().to(device), quant_type="nf4"
        )
        new.lora_A_gate_up = nn.Parameter(torch.empty(E, lora_r, h, dtype=torch.bfloat16, device=device))
        new.lora_B_gate_up = nn.Parameter(torch.zeros(E, two_i, lora_r, dtype=torch.bfloat16, device=device))
        new.lora_A_down = nn.Parameter(torch.empty(E, lora_r, inter, dtype=torch.bfloat16, device=device))
        new.lora_B_down = nn.Parameter(torch.zeros(E, h, lora_r, dtype=torch.bfloat16, device=device))
        for a in (new.lora_A_gate_up, new.lora_A_down):
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        return new

    def forward(
        self,
        hidden_states: torch.Tensor,  # (T, h)
        top_k_index: torch.Tensor,  # (T, K)
        top_k_weights: torch.Tensor,  # (T, K)
    ) -> torch.Tensor:
        T, h = hidden_states.shape
        K = top_k_index.shape[1]
        E = self.num_experts

        # Group the T*K (token, expert) assignments by expert, zero-padded to
        # the largest group, so all experts run as one bmm.
        flat_e = top_k_index.reshape(-1)
        sort_idx = torch.argsort(flat_e, stable=True)
        sorted_e = flat_e[sort_idx]
        counts = torch.bincount(flat_e, minlength=E)
        group_start = counts.cumsum(0) - counts
        pos = torch.arange(T * K, device=flat_e.device) - group_start[sorted_e]
        token_idx = sort_idx // K
        cap = int(counts.max())  # the one CPU sync per layer call

        x = hidden_states.new_zeros(E, cap, h)
        x[sorted_e, pos] = hidden_states[token_idx]

        w_gu = bnb.functional.dequantize_4bit(self.gate_up_q, self.gate_up_state).view(E, -1, h)
        h1 = torch.bmm(x, w_gu.transpose(1, 2))
        h1 = h1 + torch.bmm(
            torch.bmm(x, self.lora_A_gate_up.transpose(1, 2)),
            self.lora_B_gate_up.transpose(1, 2),
        ) * self.scaling
        gate, up = h1.chunk(2, dim=-1)
        act = F.silu(gate) * up
        w_d = bnb.functional.dequantize_4bit(self.down_q, self.down_state).view(E, h, -1)
        y = torch.bmm(act, w_d.transpose(1, 2))
        y = y + torch.bmm(
            torch.bmm(act, self.lora_A_down.transpose(1, 2)),
            self.lora_B_down.transpose(1, 2),
        ) * self.scaling

        y = y[sorted_e, pos] * top_k_weights.reshape(-1)[sort_idx].unsqueeze(-1)
        out = torch.zeros_like(hidden_states)
        out.index_add_(0, token_idx, y.to(hidden_states.dtype))
        return out


class FusedDenseExperts(nn.Module):
    """Fused bf16 experts + batched 3D LoRA, for GPUs with enough VRAM."""

    def __init__(self, num_experts: int, lora_r: int, lora_alpha: int):
        super().__init__()
        self.num_experts = num_experts
        self.scaling = lora_alpha / lora_r

    @classmethod
    def from_dense(cls, experts, lora_r: int = 8, lora_alpha: int = 16, device="cuda"):
        new = cls(experts.num_experts, lora_r, lora_alpha)
        gu = experts.gate_up_proj.data.to(device=device, dtype=torch.bfloat16).contiguous()
        dn = experts.down_proj.data.to(device=device, dtype=torch.bfloat16).contiguous()
        E = experts.num_experts
        two_i, h = gu.shape[1], gu.shape[2]
        inter = dn.shape[2]
        new.gate_up = nn.Parameter(gu, requires_grad=False)
        new.down = nn.Parameter(dn, requires_grad=False)
        new.lora_A_gate_up = nn.Parameter(torch.empty(E, lora_r, h, dtype=torch.bfloat16, device=device))
        new.lora_B_gate_up = nn.Parameter(torch.zeros(E, two_i, lora_r, dtype=torch.bfloat16, device=device))
        new.lora_A_down = nn.Parameter(torch.empty(E, lora_r, inter, dtype=torch.bfloat16, device=device))
        new.lora_B_down = nn.Parameter(torch.zeros(E, h, lora_r, dtype=torch.bfloat16, device=device))
        for a in (new.lora_A_gate_up, new.lora_A_down):
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        return new

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        T, h = hidden_states.shape
        K = top_k_index.shape[1]
        E = self.num_experts

        flat_e = top_k_index.reshape(-1)
        sort_idx = torch.argsort(flat_e, stable=True)
        sorted_e = flat_e[sort_idx]
        counts = torch.bincount(flat_e, minlength=E)
        group_start = counts.cumsum(0) - counts
        pos = torch.arange(T * K, device=flat_e.device) - group_start[sorted_e]
        token_idx = sort_idx // K
        cap = int(counts.max())

        x = hidden_states.new_zeros(E, cap, h)
        x[sorted_e, pos] = hidden_states[token_idx]

        h1 = torch.bmm(x, self.gate_up.transpose(1, 2))
        h1 = h1 + torch.bmm(
            torch.bmm(x, self.lora_A_gate_up.transpose(1, 2)),
            self.lora_B_gate_up.transpose(1, 2),
        ) * self.scaling
        gate, up = h1.chunk(2, dim=-1)
        act = F.silu(gate) * up
        y = torch.bmm(act, self.down.transpose(1, 2))
        y = y + torch.bmm(
            torch.bmm(act, self.lora_A_down.transpose(1, 2)),
            self.lora_B_down.transpose(1, 2),
        ) * self.scaling

        y = y[sorted_e, pos] * top_k_weights.reshape(-1)[sort_idx].unsqueeze(-1)
        out = torch.zeros_like(hidden_states)
        out.index_add_(0, token_idx, y.to(hidden_states.dtype))
        return out


def expert_lora_parameters(model):
    """All fused expert LoRA params (PEFT freezes these; re-enable grad)."""
    for _, block in moe_blocks(model):
        for name in EXPERT_LORA_NAMES:
            yield getattr(block.experts, name)


def save_expert_lora(model, out_dir):
    sd = {}
    for idx, block in moe_blocks(model):
        for name in EXPERT_LORA_NAMES:
            sd[f"{idx}.{name}"] = getattr(block.experts, name).detach().cpu()
    torch.save(sd, Path(out_dir) / "expert_lora.pt")


def load_expert_lora(model, adapter_dir) -> bool:
    path = Path(adapter_dir) / "expert_lora.pt"
    if not path.exists():
        return False
    sd = torch.load(path, map_location="cpu")
    with torch.no_grad():
        for idx, block in moe_blocks(model):
            for name in EXPERT_LORA_NAMES:
                getattr(block.experts, name).copy_(sd[f"{idx}.{name}"])
    return True


def _route_tokens_to_experts_maskable(self, router_logits):
    routing_weights = router_logits.sigmoid()
    if self.use_expert_bias:
        scores_for_routing = routing_weights + self.expert_bias
    else:
        scores_for_routing = routing_weights

    keep = getattr(self, "_freeze_keep", None)
    if keep is not None:
        scores_for_routing = scores_for_routing.masked_fill(~keep, float("-inf"))

    _, selected_experts = torch.topk(scores_for_routing, k=self.top_k, dim=-1)
    routing_weights = torch.gather(routing_weights, dim=1, index=selected_experts).type_as(
        router_logits
    )
    if self.norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
    routing_weights = routing_weights * self.routed_scaling_factor
    return selected_experts, routing_weights


Lfm2MoeSparseMoeBlock.route_tokens_to_experts = _route_tokens_to_experts_maskable


def moe_blocks(model) -> list[tuple[int, Lfm2MoeSparseMoeBlock]]:
    """(layer_idx, block) for every sparse MoE layer, in order."""
    out = []
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for idx, layer in enumerate(base.model.layers):
        if isinstance(layer.feed_forward, Lfm2MoeSparseMoeBlock):
            out.append((idx, layer.feed_forward))
    return out


def set_keep_masks(model, masks: dict[int, torch.Tensor] | None):
    """masks: layer_idx -> bool[num_experts] on the model device, or None to clear."""
    for idx, block in moe_blocks(model):
        block._freeze_keep = None if masks is None else masks[idx]


class RouterLogger:
    """Collects raw router logits per MoE layer for every forward pass."""

    def __init__(self, model):
        self.blocks = moe_blocks(model)
        self.logits: dict[int, list[torch.Tensor]] = {idx: [] for idx, _ in self.blocks}
        self._handles = []

    def __enter__(self):
        for idx, block in self.blocks:
            def hook(module, args, output, idx=idx):
                self.logits[idx].append(output.detach().float().cpu())
            # The wrapped module may be a peft ModulesToSaveWrapper; hook
            # fires on the outer module either way.
            self._handles.append(block.gate.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def mass(self) -> dict[int, torch.Tensor]:
        """Per-layer expert mass: per-token normalized sigmoid weights, summed
        over all logged tokens (the selection signal from step3/drift_lfm.py)."""
        out = {}
        for idx, chunks in self.logits.items():
            logits = torch.cat(chunks, dim=0)
            w = logits.sigmoid()
            w = w / w.sum(dim=-1, keepdim=True)
            out[idx] = w.sum(dim=0)
        return out


def compute_keep_masks(model, input_ids: torch.Tensor, k: int) -> dict[int, torch.Tensor]:
    """Route the prompt (unmasked, no grad), return top-k-by-mass keep masks."""
    set_keep_masks(model, None)
    with RouterLogger(model) as log, torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    device = input_ids.device
    masks = {}
    for idx, mass in log.mass().items():
        keep = torch.zeros(mass.numel(), dtype=torch.bool, device=device)
        keep[torch.topk(mass, k=k).indices] = True
        masks[idx] = keep
    return masks


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def load_quantized_model(
    device: str = "cuda",
    lora_r: int = 8,
    lora_alpha: int = 16,
    quantize_base: bool = True,
):
    """LFM2.5-8B-A1B with fused experts; optionally quantize base to 4-bit."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)

    for _, block in moe_blocks(model):
        cls = FusedQuantExperts if quantize_base else FusedDenseExperts
        block.experts = cls.from_dense(block.experts, lora_r=lora_r, lora_alpha=lora_alpha, device=device)
        gc.collect()

    if not quantize_base:
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()
        return model

    from transformers.models.lfm2_moe.modeling_lfm2_moe import (
        Lfm2MoeAttention,
        Lfm2MoeMLP,
        Lfm2MoeShortConv,
    )

    quant_targets = {
        Lfm2MoeAttention: ("q_proj", "k_proj", "v_proj", "out_proj"),
        Lfm2MoeShortConv: ("in_proj", "out_proj"),
        Lfm2MoeMLP: ("w1", "w2", "w3"),
    }
    for module in model.modules():
        suffixes = quant_targets.get(type(module))
        if suffixes is None:
            continue
        for suffix in suffixes:
            attr = getattr(module, suffix)
            if isinstance(attr, nn.Linear):
                setattr(module, suffix, _linear4bit_from_weight(attr.weight))

    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    return model

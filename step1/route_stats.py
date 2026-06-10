"""Step 1a — route once: run the first denoising step of diffusiongemma-26B-A4B
([prompt | 256-mask canvas], zero self-conditioning) layer-by-layer in numpy,
streaming dequantized weights from the GGUF, and accumulate each layer's router
softmax mass per expert. Output: the top --keep fraction of experts per layer
as JSON for step1/prune_gguf.py.

The forward mirrors llama.cpp PR #24423's UNIFIED diffusion-gemma graph
(src/models/diffusion-gemma.cpp + gemma4-common.h):
  - prompt rows: embed*sqrt(n_embd); canvas rows: rms_norm_noscale(same)  [zero SC]
  - region-aware mask: prompt causal (SWA-clipped on sliding layers);
    canvas bidirectional, canvas->prompt limited to last n_swa-1 on sliding layers
  - per-head q/k rms-norm, scale-less v rms-norm, V=K on global layers
  - NEOX rope, base 1e6 (global, with proportional rope_freqs factors) / 1e4 (SWA)
  - attention scale 1.0 (no 1/sqrt(d))
  - router on the unnormed attn_out: rms_noscale * 1/sqrt(n_embd) * gate_inp.scale
    -> gate_inp -> softmax -> top-8 renormalized
  - dense shared-expert MLP + MoE (fused gate_up, per-expert down scale), the
    five FFN norms, residual, per-layer enc/dec output scalars

Only routing *statistics* are needed, so small numeric drift vs ggml is fine.

Usage:
    python step1/route_stats.py --prompt "..." [--keep 0.5] [--out experts.json]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

DEFAULT_MODEL = Path.home() / ".lmstudio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
TOKENIZE_BIN = "/tmp/llamacpp-src/build/bin/llama-tokenize"


def rms(x, w=None, eps=1e-6):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y * w if w is not None else y


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def rope_neox(x, pos, base, factors=None):
    """x: [T, n_head, hd], rotate all hd dims, NEOX pairing (i, i+hd/2)."""
    hd = x.shape[-1]
    half = hd // 2
    inv = base ** (-np.arange(half, dtype=np.float64) * 2.0 / hd)
    if factors is not None:
        inv = inv / factors
    theta = pos[:, None].astype(np.float64) * inv[None, :]      # [T, half]
    cos = np.cos(theta)[:, None, :].astype(np.float32)
    sin = np.sin(theta)[:, None, :].astype(np.float32)
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)


class Streamer:
    """Dequantizes tensors from the mmap'd GGUF on demand."""

    def __init__(self, path):
        self.reader = GGUFReader(path)
        self.t = {t.name: t for t in self.reader.tensors}
        self.meta = {f.name: f.contents() for f in self.reader.fields.values()}

    def __call__(self, name):
        t = self.t[name]
        d = dequantize(t.data, t.tensor_type)
        return d.reshape(list(t.shape)[::-1]).astype(np.float32)  # numpy order

    def expert(self, name, e):
        """Dequantize one expert slab of blk.N.ffn_*_exps.weight."""
        t = self.t[name]
        n_expert = int(t.shape[-1])
        raw = t.data.reshape(n_expert, -1)
        d = dequantize(raw[e], t.tensor_type)
        return d.reshape(list(t.shape)[-2::-1]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("experts.json"))
    ap.add_argument("--from-mass", type=Path, default=None,
                    help="reuse a saved .npz of router mass; skip the forward pass")
    args = ap.parse_args()

    t0 = time.time()
    if args.from_mass:
        z = np.load(args.from_mass)
        mass_prompt, mass_canvas = z["mass_prompt"], z["mass_canvas"]
        n_layer, n_expert = mass_canvas.shape
        n_used = 8
        select(args, mass_prompt, mass_canvas, n_layer, n_expert, n_used, t0)
        return
    W = Streamer(args.model)
    m = W.meta
    arch = m["general.architecture"]
    n_layer = m[f"{arch}.block_count"]
    n_embd = m[f"{arch}.embedding_length"]
    n_head = m[f"{arch}.attention.head_count"]
    n_head_kv = m[f"{arch}.attention.head_count_kv"]
    n_expert = m[f"{arch}.expert_count"]
    n_used = m[f"{arch}.expert_used_count"]
    n_ff_exp = m[f"{arch}.expert_feed_forward_length"]
    eps = m[f"{arch}.attention.layer_norm_rms_epsilon"]
    n_swa = m[f"{arch}.attention.sliding_window"]
    is_swa = m[f"{arch}.attention.sliding_window_pattern"]
    base_full = m[f"{arch}.rope.freq_base"]
    base_swa = m[f"{arch}.rope.freq_base_swa"]
    canvas_len = m["diffusion.canvas_length"]
    mask_id = m["tokenizer.ggml.mask_token_id"]

    # chat-template + tokenize the prompt exactly as llama-diffusion-cli does
    import jinja2
    tpl = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(m["tokenizer.chat_template"])
    text = tpl.render(messages=[{"role": "user", "content": args.prompt}],
                      add_generation_prompt=True, bos_token="")
    r = subprocess.run([TOKENIZE_BIN, "-m", str(args.model), "-p", text,
                        "--ids", "--log-disable"], capture_output=True, text=True, check=True)
    prefix = json.loads(r.stdout.strip())
    P = len(prefix)
    tokens = np.array(prefix + [mask_id] * canvas_len, dtype=np.int64)
    T = len(tokens)
    print(f"prompt tokens: {P}, total with canvas: {T}", file=sys.stderr)

    # input embeddings: prompt scaled; canvas rms_norm_noscale(scaled) (zero SC)
    emb = W("token_embd.weight")  # [n_vocab, n_embd]
    x = emb[tokens] * np.sqrt(n_embd)
    x[P:] = rms(x[P:], eps=eps)
    del emb

    pos = np.arange(T)

    # region-aware additive masks (True = allowed)
    q = pos[:, None]
    k = pos[None, :]
    canvas_q = q >= P
    canvas_k = k >= P
    causal = (~canvas_k) & (k <= q)
    full_mask = np.where(canvas_q, True, causal)
    swa_ok = (q - k < n_swa) & (k <= q)  # is_masked_swa: standard window for prompt queries
    canvas_lo = P - n_swa + 1
    swa_mask = np.where(canvas_q, canvas_k | (k >= canvas_lo), causal & swa_ok)
    neg = np.float32(-1e30)

    mass_prompt = np.zeros((n_layer, n_expert), dtype=np.float64)
    mass_canvas = np.zeros((n_layer, n_expert), dtype=np.float64)

    for il in range(n_layer):
        t_l = time.time()
        pfx = f"blk.{il}."
        swa = bool(is_swa[il])
        nkv = n_head_kv[il]
        base = base_swa if swa else base_full
        factors = None if swa else W("rope_freqs.weight")

        h = rms(x, W(pfx + "attn_norm.weight"), eps)
        wq = W(pfx + "attn_q.weight")          # [out, n_embd]
        wk = W(pfx + "attn_k.weight")
        hd = wk.shape[0] // nkv
        qh = (h @ wq.T).reshape(T, n_head, hd)
        kh = (h @ wk.T).reshape(T, nkv, hd)
        if pfx + "attn_v.weight" in W.t:
            vh = (h @ W(pfx + "attn_v.weight").T).reshape(T, nkv, hd)
        else:
            vh = kh.copy()                      # global layers: V = K (pre-norm)
        qh = rms(qh, W(pfx + "attn_q_norm.weight"), eps)
        kh = rms(kh, W(pfx + "attn_k_norm.weight"), eps)
        vh = rms(vh, eps=eps)                   # scale-less v-norm
        qh = rope_neox(qh, pos, base, factors)
        kh = rope_neox(kh, pos, base, factors)

        # GQA attention, kq_scale = 1.0 (f_attention_scale)
        rep = n_head // nkv
        kh = np.repeat(kh, rep, axis=1)
        vh = np.repeat(vh, rep, axis=1)
        scores = np.einsum("qhd,khd->hqk", qh, kh)
        scores += np.where(swa_mask if swa else full_mask, 0.0, neg)[None]
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        att = np.einsum("hqk,khd->qhd", p, vh).reshape(T, n_head * hd)
        att = att @ W(pfx + "attn_output.weight").T
        attn_out = rms(att, W(pfx + "post_attention_norm.weight"), eps) + x

        # dense shared-expert MLP
        hm = rms(attn_out, W(pfx + "ffn_norm.weight"), eps)
        mlp = (gelu_tanh(hm @ W(pfx + "ffn_gate.weight").T) * (hm @ W(pfx + "ffn_up.weight").T)) \
              @ W(pfx + "ffn_down.weight").T
        mlp = rms(mlp, W(pfx + "post_ffw_norm_1.weight"), eps)

        # router on the unnormed attn_out
        t_r = rms(attn_out, eps=eps) / np.sqrt(n_embd) * W(pfx + "ffn_gate_inp.scale")
        logits = t_r @ W(pfx + "ffn_gate_inp.weight").T          # [T, n_expert]
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        mass_prompt[il] = probs[:P].sum(0)
        mass_canvas[il] = probs[P:].sum(0)

        top = np.argpartition(-probs, n_used - 1, axis=-1)[:, :n_used]  # [T, 8]
        wts = np.take_along_axis(probs, top, -1)
        wts /= wts.sum(-1, keepdims=True)

        # MoE: per used expert, dequantize its slab and process its tokens
        moe_in = rms(attn_out, W(pfx + "pre_ffw_norm_2.weight"), eps)
        down_s = W(pfx + "ffn_down_exps.scale")
        moe = np.zeros_like(attn_out)
        for e in np.unique(top):
            tok, slot = np.where(top == e)
            gu = W.expert(pfx + "ffn_gate_up_exps.weight", e)    # [2*n_ff, n_embd]
            dn = W.expert(pfx + "ffn_down_exps.weight", e)       # [n_embd, n_ff]
            z = moe_in[tok] @ gu.T
            act = gelu_tanh(z[:, :n_ff_exp]) * z[:, n_ff_exp:]
            moe[tok] += (act @ dn.T) * down_s[e] * wts[tok, slot][:, None]
        moe = rms(moe, W(pfx + "post_ffw_norm_2.weight"), eps)

        cur = rms(mlp + moe, W(pfx + "post_ffw_norm.weight"), eps) + attn_out
        enc_s = W(pfx + "enc_layer_output_scale.weight")[0]
        dec_s = W(pfx + "layer_output_scale.weight")[0]
        cur[:P] *= enc_s
        cur[P:] *= dec_s
        x = cur
        print(f"layer {il:2d}  {time.time()-t_l:5.1f}s  "
              f"canvas mass top64: {np.sort(mass_canvas[il])[::-1][:64].sum()/mass_canvas[il].sum():.3f}",
              file=sys.stderr)

    mass_path = args.out.with_suffix(".npz")
    np.savez(mass_path, mass_prompt=mass_prompt, mass_canvas=mass_canvas)
    print(f"saved raw router mass to {mass_path}", file=sys.stderr)
    select(args, mass_prompt, mass_canvas, n_layer, n_expert, n_used, t0)


def select(args, mass_prompt, mass_canvas, n_layer, n_expert, n_used, t0):
    n_keep = max(n_used, round(args.keep * n_expert))
    mass = mass_canvas + mass_prompt          # canvas dominates (256 rows vs P)
    sel = {il: sorted(np.argsort(-mass[il])[:n_keep].tolist()) for il in range(n_layer)}
    cov = np.mean([mass_canvas[il][sel[il]].sum() / mass_canvas[il].sum() for il in range(n_layer)])

    args.out.write_text(json.dumps({str(k): v for k, v in sel.items()}))
    print(f"kept {n_keep}/{n_expert} experts/layer; mean canvas router-mass coverage "
          f"{cov:.3f}; wrote {args.out} in {time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

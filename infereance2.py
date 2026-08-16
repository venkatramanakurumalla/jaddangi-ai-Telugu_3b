#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 TeluguGPT 3B – Simple Interactive Inference
No command‑line arguments – just run and type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import re
import sentencepiece as spm
from huggingface_hub import hf_hub_download

# ============================================================
# 1. CONFIGURATION
# ============================================================
class Config:
    vocab_size = 32000
    d_model = 1728
    n_layers = 68
    n_heads = 24
    n_kv_heads = 6
    max_seq_len = 384
    rope_theta = 10000
    expansion_factor = 4
    dropout = 0.0

config = Config()

# ============================================================
# 2. MODEL ARCHITECTURE (exact match)
# ============================================================
def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot, k_rot = torch.empty_like(q), torch.empty_like(k)
    q_rot[..., 0::2] = q[..., 0::2] * cos - q[..., 1::2] * sin
    k_rot[..., 0::2] = k[..., 0::2] * cos - k[..., 1::2] * sin
    q_rot[..., 1::2] = q[..., 1::2] * cos + q[..., 0::2] * sin
    k_rot[..., 1::2] = k[..., 1::2] * cos + k[..., 0::2] * sin
    return q_rot, k_rot

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos[:, :, :T, :], sin[:, :, :T, :])
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hd = int(2 / 3 * config.expansion_factor * config.d_model)
        hd = (hd // 256) * 256
        self.w1 = nn.Linear(config.d_model, hd, bias=False)
        self.w2 = nn.Linear(config.d_model, hd, bias=False)
        self.w3 = nn.Linear(hd, config.d_model, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))

class TeluguGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied
        head_dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(config.max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        cos = self.cos_cached[:, :, :T, :].to(x.device)
        sin = self.sin_cached[:, :, :T, :].to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm_f(x))

# ============================================================
# 3. GENERATION FUNCTION
# ============================================================
def generate_text(
    prompt,
    model,
    tokenizer,
    max_new_tokens=128,
    temperature=0.75,
    top_k=40,
    top_p=0.92,
    repetition_penalty=1.15,
    block_english=True,
    filter_unk=True,
    stop_on_punctuation=True,
):
    """Generate Telugu text from a prompt."""
    # Block English
    if block_english and re.search(r'[a-zA-Z]', prompt):
        raise ValueError("Prompt contains English letters. Use only Telugu.")

    # Tokenize and truncate if necessary
    input_ids = tokenizer.encode(prompt, out_type=int)
    max_context = config.max_seq_len - 10
    if len(input_ids) > max_context:
        input_ids = input_ids[-max_context:]
        prompt = tokenizer.decode(input_ids)  # update for later

    device = next(model.parameters()).device
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = input_ids.copy()
    UNK_ID = tokenizer.unk_id() if tokenizer.unk_id() >= 0 else 0

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.max_seq_len:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if token_id < logits.size(1):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

            # Filter unknown token
            if filter_unk and UNK_ID < logits.size(1):
                logits[0, UNK_ID] = float("-inf")

            # Top-K
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            # Top-P (nucleus)
            if top_p < 1.0 and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token)
            idx = torch.cat((idx, torch.tensor([[next_token]], dtype=torch.long, device=device)), dim=1)

            # Stop on EOS
            if tokenizer.eos_id() >= 0 and next_token == tokenizer.eos_id():
                break

            # Stop on punctuation
            if stop_on_punctuation:
                full_text = tokenizer.decode(generated_ids)
                if len(full_text) > 0:
                    last_char = full_text[-1]
                    if last_char in ('.', '?', '!'):
                        if len(full_text) > 1 and not full_text[-2].isdigit():
                            break

    full_text = tokenizer.decode(generated_ids)
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    else:
        return full_text  # fallback

# ============================================================
# 4. LOAD MODEL AND TOKENIZER (AUTO-DOWNLOAD)
# ============================================================
def load_model_and_tokenizer(
    model_path="TeluguGPT_3B_CPT_Merged.pt",
    tokenizer_path="telugu_spm.model",
    repo_id="VenkataRamanaKurumallajaddangi/Telugu"
):
    # Download missing files
    for fname in [model_path, tokenizer_path]:
        if not os.path.exists(fname):
            print(f"📥 Downloading {fname}...")
            hf_hub_download(repo_id=repo_id, filename=fname, local_dir=".")

    # Load model
    print("🧠 Loading merged 3B model...")
    model = TeluguGPT(config).half()  # FP16
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    for k in ["cos_cached", "sin_cached", "lm_head.weight"]:
        state_dict.pop(k, None)
    model.load_state_dict(state_dict, strict=False)
    model.lm_head.weight = model.wte.weight  # re-tie

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Model loaded on {device}")

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    print(f"✅ Tokenizer loaded (vocab: {tokenizer.vocab_size()})")
    return model, tokenizer

# ============================================================
# 5. MAIN – INTERACTIVE LOOP
# ============================================================
if __name__ == "__main__":
    # Load once
    model, tokenizer = load_model_and_tokenizer()

    print("\n" + "="*60)
    print("🧠 TeluguGPT 3B – Interactive Chat")
    print("   Type your prompt in Telugu (no English letters).")
    print("   Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            prompt = input("📝 Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        print("-"*50)
        try:
            output = generate_text(prompt, model, tokenizer)
            print(output)
        except ValueError as e:
            print(f"🚫 {e}")
        print("-"*50 + "\n")

    print("👋 Goodbye!")

#!/usr/bin/env python3
"""
🔧 MERGE SCRIPT: Bake CPT LoRA weights into the base 3B model
Produces a single FP16 checkpoint and uploads it to Hugging Face.
"""

import torch
import torch.nn as nn
import math, os, gc
from huggingface_hub import HfApi, login, hf_hub_download

# ============================================================
# 0. AUTH – set HF_TOKEN in environment
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("❌ HF_TOKEN not found. Please set it as an environment variable.")
login(token=HF_TOKEN)
api = HfApi()

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"  # change if needed

# ============================================================
# 1. CONFIG (exact match)
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
    lora_rank = 16
    lora_alpha = 32

config = Config()

# ============================================================
# 2. ARCHITECTURE (exact copy, FP16, NO 4-bit)
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

class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank, alpha):
        super().__init__()
        self.base = base_linear
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
    def forward(self, x):
        base_out = self.base(x)
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

def inject_lora(module, rank, alpha, target_names):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            inject_lora(child, rank, alpha, target_names)

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
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

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
        self.lm_head.weight = self.wte.weight
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
# 3. LOAD BASE + INJECT LoRA
# ============================================================
print("🔧 Loading base 3B model (FP16)...")
model = TeluguGPT(config)
model = model.half()  # FP16 for memory

# Download base if missing
BASE_MODEL = "Telugu_Model_3B_V1.pt"
if not os.path.exists(BASE_MODEL):
    print(f"📥 Downloading {BASE_MODEL} from HF...")
    hf_hub_download(repo_id=REPO_ID, filename=BASE_MODEL, local_dir=".")

print("📥 Loading base weights...")
base_ckpt = torch.load(BASE_MODEL, map_location="cpu")
base_sd = base_ckpt.get("model_state_dict", base_ckpt)
for k in ["cos_cached", "sin_cached", "lm_head.weight"]:
    base_sd.pop(k, None)
model.load_state_dict(base_sd, strict=False)
del base_ckpt, base_sd
gc.collect()

print("🔧 Injecting LoRA adapters...")
targets = ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3")
inject_lora(model, config.lora_rank, config.lora_alpha, targets)
model.lm_head.weight = model.wte.weight

# ============================================================
# 4. LOAD CPT CHECKPOINT (auto-download if missing)
# ============================================================
CPT_FILE = "checkpoints/cpt_3b_step_1000.pt"  # or cpt_3b_final.pt
if not os.path.exists(CPT_FILE):
    print(f"📥 Downloading {CPT_FILE} from HF...")
    hf_hub_download(repo_id=REPO_ID, filename=CPT_FILE, local_dir=".")

print(f"📥 Loading CPT weights from {CPT_FILE}...")
cpt = torch.load(CPT_FILE, map_location="cpu")
# The checkpoint may have "model_partial" key
cpt_state = cpt.get("model_partial", cpt)
model.load_state_dict(cpt_state, strict=False)
print("✅ CPT weights loaded.")

# ============================================================
# 5. MERGE LoRA INTO BASE
# ============================================================
print("🔨 Merging LoRA weights into base layers...")

def merge_lora(module):
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            # base.weight += scaling * (lora_B @ lora_A)
            delta = child.scaling * (child.lora_B @ child.lora_A)
            child.base.weight.data += delta
            # Replace with plain Linear
            setattr(module, name, child.base)
        else:
            merge_lora(child)

merge_lora(model)
print("✅ Merge complete. All LoRALinear layers replaced with plain Linear.")

# ============================================================
# 6. SAVE MERGED MODEL
# ============================================================
OUTPUT = "TeluguGPT_3B_CPT_Merged.pt"
print(f"💾 Saving merged model to {OUTPUT}...")
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {k: getattr(config, k) for k in dir(config) if not k.startswith("_")},
}, OUTPUT)

size_mb = os.path.getsize(OUTPUT) / (1024 ** 3)
print(f"✅ Saved! File size: {size_mb:.2f} GB")

# ============================================================
# 7. UPLOAD TO HUGGING FACE
# ============================================================
print(f"☁️ Uploading {OUTPUT} to Hugging Face ({REPO_ID})...")
api.upload_file(
    path_or_fileobj=OUTPUT,
    path_in_repo=OUTPUT,
    repo_id=REPO_ID,
    commit_message="🚀 Merged CPT 1000 steps into base 3B → clean FP16 checkpoint"
)
print("🎉 Upload complete! Your merged model is now available on Hugging Face.")

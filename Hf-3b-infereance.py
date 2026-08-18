import os
import re
import gc
import torch
import torch.nn.functional as F
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file  # ⚠️ మాన్యువల్ లోడర్

# ==========================================
# 1. మెమరీ క్లీన్-అప్
# ==========================================
torch.cuda.empty_cache()
gc.collect()

model_id = "VenkataRamanaKurumallajaddangi/Jaddangi-3B"
print("📥 కోడ్ మరియు బరువులు (Weights) చెక్ చేస్తున్నాము...")

# ఫైల్స్ అన్నీ డౌన్‌లోడ్ చేయడం
for file_name in ["configuration_jaddangi.py", "modeling_jaddangi.py", "telugu_spm.model", "model.safetensors"]:
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        hf_hub_download(repo_id=model_id, filename=file_name, local_dir=".")

# ==========================================
# 2. బగ్‌ని బైపాస్ చేస్తూ మోడల్ మాన్యువల్ లోడింగ్ 
# ==========================================
from configuration_jaddangi import JaddangiConfig
from modeling_jaddangi import JaddangiForCausalLM

print("🧠 Jaddangi-3B మోడల్ స్ట్రక్చర్ క్రియేట్ చేస్తున్నాము...")

# కాన్ఫిగరేషన్ లోడ్
config = JaddangiConfig.from_pretrained(model_id)

# మోడల్ ఇనిషియలైజ్ (FP16 లో)
model = JaddangiForCausalLM(config).half()

# నేరుగా safetensors నుండి బరువులు చదవడం (metadata బగ్ రాదు!)
print("🔧 బరువులను ఎక్కిస్తున్నాము...")
state_dict = load_file("model.safetensors")

# క్రాష్ రాకుండా అనవసరమైన కీలు తీసేయడం
for k in ["cos_cached", "sin_cached", "lm_head.weight"]:
    state_dict.pop(k, None)

# మోడల్ లోకి ఎక్కించడం
model.load_state_dict(state_dict, strict=False)

# ⚠️ CRITICAL FIX:
model.lm_head.weight = model.model.wte.weight

# మోడల్‌ను GPU కి పంపడం
model = model.to("cuda")
model.eval()
print("✅ మోడల్ విజయవంతంగా GPU లోకి లోడ్ అయ్యింది (Bypassed HF Bug)!")

# ==========================================
# 3. టోకనైజర్ లోడ్ చేయడం
# ==========================================
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("telugu_spm.model")
UNK_ID = tokenizer.unk_id() if tokenizer.unk_id() >= 0 else 0

# ==========================================
# 4. అల్టిమేట్ సేఫ్ జనరేషన్ ఫంక్షన్ 
# ==========================================
def generate_ultimate_text(prompt, max_new_tokens=150, min_new_tokens=5, temperature=0.75, top_k=40, top_p=0.92, repetition_penalty=1.15):
    if re.search(r'[a-zA-Z]', prompt):
        return "🚫 దయచేసి కేవలం తెలుగులో మాత్రమే టైప్ చేయండి."

    input_ids = tokenizer.encode(prompt, out_type=int)
    idx = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    generated_ids = input_ids.copy()

    with torch.no_grad():
        for step in range(max_new_tokens):
            idx_cond = idx[:, -384:]
            outputs = model(input_ids=idx_cond)
            logits = outputs.logits[:, -1, :].clone() / temperature

            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if token_id < logits.size(1):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

            if step < min_new_tokens and tokenizer.eos_id() >= 0:
                logits[0, tokenizer.eos_id()] = float("-inf")

            if UNK_ID < logits.size(1):
                logits[0, UNK_ID] = float("-inf")

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            if top_p < 1.0 and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() == 0:
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()
                
            generated_ids.append(next_token)
            idx = torch.cat((idx, torch.tensor([[next_token]], dtype=torch.long, device="cuda")), dim=1)

            if tokenizer.eos_id() >= 0 and next_token == tokenizer.eos_id():
                break

    new_tokens = generated_ids[len(input_ids):]
    output_text = tokenizer.decode(new_tokens).strip()
    
    if not output_text:
        return "⚠️ (మోడల్ ఏమీ రాయలేదు, దయచేసి వేరే వాక్యం ఇవ్వండి)"
        
    return output_text

# ==========================================
# 5. ఇంటరాక్టివ్ చాట్ లూప్
# ==========================================
print("\n" + "="*60)
print("🧠 Jaddangi-3B – Manual Load Stable Chat")
print("   మీరు తెలుగులో ఏదైనా సగం వాక్యం ఇవ్వండి (ఉదా: సూర్యుడు తూర్పున)")
print("   చాట్ ఆపడానికి 'quit' లేదా 'exit' అని టైప్ చేయండి.")
print("="*60 + "\n")

while True:
    try:
        user_input = input("📝 Prompt: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if user_input.lower() in ("quit", "exit", "q"):
        print("\n👋 ధన్యవాదాలు! చాట్ ముగిసింది.")
        break
        
    if not user_input:
        continue

    print("-" * 50)
    output = generate_ultimate_text(user_input, max_new_tokens=150, min_new_tokens=5)
    print(output)
    print("-" * 50 + "\n")

# Jaddangi AI Telugu 3B

A Telugu-focused large language model project that expands, adapts, and runs a 3B-parameter causal language model for Telugu text generation. The repository contains research scripts for checkpoint expansion, continual pretraining (CPT), LoRA adapter merging, and inference for interactive Telugu chat.

This project is built around a custom Telugu GPT architecture using:

- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- SwiGLU feed-forward blocks
- SentencePiece tokenizer for Telugu
- 3B-parameter model configuration
- Hugging Face integration for hosting and downloading weights

## Project Summary

The repo aims to create and deploy a Telugu language model that starts from a smaller 1B model and expands it into a larger 3B model using Net2Net-inspired structured growth, then optionally continues training with CPT and LoRA-based adaptation.

The main focus is:

- model expansion from 1B to 3B without losing learned behavior
- memory-safe checkpoint growth
- post-expansion training / wake-up
- LoRA merging into a final merged model
- Telugu-only chat generation and inference

## Model Architecture

The 3B model configuration used across the project is:

- Vocabulary size: 32,000
- Hidden size: 1,728
- Layers: 68
- Attention heads: 24
- KV heads: 6
- Max sequence length: 384
- RoPE theta: 10,000
- Expansion factor: 4

The model uses:

- RMSNorm
- Rotary position embeddings
- GQA attention with repeat-interleave behavior
- SwiGLU MLP blocks
- Tied embedding and output projection (`wte.weight` tied to `lm_head.weight`)

## Repository Structure

```text
.
├── 3B-Net-to-Net.py           # expands a 1B checkpoint into a 3B checkpoint
├── 3b-cpt.py                  # CPT / "wake-up" training script with LoRA + QLoRA
├── 3bloramerge.py             # merges LoRA weights into the base 3B model
├── Hf-3b-infereance.py        # manual Hugging Face model loading + generation loop
├── infereance.py              # interactive inference for the merged model
├── infereance2.py             # alternative interactive inference script
├── output.txt                 # sample output / generated text log
├── README.md                  # project documentation
└── .gitignore                 # repository config (if present in your clone)
```

## Why this project exists

This repository represents a practical workflow for building a Telugu LLM from an existing checkpoint:

1. Start with a smaller 1B checkpoint
2. Expand the model width and depth using Net2Net-style tensor growth
3. Preserve prior knowledge while adding new capacity
4. Continue pretraining with a Telugu corpus
5. Merge LoRA layers into the base architecture for a clean final model
6. Inference on text prompts in Telugu

This is especially useful for low-resource language modeling where training a large model from scratch is expensive and difficult.

## Core Scripts

### 1) 3B-Net-to-Net.py

This script performs the model expansion from a 1B checkpoint to a 3B checkpoint.

Key features:

- downloads the old checkpoint from Hugging Face if missing
- defines old and new model configurations
- width-expands existing embedding and transformer weights
- adds newly initialized deeper layers with near-zero residual identity behavior
- keeps memory usage under control using FP16 tensors and garbage collection
- validates the expanded checkpoint for NaN/Inf values
- saves and uploads the final 3B checkpoint

This is the main "growth" phase where the model is scaled up structurally while keeping learned information from the smaller model.

### 2) 3b-cpt.py

This is the continual pretraining / wake-up script.

It does the following:

- loads the expanded 3B model
- converts selected layers to 4-bit quantized format using bitsandbytes
- injects LoRA adapters into attention and MLP projections
- loads a Telugu corpus from a Hugging Face dataset repository
- tokenizes the corpus using SentencePiece
- trains with cross-entropy loss
- saves checkpoints periodically
- uploads intermediate checkpoints to Hugging Face

The script is designed for memory-conscious training on GPU hardware and is intended to adapt and improve the 3B model on Telugu data.

### 3) 3bloramerge.py

This script merges LoRA adapters into the base model weights.

Key steps:

- loads the base 3B model
- injects LoRA adapters
- loads CPT-trainable weights
- merges the low-rank updates back into the base linear layers
- removes the extra LoRA wrappers
- saves the final merged checkpoint
- uploads it to Hugging Face

This produces a cleaner inference-ready base model without needing to keep LoRA adapters separately.

### 4) Inference Scripts

#### infereance.py

A simple interactive inference script that:

- loads a merged model checkpoint
- loads a SentencePiece tokenizer
- validates the prompt is in Telugu (optionally)
- generates text using sampling with temperature, top-k, and top-p
- supports chat-style prompting from the terminal

#### infereance2.py

A second inference implementation with the same architecture and logic, mostly a duplicate or alternative version of the main inference script.

#### Hf-3b-infereance.py

This script shows how to manually download model files from Hugging Face and load the custom model without relying on the normal HF pipeline bug workaround.

It includes:

- manual safetensors-based loading
- config and model import from custom files
- manual weight injection logic
- stable chat generation loop
- strict Telugu-only prompt filtering

## Training and Data Pipeline

The repository expects a Telugu dataset stream and a SentencePiece tokenizer file such as:

- `telugu_spm.model`
- dataset from a Hugging Face repository, such as `VenkataRamanaKurumallajaddangi/Telugu-Pretraining-Corpus`

The training logic uses:

- streaming dataset loading via `datasets`
- token sequence chunking into fixed-length context windows
- causal language modeling objective
- periodic checkpoint saving and resume support

## Required Dependencies

Install the core Python packages:

```bash
pip install torch sentencepiece transformers datasets huggingface_hub safetensors bitsandbytes
```

Depending on the environment, you may also need:

```bash
pip install accelerate peft
```

If using GPU training, ensure CUDA is configured correctly for your PyTorch build.

## Environment Setup

Most scripts assume:

- Python 3.10+
- CUDA-capable GPU for reasonable training/inference speed
- Hugging Face authentication token
- access to the model repository and data repository

Example environment variables:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Some scripts also directly call `login(token=HF_TOKEN)` and expect the token to be set in code or environment.

## Usage Guide

### Run the Net2Net expansion

```bash
python 3B-Net-to-Net.py
```

This creates a larger expanded checkpoint from the original 1B model.

### Run CPT training

```bash
python 3b-cpt.py
```

This performs the model "wake-up" training and saves checkpoints.

### Merge LoRA into the base model

```bash
python 3bloramerge.py
```

This creates a cleaner, merged model checkpoint for deployment or inference.

### Run interactive inference

```bash
python infereance.py
```

or

```bash
python infereance2.py
```

Then type Telugu prompts such as:

```text
సూర్యుడు తూర్పున ఉదయించేడు
తెలుగు exemplars మరియు సరళమైన వాక్యనిర్మాణం
హైస్కూల్ విద్య తరచుగా చిన్న విషయాలను బాగా అర్థం చేసుకుంటుంది
```

## Notes on Prompting

The model is intended primarily for Telugu text generation. Some scripts explicitly reject prompts containing Latin letters to reduce mixing of English and Telugu.

Recommended usage pattern:

- Use fully Telugu prompts
- Keep prompts concise and natural
- Tune `temperature`, `top_k`, and `top_p` for generation style
- Use longer context windows for more coherent output

## Model Behavior and Quality Expectations

This project is a research-oriented model-building pipeline rather than a turnkey commercial product. Results will vary based on:

- dataset quality and coverage
- training duration
- checkpoint quality
- generation parameters
- GPU memory and precision settings

A good model depends not only on architecture, but also on the quality of the Telugu corpus and the training recipe.

## Known Characteristics of the Codebase

The repository contains several scripts that are closely related and intentionally re-implemented in slightly different forms:

- repeated architecture definitions across scripts
- multiple inference variants
- custom memory-safe checkpoint loading logic
- Hugging Face download and upload flows
- research/debugging scripts rather than a single packaged CLI

This is common in exploratory model development work and indicates an active research workflow rather than a production package.

## Recommended Future Improvements

To make the repo more polished for broader usage, future work could include:

- creating a single `requirements.txt`
- adding a proper `config.yaml` or CLI interface
- standardizing naming (`inference.py` vs `infereance.py`)
- adding a proper training command runner
- documenting expected HF repo structure
- adding example outputs and benchmark notes
- creating a reusable package structure for model loading

## License and Use

This repository is a research and experimentation project focused on Telugu language modeling. Use it responsibly and respect the licensing terms of any datasets, model weights, or third-party libraries you download or use.

## Acknowledgements

The project integrates:

- PyTorch
- Hugging Face Hub / datasets
- SentencePiece
- bitsandbytes
- custom research model code for Telugu text generation

## Final Note

The repository is best understood as a full experimental pipeline for building a Telugu 3B LM from a smaller checkpoint, training it on Telugu text, and running local generation. It combines model-scaling techniques, continual pretraining, LoRA adaptation, and interactive inference in a single focused codebase.

If you want, I can also turn this into a more polished GitHub README in a specific style such as:

- research-paper style
- startup/product style
- minimal professional README
- Telugu README version
- badge-heavy open-source README

I can also generate a `requirements.txt` and a `run.sh` helper for this repo.

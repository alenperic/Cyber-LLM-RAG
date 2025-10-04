# Models Directory

This directory stores LLM models, adapters, and training checkpoints.

## Structure

```
models/
├── base/          # Base pretrained models (e.g., Llama-2-7b)
├── adapters/      # QLoRA/LoRA adapter weights
└── checkpoints/   # Training checkpoints
```

## Model Storage

Models are typically downloaded from HuggingFace Hub on first use:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    cache_dir="./models/base"
)
```

**Note:** Model files are excluded from git via `.gitignore` due to large sizes (5-50GB).

## Supported Models

- Llama-2 (7B, 13B, 70B)
- Mistral (7B)
- Mixtral (8x7B)
- Any HuggingFace causal LM

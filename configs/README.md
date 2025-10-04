# Configs Directory

This directory stores configuration files for training, inference, and deployment.

## Configuration Files

Add YAML or JSON configuration files here for:

- **Training configs**: Hyperparameters for QLoRA/CPT
- **Model configs**: Model architecture settings
- **RAG configs**: Retrieval parameters (top-k, chunk size, etc.)
- **Deployment configs**: Ray Serve autoscaling, resource limits

## Example

```yaml
# configs/qlora_training.yaml
model_name: "meta-llama/Llama-2-7b-chat-hf"
lora_rank: 16
lora_alpha: 32
learning_rate: 2e-4
batch_size: 4
gradient_accumulation_steps: 4
max_steps: 1000
```

Usage:

```python
import yaml

with open("configs/qlora_training.yaml") as f:
    config = yaml.safe_load(f)
```

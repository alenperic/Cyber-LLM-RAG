# Notebooks Directory

This directory stores Jupyter notebooks for exploration, experimentation, and demos.

## Suggested Notebooks

Create notebooks for:

- **Data Exploration**: Analyze ATT&CK, CWE, CVE datasets
- **RAG Evaluation**: Test retrieval quality and generation
- **Model Fine-tuning**: Experiment with QLoRA hyperparameters
- **Inference Demos**: Interactive chatbot interface

## Example Notebook Structure

```python
# 01_data_exploration.ipynb

import sys
sys.path.append("..")

from src.rag.data_processing import ATTACKProcessor

processor = ATTACKProcessor()
docs = processor.load_attack_data("data/raw/attack/")
print(f"Loaded {len(docs)} ATT&CK documents")
```

## Running Notebooks

```bash
pip install jupyter
jupyter notebook
```

**Note:** Notebook checkpoints (`.ipynb_checkpoints/`) are excluded from git.

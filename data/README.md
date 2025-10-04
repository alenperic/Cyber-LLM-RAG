# Data Directory

This directory stores the cybersecurity knowledge base for the RAG system.

## Structure

```
data/
├── raw/           # Downloaded raw data from sources
├── processed/     # Processed and chunked documents
└── embeddings/    # Vector store (ChromaDB database)
```

## Populating Data

Run the data download script:

```bash
python scripts/download_data.py
```

This will download:
- MITRE ATT&CK (STIX 2.1 JSON)
- CWE (XML)
- CAPEC (XML)
- NVD CVEs 2020-2025 (JSON)
- Sigma Detection Rules (YAML)

**Note:** These directories are excluded from git via `.gitignore` due to large file sizes (~2-5GB).

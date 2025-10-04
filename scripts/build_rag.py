#!/usr/bin/env python3
"""
End-to-end script to build RAG MVP.
Downloads data, processes it, builds vector store.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from download_data import main as download_data
from rag.data_processing import process_all_data
from rag.vector_store import build_vector_store


def main():
    """Build complete RAG system"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    embeddings_dir = project_root / "data" / "embeddings"

    print("=" * 70)
    print("BUILDING CYBERRAG MVP")
    print("=" * 70)

    # Step 1: Download data
    print("\n[STEP 1/3] Downloading cybersecurity datasets...")
    download_data()

    # Step 2: Process data
    print("\n[STEP 2/3] Processing and extracting documents...")
    documents = process_all_data(data_dir)

    # Step 3: Build vector store
    print("\n[STEP 3/3] Building vector store with embeddings...")
    vector_store = build_vector_store(
        documents=documents,
        persist_dir=embeddings_dir,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\n" + "=" * 70)
    print("✓ RAG MVP BUILD COMPLETE!")
    print("=" * 70)
    print(f"\nVector store saved to: {embeddings_dir}")
    print(f"Total documents indexed: {len(documents)}")
    print("\nNext steps:")
    print("1. Deploy RAG service:")
    print(f"   python src/serving/ray_serve_app.py --vector-store {embeddings_dir}")
    print("\n2. Or run CPT + QLoRA fine-tuning:")
    print(f"   python src/training/continual_pretrain.py --data-dir {data_dir}")
    print("   python src/training/qlora_finetune.py --base-model <cpt-model-path>")


if __name__ == "__main__":
    main()

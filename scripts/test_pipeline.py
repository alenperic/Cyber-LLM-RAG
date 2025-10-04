#!/usr/bin/env python3
"""
Test script to validate the RAG pipeline components.
Run this after building the vector store to ensure everything works.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test all required imports"""
    print("\n[TEST 1/5] Testing imports...")
    try:
        import torch
        import transformers
        import chromadb
        from sentence_transformers import SentenceTransformer
        print("  ✓ Core ML libraries imported")

        from rag.data_processing import Document, AttackProcessor
        from rag.vector_store import CyberRAGStore
        from rag.rag_pipeline import CyberRAGPipeline
        print("  ✓ Custom RAG modules imported")

        print("✓ All imports successful\n")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}\n")
        return False


def test_data_processing():
    """Test data processing functions"""
    print("[TEST 2/5] Testing data processing...")
    try:
        from rag.data_processing import Document

        # Create test document
        doc = Document(
            id="test-001",
            content="Test cybersecurity document about T1059",
            metadata={"name": "Test", "type": "technique"},
            source_type="attack"
        )

        assert doc.id == "test-001"
        assert "T1059" in doc.content
        print("  ✓ Document creation works")
        print("✓ Data processing test passed\n")
        return True
    except Exception as e:
        print(f"✗ Data processing failed: {e}\n")
        return False


def test_vector_store():
    """Test vector store operations"""
    print("[TEST 3/5] Testing vector store...")
    try:
        from rag.vector_store import CyberRAGStore
        from rag.data_processing import Document
        import tempfile

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize store
            store = CyberRAGStore(
                persist_dir=Path(tmpdir),
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("  ✓ Vector store initialized")

            # Create test documents
            test_docs = [
                Document(
                    id="T1059",
                    content="Command and Scripting Interpreter: Adversaries may abuse command interpreters",
                    metadata={"name": "T1059", "type": "technique"},
                    source_type="attack"
                ),
                Document(
                    id="CVE-2021-44228",
                    content="Log4Shell vulnerability in Apache Log4j allows remote code execution",
                    metadata={"cve_id": "CVE-2021-44228", "cvss_score": 10.0},
                    source_type="cve"
                ),
            ]

            # Add documents
            store.add_documents(test_docs)
            print("  ✓ Documents added to vector store")

            # Test search
            results = store.search("command execution vulnerability", n_results=2)
            assert len(results['ids'][0]) == 2
            print("  ✓ Search works")

            # Get stats
            stats = store.get_stats()
            assert stats['total_documents'] == 2
            print("  ✓ Statistics retrieval works")

        print("✓ Vector store test passed\n")
        return True
    except Exception as e:
        print(f"✗ Vector store test failed: {e}\n")
        return False


def test_vector_store_persistence():
    """Test if vector store was built"""
    print("[TEST 4/5] Testing vector store persistence...")
    try:
        project_root = Path(__file__).parent.parent
        embeddings_dir = project_root / "data" / "embeddings"

        if not embeddings_dir.exists():
            print(f"  ⚠ Vector store not found at {embeddings_dir}")
            print("  → Run 'python scripts/build_rag.py' first")
            return False

        from rag.vector_store import CyberRAGStore

        # Load existing store
        store = CyberRAGStore(
            persist_dir=embeddings_dir,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Get stats
        stats = store.get_stats()
        print(f"  ✓ Loaded vector store with {stats['total_documents']} documents")
        print(f"  ✓ Source distribution: {json.dumps(stats['source_distribution'], indent=4)}")

        # Test search
        results = store.search("What is ATT&CK technique T1059?", n_results=3)
        print(f"  ✓ Search returned {len(results['ids'][0])} results")

        print("✓ Vector store persistence test passed\n")
        return True
    except Exception as e:
        print(f"✗ Vector store persistence test failed: {e}\n")
        return False


def test_rag_pipeline_mock():
    """Test RAG pipeline with mock (no GPU required)"""
    print("[TEST 5/5] Testing RAG pipeline (mock mode)...")
    try:
        from rag.vector_store import CyberRAGStore
        from rag.data_processing import Document
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal vector store
            store = CyberRAGStore(
                persist_dir=Path(tmpdir),
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            )

            test_docs = [
                Document(
                    id="T1059",
                    content="ATT&CK Technique T1059: Command and Scripting Interpreter",
                    metadata={"name": "T1059"},
                    source_type="attack"
                ),
            ]
            store.add_documents(test_docs)

            # Test retrieval (without LLM)
            from rag.rag_pipeline import CyberRAGPipeline

            # We can't test full pipeline without GPU, but we can test retrieval
            class MockRAGPipeline:
                def __init__(self, vector_store):
                    self.vector_store = vector_store

                def retrieve(self, query, top_k=5):
                    results = self.vector_store.search(query, n_results=top_k)
                    retrieved_docs = []
                    for i in range(len(results['ids'][0])):
                        retrieved_docs.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i]
                        })
                    return retrieved_docs

            pipeline = MockRAGPipeline(store)
            results = pipeline.retrieve("What is T1059?", top_k=1)

            assert len(results) == 1
            assert "T1059" in results[0]['content']
            print("  ✓ Retrieval pipeline works")

        print("✓ RAG pipeline test passed (mock mode)\n")
        print("  ℹ Full pipeline test requires GPU and model download")
        return True
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("CYBERRAG PIPELINE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Data Processing", test_data_processing),
        ("Vector Store", test_vector_store),
        ("Vector Store Persistence", test_vector_store_persistence),
        ("RAG Pipeline (Mock)", test_rag_pipeline_mock),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}\n")
            results.append((name, False))

    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! Pipeline is ready.")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

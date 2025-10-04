#!/usr/bin/env python3
"""
Core functionality tests (no heavy ML dependencies required).
Tests data processing, file I/O, and basic pipeline logic.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test core imports"""
    print("\n[TEST 1/6] Testing core imports...")
    try:
        import json
        import xml.etree.ElementTree as ET
        import requests
        import yaml
        from pathlib import Path
        print("  ✓ Standard library imports OK")

        from rag.data_processing import Document, AttackProcessor
        print("  ✓ Custom modules import OK")

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_document_class():
    """Test Document dataclass"""
    print("\n[TEST 2/6] Testing Document class...")
    try:
        from rag.data_processing import Document

        doc = Document(
            id="test-001",
            content="Test content about T1059",
            metadata={"name": "Test", "type": "technique"},
            source_type="attack"
        )

        assert doc.id == "test-001"
        assert "T1059" in doc.content
        assert doc.metadata["type"] == "technique"
        assert doc.source_type == "attack"

        print("  ✓ Document creation works")
        print("  ✓ Document attributes accessible")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_attack_processor():
    """Test ATT&CK STIX processor"""
    print("\n[TEST 3/6] Testing ATT&CK processor...")
    try:
        import json
        import tempfile
        from rag.data_processing import AttackProcessor

        # Create sample STIX
        sample = {
            "type": "bundle",
            "objects": [
                {
                    "type": "attack-pattern",
                    "id": "attack-pattern--test",
                    "name": "Test Technique",
                    "description": "Test description",
                    "kill_chain_phases": [{
                        "kill_chain_name": "mitre-attack",
                        "phase_name": "execution"
                    }],
                    "external_references": [{
                        "source_name": "mitre-attack",
                        "external_id": "T1234"
                    }]
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample, f)
            temp_file = f.name

        try:
            processor = AttackProcessor(Path(temp_file))
            docs = processor.process()

            assert len(docs) == 1
            assert docs[0].id == "T1234"
            assert "Test Technique" in docs[0].content

            print(f"  ✓ Processed {len(docs)} document(s)")
            print(f"  ✓ Extracted technique: {docs[0].id}")
            return True
        finally:
            Path(temp_file).unlink()

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_real_attack_data():
    """Test processing real ATT&CK data if available"""
    print("\n[TEST 4/6] Testing real ATT&CK data...")
    try:
        from rag.data_processing import AttackProcessor

        # Check if we downloaded ATT&CK data
        data_dir = Path(__file__).parent.parent / "data" / "raw" / "attack"
        stix_file = data_dir / "enterprise-attack.json"

        if not stix_file.exists():
            print("  ⚠ ATT&CK data not found (skipping)")
            return True

        processor = AttackProcessor(stix_file)
        docs = processor.process()

        # Count by type
        types = {}
        for doc in docs:
            doc_type = doc.metadata.get('type', 'unknown')
            types[doc_type] = types.get(doc_type, 0) + 1

        assert len(docs) > 0, "No documents extracted"
        assert types.get('technique', 0) > 0, "No techniques found"

        print(f"  ✓ Processed {len(docs)} documents")
        print(f"  ✓ Techniques: {types.get('technique', 0)}")
        print(f"  ✓ Mitigations: {types.get('mitigation', 0)}")
        print(f"  ✓ Groups: {types.get('group', 0)}")

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test project file structure"""
    print("\n[TEST 5/6] Testing file structure...")
    try:
        project_root = Path(__file__).parent.parent

        # Check key directories
        required_dirs = [
            "src/rag",
            "src/training",
            "src/serving",
            "scripts",
            "k8s",
            "data",
            "models",
        ]

        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Missing directory: {dir_path}"

        print(f"  ✓ All required directories exist")

        # Check key files
        required_files = [
            "README.md",
            "requirements.txt",
            "src/rag/data_processing.py",
            "src/rag/vector_store.py",
            "src/rag/rag_pipeline.py",
            "src/training/qlora_finetune.py",
            "src/training/continual_pretrain.py",
            "src/serving/ray_serve_app.py",
            "scripts/download_data.py",
            "scripts/build_rag.py",
        ]

        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Missing file: {file_path}"

        print(f"  ✓ All key files exist")

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_python_syntax():
    """Test Python syntax on all files"""
    print("\n[TEST 6/6] Testing Python syntax...")
    try:
        import py_compile

        project_root = Path(__file__).parent.parent
        python_files = list(project_root.rglob("*.py"))

        # Exclude venv
        python_files = [f for f in python_files if 'venv' not in str(f)]

        failed = []
        for py_file in python_files:
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError:
                failed.append(py_file)

        if failed:
            print(f"  ✗ Syntax errors in {len(failed)} file(s):")
            for f in failed:
                print(f"    - {f}")
            return False

        print(f"  ✓ All {len(python_files)} Python files have valid syntax")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("CYBERRAG CORE FUNCTIONALITY TEST SUITE")
    print("=" * 70)
    print("\nTesting core functionality without heavy ML dependencies...")

    tests = [
        ("Core Imports", test_imports),
        ("Document Class", test_document_class),
        ("ATT&CK Processor", test_attack_processor),
        ("Real ATT&CK Data", test_real_attack_data),
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All core tests passed!")
        print("\nNote: Full RAG pipeline requires:")
        print("  - PyTorch")
        print("  - Transformers")
        print("  - ChromaDB")
        print("  - Sentence-Transformers")
        print("\nInstall with: pip install -r requirements.txt")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

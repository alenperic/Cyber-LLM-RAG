"""
Vector store implementation using ChromaDB for RAG.
Handles document embedding, indexing, and retrieval.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from tqdm import tqdm

from .data_processing import Document


class CyberRAGStore:
    """Vector store for cybersecurity knowledge base"""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "cyber_knowledge",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store with ChromaDB.

        Args:
            persist_dir: Directory to persist the vector database
            collection_name: Name of the collection
            embedding_model: HuggingFace model for embeddings
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.persist_dir)
        ))

        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Cybersecurity knowledge base (ATT&CK, CWE, CVE, Sigma)"}
        )

    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """
        Add documents to vector store with embeddings.

        Args:
            documents: List of Document objects
            batch_size: Number of documents to process in each batch
        """
        print(f"Adding {len(documents)} documents to vector store...")

        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]

            # Prepare batch data
            ids = [doc.id for doc in batch]
            contents = [doc.content for doc in batch]
            metadatas = [
                {**doc.metadata, "source_type": doc.source_type}
                for doc in batch
            ]

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                contents,
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )

        # Persist to disk
        self.client.persist()
        print(f"✓ Added {len(documents)} documents to vector store")

    def search(
        self,
        query: str,
        n_results: int = 5,
        source_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Semantic search over knowledge base.

        Args:
            query: Search query
            n_results: Number of results to return
            source_filter: Filter by source types (e.g., ["attack", "cve"])

        Returns:
            Dictionary with results, distances, and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()

        # Build where clause for filtering
        where = None
        if source_filter:
            where = {"source_type": {"$in": source_filter}}

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        count = self.collection.count()

        # Get source type distribution
        source_types = {}
        if count > 0:
            # Sample to get source types
            sample = self.collection.get(limit=min(count, 1000))
            for metadata in sample.get("metadatas", []):
                source = metadata.get("source_type", "unknown")
                source_types[source] = source_types.get(source, 0) + 1

        return {
            "total_documents": count,
            "source_distribution": source_types,
            "collection_name": self.collection.name,
            "persist_dir": str(self.persist_dir)
        }


def build_vector_store(
    documents: List[Document],
    persist_dir: Path,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> CyberRAGStore:
    """
    Build vector store from documents.

    Args:
        documents: List of processed documents
        persist_dir: Directory to persist vector store
        embedding_model: Embedding model to use

    Returns:
        CyberRAGStore instance
    """
    store = CyberRAGStore(
        persist_dir=persist_dir,
        embedding_model=embedding_model
    )

    store.add_documents(documents)

    # Print statistics
    stats = store.get_stats()
    print("\nVector Store Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Source distribution: {json.dumps(stats['source_distribution'], indent=2)}")

    return store

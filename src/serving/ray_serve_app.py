"""
Ray Serve deployment for CyberRAG API.
Provides REST API for cybersecurity Q&A with RAG.
"""

import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.vector_store import CyberRAGStore
from rag.rag_pipeline import CyberRAGPipeline, CyberRAGStreamingPipeline


# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question about cybersecurity")
    top_k: int = Field(5, description="Number of documents to retrieve")
    source_filter: Optional[List[str]] = Field(
        None,
        description="Filter by source types: attack, cwe, capec, cve, sigma"
    )
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    stream: bool = Field(False, description="Enable streaming response")


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    vector_store_stats: Dict[str, Any]


app = FastAPI(
    title="CyberRAG API",
    description="Cybersecurity RAG system powered by ATT&CK, CWE, CVE, and Sigma",
    version="1.0.0"
)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 5,
    }
)
@serve.ingress(app)
class CyberRAGService:
    """Ray Serve deployment for CyberRAG"""

    def __init__(
        self,
        vector_store_path: str,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG service.

        Args:
            vector_store_path: Path to persisted vector store
            model_name: LLM model name or path to adapter
            embedding_model: Embedding model name
        """
        print(f"Initializing CyberRAG Service...")
        print(f"Vector store: {vector_store_path}")
        print(f"Model: {model_name}")

        # Load vector store
        self.vector_store = CyberRAGStore(
            persist_dir=Path(vector_store_path),
            embedding_model=embedding_model
        )

        # Initialize RAG pipeline
        self.rag_pipeline = CyberRAGStreamingPipeline(
            vector_store=self.vector_store,
            model_name=model_name,
            load_in_4bit=True
        )

        self.model_name = model_name

        print("✓ CyberRAG Service initialized")

    @app.get("/health", response_model=HealthResponse)
    async def health(self):
        """Health check endpoint"""
        stats = self.vector_store.get_stats()
        return HealthResponse(
            status="healthy",
            model=self.model_name,
            vector_store_stats=stats
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(self, request: QueryRequest):
        """
        Query endpoint for RAG-based Q&A.

        Returns JSON response with answer and sources.
        """
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Use /query/stream endpoint for streaming responses"
            )

        try:
            result = self.rag_pipeline.query(
                question=request.question,
                top_k=request.top_k,
                source_filter=request.source_filter,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                return_sources=True
            )

            return QueryResponse(**result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query/stream")
    async def query_stream(self, request: QueryRequest):
        """
        Streaming query endpoint for real-time responses.

        Returns Server-Sent Events (SSE) stream.
        """
        try:
            # Retrieve documents
            retrieved_docs = self.rag_pipeline.retrieve(
                query=request.question,
                top_k=request.top_k,
                source_filter=request.source_filter
            )

            # Build prompt
            prompt = self.rag_pipeline.build_prompt(
                request.question,
                retrieved_docs
            )

            # Stream generator
            async def generate():
                # First send sources
                sources = [
                    {
                        "id": doc['id'],
                        "source_type": doc['metadata'].get('source_type'),
                        "relevance": 1.0 - doc['distance']
                    }
                    for doc in retrieved_docs
                ]
                yield f"data: {json.dumps({'sources': sources})}\n\n"

                # Stream answer tokens
                for token in self.rag_pipeline.generate_stream(
                    prompt=prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/search")
    async def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None
    ):
        """
        Direct vector search endpoint (no LLM generation).

        Args:
            query: Search query
            top_k: Number of results
            source_filter: Comma-separated source types

        Returns:
            Retrieved documents
        """
        try:
            filters = source_filter.split(",") if source_filter else None

            retrieved = self.rag_pipeline.retrieve(
                query=query,
                top_k=top_k,
                source_filter=filters
            )

            return {"results": retrieved}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def deploy_service(
    vector_store_path: str,
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    host: str = "0.0.0.0",
    port: int = 8000
):
    """
    Deploy CyberRAG service with Ray Serve.

    Args:
        vector_store_path: Path to vector store
        model_name: LLM model name or adapter path
        embedding_model: Embedding model name
        host: Host to bind to
        port: Port to bind to
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Deploy service
    serve.start(detached=True, http_options={"host": host, "port": port})

    handle = serve.run(
        CyberRAGService.bind(
            vector_store_path=vector_store_path,
            model_name=model_name,
            embedding_model=embedding_model
        ),
        name="cyber_rag",
        route_prefix="/"
    )

    print(f"\n✓ CyberRAG API deployed at http://{host}:{port}")
    print(f"  - Health: http://{host}:{port}/health")
    print(f"  - Query: http://{host}:{port}/query")
    print(f"  - Stream: http://{host}:{port}/query/stream")
    print(f"  - Search: http://{host}:{port}/search")
    print(f"  - Docs: http://{host}:{port}/docs")

    return handle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy CyberRAG with Ray Serve")
    parser.add_argument(
        "--vector-store",
        type=str,
        required=True,
        help="Path to vector store directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model name or path to LoRA adapter"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")

    args = parser.parse_args()

    deploy_service(
        vector_store_path=args.vector_store,
        model_name=args.model,
        embedding_model=args.embedding_model,
        host=args.host,
        port=args.port
    )

    # Keep service running
    import time
    while True:
        time.sleep(10)

"""
RAG pipeline for cybersecurity Q&A.
Combines retrieval from vector store with LLM generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
from pathlib import Path

from .vector_store import CyberRAGStore


class CyberRAGPipeline:
    """End-to-end RAG pipeline for cybersecurity queries"""

    def __init__(
        self,
        vector_store: CyberRAGStore,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        load_in_4bit: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize RAG pipeline.

        Args:
            vector_store: CyberRAGStore instance
            model_name: HuggingFace model name or path to LoRA adapter
            load_in_4bit: Use 4-bit quantization
            device: Device to use for inference
        """
        self.vector_store = vector_store
        self.device = device

        print(f"Loading model: {model_name}")

        # Configure quantization
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None
        )
        self.model.eval()

        print("✓ Model loaded successfully")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            source_filter: Filter by source type

        Returns:
            List of retrieved documents with metadata
        """
        results = self.vector_store.search(
            query=query,
            n_results=top_k,
            source_filter=source_filter
        )

        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        return retrieved_docs

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build RAG prompt with retrieved context.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            system_prompt: Optional custom system prompt

        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = (
                "You are a cybersecurity expert assistant with access to "
                "knowledge from MITRE ATT&CK, CWE, CAPEC, NVD CVEs, and Sigma detection rules. "
                "Provide accurate, detailed answers based on the provided context. "
                "If you're unsure or the context doesn't contain relevant information, say so."
            )

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source_type', 'unknown')
            content = doc['content'][:500]  # Truncate for token limits
            context_parts.append(f"[{i}] Source: {source}\n{content}\n")

        context = "\n".join(context_parts)

        # Format prompt (Llama-2 chat format)
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Context:
{context}

Question: {query} [/INST]"""

        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response using LLM.

        Args:
            prompt: Formatted prompt with context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def query(
        self,
        question: str,
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        End-to-end RAG query.

        Args:
            question: User question
            top_k: Number of documents to retrieve
            source_filter: Filter by source type
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            return_sources: Include source documents in response

        Returns:
            Dictionary with answer and optionally sources
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(
            query=question,
            top_k=top_k,
            source_filter=source_filter
        )

        # Build prompt
        prompt = self.build_prompt(question, retrieved_docs)

        # Generate response
        answer = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        result = {"answer": answer}

        if return_sources:
            result["sources"] = [
                {
                    "id": doc['id'],
                    "source_type": doc['metadata'].get('source_type'),
                    "content_preview": doc['content'][:200] + "...",
                    "relevance_score": 1.0 - doc['distance']  # Convert distance to similarity
                }
                for doc in retrieved_docs
            ]

        return result


class CyberRAGStreamingPipeline(CyberRAGPipeline):
    """Streaming version of RAG pipeline for real-time responses"""

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Generate response with streaming.

        Args:
            prompt: Formatted prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Yields:
            Generated tokens as they are produced
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()

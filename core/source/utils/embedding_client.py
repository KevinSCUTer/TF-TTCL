"""
Embedding Client
Abstraction layer for semantic similarity calculations.
Responsible for calling the vLLM Embedding API to compute text similarities.
"""

import os
from typing import Optional, List, Union
from dataclasses import dataclass
from openai import OpenAI
import numpy as np


@dataclass
class EmbeddingResult:
    """Embedding Result Data Structure"""
    embeddings: np.ndarray  # shape: (n_texts, embedding_dim)
    texts: List[str]        # Original text list


class EmbeddingClient:
    """
    Embedding Client Wrapper
    - Calls vLLM Embedding API.
    - Computes semantic similarity between texts.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:10000/v1",
        model_name: str = "Qwen3-Embedding-0.6B"
    ):
        """
        Initialize Embedding Client
        
        Args:
            api_key: API Key, defaults to "EMPTY" if None.
            base_url: Embedding API service URL.
            model_name: Embedding model name.
        """
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "EMPTY")
        
        # Handle base_url: strip "/embeddings" suffix for OpenAI client compatibility.
        if base_url.endswith("/embeddings"):
            self.base_url = base_url[:-11]  # Remove "/embeddings"
        elif base_url.endswith("/embeddings/"):
            self.base_url = base_url[:-12]
        else:
            self.base_url = base_url
            
        self.model_name = model_name
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Retrieve embedding vectors for text.
        
        Args:
            texts: A single string or a list of strings.
            
        Returns:
            Numpy array of embeddings, shape: (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        
        # Sort by index to maintain correct order.
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = np.array([data.embedding for data in sorted_data])
        
        return embeddings
    
    def compute_similarity(
        self,
        query: str,
        documents: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute semantic similarity between query and multiple documents.
        
        Args:
            query: Query text (e.g., Teacher's answer).
            documents: List of documents (e.g., Student answers).
            normalize: Whether to apply L2 normalization to embeddings.
            
        Returns:
            Similarity array, shape: (n_documents,)
        """
        # Get embeddings for all texts.
        all_texts = [query] + documents
        embeddings = self.get_embeddings(all_texts)
        
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        if normalize:
            # L2 Normalization
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute Cosine Similarity (Dot Product)
        similarities = doc_embeddings @ query_embedding
        
        return similarities
    
    def compute_pairwise_similarity(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all texts.
        
        Args:
            texts: List of texts.
            normalize: Whether to apply L2 normalization.
            
        Returns:
            Similarity matrix, shape: (n_texts, n_texts)
        """
        embeddings = self.get_embeddings(texts)
        
        if normalize:
            # L2 Normalization
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute Similarity Matrix
        similarity_matrix = embeddings @ embeddings.T
        
        return similarity_matrix


# Helper function for scenarios where a persistent client is not needed.
def compute_semantic_similarity(
    query: str,
    documents: List[str],
    base_url: str = "http://localhost:10000/v1",
    model_name: str = "Qwen3-Embedding-0.6B"
) -> np.ndarray:
    """
    Helper function: compute semantic similarity.
    
    Args:
        query: Query text.
        documents: List of documents.
        base_url: Embedding API service URL.
        model_name: Embedding model name.
        
    Returns:
        Similarity array.
    """
    client = EmbeddingClient(base_url=base_url, model_name=model_name)
    return client.compute_similarity(query, documents)


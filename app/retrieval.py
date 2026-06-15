"""Top-K cosine similarity retrieval over chunk embeddings."""

import numpy as np

from app.chunking import Chunk


class RetrievalIndex:
    """NumPy-based retrieval index for transcript chunks using cosine similarity."""

    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]]):
        """
        Initialize the retrieval index.

        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (same order as chunks)

        Raises:
            ValueError: If chunks and embeddings have mismatched lengths or are empty
        """
        if not chunks or not embeddings:
            raise ValueError("Chunks and embeddings cannot be empty")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        self.chunks = chunks
        self.embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Handle zero-norm embeddings (prevent division by zero)
        norms[norms == 0] = 1
        self.normalized_embeddings = self.embeddings / norms

    def retrieve(
        self, query_embedding: list[float], k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        Retrieve top-K most similar chunks using cosine similarity.

        Args:
            query_embedding: Embedding vector of the query
            k: Number of results to return

        Returns:
            List of (Chunk, score) tuples sorted by relevance

        Raises:
            ValueError: If query dimension doesn't match stored embeddings
        """
        # Adjust k if it exceeds available chunks
        if k > len(self.chunks):
            k = len(self.chunks)

        # Convert query to numpy and normalize
        query_vec = np.array(query_embedding, dtype=np.float32)

        if query_vec.ndim != 1:
            raise ValueError("Query embedding must be 1-dimensional")

        if query_vec.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension ({query_vec.shape[0]}) != "
                f"embedding dimension ({self.embeddings.shape[1]})"
            )

        # Normalize query vector (handle zero norm)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            query_norm = 1
        query_vec = query_vec / query_norm

        # Compute cosine similarity
        similarities = np.dot(self.normalized_embeddings, query_vec)

        # Get top-K indices
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Return (chunk, score) tuples
        results = [
            (self.chunks[idx], float(similarities[idx])) for idx in top_k_indices
        ]

        return results

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self.embeddings.shape[1] if len(self.embeddings) > 0 else 0

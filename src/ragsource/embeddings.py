from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.embed(texts)
        return [np.asarray(vector, dtype=np.float32).tolist() for vector in vectors]


def cosine_similarity(query: list[float], candidates: list[list[float]]) -> np.ndarray:
    if not candidates:
        return np.array([], dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(candidates, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(row_norms * q_norm, 1e-12)
    return (matrix @ q) / denominator

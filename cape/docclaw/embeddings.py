"""Small embedding abstraction and a dependency-free default implementation."""

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, Sequence


class Embedder(ABC):
    """An embedding provider. Implementations must return unit-normalized vectors."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier recorded in each corpus."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of float components in each vector."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts in input order."""


class HashingEmbedder(Embedder):
    """Deterministic token hashing baseline requiring no model download.

    This makes a fresh installation fully functional. Deployments can provide
    any local or remote :class:`Embedder` implementation.
    """

    def __init__(self, dimensions: int = 384):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return "docclaw-hashing-v1/%d" % self.dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w.-]+", text.casefold())
        features = tokens + [a + "\x1f" + b for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self.dimensions] += -1.0 if value & (1 << 63) else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

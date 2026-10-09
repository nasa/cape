"""DocClaw: small-corpus document retrieval for agents and humans."""

from .api import DocClaw
from .corpus import Corpus
from .embeddings import Embedder, HashingEmbedder
from .models import CorpusInfo, DocumentInfo, SearchResult

__all__ = [
    "Corpus",
    "CorpusInfo",
    "DocClaw",
    "DocumentInfo",
    "Embedder",
    "HashingEmbedder",
    "SearchResult",
]

__version__ = "0.1.0"

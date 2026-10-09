"""Installation-level, transport-independent DocClaw API."""

from pathlib import Path
import sqlite3
from typing import List, Optional, Sequence, Union

from .corpus import Corpus
from .embeddings import Embedder, HashingEmbedder
from .models import CorpusInfo, DocumentInfo, SearchResult


class DocClaw:
    def __init__(self, root: Union[str, Path], embedder: Optional[Embedder] = None):
        self.root = Path(root)
        self.embedder = embedder or HashingEmbedder()

    def create_corpus(self, name: str, *, title: Optional[str] = None,
                      description: str = "") -> CorpusInfo:
        self.root.mkdir(parents=True, exist_ok=True)
        return Corpus.create(self.root / name, name=name, title=title,
                             description=description, embedder=self.embedder).info

    def list_corpora(self) -> List[CorpusInfo]:
        if not self.root.exists():
            return []
        output = []
        for child in self.root.iterdir():
            try:
                output.append(Corpus(child, self.embedder).info)
            except (FileNotFoundError, ValueError, KeyError, OSError, sqlite3.Error):
                continue
        return sorted(output, key=lambda item: item.name)

    def corpus(self, name: str) -> Corpus:
        return Corpus(self.root / name, self.embedder)

    def list_documents(self, corpus: str, **kwargs) -> List[DocumentInfo]:
        return self.corpus(corpus).list_documents(**kwargs)

    def search(self, query: str, *, corpus: Union[str, Sequence[str]], n: int = 10,
               mode: str = "hybrid", **kwargs) -> List[SearchResult]:
        names = [corpus] if isinstance(corpus, str) else list(corpus)
        if not names or n <= 0:
            return []
        per_corpus = max(n, 10)
        results = []
        for name in names:
            results.extend(self.corpus(name).search(query, n=per_corpus, mode=mode, **kwargs))
        return sorted(results, key=lambda item: (-item.score, item.corpus, item.chunk_id))[:n]

    def read_chunk(self, corpus: str, chunk_id: str):
        return self.corpus(corpus).read_chunk(chunk_id)

    def read_context(self, corpus: str, chunk_id: str, *, before: int = 2, after: int = 2):
        return self.corpus(corpus).read_context(chunk_id, before=before, after=after)

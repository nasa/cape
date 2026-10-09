"""Public value objects returned by DocClaw."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CorpusInfo:
    name: str
    title: str
    description: str
    embedding_model: str
    documents: int
    pages: int
    created: str
    updated: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentInfo:
    document: str
    title: str
    author: Optional[str]
    document_date: Optional[str]
    pages: int
    chunks: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    corpus: str
    chunk_id: str
    document: str
    title: str
    page_start: Optional[int]
    page_end: Optional[int]
    section: Optional[str]
    subsection: Optional[str]
    score: float
    text: str
    lexical_rank: Optional[int] = None
    semantic_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

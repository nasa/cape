"""A single portable DocClaw corpus."""

import hashlib
import json
import re
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .chunking import chunk_text
from .embeddings import Embedder, HashingEmbedder
from .models import CorpusInfo, DocumentInfo, SearchResult

SCHEMA_VERSION = 1
INDEX_VERSION = 1
MANIFEST_FILE = "manifest.json"
DATABASE_FILE = "index.sqlite"

SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    document_key TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    author TEXT,
    document_date TEXT,
    page_count INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX documents_date_idx ON documents(document_date);
CREATE INDEX documents_author_idx ON documents(author);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section TEXT,
    subsection TEXT,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE(document_id, ordinal)
);
CREATE INDEX chunks_document_order_idx ON chunks(document_id, ordinal);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED, title, section, subsection, text,
    tokenize='unicode61 remove_diacritics 2'
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Corpus:
    """Create, ingest, and search one corpus directory."""

    def __init__(self, path: Path, embedder: Optional[Embedder] = None):
        self.path = Path(path)
        self.manifest_path = self.path / MANIFEST_FILE
        self.database_path = self.path / DATABASE_FILE
        if not self.manifest_path.is_file() or not self.database_path.is_file():
            raise FileNotFoundError("not a DocClaw corpus: %s" % self.path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.embedder = embedder or HashingEmbedder()
        self._validate()

    @classmethod
    def create(cls, path: Path, *, name: str, title: Optional[str] = None,
               description: str = "", embedder: Optional[Embedder] = None) -> "Corpus":
        path = Path(path)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError("corpus name must use letters, digits, dot, underscore, or dash")
        path.mkdir(parents=True, exist_ok=False)
        chosen = embedder or HashingEmbedder()
        now = _now()
        manifest = {
            "name": name,
            "title": title or name,
            "description": description,
            "embedding_model": chosen.model_id,
            "embedding_dimensions": chosen.dimensions,
            "schema_version": SCHEMA_VERSION,
            "index_version": INDEX_VERSION,
            "chunking": {"strategy": "structure-v1", "target_tokens": 600, "max_tokens": 800},
            "documents": 0,
            "pages": 0,
            "created": now,
            "updated": now,
        }
        (path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        connection = sqlite3.connect(path / DATABASE_FILE)
        try:
            connection.executescript(SCHEMA)
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("index_version", str(INDEX_VERSION)),
                    ("embedding_model", chosen.model_id),
                    ("embedding_dimensions", str(chosen.dimensions)),
                    ("created", now),
                ],
            )
            connection.commit()
        finally:
            connection.close()
        return cls(path, chosen)

    @property
    def info(self) -> CorpusInfo:
        data = self.manifest
        return CorpusInfo(**{key: data[key] for key in CorpusInfo.__dataclass_fields__})

    def ingest_file(self, source: Path, *, document: Optional[str] = None,
                    title: Optional[str] = None, author: Optional[str] = None,
                    document_date: Optional[str] = None,
                    metadata: Optional[Mapping[str, Any]] = None) -> DocumentInfo:
        source = Path(source)
        suffix = source.suffix.casefold()
        if suffix not in {".md", ".markdown", ".txt", ".text"}:
            raise ValueError("first milestone supports Markdown and plain text only")
        return self.ingest_text(
            source.read_text(encoding="utf-8"),
            document=document or source.name,
            title=title or source.stem,
            markdown=suffix in {".md", ".markdown"},
            author=author,
            document_date=document_date,
            metadata=metadata,
        )

    def ingest_text(self, text: str, *, document: str, title: Optional[str] = None,
                    markdown: bool = False, author: Optional[str] = None,
                    document_date: Optional[str] = None,
                    metadata: Optional[Mapping[str, Any]] = None) -> DocumentInfo:
        normalized = document.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        invalid_path = any((
            not normalized,
            normalized.startswith("/"),
            ".." in Path(normalized).parts,
        ))
        if invalid_path:
            raise ValueError("document must be a relative logical path")
        settings = self.manifest["chunking"]
        chunks = chunk_text(
            text,
            markdown=markdown,
            target_tokens=int(settings["target_tokens"]),
            max_tokens=int(settings["max_tokens"]),
        )
        if not chunks:
            raise ValueError("document contains no indexable text")
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned the wrong number of vectors")
        document_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        now = _now()
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadata_json = json.dumps(dict(metadata or {}), sort_keys=True)
        connection = self._connect()
        try:
            with connection:
                old = connection.execute(
                    "SELECT id FROM documents WHERE source_path = ?", (normalized,)
                ).fetchone()
                if old:
                    ids = [row[0] for row in connection.execute(
                        "SELECT chunk_id FROM chunks WHERE document_id = ?", (old["id"],)
                    )]
                    connection.executemany(
                        "DELETE FROM chunks_fts WHERE chunk_id = ?",
                        [(item,) for item in ids],
                    )
                    connection.execute("DELETE FROM documents WHERE id = ?", (old["id"],))
                cursor = connection.execute(
                    """INSERT INTO documents
                       (document_key, source_path, title, author, document_date,
                        content_hash, metadata_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (document_key, normalized, title or Path(normalized).stem, author,
                     document_date, content_hash, metadata_json, now, now),
                )
                document_id = cursor.lastrowid
                for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    if len(vector) != self.embedder.dimensions:
                        raise ValueError("embedder returned a vector with the wrong dimensions")
                    chunk_id = "%s:%04d" % (document_key, ordinal)
                    blob = struct.pack("<%df" % len(vector), *vector)
                    connection.execute(
                        """INSERT INTO chunks
                           (chunk_id, document_id, ordinal, section, subsection,
                            text, token_count, embedding)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (chunk_id, document_id, ordinal, chunk.section, chunk.subsection,
                         chunk.text, chunk.token_count, blob),
                    )
                    connection.execute(
                        """INSERT INTO chunks_fts
                           (chunk_id, title, section, subsection, text)
                           VALUES (?, ?, ?, ?, ?)""",
                        (chunk_id, title or Path(normalized).stem, chunk.section or "",
                         chunk.subsection or "", chunk.text),
                    )
            self._refresh_manifest(connection)
        finally:
            connection.close()
        return next(item for item in self.list_documents() if item.document == normalized)

    def list_documents(self, *, sort: str = "document", descending: bool = False,
                       author: Optional[str] = None, after: Optional[str] = None,
                       before: Optional[str] = None) -> List[DocumentInfo]:
        columns = {"document": "d.source_path", "title": "d.title", "date": "d.document_date"}
        if sort not in columns:
            raise ValueError("sort must be document, title, or date")
        clauses, parameters = [], []
        if author is not None:
            clauses.append("d.author = ?")
            parameters.append(author)
        if after is not None:
            clauses.append("d.document_date >= ?")
            parameters.append(after)
        if before is not None:
            clauses.append("d.document_date <= ?")
            parameters.append(before)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = """SELECT d.*, COUNT(c.chunk_id) AS chunks
                 FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"""
        sql += where + " GROUP BY d.id ORDER BY " + columns[sort]
        sql += " DESC" if descending else " ASC"
        connection = self._connect()
        try:
            rows = connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()
        return [DocumentInfo(row["source_path"], row["title"], row["author"],
                             row["document_date"], row["page_count"], row["chunks"],
                             json.loads(row["metadata_json"])) for row in rows]

    def search(self, query: str, *, n: int = 10, mode: str = "hybrid",
               document: Optional[str] = None, author: Optional[str] = None,
               after: Optional[str] = None, before: Optional[str] = None,
               excerpt_chars: int = 600) -> List[SearchResult]:
        if mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("mode must be lexical, semantic, or hybrid")
        if n <= 0:
            return []
        filters = (document, author, after, before)
        limit = max(50, n * 5)
        lexical = self._lexical(query, limit, filters) if mode != "semantic" else []
        semantic = self._semantic(query, limit, filters) if mode != "lexical" else []
        ranks: Dict[str, Dict[str, Any]] = {}
        for source, rows in (("lexical", lexical), ("semantic", semantic)):
            for rank, row in enumerate(rows, 1):
                entry = ranks.setdefault(row["chunk_id"], {"row": row, "score": 0.0})
                entry[source + "_rank"] = rank
                entry["score"] += 1.0 / (60 + rank)
        ordered = sorted(
            ranks.values(),
            key=lambda item: (-item["score"], item["row"]["chunk_id"]),
        )[:n]
        terms = re.findall(r"\w+", query.casefold())
        return [self._result(item, terms, excerpt_chars) for item in ordered]

    def read_chunk(self, chunk_id: str) -> Dict[str, Any]:
        row = self._get_chunk(chunk_id)
        return self._chunk_dict(row)

    def read_context(
        self, chunk_id: str, *, before: int = 2, after: int = 2
    ) -> List[Dict[str, Any]]:
        if before < 0 or after < 0:
            raise ValueError("before and after must be non-negative")
        row = self._get_chunk(chunk_id)
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT c.*, d.source_path, d.title FROM chunks c
                   JOIN documents d ON d.id = c.document_id
                   WHERE c.document_id = ? AND c.ordinal BETWEEN ? AND ?
                   ORDER BY c.ordinal""",
                (row["document_id"], row["ordinal"] - before, row["ordinal"] + after),
            ).fetchall()
        finally:
            connection.close()
        return [self._chunk_dict(item) for item in rows]

    def _lexical(self, query: str, limit: int, filters: Tuple[Any, ...]):
        tokens = re.findall(r"[\w.-]+", query, re.UNICODE)
        if not tokens:
            return []
        match = " OR ".join('"%s"' % token.replace('"', '""') for token in tokens)
        where, parameters = self._filter_sql(filters)
        sql = """SELECT c.*, d.source_path, d.title, bm25(chunks_fts) AS raw_score
                 FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                 JOIN documents d ON d.id = c.document_id
                 WHERE chunks_fts MATCH ?""" + where + " ORDER BY raw_score LIMIT ?"
        connection = self._connect()
        try:
            return connection.execute(sql, [match] + parameters + [limit]).fetchall()
        finally:
            connection.close()

    def _semantic(self, query: str, limit: int, filters: Tuple[Any, ...]):
        vector = self.embedder.embed([query])[0]
        where, parameters = self._filter_sql(filters)
        sql = """SELECT c.*, d.source_path, d.title FROM chunks c
                 JOIN documents d ON d.id = c.document_id WHERE 1=1""" + where
        connection = self._connect()
        try:
            rows = connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()
        scored = []
        for row in rows:
            stored = struct.unpack("<%df" % self.embedder.dimensions, row["embedding"])
            scored.append((sum(a * b for a, b in zip(vector, stored)), row))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["chunk_id"]))
        return [row for _, row in scored[:limit]]

    @staticmethod
    def _filter_sql(filters: Tuple[Any, ...]):
        document, author, after, before = filters
        clauses, parameters = [], []
        for value, clause in ((document, "d.source_path = ?"), (author, "d.author = ?"),
                              (after, "d.document_date >= ?"), (before, "d.document_date <= ?")):
            if value is not None:
                clauses.append(clause)
                parameters.append(value)
        return ((" AND " + " AND ".join(clauses)) if clauses else "", parameters)

    def _result(self, entry, terms, excerpt_chars):
        row = entry["row"]
        return SearchResult(
            corpus=self.manifest["name"], chunk_id=row["chunk_id"],
            document=row["source_path"], title=row["title"],
            page_start=row["page_start"], page_end=row["page_end"],
            section=row["section"], subsection=row["subsection"],
            score=entry["score"], text=_excerpt(row["text"], terms, excerpt_chars),
            lexical_rank=entry.get("lexical_rank"), semantic_rank=entry.get("semantic_rank"),
        )

    def _get_chunk(self, chunk_id):
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT c.*, d.source_path, d.title FROM chunks c
                   JOIN documents d ON d.id = c.document_id WHERE c.chunk_id = ?""", (chunk_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise KeyError("unknown chunk: %s" % chunk_id)
        return row

    def _chunk_dict(self, row):
        return {
            "corpus": self.manifest["name"],
            "chunk_id": row["chunk_id"],
            "document": row["source_path"],
            "title": row["title"],
            "ordinal": row["ordinal"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "section": row["section"],
            "subsection": row["subsection"],
            "text": row["text"],
        }

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _validate(self):
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported corpus schema version")
        if self.manifest.get("index_version") != INDEX_VERSION:
            raise ValueError("unsupported corpus index version")
        if self.manifest.get("embedding_model") != self.embedder.model_id:
            raise ValueError("corpus uses embedding model %r, not %r" %
                             (self.manifest.get("embedding_model"), self.embedder.model_id))
        if self.manifest.get("embedding_dimensions") != self.embedder.dimensions:
            raise ValueError("embedding dimension mismatch")
        connection = self._connect()
        try:
            stored = dict(connection.execute("SELECT key, value FROM metadata"))
        finally:
            connection.close()
        expected = {
            "schema_version": str(SCHEMA_VERSION),
            "index_version": str(INDEX_VERSION),
            "embedding_model": self.embedder.model_id,
            "embedding_dimensions": str(self.embedder.dimensions),
        }
        for key, value in expected.items():
            if stored.get(key) != value:
                raise ValueError("database metadata mismatch for %s" % key)

    def _refresh_manifest(self, connection):
        row = connection.execute(
            """SELECT COUNT(*) AS documents,
                      COALESCE(SUM(page_count), 0) AS pages
               FROM documents"""
        ).fetchone()
        self.manifest.update(documents=row["documents"], pages=row["pages"], updated=_now())
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)


def _excerpt(text: str, terms: Sequence[str], maximum: int) -> str:
    if maximum <= 0 or len(text) <= maximum:
        return text
    folded = text.casefold()
    positions = [folded.find(term.casefold()) for term in terms]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - maximum // 3)
    end = min(len(text), start + maximum)
    start = max(0, end - maximum)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")

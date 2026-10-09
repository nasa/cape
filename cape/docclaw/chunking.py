"""Structure-aware chunking for Markdown and plain text."""

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class TextChunk:
    text: str
    section: Optional[str]
    subsection: Optional[str]
    token_count: int


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))


def chunk_text(text: str, *, markdown: bool, target_tokens: int = 600,
               max_tokens: int = 800) -> List[TextChunk]:
    """Pack paragraphs without crossing Markdown heading boundaries."""
    if target_tokens <= 0 or max_tokens < target_tokens:
        raise ValueError("require 0 < target_tokens <= max_tokens")
    sections = _markdown_sections(text) if markdown else [(None, None, text)]
    chunks: List[TextChunk] = []
    for section, subsection, body in sections:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        units = []
        for paragraph in paragraphs:
            units.extend(_split_long(paragraph, max_tokens))
        current: List[str] = []
        current_size = 0
        for unit in units:
            size = estimate_tokens(unit)
            if current and current_size + size > target_tokens:
                joined = "\n\n".join(current)
                chunks.append(TextChunk(joined, section, subsection, estimate_tokens(joined)))
                current, current_size = [], 0
            current.append(unit)
            current_size += size
        if current:
            joined = "\n\n".join(current)
            chunks.append(TextChunk(joined, section, subsection, estimate_tokens(joined)))
    return chunks


def _markdown_sections(text: str):
    section = subsection = None
    body: List[str] = []
    output = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            if body and any(item.strip() for item in body):
                output.append((section, subsection, "\n".join(body)))
            level, heading = len(match.group(1)), match.group(2).strip()
            if level <= 2:
                section, subsection = heading, None
            else:
                subsection = heading
            body = []
        else:
            body.append(line)
    if body and any(item.strip() for item in body):
        output.append((section, subsection, "\n".join(body)))
    return output


def _split_long(text: str, max_tokens: int) -> Iterable[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1:
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    output, current, size = [], [], 0
    for sentence in sentences:
        sentence_size = estimate_tokens(sentence)
        if current and size + sentence_size > max_tokens:
            output.append(" ".join(current))
            current, size = [], 0
        if sentence_size > max_tokens:
            words = sentence.split()
            output.extend(
                " ".join(words[i:i + max_tokens])
                for i in range(0, len(words), max_tokens)
            )
        else:
            current.append(sentence)
            size += sentence_size
    if current:
        output.append(" ".join(current))
    return output

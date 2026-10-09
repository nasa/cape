"""JSON-oriented command-line interface for DocClaw."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .api import DocClaw


def _json(value: Any) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif isinstance(value, list):
        value = [item.to_dict() if hasattr(item, "to_dict") else item for item in value]
    print(json.dumps(value, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docclaw", description="Small-corpus document search")
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("DOCCLAW_ROOT", "corpora")),
                        help="corpus collection directory (default: ./corpora)")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a corpus")
    create.add_argument("name")
    create.add_argument("--title")
    create.add_argument("--description", default="")
    commands.add_parser("corpora", help="list corpora")
    ingest = commands.add_parser("ingest", help="ingest a Markdown or text file")
    ingest.add_argument("corpus")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--document", help="logical source path stored in the index")
    ingest.add_argument("--title")
    ingest.add_argument("--author")
    ingest.add_argument("--date", dest="document_date")
    ingest.add_argument("--metadata", default="{}", help="JSON object")
    documents = commands.add_parser("documents", help="list documents")
    documents.add_argument("corpus")
    documents.add_argument("--sort", choices=("document", "title", "date"), default="document")
    documents.add_argument("--descending", action="store_true")
    documents.add_argument("--author")
    documents.add_argument("--after")
    documents.add_argument("--before")
    search = commands.add_parser("search", help="search one or more corpora")
    search.add_argument("query")
    search.add_argument("--corpus", action="append", required=True,
                        help="repeat to search multiple corpora")
    search.add_argument("-n", type=int, default=10)
    search.add_argument("--mode", choices=("lexical", "semantic", "hybrid"), default="hybrid")
    search.add_argument("--document")
    search.add_argument("--author")
    search.add_argument("--after")
    search.add_argument("--before")
    read = commands.add_parser("read", help="read a complete chunk")
    read.add_argument("corpus")
    read.add_argument("chunk_id")
    context = commands.add_parser("context", help="read neighboring chunks")
    context.add_argument("corpus")
    context.add_argument("chunk_id")
    context.add_argument("--before", type=int, default=2)
    context.add_argument("--after", type=int, default=2)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    api = DocClaw(args.root)
    if args.command == "create":
        _json(api.create_corpus(args.name, title=args.title, description=args.description))
    elif args.command == "corpora":
        _json(api.list_corpora())
    elif args.command == "ingest":
        metadata = json.loads(args.metadata)
        if not isinstance(metadata, dict):
            raise SystemExit("--metadata must be a JSON object")
        _json(api.corpus(args.corpus).ingest_file(
            args.path, document=args.document, title=args.title, author=args.author,
            document_date=args.document_date, metadata=metadata,
        ))
    elif args.command == "documents":
        _json(api.list_documents(args.corpus, sort=args.sort, descending=args.descending,
                                 author=args.author, after=args.after, before=args.before))
    elif args.command == "search":
        _json(api.search(args.query, corpus=args.corpus, n=args.n, mode=args.mode,
                         document=args.document, author=args.author,
                         after=args.after, before=args.before))
    elif args.command == "read":
        _json(api.read_chunk(args.corpus, args.chunk_id))
    elif args.command == "context":
        _json(api.read_context(args.corpus, args.chunk_id,
                               before=args.before, after=args.after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

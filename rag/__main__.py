from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import get_settings
from rag.pipeline import ingest, retrieve_dicts


def _doc_id_from_path(path: Path) -> str:
    """Use filename stem as doc_id (e.g. notes.txt -> notes)."""
    stem = path.resolve().stem
    if stem:
        return stem
    name = path.resolve().name
    if name:
        return name
    return "document"


def _cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.file)
    doc_id = args.doc_id if args.doc_id is not None else _doc_id_from_path(path)
    text = path.read_text(encoding="utf-8")
    n = ingest(doc_id, text)
    print(f"Ingested {n} chunks for doc_id={doc_id!r}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    hits = retrieve_dicts(args.query, doc_id=args.doc_id)
    for i, h in enumerate(hits, 1):
        print(f"--- {i} score={h['score']:.4f} ---")
        print(h["content"])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Light RAG: ingest and query")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Chunk, embed, and store a text file")
    pi.add_argument("file", help="Path to UTF-8 text file (doc_id defaults to filename stem)")
    pi.add_argument(
        "--doc-id",
        dest="doc_id",
        default=None,
        metavar="ID",
        help="Override document id (default: filename without extension)",
    )
    pi.set_defaults(func=_cmd_ingest)

    pq = sub.add_parser("query", help="Embed query and retrieve top chunks")
    pq.add_argument("query", help="Search query text")
    pq.add_argument(
        "--doc-id",
        dest="doc_id",
        default=None,
        help="Optional doc_id filter (metadata)",
    )
    pq.set_defaults(func=_cmd_query)

    args = p.parse_args()
    try:
        get_settings()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

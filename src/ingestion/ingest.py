"""
CLI entry point for ingesting custom contract directories into ChromaDB.

Replaces the CUAD-specific chunker.py main() flow. Point it at a directory
of .pdf and/or .docx files and it extracts text, chunks it, and builds a
ChromaDB collection that the MCP server can query.

Usage:
    python3 ingest.py ./my-contracts/
    python3 ingest.py ./my-contracts/ --metadata metadata.json
    python3 ingest.py ./my-contracts/ --collection client_a_contracts
    python3 ingest.py ./my-contracts/ --rebuild
"""

import argparse
import sys

from ingestion.extract import load_contracts_from_directory
from rag.chunker import chunk_contracts, create_collection
from rag.rag_core import CLIENT


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a directory of contracts into ChromaDB for RAG retrieval."
    )
    parser.add_argument(
        "directory",
        help="Path to a directory containing .pdf and/or .docx contract files.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Path to a JSON metadata file mapping filenames to titles and/or "
            "party names. See extract.load_metadata for the expected format."
        ),
    )
    parser.add_argument(
        "--collection",
        default="legal_contracts",
        help=(
            "Name of the ChromaDB collection to create or add to. "
            "Defaults to 'legal_contracts'."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Delete the existing ChromaDB collection before indexing. "
            "Without this flag, ingestion is skipped if the collection "
            "already contains data."
        ),
    )

    args = parser.parse_args()

    # --- Extract ---
    try:
        contracts = load_contracts_from_directory(args.directory, args.metadata)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not contracts:
        print("No contracts were extracted. Nothing to index.", file=sys.stderr)
        sys.exit(1)

    # --- Chunk ---
    chunks = chunk_contracts(contracts)
    print(f"Created {len(chunks)} chunks from {len(contracts)} contracts")

    # --- Index ---
    if args.rebuild:
        try:
            CLIENT.delete_collection(args.collection)
            print(f"Deleted existing collection '{args.collection}' (--rebuild)")
        except Exception:
            pass  # collection didn't exist, which is fine for --rebuild
    else:
        existing = CLIENT.get_or_create_collection(args.collection)
        if existing.count() > 0:
            print(
                f"Collection '{args.collection}' already contains "
                f"{existing.count()} chunks, skipping indexing. "
                f"Use --rebuild to re-index."
            )
            return

    collection = create_collection(chunks, collection_name=args.collection)
    print(f"Collection '{collection.name}' now contains {collection.count()} chunks")


if __name__ == "__main__":
    main()

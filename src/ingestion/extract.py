"""
Document text extraction for legal contracts.

Extracts clean text from PDF and DOCX files in a directory and produces the
same list-of-dicts structure that chunker.py's load_contracts returns, so
the output can be passed directly to chunk_contracts and create_collection.

Supported formats:
  - PDF  (.pdf)  - via pymupdf (text-based PDFs; scanned/image PDFs are skipped)
  - DOCX (.docx) - via python-docx

Usage:
    from extract import load_contracts_from_directory
    contracts = load_contracts_from_directory("./my-contracts/")
"""

from pathlib import Path
from urllib.parse import unquote
import json
import re
import logging

import pymupdf             # PyMuPDF - installed as 'pymupdf'
from docx import Document  # install python-docx, *not* docx

logger = logging.getLogger(__name__)

# File extensions this module can handle
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# Minimum character count for a contract to be considered valid. Contracts shorter
# than this are likely extraction failures (scanned PDFs, empty files, etc.)
MIN_TEXT_LENGTH = 200


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: Path) -> str | None:
    """
    Extract text from a PDF using PyMuPDF. Returns None if the file can't be
    read or produces too little text (likely a scanned/image-only PDF).

    Args:
        filepath: Path to the PDF file.
    """
    try:
        doc = pymupdf.open(str(filepath))
    except Exception as e:
        logger.warning(f"Could not open PDF '{filepath.name}': {e}")
        return None

    pages = []
    for page in doc:
        text = page.get_text()
        if text:
            pages.append(text)
    doc.close()

    full_text = "\n\n".join(pages)

    if len(full_text.strip()) < MIN_TEXT_LENGTH:
        logger.warning(
            f"Skipping '{filepath.name}': extracted only {len(full_text.strip())} chars "
            f"(minimum is {MIN_TEXT_LENGTH}). This may be a scanned or image-only PDF."
        )
        return None

    return full_text


def extract_text_from_docx(filepath: Path) -> str | None:
    """
    Extract text from a Word document. Returns None if the file can't be read
    or contains too little text.

    Args:
        filepath: Path to the DOCX file.
    """
    try:
        doc = Document(str(filepath))
    except Exception as e:
        logger.warning(f"Could not open DOCX '{filepath.name}': {e}")
        return None

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)

    if len(full_text.strip()) < MIN_TEXT_LENGTH:
        logger.warning(
            f"Skipping '{filepath.name}': extracted only {len(full_text.strip())} chars "
            f"(minimum is {MIN_TEXT_LENGTH})."
        )
        return None

    return full_text


# Maps file extensions to their extraction functions
EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
}


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize extracted text so it chunks cleanly downstream.

    Handles common extraction artifacts: form-feed characters from PDF page
    breaks, runs of blank lines, trailing whitespace, and non-breaking spaces.

    Args:
        text: Raw extracted text.
    """
    # Replace form-feed / vertical-tab page break characters with double newlines
    text = text.replace("\f", "\n\n").replace("\v", "\n\n")

    # Normalize non-breaking spaces and other Unicode whitespace to regular spaces
    text = text.replace("\u00a0", " ").replace("\u2003", " ")

    # Collapse runs of 3+ newlines down to 2 (preserves paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


# ---------------------------------------------------------------------------
# Title helpers
# ---------------------------------------------------------------------------

def title_from_filename(filepath: Path) -> str:
    """
    Derive a human-readable contract title from a filename.

    Strips the extension and replaces underscores/hyphens with spaces. For
    example, 'Acme_Corp-NDA_2024.pdf' becomes 'Acme Corp NDA 2024'.

    Args:
        filepath: Path to the contract file.
    """
    stem = filepath.stem  # filename without extension
    # Remove URL encoded characters
    title = unquote(stem)
    # Replace underscores, hyphens, and other separators with spaces
    title = re.sub(r"[_\-]", " ", title)
    # Remove all remaining non-alphanumeric, non-space characters
    # (commas, periods, parentheses, percent signs, etc.)
    title = re.sub(r"[^a-zA-Z0-9 ]", "", title)
    # Collapse multiple spaces
    title = re.sub(r" {2,}", " ", title)

    return title.strip()


# ---------------------------------------------------------------------------
# Optional metadata loading
# ---------------------------------------------------------------------------

def load_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    """
    Load optional per-contract metadata from a JSON file.

    The file should be a JSON object mapping filenames (including extension)
    to metadata dicts. Supported metadata fields are 'title' and 'parties'.

    Example metadata.json:
    {
        "acme_nda.pdf": {
            "title": "Acme Corp Non-Disclosure Agreement",
            "parties": "Acme Corp, John Smith"
        },
        "globex_services.docx": {
            "parties": "Globex Corporation, Initech LLC"
        }
    }

    Filenames that don't appear in the metadata file will use defaults
    (title derived from filename, parties left blank).

    Args:
        metadata_path: Path to the JSON metadata file.
    """
    try:
        with open(metadata_path, "r") as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load metadata file '{metadata_path}': {e}")
        return {}

    if not isinstance(raw, dict):
        logger.warning(f"Metadata file should be a JSON object, got {type(raw).__name__}")
        return {}

    return raw


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def load_contracts_from_directory(
    directory: str,
    metadata_file: str = None,
) -> list[dict[str, str]]:
    """
    Extract text from all supported contract files in a directory and return
    them in the same format as chunker.load_contracts:

        [{"title": ..., "text": ..., "parties": ...}, ...]

    This output can be passed directly to chunk_contracts and create_collection.

    Args:
        directory: Path to a directory containing .pdf and/or .docx files.
        metadata_file: Optional. Path to a JSON file with per-contract metadata
                       (titles and/or party names). See load_metadata for format.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Gather all supported files, sorted for deterministic ordering. Sorting is 
    # necessary here because iterdir doesn't guarantee any particular order.
    # This could create confusion if contract collection needs to be re-chunked,
    # because each contract would end up with a different ID than it had before.
    # Deterministic sorting also makes debugging easier
    files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise ValueError(
            f"No supported files found in '{directory}'. "
            f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Load optional metadata
    metadata = {}
    if metadata_file:
        metadata = load_metadata(Path(metadata_file))

    contracts = []
    skipped = []

    for filepath in files:
        # Extract text using the appropriate extractor for this file type
        extractor = EXTRACTORS[filepath.suffix.lower()]
        raw_text = extractor(filepath)

        if raw_text is None:
            skipped.append(filepath.name)
            continue

        text = clean_text(raw_text)

        # Look up optional metadata for this file
        file_meta = metadata.get(filepath.name, {})
        title = file_meta.get("title", title_from_filename(filepath))
        parties = file_meta.get("parties", "")

        contracts.append({
            "title": title,
            "text": text,
            "parties": parties,
        })

    print(f"Loaded {len(contracts)} contracts from '{directory}'")
    if skipped:
        print(f"Skipped {len(skipped)} files: {', '.join(skipped)}")

    return contracts

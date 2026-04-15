"""
Smoke tests for the document extraction module.

Quick functional checks that the extraction code handles expected cases
correctly: unsupported file types are ignored, empty directories raise
errors, metadata loading works, clean_text normalizes artifacts, and
title_from_filename produces readable titles.

These tests create temporary files and directories so they don't depend
on the test-contracts folder or any external data.

Usage (from the src folder):
    python -m testing.extraction_smoke_test
"""

import json
import tempfile
from pathlib import Path

from ingestion.extract import (
    load_contracts_from_directory,
    clean_text,
    title_from_filename,
    MIN_TEXT_LENGTH,
)

PASS = "PASS"
FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Helper to create a minimal valid DOCX file for testing. PDFs are harder to
# generate programmatically, so most tests that need a real extractable file
# use DOCX.
# ---------------------------------------------------------------------------

def _create_test_docx(filepath: Path, text: str) -> None:
    """Create a minimal DOCX file with the given text content."""
    from docx import Document
    doc = Document()
    doc.add_paragraph(text)
    doc.save(str(filepath))


# A block of text long enough to pass the MIN_TEXT_LENGTH threshold
_SAMPLE_TEXT = (
    "This Agreement is entered into as of January 1, 2024, by and between "
    "Acme Corporation, a Delaware corporation, and Globex Industries, a "
    "California limited liability company. The parties agree to the following "
    "terms and conditions governing the provision of services described herein. "
    "This contract shall remain in effect for a period of twenty-four months "
    "from the effective date unless terminated earlier in accordance with the "
    "provisions set forth in Section 12 of this Agreement."
)


def test_unsupported_files_ignored():
    """load_contracts_from_directory ignores files with unsupported extensions."""
    name = "unsupported files ignored"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid DOCX so the directory isn't empty of supported files
        _create_test_docx(Path(tmpdir) / "valid.docx", _SAMPLE_TEXT)

        # Create files with unsupported extensions
        for ext in [".txt", ".csv", ".json", ".html", ".xlsx"]:
            (Path(tmpdir) / f"ignored{ext}").write_text("some content")

        contracts = load_contracts_from_directory(tmpdir)

        if len(contracts) != 1:
            print(f"  [{FAIL}] {name}: expected 1 contract, got {len(contracts)}")
            return

        if contracts[0]["title"] != "valid":
            print(f"  [{FAIL}] {name}: expected title 'valid', got '{contracts[0]['title']}'")
            return

        print(f"  [{PASS}] {name}: only the .docx file was extracted")


def test_nonexistent_directory():
    """load_contracts_from_directory raises FileNotFoundError for a missing directory."""
    name = "nonexistent directory"

    try:
        load_contracts_from_directory("/tmp/this_directory_does_not_exist_12345")
        print(f"  [{FAIL}] {name}: no exception raised")
    except FileNotFoundError:
        print(f"  [{PASS}] {name}: raised FileNotFoundError")
    except Exception as e:
        print(f"  [{FAIL}] {name}: expected FileNotFoundError, got {type(e).__name__}: {e}")


def test_no_supported_files():
    """load_contracts_from_directory raises ValueError when directory has no supported files."""
    name = "no supported files"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Only unsupported files
        (Path(tmpdir) / "notes.txt").write_text("not a contract")
        (Path(tmpdir) / "data.csv").write_text("a,b,c")

        try:
            load_contracts_from_directory(tmpdir)
            print(f"  [{FAIL}] {name}: no exception raised")
        except ValueError:
            print(f"  [{PASS}] {name}: raised ValueError")
        except Exception as e:
            print(f"  [{FAIL}] {name}: expected ValueError, got {type(e).__name__}: {e}")


def test_short_docx_skipped():
    """A DOCX file with text below MIN_TEXT_LENGTH is skipped."""
    name = "short DOCX skipped"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create one valid file and one too-short file
        _create_test_docx(Path(tmpdir) / "valid.docx", _SAMPLE_TEXT)
        _create_test_docx(Path(tmpdir) / "tiny.docx", "Short.")

        contracts = load_contracts_from_directory(tmpdir)

        titles = [c["title"] for c in contracts]
        if "tiny" in titles:
            print(f"  [{FAIL}] {name}: short file was not skipped")
            return

        if "valid" not in titles:
            print(f"  [{FAIL}] {name}: valid file was skipped")
            return

        print(f"  [{PASS}] {name}: short file skipped, valid file kept")


def test_metadata_overrides_title_and_parties():
    """Metadata file overrides the filename-derived title and provides parties."""
    name = "metadata overrides"

    with tempfile.TemporaryDirectory() as tmpdir:
        docx_name = "some_contract.docx"
        _create_test_docx(Path(tmpdir) / docx_name, _SAMPLE_TEXT)

        metadata = {
            docx_name: {
                "title": "Acme-Globex Services Agreement",
                "parties": "Acme Corporation, Globex Industries",
            }
        }
        meta_path = Path(tmpdir) / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        contracts = load_contracts_from_directory(tmpdir, metadata_file=str(meta_path))

        if len(contracts) != 1:
            print(f"  [{FAIL}] {name}: expected 1 contract, got {len(contracts)}")
            return

        c = contracts[0]
        if c["title"] != "Acme-Globex Services Agreement":
            print(f"  [{FAIL}] {name}: title not overridden, got '{c['title']}'")
            return

        if c["parties"] != "Acme Corporation, Globex Industries":
            print(f"  [{FAIL}] {name}: parties not set, got '{c['parties']}'")
            return

        print(f"  [{PASS}] {name}: title and parties set from metadata")


def test_metadata_partial_override():
    """Metadata with only 'parties' still uses the filename-derived title."""
    name = "metadata partial override"

    with tempfile.TemporaryDirectory() as tmpdir:
        docx_name = "acme_nda.docx"
        _create_test_docx(Path(tmpdir) / docx_name, _SAMPLE_TEXT)

        metadata = {
            docx_name: {
                "parties": "Acme Corp, John Smith",
            }
        }
        meta_path = Path(tmpdir) / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        contracts = load_contracts_from_directory(tmpdir, metadata_file=str(meta_path))
        c = contracts[0]

        # Title should come from filename since metadata didn't provide one
        if "acme" not in c["title"].lower():
            print(f"  [{FAIL}] {name}: expected filename-derived title, got '{c['title']}'")
            return

        if c["parties"] != "Acme Corp, John Smith":
            print(f"  [{FAIL}] {name}: parties not set from metadata")
            return

        print(f"  [{PASS}] {name}: filename title preserved, parties from metadata")


def test_metadata_missing_file_uses_defaults():
    """Files not listed in metadata get default title and empty parties."""
    name = "metadata missing file uses defaults"

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_test_docx(Path(tmpdir) / "unlisted_contract.docx", _SAMPLE_TEXT)

        # Metadata references a different file entirely
        metadata = {
            "other_file.docx": {"title": "Other", "parties": "Someone"}
        }
        meta_path = Path(tmpdir) / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        contracts = load_contracts_from_directory(tmpdir, metadata_file=str(meta_path))
        c = contracts[0]

        if c["parties"] != "":
            print(f"  [{FAIL}] {name}: expected empty parties, got '{c['parties']}'")
            return

        if "unlisted" not in c["title"].lower():
            print(f"  [{FAIL}] {name}: expected filename-derived title, got '{c['title']}'")
            return

        print(f"  [{PASS}] {name}: defaults used for unlisted file")


def test_invalid_metadata_file():
    """A malformed metadata file is handled gracefully (not a crash)."""
    name = "invalid metadata file"

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_test_docx(Path(tmpdir) / "contract.docx", _SAMPLE_TEXT)

        # Write invalid JSON
        meta_path = Path(tmpdir) / "metadata.json"
        meta_path.write_text("this is not valid json {{{")

        try:
            contracts = load_contracts_from_directory(tmpdir, metadata_file=str(meta_path))
        except Exception as e:
            print(f"  [{FAIL}] {name}: crashed instead of recovering: {e}")
            return

        # Should still extract the contract, just without metadata
        if len(contracts) != 1:
            print(f"  [{FAIL}] {name}: expected 1 contract, got {len(contracts)}")
            return

        print(f"  [{PASS}] {name}: recovered gracefully, extracted 1 contract")


def test_clean_text_normalizes_artifacts():
    """clean_text handles form-feeds, non-breaking spaces, and excessive newlines."""
    name = "clean_text normalization"

    dirty = "Section 1.\f\fSection 2.\v\vSection 3.\u00a0\u2003Extra spaces.\n\n\n\n\nToo many newlines."
    cleaned = clean_text(dirty)

    # Form-feeds and vertical tabs should become double newlines, not be left as-is
    if "\f" in cleaned or "\v" in cleaned:
        print(f"  [{FAIL}] {name}: form-feed or vertical tab not removed")
        return

    # Non-breaking spaces should become regular spaces
    if "\u00a0" in cleaned or "\u2003" in cleaned:
        print(f"  [{FAIL}] {name}: non-breaking spaces not normalized")
        return

    # Runs of 3+ newlines should collapse to exactly 2
    if "\n\n\n" in cleaned:
        print(f"  [{FAIL}] {name}: excessive newlines not collapsed")
        return

    # Core content should still be present
    if "Section 1." not in cleaned or "Section 3." not in cleaned:
        print(f"  [{FAIL}] {name}: content was lost during cleaning")
        return

    print(f"  [{PASS}] {name}: artifacts normalized, content preserved")


def test_title_from_filename():
    """title_from_filename produces clean human-readable titles."""
    name = "title_from_filename"

    cases = [
        ("Acme_Corp-NDA_2024.pdf", "Acme Corp NDA 2024"),
        ("simple.docx", "simple"),
        ("multiple___underscores---hyphens.pdf", "multiple underscores hyphens"),
    ]

    for filename, expected in cases:
        result = title_from_filename(Path(filename))
        if result != expected:
            print(f"  [{FAIL}] {name}: '{filename}' -> '{result}', expected '{expected}'")
            return

    print(f"  [{PASS}] {name}: all {len(cases)} cases correct")


def test_deterministic_ordering():
    """Contracts are returned in the same order regardless of filesystem ordering."""
    name = "deterministic ordering"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with names that sort differently alphabetically vs. by creation time
        for fname in ["charlie.docx", "alpha.docx", "bravo.docx"]:
            _create_test_docx(Path(tmpdir) / fname, _SAMPLE_TEXT)

        contracts_1 = load_contracts_from_directory(tmpdir)
        contracts_2 = load_contracts_from_directory(tmpdir)

        titles_1 = [c["title"] for c in contracts_1]
        titles_2 = [c["title"] for c in contracts_2]

        if titles_1 != titles_2:
            print(f"  [{FAIL}] {name}: ordering differs between runs")
            return

        # Should be alphabetical
        if titles_1 != sorted(titles_1):
            print(f"  [{FAIL}] {name}: not alphabetically sorted: {titles_1}")
            return

        print(f"  [{PASS}] {name}: consistent alphabetical ordering")


if __name__ == "__main__":
    print("\n--- Extraction Smoke Tests ---\n")
    test_unsupported_files_ignored()
    test_nonexistent_directory()
    test_no_supported_files()
    test_short_docx_skipped()
    test_metadata_overrides_title_and_parties()
    test_metadata_partial_override()
    test_metadata_missing_file_uses_defaults()
    test_invalid_metadata_file()
    test_clean_text_normalizes_artifacts()
    test_title_from_filename()
    test_deterministic_ordering()
    print("\nDone.\n")

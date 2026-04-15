"""
Validation script for the document extraction module.

Compares text extracted from CUAD PDF files against the ground truth text in
CUADv1.json. For each matched contract, samples phrases from the ground truth
and checks what percentage appear in the extracted text. This gives a "recall"
score that measures how much of the original content the extractor recovers.

Contracts in the test directory that don't match a CUAD entry (e.g., contracts
sourced from ResourceContracts.org) are reported but not scored, since there's
no ground truth to compare against.

Usage (from the src folder):
    python -m testing.extraction_validation
"""

import json
import re
from pathlib import Path
from urllib.parse import unquote
from ingestion.extract import load_contracts_from_directory

PASS = "PASS"
FAIL = "FAIL"

_CUAD_JSON = str(Path(__file__).parent.parent / "cuad" / "data" / "CUADv1.json")
_TEST_CONTRACTS = str(Path(__file__).parent / "test-contracts")

# Minimum phrase recall to consider extraction acceptable. 0.7 means at least
# 70% of sampled ground-truth phrases must appear in the extracted text.
RECALL_THRESHOLD = 0.7


def load_cuad_ground_truth(json_path: str) -> dict[str, str]:
    """
    Load the CUAD JSON and return a dict mapping normalized contract titles
    to their full text.

    Args:
        json_path: Path to CUADv1.json.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ground_truth = {}

    for contract in data["data"]:
        title = contract["title"]
        full_text = "\n\n".join(
            p["context"] for p in contract["paragraphs"]
        )
        # Normalize the title for matching: lowercase, collapse whitespace
        normalized = normalize_title(title)
        ground_truth[normalized] = {
            "original_title": title,
            "text": full_text,
        }

    return ground_truth


def normalize_title(title: str) -> str:
    """
    Normalize a contract title for fuzzy matching between CUAD JSON titles and
    filename-derived titles. Removes URL encoded characters, lowercases, strips common suffixes,
    replaces separators with spaces, etc.

    Args:
        title: The raw title string.
    """
    # Remove URL encoded characters
    title = unquote(title)
    title = title.lower().strip()
    title = re.sub(r"\.(pdf|docx|doc|txt)$", "", title)
    # Replace underscores, hyphens, and other separators with spaces
    title = re.sub(r"[_\-]", " ", title)
    # Remove all remaining non-alphanumeric, non-space characters
    # (commas, periods, parentheses, percent signs, etc.)
    title = re.sub(r"[^a-z0-9 ]", "", title)
    # Collapse multiple spaces
    title = re.sub(r" {2,}", " ", title)

    return title.strip()


def match_to_ground_truth(
    extracted_title: str, ground_truth: dict[str, str]
) -> str | None:
    """
    Try to match an extracted contract title to a CUAD ground truth entry.
    First attempts an exact match on normalized titles, then falls back to
    substring matching (either direction) to handle cases where the PDF
    filename is a truncated or extended version of the JSON title.

    Args:
        extracted_title: The title derived from the extracted contract's filename.
        ground_truth: The dict of normalized CUAD titles to ground truth data.

    Returns:
        The normalized ground truth key if a match is found, None otherwise.
    """
    normalized = normalize_title(extracted_title)

    # Exact match
    if normalized in ground_truth:
        return normalized

    # Substring match: check if either title contains the other
    for gt_key in ground_truth:
        if normalized in gt_key or gt_key in normalized:
            return gt_key

    return None


def sample_phrases(text: str, num_phrases: int = 20, phrase_length: int = 5) -> list[str]:
    """
    Sample multi-word phrases from the ground truth text for use as probes.
    
    Rather than comparing full text (which won't match due to whitespace and
    formatting differences from PDF extraction), we check whether short phrases
    from the ground truth appear in the extracted text. This is resilient to
    differences in line breaks, spacing, and minor OCR artifacts while still
    catching cases where content is missing or garbled.

    Args:
        text: The ground truth text to sample from.
        num_phrases: Number of phrases to sample. Defaults to 20.
        phrase_length: Number of words per phrase. Defaults to 5.
    """
    # Normalize whitespace so we're sampling clean word sequences
    words = text.split()

    if len(words) < phrase_length:
        return [" ".join(words)] if words else []

    # Sample phrases evenly spaced through the document to get coverage
    # across the beginning, middle, and end
    max_start = len(words) - phrase_length
    step = max(1, max_start // num_phrases)
    
    phrases = []
    for i in range(0, max_start, step):
        phrase = " ".join(words[i:i + phrase_length])
        phrases.append(phrase)
        if len(phrases) >= num_phrases:
            break

    return phrases


def compute_phrase_recall(extracted_text: str, phrases: list[str]) -> float:
    """
    Compute what fraction of ground truth phrases appear in the extracted text.

    Args:
        extracted_text: The text produced by the extraction module.
        phrases: List of ground truth phrases to search for.
    """
    if not phrases:
        return 0.0

    # Normalize the extracted text's whitespace to match how phrases were sampled
    normalized_extracted = " ".join(extracted_text.split())

    found = sum(1 for phrase in phrases if phrase in normalized_extracted)
    return found / len(phrases)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_cuad_phrase_recall(contracts: list[dict], ground_truth: dict) -> None:
    """
    For each extracted contract that matches a CUAD entry, sample phrases from
    the ground truth and measure what percentage appear in the extracted text.

    Args:
        contracts: List of contracts produced by load_contracts_from_directory.
        ground_truth: Dict of normalized CUAD titles to ground truth data.
    """
    name = "CUAD phrase recall"
    matched = 0
    passed = 0
    failed = 0

    for contract in contracts:
        gt_key = match_to_ground_truth(contract["title"], ground_truth)

        if gt_key is None:
            # Not a CUAD contract (e.g., from ResourceContracts.org) - skip
            continue

        matched += 1
        gt_text = ground_truth[gt_key]["text"]
        gt_title = ground_truth[gt_key]["original_title"]
        phrases = sample_phrases(gt_text)
        recall = compute_phrase_recall(contract["text"], phrases)

        if recall >= RECALL_THRESHOLD:
            passed += 1
            print(f"  [{PASS}] {gt_title}: recall {recall:.0%} ({len(phrases)} phrases)")
        else:
            failed += 1
            # Show which phrases were missed to help diagnose extraction problems
            normalized_extracted = " ".join(contract["text"].split())
            missed = [p for p in phrases if p not in normalized_extracted]
            print(f"  [{FAIL}] {gt_title}: recall {recall:.0%} ({len(phrases)} phrases)")
            print(f"         Missed phrases (first 3): {missed[:3]}")

    if matched == 0:
        print(f"  [SKIP] {name}: no CUAD contracts found in test directory")
        return

    print(f"\n  {name} summary: {passed} passed, {failed} failed out of {matched} matched")


def test_non_cuad_contracts_extracted(contracts: list[dict], ground_truth: dict) -> None:
    """
    Report on contracts in the test directory that don't match any CUAD entry.
    These can't be validated against ground truth, but we confirm they were at
    least extracted (non-empty text and correct dict structure).

    Args:
        contracts: List of contracts produced by load_contracts_from_directory.
        ground_truth: Dict of normalized CUAD titles to ground truth data.
    """
    name = "non-CUAD contract extraction"

    non_cuad = [
        c for c in contracts
        if match_to_ground_truth(c["title"], ground_truth) is None
    ]

    if not non_cuad:
        print(f"  [SKIP] {name}: all contracts matched CUAD entries")
        return

    for contract in non_cuad:
        text_len = len(contract["text"])
        has_keys = all(k in contract for k in ("title", "text", "parties"))

        if has_keys and text_len > 0:
            print(f"  [{PASS}] {contract['title']}: extracted {text_len} chars (no ground truth to validate)")
        else:
            print(f"  [{FAIL}] {contract['title']}: missing keys or empty text")


def test_output_structure(contracts: list[dict]) -> None:
    """
    Verify that every extracted contract has the expected dict keys and non-empty
    text, matching the structure that chunk_contracts expects.

    Args:
        contracts: List of contracts produced by load_contracts_from_directory.
    """
    name = "output structure"
    required_keys = {"title", "text", "parties"}

    for contract in contracts:
        missing = required_keys - set(contract.keys())
        if missing:
            print(f"  [{FAIL}] {name}: '{contract.get('title', '?')}' missing keys: {missing}")
            return

        if not contract["text"].strip():
            print(f"  [{FAIL}] {name}: '{contract['title']}' has empty text")
            return

        if not isinstance(contract["parties"], str):
            print(f"  [{FAIL}] {name}: '{contract['title']}' parties should be str, "
                  f"got {type(contract['parties']).__name__}")
            return

    print(f"  [{PASS}] {name}: all {len(contracts)} contracts have correct structure")


if __name__ == "__main__":
    print("\n--- Extraction Validation Tests ---\n")

    ground_truth = load_cuad_ground_truth(_CUAD_JSON)
    print(f"Loaded {len(ground_truth)} CUAD ground truth entries\n")

    contracts = load_contracts_from_directory(_TEST_CONTRACTS)
    print()

    test_output_structure(contracts)
    test_cuad_phrase_recall(contracts, ground_truth)
    test_non_cuad_contracts_extracted(contracts, ground_truth)

    print("--- Sample CUAD JSON titles (normalized) ---")

    for key in list(ground_truth.keys())[:5]:
        print(f"  '{key}'")

    print("\n--- Extracted titles (normalized) ---")
    for c in contracts[:5]:
        print(f"  '{normalize_title(c['title'])}'")

    print("\nDone.\n")

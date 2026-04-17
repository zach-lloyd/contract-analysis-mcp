from rag.rag_core import (
    CLIENT,
    _collections,
    get_collection,
    list_collections,
    list_matching_contracts,
    query_clauses,
)
from rag.chunker import create_collection

PASS = "PASS"
FAIL = "FAIL"

# A unique collection name that won't collide with anything real
TEST_COLLECTION = "smoke_test_collection_xyz"

# Small fake contracts loaded into the test collection. Two distinct titles
# so list_matching_contracts has something interesting to dedupe, and two
# distinct parties so the party-filter test has something to filter on.
TEST_CHUNKS = [
    {
        "chunk_text": (
            "This Test Agreement concerns governing law in the jurisdiction "
            "of the State of TestLand and shall be construed accordingly."
        ),
        "contract_title": "SMOKE_TEST_ALPHA",
        "parties": "Test Alpha Corp, Test Bravo LLC",
    },
    {
        "chunk_text": (
            "Termination provisions under this agreement allow either party "
            "to terminate with thirty days written notice to the other party."
        ),
        "contract_title": "SMOKE_TEST_ALPHA",
        "parties": "Test Alpha Corp, Test Bravo LLC",
    },
    {
        "chunk_text": (
            "This second test contract contains an indemnification clause "
            "whereby each party shall indemnify the other against third-party "
            "claims."
        ),
        "contract_title": "SMOKE_TEST_BETA",
        "parties": "Test Charlie Inc",
    },
]


def setup_test_collection():
    """Create the test collection with known fixture data. Idempotent: if a
    stale test collection exists from a previous run, it's deleted first."""
    # Drop any cached reference so we don't reuse a stale handle
    _collections.pop(TEST_COLLECTION, None)
    try:
        CLIENT.delete_collection(TEST_COLLECTION)
    except Exception:
        # Didn't exist, which is fine
        pass
    create_collection(TEST_CHUNKS, collection_name=TEST_COLLECTION)


def teardown_test_collection():
    """Remove the test collection after tests complete."""
    _collections.pop(TEST_COLLECTION, None)
    try:
        CLIENT.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


def test_query_clauses_returns_results():
    """query_clauses returns results across all contracts when no title is given."""
    name = "query_clauses (unfiltered)"
    results = query_clauses(
        "termination and governing law", num_results=3,
        collection_name=TEST_COLLECTION,
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        print(f"  [{FAIL}] {name}: no results returned")
        return

    # All returned chunks should be from one of the test contracts
    test_titles = {"SMOKE_TEST_ALPHA", "SMOKE_TEST_BETA"}
    bad = [m for m in metas if m["contract_title"] not in test_titles]
    if bad:
        print(f"  [{FAIL}] {name}: got results from outside test collection: "
              f"{[m['contract_title'] for m in bad]}")
        return

    # The fixture has two distinct contracts; unfiltered results should touch both
    returned_titles = set(m["contract_title"] for m in metas)
    if len(returned_titles) < 2:
        print(f"  [{FAIL}] {name}: expected multiple contracts in results, "
              f"got {returned_titles}")
        return

    print(f"  [{PASS}] {name}: got {len(docs)} results from {len(returned_titles)} contracts")


def test_query_clauses_filtered_by_title():
    """query_clauses filters to a single contract when a title is provided."""
    name = "query_clauses (filtered by title)"
    target = "SMOKE_TEST_ALPHA"
    results = query_clauses(
        "termination", num_results=5,
        collection_name=TEST_COLLECTION,
        contract_title=target,
    )
    metas = results["metadatas"][0]

    if not metas:
        print(f"  [{FAIL}] {name}: no results returned")
        return

    off_target = [m for m in metas if m["contract_title"] != target]
    if off_target:
        print(f"  [{FAIL}] {name}: got results from wrong contracts: "
              f"{set(m['contract_title'] for m in off_target)}")
        return

    print(f"  [{PASS}] {name}: all {len(metas)} results from '{target}'")


def test_list_matching_contracts_all():
    """list_matching_contracts returns all contracts in the collection."""
    name = "list_matching_contracts (all)"
    contracts = list_matching_contracts(collection_name=TEST_COLLECTION)

    titles = {c["contract_title"] for c in contracts}
    expected = {"SMOKE_TEST_ALPHA", "SMOKE_TEST_BETA"}

    if titles != expected:
        print(f"  [{FAIL}] {name}: expected {expected}, got {titles}")
        return

    # Each entry should have both expected keys
    sample = contracts[0]
    if "contract_title" not in sample or "parties" not in sample:
        print(f"  [{FAIL}] {name}: missing keys, got {list(sample.keys())}")
        return

    print(f"  [{PASS}] {name}: found {len(contracts)} contracts with correct structure")


def test_list_matching_contracts_filtered_by_party():
    """list_matching_contracts filters by party name."""
    name = "list_matching_contracts (filtered by party)"

    # Test Alpha Corp only appears on SMOKE_TEST_ALPHA, so filtering by it
    # should exclude SMOKE_TEST_BETA
    filtered = list_matching_contracts(
        collection_name=TEST_COLLECTION,
        party_name="Test Alpha Corp",
    )
    titles = {c["contract_title"] for c in filtered}

    if titles != {"SMOKE_TEST_ALPHA"}:
        print(f"  [{FAIL}] {name}: expected only SMOKE_TEST_ALPHA, got {titles}")
        return

    print(f"  [{PASS}] {name}: 'Test Alpha Corp' matched only the expected contract")


def test_list_matching_contracts_filter_is_case_insensitive():
    """list_matching_contracts matches parties regardless of case."""
    name = "list_matching_contracts (case-insensitive filter)"

    # Same party, lowercased
    filtered = list_matching_contracts(
        collection_name=TEST_COLLECTION,
        party_name="test alpha corp",
    )

    if not any(c["contract_title"] == "SMOKE_TEST_ALPHA" for c in filtered):
        print(f"  [{FAIL}] {name}: case-insensitive match failed, got "
              f"{[c['contract_title'] for c in filtered]}")
        return

    print(f"  [{PASS}] {name}: matched party name regardless of case")


def test_list_matching_contracts_nonexistent_party():
    """list_matching_contracts returns an empty list for a party that doesn't exist."""
    name = "list_matching_contracts (nonexistent party)"
    filtered = list_matching_contracts(
        collection_name=TEST_COLLECTION,
        party_name="ZZZ_FAKE_PARTY_XYZ",
    )

    if filtered:
        print(f"  [{FAIL}] {name}: expected empty list, got {filtered}")
        return

    print(f"  [{PASS}] {name}: correctly returned empty list")


def test_list_collections_includes_test_collection():
    """list_collections includes the test collection after it's created."""
    name = "list_collections"
    names = list_collections()

    if TEST_COLLECTION not in names:
        print(f"  [{FAIL}] {name}: test collection '{TEST_COLLECTION}' not "
              f"listed, got: {names}")
        return

    print(f"  [{PASS}] {name}: test collection appears in listing")


def test_get_collection_caches():
    """get_collection returns the same cached instance on repeated calls."""
    name = "get_collection (cache)"
    first = get_collection(TEST_COLLECTION)
    second = get_collection(TEST_COLLECTION)

    if first is not second:
        print(f"  [{FAIL}] {name}: returned different objects on second call")
        return

    if TEST_COLLECTION not in _collections:
        print(f"  [{FAIL}] {name}: collection not stored in cache")
        return

    print(f"  [{PASS}] {name}: returned cached object on repeat call")


if __name__ == "__main__":
    print("\n--- RAG Smoke Tests ---\n")
    try:
        setup_test_collection()
        test_query_clauses_returns_results()
        test_query_clauses_filtered_by_title()
        test_list_matching_contracts_all()
        test_list_matching_contracts_filtered_by_party()
        test_list_matching_contracts_filter_is_case_insensitive()
        test_list_matching_contracts_nonexistent_party()
        test_list_collections_includes_test_collection()
        test_get_collection_caches()
    finally:
        teardown_test_collection()

    print("\nDone.\n")
    
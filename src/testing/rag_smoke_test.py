from rag.rag_core import (
    CLIENT,
    DEFAULT_COLLECTION,
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
# A sentinel string that won't appear in any real contract chunk - used to
# verify that test data doesn't leak into the default collection's results
SENTINEL = "ZZZQUARKLE"

# Small fake contracts loaded into the test collection. Two distinct titles
# so list_matching_contracts has something interesting to dedupe.
TEST_CHUNKS = [
    {
        "chunk_text": (
            f"This Test Agreement {SENTINEL} concerns governing law in the "
            f"jurisdiction of the State of TestLand and shall be construed "
            f"accordingly."
        ),
        "contract_title": "SMOKE_TEST_ALPHA",
        "parties": "Test Alpha Corp, Test Bravo LLC",
    },
    {
        "chunk_text": (
            f"Termination provisions under this {SENTINEL} agreement allow "
            f"either party to terminate with thirty days written notice to "
            f"the other party."
        ),
        "contract_title": "SMOKE_TEST_ALPHA",
        "parties": "Test Alpha Corp, Test Bravo LLC",
    },
    {
        "chunk_text": (
            f"This second test contract {SENTINEL} contains an indemnification "
            f"clause whereby each party shall indemnify the other against "
            f"third-party claims."
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


def test_query_clauses_unfiltered():
    """query_clauses returns results across all contracts when no title is given."""
    name = "query_clauses (unfiltered)"
    results = query_clauses("What is the governing law?", num_results=5)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Should return the requested number of results
    if len(docs) != 5:
        print(f"  [{FAIL}] {name}: expected 5 results, got {len(docs)}")
        return

    # Results should come from more than one contract
    titles = set(m["contract_title"] for m in metas)
    if len(titles) < 2:
        print(f"  [{FAIL}] {name}: expected multiple contracts, got {titles}")
        return

    print(f"  [{PASS}] {name}: got {len(docs)} results from {len(titles)} contracts")


def test_query_clauses_filtered():
    """query_clauses filters to a single contract when a title is provided."""
    name = "query_clauses (filtered)"

    # Grab a real contract title from the database to use as our filter
    all_contracts = list_matching_contracts()
    if not all_contracts:
        print(f"  [{FAIL}] {name}: no contracts in database")
        return

    target_title = all_contracts[0]["contract_title"]
    results = query_clauses("termination", num_results=5, contract_title=target_title)
    metas = results["metadatas"][0]

    # Every result should be from the target contract
    off_target = [m for m in metas if m["contract_title"] != target_title]
    if off_target:
        print(f"  [{FAIL}] {name}: got results from wrong contracts: "
              f"{set(m['contract_title'] for m in off_target)}")
        return

    print(f"  [{PASS}] {name}: all results from '{target_title}'")


def test_list_matching_contracts_all():
    """list_matching_contracts returns all contracts when no party is given."""
    name = "list_matching_contracts (all)"
    contracts = list_matching_contracts()

    if not contracts:
        print(f"  [{FAIL}] {name}: returned empty list")
        return

    # Each entry should have both expected keys
    sample = contracts[0]
    if "contract_title" not in sample or "parties" not in sample:
        print(f"  [{FAIL}] {name}: missing keys, got {sample.keys()}")
        return

    print(f"  [{PASS}] {name}: found {len(contracts)} contracts")


def test_list_matching_contracts_filtered():
    """list_matching_contracts filters by party name."""
    name = "list_matching_contracts (filtered)"

    # Find a contract that has a non-empty parties field to use as our test case
    all_contracts = list_matching_contracts()
    contracts_with_parties = [c for c in all_contracts if c["parties"]]

    if not contracts_with_parties:
        print(f"  [SKIP] {name}: no contracts have party metadata")
        return

    # Use the first party name from the first contract that has one
    sample_parties = contracts_with_parties[0]["parties"]
    # Take the first comma-separated party name
    test_party = sample_parties.split(",")[0].strip()

    filtered = list_matching_contracts(party_name=test_party)

    if not filtered:
        print(f"  [{FAIL}] {name}: no results for party '{test_party}'")
        return

    # Every result should contain the party name
    bad = [c for c in filtered if test_party.lower() not in c["parties"].lower()]
    if bad:
        print(f"  [{FAIL}] {name}: results missing party '{test_party}': {bad}")
        return

    # Filtered list should be smaller than or equal to the full list
    if len(filtered) > len(all_contracts):
        print(f"  [{FAIL}] {name}: filtered ({len(filtered)}) > total ({len(all_contracts)})")
        return

    print(f"  [{PASS}] {name}: '{test_party}' matched {len(filtered)} contracts")


# ---------------------------------------------------------------------------
# Multi-collection tests. These rely on the test collection being set up via
# setup_test_collection() before they run.
# ---------------------------------------------------------------------------

def test_query_clauses_custom_collection():
    """query_clauses returns results only from the named collection."""
    name = "query_clauses (custom collection)"
    results = query_clauses(
        "termination", num_results=3, collection_name=TEST_COLLECTION
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

    print(f"  [{PASS}] {name}: got {len(docs)} results, all from test collection")


def test_query_clauses_collection_isolation():
    """Test data does not leak into the default collection's query results."""
    name = "query_clauses (collection isolation)"
    # Search the default collection for the sentinel string. If the test
    # collection is properly isolated, none of the default collection's
    # chunks should contain it.
    results = query_clauses(SENTINEL, num_results=10)
    docs = results["documents"][0]

    leaked = [d for d in docs if SENTINEL in d]
    if leaked:
        print(f"  [{FAIL}] {name}: sentinel '{SENTINEL}' leaked into "
              f"default collection results")
        return

    print(f"  [{PASS}] {name}: default collection returned no test data")


def test_list_matching_contracts_custom_collection():
    """list_matching_contracts returns only contracts from the named collection."""
    name = "list_matching_contracts (custom collection)"
    contracts = list_matching_contracts(collection_name=TEST_COLLECTION)

    titles = {c["contract_title"] for c in contracts}
    expected = {"SMOKE_TEST_ALPHA", "SMOKE_TEST_BETA"}

    if titles != expected:
        print(f"  [{FAIL}] {name}: expected {expected}, got {titles}")
        return

    print(f"  [{PASS}] {name}: got expected contracts from test collection")


def test_list_matching_contracts_collection_isolation():
    """Default collection does not include test contracts."""
    name = "list_matching_contracts (collection isolation)"
    contracts = list_matching_contracts()  # Default collection
    titles = {c["contract_title"] for c in contracts}

    leaked = titles & {"SMOKE_TEST_ALPHA", "SMOKE_TEST_BETA"}
    if leaked:
        print(f"  [{FAIL}] {name}: test contracts leaked into default: {leaked}")
        return

    print(f"  [{PASS}] {name}: default collection has no test contracts")


def test_list_collections_includes_both():
    """list_collections returns both the default and the test collection."""
    name = "list_collections"
    names = list_collections()

    if TEST_COLLECTION not in names:
        print(f"  [{FAIL}] {name}: test collection '{TEST_COLLECTION}' not "
              f"listed, got: {names}")
        return

    if DEFAULT_COLLECTION not in names:
        print(f"  [{FAIL}] {name}: default collection not listed, got: {names}")
        return

    print(f"  [{PASS}] {name}: both default and test collections listed")


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
    test_query_clauses_unfiltered()
    test_query_clauses_filtered()
    test_list_matching_contracts_all()
    test_list_matching_contracts_filtered()

    print("\n--- Multi-Collection Tests ---\n")
    try:
        setup_test_collection()
        test_query_clauses_custom_collection()
        test_query_clauses_collection_isolation()
        test_list_matching_contracts_custom_collection()
        test_list_matching_contracts_collection_isolation()
        test_list_collections_includes_both()
        test_get_collection_caches()
    finally:
        teardown_test_collection()

    print("\nDone.\n")
    
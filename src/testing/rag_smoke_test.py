from rag.rag_core import (
    query_clauses,
    list_matching_contracts
)

PASS = "PASS"
FAIL = "FAIL"


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


if __name__ == "__main__":
    print("\n--- RAG Smoke Tests ---\n")
    test_query_clauses_unfiltered()
    test_query_clauses_filtered()
    test_list_matching_contracts_all()
    test_list_matching_contracts_filtered()
    print("\nDone.\n")

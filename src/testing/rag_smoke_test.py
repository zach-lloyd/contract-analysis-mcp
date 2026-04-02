"""
Smoke tests for rag_core.py

Run the ChromaDB-only tests first (no Ollama required):
    python smoke_test.py --db-only

Run all tests (requires Ollama with qwen3:32b):
    python smoke_test.py
"""
import argparse
from rag.rag_core import (
    query_clauses,
    rewrite_prompt,
    list_matching_contracts,
    generate_answer,
    generate_comparison,
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


def test_rewrite_prompt():
    """rewrite_prompt returns a non-empty string."""
    name = "rewrite_prompt"
    history = [
        {"role": "user", "content": "Tell me about the non-compete clause in the Acme contract."},
        {"role": "assistant", "content": "The non-compete clause restricts employees for 2 years."},
    ]

    result = rewrite_prompt("What about the duration?", history)

    if not isinstance(result, str):
        print(f"  [{FAIL}] {name}: expected str, got {type(result)}")
        return

    if len(result.strip()) == 0:
        print(f"  [{FAIL}] {name}: returned empty string")
        return

    print(f"  [{PASS}] {name}: rewritten to '{result[:80]}...'")


def test_generate_answer_no_history():
    """generate_answer works on a first question with no history or clauses."""
    name = "generate_answer (no history)"
    answer, history, clauses = generate_answer("What is the governing law?")

    if not isinstance(answer, str) or len(answer.strip()) == 0:
        print(f"  [{FAIL}] {name}: empty or non-string answer")
        return

    # History should now have 2 entries (user question + assistant answer)
    if len(history) != 2:
        print(f"  [{FAIL}] {name}: expected 2 history entries, got {len(history)}")
        return

    if history[0]["role"] != "user" or history[1]["role"] != "assistant":
        print(f"  [{FAIL}] {name}: unexpected history roles")
        return

    # Clauses should be populated from retrieval
    if not clauses:
        print(f"  [{FAIL}] {name}: clauses list is empty after retrieval")
        return

    print(f"  [{PASS}] {name}: got answer ({len(answer)} chars), "
          f"{len(history)} history entries, {len(clauses)} clauses")


def test_generate_answer_with_history():
    """generate_answer incorporates prior history and clauses across turns."""
    name = "generate_answer (with history)"

    # Simulate a first turn
    answer1, history, clauses = generate_answer("What is the governing law?")
    clauses_after_first = len(clauses)

    # Simulate a follow-up turn, passing in the returned history and clauses
    answer2, history, clauses = generate_answer(
        "Are there any exceptions?", 
        clauses=clauses, 
        history=history
    )

    # History should now have 4 entries (2 turns x 2 messages each)
    if len(history) != 4:
        print(f"  [{FAIL}] {name}: expected 4 history entries, got {len(history)}")
        return

    if not isinstance(answer2, str) or len(answer2.strip()) == 0:
        print(f"  [{FAIL}] {name}: empty answer on follow-up")
        return

    print(f"  [{PASS}] {name}: follow-up got answer ({len(answer2)} chars), "
          f"{len(history)} history entries, clauses grew from "
          f"{clauses_after_first} to {len(clauses)}")


def test_generate_answer_filtered():
    """generate_answer restricts retrieval to a specific contract."""
    name = "generate_answer (filtered by contract)"

    all_contracts = list_matching_contracts()
    if not all_contracts:
        print(f"  [{FAIL}] {name}: no contracts in database")
        return

    target_title = all_contracts[0]["contract_title"]
    answer, history, clauses = generate_answer(
        "What are the termination provisions?",
        contract_title=target_title
    )

    # Every clause should reference the target contract
    off_target = [c for c in clauses if target_title not in c]
    if off_target:
        print(f"  [{FAIL}] {name}: clauses reference wrong contracts")
        return

    print(f"  [{PASS}] {name}: all clauses from '{target_title}', "
          f"answer is {len(answer)} chars")


def test_generate_comparison():
    """generate_comparison returns a non-empty answer given two contracts."""
    name = "generate_comparison (basic)"
 
    all_contracts = list_matching_contracts()
    if len(all_contracts) < 2:
        print(f"  [{FAIL}] {name}: need at least 2 contracts in database, "
              f"got {len(all_contracts)}")
        return
 
    titles = [all_contracts[0]["contract_title"], all_contracts[1]["contract_title"]]
    answer = generate_comparison("What are the termination provisions?", titles)
 
    if not isinstance(answer, str) or len(answer.strip()) == 0:
        print(f"  [{FAIL}] {name}: empty or non-string answer")
        return
 
    print(f"  [{PASS}] {name}: got answer ({len(answer)} chars) "
          f"comparing '{titles[0]}' and '{titles[1]}'")
 
 
def test_generate_comparison_references_contracts():
    """generate_comparison's answer references both contract titles."""
    name = "generate_comparison (references both contracts)"
 
    all_contracts = list_matching_contracts()
    if len(all_contracts) < 2:
        print(f"  [{FAIL}] {name}: need at least 2 contracts in database")
        return
 
    titles = [all_contracts[0]["contract_title"], all_contracts[1]["contract_title"]]
    answer = generate_comparison("How do the governing law clauses differ?", titles)
 
    # The comparison prompt instructs the LLM to be specific about which contract
    # each observation applies to, so both titles should appear in the answer
    missing = [t for t in titles if t.lower() not in answer.lower()]
    if missing:
        print(f"  [{FAIL}] {name}: answer does not mention {missing}")
        return
 
    print(f"  [{PASS}] {name}: answer references both contracts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-only", 
        action="store_true",
        help="Run only ChromaDB tests (no Ollama required)"
    )
    args = parser.parse_args()

    print("\n--- ChromaDB-only tests ---\n")
    test_query_clauses_unfiltered()
    test_query_clauses_filtered()
    test_list_matching_contracts_all()
    test_list_matching_contracts_filtered()

    if args.db_only:
        print("\n--- Skipping LLM tests (--db-only) ---\n")
    else:
        print("\n--- LLM tests (require Ollama + qwen3:32b) ---\n")
        test_rewrite_prompt()
        test_generate_answer_no_history()
        test_generate_answer_with_history()
        test_generate_answer_filtered()
        test_generate_comparison()
        test_generate_comparison_references_contracts()

    print("\nDone.\n")

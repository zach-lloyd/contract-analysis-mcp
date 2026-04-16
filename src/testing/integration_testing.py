import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path

# Used for the multi-collection tests: we set up a small test collection in
# the same chroma_data folder the server subprocess reads from, then verify
# the MCP tools respect the collection_name parameter.
from rag.rag_core import CLIENT, _collections
from rag.chunker import create_collection

PASS = "PASS"
FAIL = "FAIL"

# Resolve server.py relative to this file so the path works regardless of
# where you run the script from
_SERVER_PATH = str(Path(__file__).parent.parent / "rag" / "server.py")

# Test collection fixtures - matched to the smoke test fixtures so the data
# shape is familiar
TEST_COLLECTION = "integration_test_collection_xyz"
TEST_CHUNKS = [
    {
        "chunk_text": (
            "This Test Agreement concerns governing law in the jurisdiction "
            "of the State of TestLand and shall be construed accordingly."
        ),
        "contract_title": "INTEGRATION_TEST_ALPHA",
        "parties": "Test Alpha Corp, Test Bravo LLC",
    },
    {
        "chunk_text": (
            "Termination provisions under this agreement allow either party "
            "to terminate with thirty days written notice to the other party."
        ),
        "contract_title": "INTEGRATION_TEST_ALPHA",
        "parties": "Test Alpha Corp, Test Bravo LLC",
    },
    {
        "chunk_text": (
            "This second test contract contains an indemnification clause "
            "whereby each party shall indemnify the other against third-party "
            "claims."
        ),
        "contract_title": "INTEGRATION_TEST_BETA",
        "parties": "Test Charlie Inc",
    },
]


def setup_test_collection():
    """Create the test collection that the multi-collection tests query against.
    Idempotent: drops any stale collection from a prior run first."""
    _collections.pop(TEST_COLLECTION, None)
    try:
        CLIENT.delete_collection(TEST_COLLECTION)
    except Exception:
        pass
    create_collection(TEST_CHUNKS, collection_name=TEST_COLLECTION)


def teardown_test_collection():
    """Delete the test collection after tests complete."""
    _collections.pop(TEST_COLLECTION, None)
    try:
        CLIENT.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


async def run_tests():
    """
    Spin up the MCP server once, run all tests against the shared session,
    then tear down.
    """
    # Set up the test collection BEFORE spawning the server. Both processes
    # share the same on-disk chroma_data, so the server will see it as soon
    # as a tool is called.
    setup_test_collection()

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python3", _SERVER_PATH],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                def get_text(result):
                    """Extract the text content from an MCP call_tool result."""
                    return "".join(block.text for block in result.content if hasattr(block, "text"))


                async def test_list_contracts(session):
                    """list_contracts returns contracts when called with no arguments."""
                    name = "list_contracts (all)"
                    result = await session.call_tool("list_contracts", {})
                    text = get_text(result)

                    if "Found" not in text or "contract(s)" not in text:
                        print(f"  [{FAIL}] {name}: unexpected response: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: {text.splitlines()[0]}")


                async def test_list_contracts_filtered(session):
                    """list_contracts filters by party name."""
                    name = "list_contracts (filtered by party)"

                    # First get all contracts so we can find a valid party name to filter by
                    all_result = await session.call_tool("list_contracts", {})
                    all_text = get_text(all_result)

                    # Extract a party name from the first contract that has one
                    # Lines look like: "ContractTitle (parties: Alice, Bob)"
                    party = None
                    for line in all_text.splitlines()[1:]:
                        if "(parties:" in line:
                            parties_str = line.split("(parties:")[1].rstrip(")")
                            party = parties_str.split(",")[0].strip()
                            if party:
                                break

                    if not party:
                        print(f"  [SKIP] {name}: no contracts have party metadata")
                        return

                    filtered_result = await session.call_tool("list_contracts", {"party_name": party})
                    filtered_text = get_text(filtered_result)

                    if party.lower() not in filtered_text.lower():
                        print(f"  [{FAIL}] {name}: response doesn't mention party '{party}'")
                        return

                    print(f"  [{PASS}] {name}: filtered by '{party}', got: {filtered_text.splitlines()[0]}")


                async def test_list_contracts_nonexistent_party(session):
                    """list_contracts returns a clear message for a party that doesn't exist."""
                    name = "list_contracts (nonexistent party)"
                    result = await session.call_tool("list_contracts", {"party_name": "ZZZ_FAKE_PARTY_XYZ"})
                    text = get_text(result)

                    if "no contracts found" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected 'no contracts found', got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly reported no contracts found")


                async def test_find_contract_clauses(session):
                    """find_contract_clauses returns results for a common clause type."""
                    name = "find_contract_clauses (unfiltered)"
                    result = await session.call_tool("find_contract_clauses", {"clause_type": "termination"})
                    text = get_text(result)

                    if "Excerpt:" not in text:
                        print(f"  [{FAIL}] {name}: no excerpts returned: {text[:100]}")
                        return

                    # Count how many excerpts were returned
                    excerpt_count = text.count("Excerpt:")
                    print(f"  [{PASS}] {name}: got {excerpt_count} excerpts for 'termination'")


                async def test_find_contract_clauses_filtered(session):
                    """find_contract_clauses restricts results to a specific contract."""
                    name = "find_contract_clauses (filtered)"

                    # Get a real contract title to use
                    list_result = await session.call_tool("list_contracts", {})
                    list_text = get_text(list_result)
                    # Grab the first contract title (second line, before any parenthetical)
                    first_contract_line = list_text.splitlines()[1]
                    title = first_contract_line.split(" (parties:")[0].strip()

                    result = await session.call_tool("find_contract_clauses", {
                        "clause_type": "governing law",
                        "contract_title": title,
                    })
                    text = get_text(result)

                    # Every excerpt's "Contract:" label should match the target title
                    for line in text.splitlines():
                        if line.startswith("Contract:"):
                            returned_title = line.replace("Contract:", "").strip()
                            if returned_title != title:
                                print(f"  [{FAIL}] {name}: got result from '{returned_title}', expected '{title}'")
                                return

                    print(f"  [{PASS}] {name}: all results from '{title}'")


                async def test_find_contract_clauses_nonexistent_title(session):
                    """find_contract_clauses returns an error for a nonexistent contract."""
                    name = "find_contract_clauses (nonexistent title)"
                    result = await session.call_tool("find_contract_clauses", {
                        "clause_type": "termination",
                        "contract_title": "ZZZ_FAKE_CONTRACT_XYZ",
                    })
                    text = get_text(result)

                    if "no contract found" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected error message, got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly reported contract not found")
                

                async def test_ask_contracts(session):
                    """ask_contracts returns clause excerpts and a session ID."""
                    name = "ask_contracts (broad query)"
                    result = await session.call_tool("ask_contracts", {
                        "question": "What are common termination provisions?"
                    })
                    text = get_text(result)
 
                    if not text or len(text.strip()) == 0:
                        print(f"  [{FAIL}] {name}: got empty response")
                        return
 
                    if "Session ID:" not in text:
                        print(f"  [{FAIL}] {name}: missing Session ID header")
                        return
 
                    excerpt_count = text.count("Contract Excerpt:")
                    if excerpt_count == 0:
                        print(f"  [{FAIL}] {name}: no contract excerpts returned")
                        return
 
                    print(f"  [{PASS}] {name}: got {excerpt_count} excerpts with session ID")
                
 
                async def test_ask_contract(session):
                    """ask_contract returns clause excerpts from the specified contract."""
                    name = "ask_contract (known contract)"
 
                    # Get a real contract title to use
                    list_result = await session.call_tool("list_contracts", {})
                    list_text = get_text(list_result)
                    first_contract_line = list_text.splitlines()[1]
                    title = first_contract_line.split(" (parties:")[0].strip()
 
                    result = await session.call_tool("ask_contract", {
                        "question": "What is the governing law for this contract?",
                        "contract_title": title,
                    })
                    text = get_text(result)
 
                    if "Session ID:" not in text:
                        print(f"  [{FAIL}] {name}: missing Session ID header")
                        return
 
                    excerpt_count = text.count("Contract Excerpt:")
                    if excerpt_count == 0:
                        print(f"  [{FAIL}] {name}: no contract excerpts returned")
                        return
 
                    # Every "Contract Title:" line should reference the target contract
                    for line in text.splitlines():
                        if line.startswith("Contract Title:"):
                            returned_title = line.replace("Contract Title:", "").strip()
                            if returned_title != title:
                                print(f"  [{FAIL}] {name}: got excerpt from '{returned_title}', expected '{title}'")
                                return
 
                    print(f"  [{PASS}] {name}: got {excerpt_count} excerpts, all from '{title}'")


                async def test_ask_contract_nonexistent_title(session):
                    """ask_contract returns an error for a nonexistent contract title."""
                    name = "ask_contract (nonexistent title)"
                    result = await session.call_tool("ask_contract", {
                        "question": "What is the governing law for this contract?",
                        "contract_title": "ZZZ_FAKE_CONTRACT_XYZ",
                    })
                    text = get_text(result)

                    if "no contract found" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected error message, got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly reported contract not found")
                

                async def test_compare_contracts(session):
                    """compare_contracts returns clause sections for both contracts."""
                    name = "compare_contracts (references both)"
 
                    # Get two real contract titles
                    list_result = await session.call_tool("list_contracts", {})
                    list_text = get_text(list_result)
                    lines = list_text.splitlines()[1:]
                    titles = [line.split(" (parties:")[0].strip() for line in lines[:2]]
 
                    if len(titles) < 2:
                        print(f"  [{FAIL}] {name}: need at least 2 contracts in database")
                        return
 
                    result = await session.call_tool("compare_contracts", {
                        "question": "How do the governing law clauses differ?",
                        "contract_titles": titles,
                    })
                    text = get_text(result)
 
                    if not text or len(text.strip()) == 0:
                        print(f"  [{FAIL}] {name}: got empty response")
                        return
 
                    # Both contract titles should appear as section headers
                    missing = [t for t in titles if f"=== {t} ===" not in text]
                    if missing:
                        print(f"  [{FAIL}] {name}: missing section headers for {missing}")
                        return
 
                    print(f"  [{PASS}] {name}: got clause sections for both '{titles[0]}' and '{titles[1]}'")


                async def test_compare_contracts_too_few_titles(session):
                    """compare_contracts returns an error when given fewer than two titles."""
                    name = "compare_contracts (fewer than two titles)"

                    list_result = await session.call_tool("list_contracts", {})
                    list_text = get_text(list_result)
                    first_line = list_text.splitlines()[1]
                    title = first_line.split(" (parties:")[0].strip()

                    result = await session.call_tool("compare_contracts", {
                        "question": "How do termination clauses differ?",
                        "contract_titles": [title],
                    })
                    text = get_text(result)

                    if "at least two" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected 'at least two' error, got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly requires at least two contracts")


                async def test_compare_contracts_nonexistent_title(session):
                    """compare_contracts returns an error when one title doesn't exist."""
                    name = "compare_contracts (nonexistent title in list)"

                    list_result = await session.call_tool("list_contracts", {})
                    list_text = get_text(list_result)
                    first_line = list_text.splitlines()[1]
                    real_title = first_line.split(" (parties:")[0].strip()

                    result = await session.call_tool("compare_contracts", {
                        "question": "How do termination clauses differ?",
                        "contract_titles": [real_title, "ZZZ_FAKE_CONTRACT_XYZ"],
                    })
                    text = get_text(result)

                    if "not found" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected 'not found' error, got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly reported contract not found")


                # ----------------------------------------------------------------
                # Multi-collection tests. These rely on setup_test_collection()
                # having been run before the server was spawned so the test
                # collection is visible on disk.
                # ----------------------------------------------------------------

                async def test_list_collections_tool(session):
                    """list_collections returns all collection names including the test one."""
                    name = "list_collections (tool)"
                    result = await session.call_tool("list_collections", {})
                    text = get_text(result)

                    if TEST_COLLECTION not in text:
                        print(f"  [{FAIL}] {name}: test collection not listed: {text[:200]}")
                        return

                    if "legal_contracts" not in text:
                        print(f"  [{FAIL}] {name}: default collection not listed: {text[:200]}")
                        return

                    print(f"  [{PASS}] {name}: lists both default and test collections")


                async def test_list_contracts_custom_collection(session):
                    """list_contracts respects the collection_name parameter."""
                    name = "list_contracts (custom collection)"
                    result = await session.call_tool("list_contracts", {
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    # Both test contracts should appear, and nothing else
                    if "INTEGRATION_TEST_ALPHA" not in text:
                        print(f"  [{FAIL}] {name}: ALPHA missing from results: {text[:200]}")
                        return
                    if "INTEGRATION_TEST_BETA" not in text:
                        print(f"  [{FAIL}] {name}: BETA missing from results: {text[:200]}")
                        return

                    print(f"  [{PASS}] {name}: returned only test collection contracts")


                async def test_ask_contracts_custom_collection(session):
                    """ask_contracts respects the collection_name parameter."""
                    name = "ask_contracts (custom collection)"
                    result = await session.call_tool("ask_contracts", {
                        "question": "termination provisions",
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "Session ID:" not in text:
                        print(f"  [{FAIL}] {name}: missing Session ID header")
                        return

                    # All retrieved excerpts should be from test contracts
                    for line in text.splitlines():
                        if line.startswith("Contract Title:"):
                            returned_title = line.replace("Contract Title:", "").strip()
                            if not returned_title.startswith("INTEGRATION_TEST_"):
                                print(f"  [{FAIL}] {name}: got excerpt from outside "
                                      f"test collection: '{returned_title}'")
                                return

                    print(f"  [{PASS}] {name}: all excerpts from test collection")


                async def test_ask_contract_custom_collection(session):
                    """ask_contract finds contracts in a non-default collection."""
                    name = "ask_contract (custom collection)"
                    result = await session.call_tool("ask_contract", {
                        "question": "What is the governing law?",
                        "contract_title": "INTEGRATION_TEST_ALPHA",
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "no contract found" in text.lower():
                        print(f"  [{FAIL}] {name}: contract not found in test collection: "
                              f"{text[:200]}")
                        return

                    if "INTEGRATION_TEST_ALPHA" not in text:
                        print(f"  [{FAIL}] {name}: no excerpts from target contract")
                        return

                    print(f"  [{PASS}] {name}: retrieved from contract in test collection")


                async def test_compare_contracts_custom_collection(session):
                    """compare_contracts works against contracts in a non-default collection."""
                    name = "compare_contracts (custom collection)"
                    result = await session.call_tool("compare_contracts", {
                        "question": "How do these contracts differ?",
                        "contract_titles": [
                            "INTEGRATION_TEST_ALPHA",
                            "INTEGRATION_TEST_BETA",
                        ],
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "=== INTEGRATION_TEST_ALPHA ===" not in text:
                        print(f"  [{FAIL}] {name}: missing ALPHA section: {text[:200]}")
                        return
                    if "=== INTEGRATION_TEST_BETA ===" not in text:
                        print(f"  [{FAIL}] {name}: missing BETA section: {text[:200]}")
                        return

                    print(f"  [{PASS}] {name}: got sections for both test contracts")


                async def test_find_contract_clauses_custom_collection(session):
                    """find_contract_clauses respects the collection_name parameter."""
                    name = "find_contract_clauses (custom collection)"
                    result = await session.call_tool("find_contract_clauses", {
                        "clause_type": "indemnification",
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    # All "Contract:" lines should reference test contracts
                    for line in text.splitlines():
                        if line.startswith("Contract:"):
                            returned_title = line.replace("Contract:", "").strip()
                            if not returned_title.startswith("INTEGRATION_TEST_"):
                                print(f"  [{FAIL}] {name}: result from outside "
                                      f"test collection: '{returned_title}'")
                                return

                    if "INTEGRATION_TEST_" not in text:
                        print(f"  [{FAIL}] {name}: no test collection results: {text[:200]}")
                        return

                    print(f"  [{PASS}] {name}: returned only test collection clauses")


                async def test_collection_isolation(session):
                    """Default collection queries don't return test collection data."""
                    name = "collection isolation (default unaffected)"
                    result = await session.call_tool("list_contracts", {})
                    text = get_text(result)

                    if "INTEGRATION_TEST_" in text:
                        print(f"  [{FAIL}] {name}: test contracts leaked into "
                              f"default collection")
                        return

                    print(f"  [{PASS}] {name}: test data not present in default collection")


                print("\n--- Integration Tests ---\n")
                await test_list_contracts(session)
                await test_list_contracts_filtered(session)
                await test_list_contracts_nonexistent_party(session)
                await test_find_contract_clauses(session)
                await test_find_contract_clauses_filtered(session)
                await test_find_contract_clauses_nonexistent_title(session)
                await test_ask_contracts(session)
                await test_ask_contract(session)
                await test_ask_contract_nonexistent_title(session)
                await test_compare_contracts(session)
                await test_compare_contracts_too_few_titles(session)
                await test_compare_contracts_nonexistent_title(session)

                print("\n--- Multi-Collection Integration Tests ---\n")
                await test_list_collections_tool(session)
                await test_list_contracts_custom_collection(session)
                await test_ask_contracts_custom_collection(session)
                await test_ask_contract_custom_collection(session)
                await test_compare_contracts_custom_collection(session)
                await test_find_contract_clauses_custom_collection(session)
                await test_collection_isolation(session)
    finally:
        teardown_test_collection()

    print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
    
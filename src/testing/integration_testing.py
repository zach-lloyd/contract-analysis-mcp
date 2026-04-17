import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path

# We set up a dedicated test collection in the same chroma_data folder the
# server subprocess reads from, then run every tool call against it. This
# keeps tests isolated from any real collections (CUAD, user data, etc.)
# that may exist on disk.
from rag.rag_core import CLIENT, _collections
from rag.chunker import create_collection

PASS = "PASS"
FAIL = "FAIL"

# Resolve server.py relative to this file so the path works regardless of
# where you run the script from
_SERVER_PATH = str(Path(__file__).parent.parent / "rag" / "server.py")

# Test collection fixtures. Two contracts with distinct titles and parties
# so list/filter/compare tests have meaningful data to work with.
TEST_COLLECTION = "integration_test_collection_xyz"
SECOND_TEST_COLLECTION = "integration_test_collection_abc"
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
# A chunk for the second collection, used to verify collection isolation
SECOND_COLLECTION_CHUNKS = [
    {
        "chunk_text": (
            "This contract belongs to a different collection and should never "
            "appear in results when the first test collection is queried."
        ),
        "contract_title": "SECOND_COLLECTION_CONTRACT",
        "parties": "Other Party LLC",
    },
]


def setup_test_collections():
    """Create the test collections that all tests query against.
    Idempotent: drops any stale collections from a prior run first."""
    for coll_name, chunks in [
        (TEST_COLLECTION, TEST_CHUNKS),
        (SECOND_TEST_COLLECTION, SECOND_COLLECTION_CHUNKS),
    ]:
        _collections.pop(coll_name, None)
        try:
            CLIENT.delete_collection(coll_name)
        except Exception:
            pass
        create_collection(chunks, collection_name=coll_name)


def teardown_test_collections():
    """Delete the test collections after tests complete."""
    for coll_name in (TEST_COLLECTION, SECOND_TEST_COLLECTION):
        _collections.pop(coll_name, None)
        try:
            CLIENT.delete_collection(coll_name)
        except Exception:
            pass


async def run_tests():
    """
    Spin up the MCP server once, run all tests against the shared session,
    then tear down.
    """
    # Set up the test collections BEFORE spawning the server. Both processes
    # share the same on-disk chroma_data, so the server will see them as soon
    # as a tool is called.
    setup_test_collections()

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


                # ----------------------------------------------------------------
                # list_collections: the only tool that doesn't take collection_name
                # ----------------------------------------------------------------

                async def test_list_collections_tool(session):
                    """list_collections returns all collection names including the test ones."""
                    name = "list_collections"
                    result = await session.call_tool("list_collections", {})
                    text = get_text(result)

                    if TEST_COLLECTION not in text:
                        print(f"  [{FAIL}] {name}: primary test collection not listed: {text[:200]}")
                        return

                    if SECOND_TEST_COLLECTION not in text:
                        print(f"  [{FAIL}] {name}: secondary test collection not listed: {text[:200]}")
                        return

                    print(f"  [{PASS}] {name}: lists both test collections")


                # ----------------------------------------------------------------
                # list_contracts
                # ----------------------------------------------------------------

                async def test_list_contracts(session):
                    """list_contracts returns the contracts in the specified collection."""
                    name = "list_contracts (all in collection)"
                    result = await session.call_tool("list_contracts", {
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "INTEGRATION_TEST_ALPHA" not in text:
                        print(f"  [{FAIL}] {name}: ALPHA missing from results: {text[:200]}")
                        return
                    if "INTEGRATION_TEST_BETA" not in text:
                        print(f"  [{FAIL}] {name}: BETA missing from results: {text[:200]}")
                        return
                    # Make sure the other test collection's data isn't leaking in
                    if "SECOND_COLLECTION_CONTRACT" in text:
                        print(f"  [{FAIL}] {name}: data from other collection leaked in")
                        return

                    print(f"  [{PASS}] {name}: returned only target collection contracts")


                async def test_list_contracts_filtered(session):
                    """list_contracts filters by party name within the specified collection."""
                    name = "list_contracts (filtered by party)"

                    result = await session.call_tool("list_contracts", {
                        "collection_name": TEST_COLLECTION,
                        "party_name": "Test Alpha Corp",
                    })
                    text = get_text(result)

                    if "INTEGRATION_TEST_ALPHA" not in text:
                        print(f"  [{FAIL}] {name}: ALPHA missing (should match party filter)")
                        return
                    if "INTEGRATION_TEST_BETA" in text:
                        print(f"  [{FAIL}] {name}: BETA shouldn't match 'Test Alpha Corp' party filter")
                        return

                    print(f"  [{PASS}] {name}: correctly filtered to party's contracts")


                async def test_list_contracts_nonexistent_party(session):
                    """list_contracts returns a clear message for a party that doesn't exist."""
                    name = "list_contracts (nonexistent party)"
                    result = await session.call_tool("list_contracts", {
                        "collection_name": TEST_COLLECTION,
                        "party_name": "ZZZ_FAKE_PARTY_XYZ",
                    })
                    text = get_text(result)

                    if "no contracts found" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected 'no contracts found', got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly reported no contracts found")


                async def test_list_contracts_collection_isolation(session):
                    """list_contracts on one collection doesn't return data from another."""
                    name = "list_contracts (collection isolation)"
                    result = await session.call_tool("list_contracts", {
                        "collection_name": SECOND_TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "INTEGRATION_TEST_ALPHA" in text or "INTEGRATION_TEST_BETA" in text:
                        print(f"  [{FAIL}] {name}: primary collection data leaked into secondary")
                        return
                    if "SECOND_COLLECTION_CONTRACT" not in text:
                        print(f"  [{FAIL}] {name}: secondary collection's contract missing")
                        return

                    print(f"  [{PASS}] {name}: collections properly isolated")


                # ----------------------------------------------------------------
                # find_contract_clauses
                # ----------------------------------------------------------------

                async def test_find_contract_clauses(session):
                    """find_contract_clauses returns results for a common clause type."""
                    name = "find_contract_clauses (unfiltered)"
                    result = await session.call_tool("find_contract_clauses", {
                        "clause_type": "termination",
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "Excerpt:" not in text:
                        print(f"  [{FAIL}] {name}: no excerpts returned: {text[:100]}")
                        return

                    # All excerpts should come from the test collection
                    for line in text.splitlines():
                        if line.startswith("Contract:"):
                            returned_title = line.replace("Contract:", "").strip()
                            if not returned_title.startswith("INTEGRATION_TEST_"):
                                print(f"  [{FAIL}] {name}: result from outside test "
                                      f"collection: '{returned_title}'")
                                return

                    excerpt_count = text.count("Excerpt:")
                    print(f"  [{PASS}] {name}: got {excerpt_count} excerpts for 'termination'")


                async def test_find_contract_clauses_filtered(session):
                    """find_contract_clauses restricts results to a specific contract."""
                    name = "find_contract_clauses (filtered by title)"

                    target = "INTEGRATION_TEST_ALPHA"
                    result = await session.call_tool("find_contract_clauses", {
                        "clause_type": "governing law",
                        "contract_title": target,
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    # Every excerpt's "Contract:" label should match the target title
                    for line in text.splitlines():
                        if line.startswith("Contract:"):
                            returned_title = line.replace("Contract:", "").strip()
                            if returned_title != target:
                                print(f"  [{FAIL}] {name}: got result from '{returned_title}', "
                                      f"expected '{target}'")
                                return

                    print(f"  [{PASS}] {name}: all results from '{target}'")


                async def test_find_contract_clauses_nonexistent_title(session):
                    """find_contract_clauses returns an error for a nonexistent contract."""
                    name = "find_contract_clauses (nonexistent title)"
                    result = await session.call_tool("find_contract_clauses", {
                        "clause_type": "termination",
                        "contract_title": "ZZZ_FAKE_CONTRACT_XYZ",
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "no contract found" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected error message, got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly reported contract not found")


                # ----------------------------------------------------------------
                # ask_contracts
                # ----------------------------------------------------------------

                async def test_ask_contracts(session):
                    """ask_contracts returns clause excerpts and a session ID."""
                    name = "ask_contracts (broad query)"
                    result = await session.call_tool("ask_contracts", {
                        "question": "What are common termination provisions?",
                        "collection_name": TEST_COLLECTION,
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

                    # All excerpts should be from the test collection
                    for line in text.splitlines():
                        if line.startswith("Contract Title:"):
                            returned_title = line.replace("Contract Title:", "").strip()
                            if not returned_title.startswith("INTEGRATION_TEST_"):
                                print(f"  [{FAIL}] {name}: excerpt from outside "
                                      f"test collection: '{returned_title}'")
                                return

                    print(f"  [{PASS}] {name}: got {excerpt_count} excerpts with session ID")


                # ----------------------------------------------------------------
                # ask_contract
                # ----------------------------------------------------------------

                async def test_ask_contract(session):
                    """ask_contract returns clause excerpts from the specified contract."""
                    name = "ask_contract (known contract)"

                    target = "INTEGRATION_TEST_ALPHA"
                    result = await session.call_tool("ask_contract", {
                        "question": "What is the governing law for this contract?",
                        "contract_title": target,
                        "collection_name": TEST_COLLECTION,
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
                            if returned_title != target:
                                print(f"  [{FAIL}] {name}: got excerpt from '{returned_title}', "
                                      f"expected '{target}'")
                                return

                    print(f"  [{PASS}] {name}: got {excerpt_count} excerpts, all from '{target}'")


                async def test_ask_contract_nonexistent_title(session):
                    """ask_contract returns an error for a nonexistent contract title."""
                    name = "ask_contract (nonexistent title)"
                    result = await session.call_tool("ask_contract", {
                        "question": "What is the governing law for this contract?",
                        "contract_title": "ZZZ_FAKE_CONTRACT_XYZ",
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "no contract found" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected error message, got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly reported contract not found")


                async def test_ask_contract_wrong_collection(session):
                    """ask_contract reports not-found when a real title is queried in
                    the wrong collection. This is the core multi-collection isolation
                    guarantee: contracts don't leak across collections."""
                    name = "ask_contract (contract in different collection)"
                    # ALPHA lives in TEST_COLLECTION, not SECOND_TEST_COLLECTION
                    result = await session.call_tool("ask_contract", {
                        "question": "What is the governing law?",
                        "contract_title": "INTEGRATION_TEST_ALPHA",
                        "collection_name": SECOND_TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "no contract found" not in text.lower():
                        print(f"  [{FAIL}] {name}: should not find contract from other "
                              f"collection, got: {text[:150]}")
                        return

                    print(f"  [{PASS}] {name}: correctly isolated across collections")


                # ----------------------------------------------------------------
                # compare_contracts
                # ----------------------------------------------------------------

                async def test_compare_contracts(session):
                    """compare_contracts returns clause sections for both contracts."""
                    name = "compare_contracts (references both)"

                    titles = ["INTEGRATION_TEST_ALPHA", "INTEGRATION_TEST_BETA"]
                    result = await session.call_tool("compare_contracts", {
                        "question": "How do the clauses differ?",
                        "contract_titles": titles,
                        "collection_name": TEST_COLLECTION,
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

                    print(f"  [{PASS}] {name}: got clause sections for both contracts")


                async def test_compare_contracts_too_few_titles(session):
                    """compare_contracts returns an error when given fewer than two titles."""
                    name = "compare_contracts (fewer than two titles)"

                    result = await session.call_tool("compare_contracts", {
                        "question": "How do termination clauses differ?",
                        "contract_titles": ["INTEGRATION_TEST_ALPHA"],
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "at least two" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected 'at least two' error, got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly requires at least two contracts")


                async def test_compare_contracts_nonexistent_title(session):
                    """compare_contracts returns an error when one title doesn't exist."""
                    name = "compare_contracts (nonexistent title in list)"

                    result = await session.call_tool("compare_contracts", {
                        "question": "How do termination clauses differ?",
                        "contract_titles": ["INTEGRATION_TEST_ALPHA", "ZZZ_FAKE_CONTRACT_XYZ"],
                        "collection_name": TEST_COLLECTION,
                    })
                    text = get_text(result)

                    if "not found" not in text.lower():
                        print(f"  [{FAIL}] {name}: expected 'not found' error, got: {text[:100]}")
                        return

                    print(f"  [{PASS}] {name}: correctly reported contract not found")


                # ----------------------------------------------------------------
                # Required-parameter enforcement. collection_name is now required
                # on every tool except list_collections, so calls omitting it
                # should fail at the MCP protocol level before reaching our code.
                # ----------------------------------------------------------------

                async def test_collection_name_required(session):
                    """Calls omitting collection_name are rejected by the MCP layer."""
                    name = "collection_name required"

                    tools_and_args = [
                        ("list_contracts", {}),
                        ("ask_contracts", {"question": "anything"}),
                        ("ask_contract", {"question": "x", "contract_title": "y"}),
                        ("find_contract_clauses", {"clause_type": "termination"}),
                        ("compare_contracts", {
                            "question": "x",
                            "contract_titles": ["a", "b"],
                        }),
                    ]

                    for tool_name, args in tools_and_args:
                        try:
                            result = await session.call_tool(tool_name, args)
                            text = get_text(result)
                            # MCP may surface missing-argument errors via isError or
                            # via a text response. Either way, the call should not
                            # succeed normally.
                            is_error = getattr(result, "isError", False)
                            if not is_error and "error" not in text.lower() and "required" not in text.lower():
                                print(f"  [{FAIL}] {name}: {tool_name} accepted missing "
                                      f"collection_name; got: {text[:150]}")
                                return
                        except Exception:
                            # An exception from the client is also acceptable - it
                            # means the protocol rejected the call
                            pass

                    print(f"  [{PASS}] {name}: all tools reject missing collection_name")


                # ----------------------------------------------------------------
                # Run everything
                # ----------------------------------------------------------------

                print("\n--- Integration Tests ---\n")

                # list_collections (the one tool without collection_name)
                await test_list_collections_tool(session)

                # list_contracts
                await test_list_contracts(session)
                await test_list_contracts_filtered(session)
                await test_list_contracts_nonexistent_party(session)
                await test_list_contracts_collection_isolation(session)

                # find_contract_clauses
                await test_find_contract_clauses(session)
                await test_find_contract_clauses_filtered(session)
                await test_find_contract_clauses_nonexistent_title(session)

                # ask_contracts / ask_contract
                await test_ask_contracts(session)
                await test_ask_contract(session)
                await test_ask_contract_nonexistent_title(session)
                await test_ask_contract_wrong_collection(session)

                # compare_contracts
                await test_compare_contracts(session)
                await test_compare_contracts_too_few_titles(session)
                await test_compare_contracts_nonexistent_title(session)

                # Protocol-level required-parameter enforcement
                await test_collection_name_required(session)
    finally:
        teardown_test_collections()

    print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
    
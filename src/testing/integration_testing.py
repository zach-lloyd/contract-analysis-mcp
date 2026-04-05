"""
Integration tests for MCP server tools.

Run DB-only tests (no Ollama required):
    python mcp_integration_test.py --db-only

Run all tests (requires Ollama with qwen3:32b):
    python mcp_integration_test.py
"""
import asyncio
import argparse
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"

# Resolve server.py relative to this file so the path works regardless of
# where you run the script from
_SERVER_PATH = str(Path(__file__).parent.parent / "rag" / "server.py")


async def run_tests(db_only=False):
    """
    Spin up the MCP server once, run all tests against the shared session,
    then tear down.

    Args:
        db_only: Optional. If true, skips the LLM tests.
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python3", _SERVER_PATH],
    )

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

            print("\n--- DB-only tests ---\n")
            await test_list_contracts(session)
            await test_list_contracts_filtered(session)
            await test_list_contracts_nonexistent_party(session)
            await test_find_contract_clauses(session)
            await test_find_contract_clauses_filtered(session)
            await test_find_contract_clauses_nonexistent_title(session)
            
            if db_only:
                print("\n--- Skipping LLM tests (--db-only) ---\n")
            else:
                print("\n--- LLM tests (require Ollama + qwen3:32b) ---\n")
                # LLM tests will go here

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Run only DB tests (no Ollama required)"
    )
    args = parser.parse_args()
    asyncio.run(run_tests(db_only=args.db_only))

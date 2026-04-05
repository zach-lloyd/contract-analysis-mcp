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
            

            async def test_ask_contracts(session):
                """ask_contracts returns a non-empty answer for a broad question."""
                name = "ask_contracts (broad query)"
                result = await session.call_tool("ask_contracts", {
                    "question": "What are common termination provisions?"
                })
                text = get_text(result)

                if not text or len(text.strip()) == 0:
                    print(f"  [{FAIL}] {name}: got empty response")
                    return

                # A meaningful answer should be more than a one-word reply
                if len(text.split()) < 10:
                    print(f"  [{FAIL}] {name}: response suspiciously short ({len(text.split())} words)")
                    return

                print(f"  [{PASS}] {name}: got answer ({len(text)} chars)")
            

            async def test_ask_contract(session):
                """ask_contract returns a relevant answer for a known contract."""
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

                if not text or len(text.strip()) == 0:
                    print(f"  [{FAIL}] {name}: got empty response")
                    return

                if len(text.split()) < 10:
                    print(f"  [{FAIL}] {name}: response suspiciously short ({len(text.split())} words)")
                    return

                print(f"  [{PASS}] {name}: got answer ({len(text)} chars) for '{title}'")


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
                """compare_contracts returns an answer that references both contracts."""
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

                # Both contract titles should appear in the comparison
                missing = [t for t in titles if t.lower() not in text.lower()]
                if missing:
                    print(f"  [{FAIL}] {name}: response doesn't mention {missing}")
                    return

                print(f"  [{PASS}] {name}: answer references both '{titles[0]}' and '{titles[1]}'")


            async def test_compare_contracts_uses_comparison_language(session):
                """compare_contracts actually compares rather than just summarizing each contract."""
                name = "compare_contracts (comparison language)"

                list_result = await session.call_tool("list_contracts", {})
                list_text = get_text(list_result)
                lines = list_text.splitlines()[1:]
                titles = [line.split(" (parties:")[0].strip() for line in lines[:2]]

                if len(titles) < 2:
                    print(f"  [{FAIL}] {name}: need at least 2 contracts in database")
                    return

                result = await session.call_tool("compare_contracts", {
                    "question": "How do the termination clauses differ?",
                    "contract_titles": titles,
                })
                text = get_text(result).lower()

                # Theoretically, the contracts could be compared without using any
                # of these words. But in practice, if none of these appear in the
                # response, it's a red flag that the LLM may just be summarizing
                # each contract rather than actually comparing them
                comparison_indicators = [
                    "differ", "whereas", "both", "unlike", "in contrast",
                    "similarly", "however", "on the other hand", "while",
                    "compared", "distinction", "common", "same", "different",
                ]

                matches = [word for word in comparison_indicators if word in text]

                if not matches:
                    print(f"  [{FAIL}] {name}: no comparison language found — LLM may be "
                        f"summarizing rather than comparing")
                    return

                print(f"  [{PASS}] {name}: found comparison language: {', '.join(matches)}")


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
                await test_ask_contracts(session)
                await test_ask_contracts(session)
                await test_ask_contract(session)
                await test_ask_contract_nonexistent_title(session)
                await test_compare_contracts(session)
                await test_compare_contracts_uses_comparison_language(session)
                await test_compare_contracts_too_few_titles(session)
                await test_compare_contracts_nonexistent_title(session)

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

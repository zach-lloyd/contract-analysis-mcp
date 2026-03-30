from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("contracts")

@mcp.tool()
async def ask_contracts(question: str) -> str:
    """
    Ask a question across all contracts in the database. Retrieves the most
    relevant clauses from any contract and uses an LLM to generate an answer
    based on those clauses. Use this when the user's question is not specific
    to a single contract or when they want to search broadly.

    Args:
        question: The user's natural language question about their contracts.
    """
    return "not implemented"

@mcp.tool()
async def ask_contract(question: str, contract_title: str) -> str:
    """
    Ask a question about a specific contract. Retrieves the most relevant
    clauses from the named contract and uses an LLM to generate an answer.
    Use this when the user's question targets a single known contract.

    Args:
        question: The user's natural language question about the contract.
        contract_title: The exact title of the contract to search within.
    """
    return "not implemented"

@mcp.tool()
async def compare_contracts(question: str, contract_titles: list[str]) -> str:
    """
    Compare two or more contracts with respect to a specific question or topic.
    Retrieves relevant clauses from each named contract and uses an LLM to
    generate a comparative analysis. Use this when the user wants to understand
    how contracts differ on a particular provision, term, or obligation.

    Args:
        question: The user's question or topic to compare across contracts
                  (e.g., "How do the termination clauses differ?").
        contract_titles: A list of exact contract titles to compare.
    """
    return "not implemented"

@mcp.tool()
async def find_contract_clauses(clause_type: str, contract_title: str = None) -> str:
    """
    Find clauses of a specific type across all contracts or within a single
    contract. Returns the relevant clause excerpts along with their source
    contract titles. Use this when the user wants to locate specific clause
    types like indemnification, termination, non-compete, governing law, etc.

    Args:
        clause_type: The type of clause to search for (e.g., "non-compete",
                     "termination", "indemnification", "governing law").
        contract_title: Optional. If provided, restricts the search to this
                        specific contract. If omitted, searches all contracts.
    """
    return "not implemented"

@mcp.tool()
async def list_contracts(party_name: str = None) -> str:
    """
    List all contracts available in the database or all contracts to which the named 
    party is party. Returns the titles of every contract that has been indexed. Use 
    this when the user wants to know what contracts are available to query, or when 
    you need to look up exact contract titles before calling other tools.

    Args:
        party_name: Optional. If provided, lists all contracts where the specified
                    party is a party to the contract. If omitted, lists all contracts.
    """
    return "not implemented"

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
from mcp.server.fastmcp import FastMCP
from rag_core import (
    query_clauses,
    list_matching_contracts,
    NUM_RESULTS,
)
from llm_provider import set_provider, VALID_PROVIDERS
from uuid import uuid4
import argparse
import asyncio

# For debugging server connection to Claude Desktop
import sys
print("server.py: starting imports", file=sys.stderr)

# Initialize FastMCP server
mcp = FastMCP("contracts")

# In-memory session storage for storing up to 30 previously retrieved clauses.
session_clauses: dict[str, list[str]] = {}


def _get_or_create_session(
    session_id: str = None,
) -> tuple[str, list[dict], list[str]]:
    """
    Retrieve an existing session or create a new one. Centralizes the
    session-lookup logic so ask_contracts and ask_contract stay clean.

    Args:
        session_id: An existing session ID, or None to create a new session.
    """
    if not session_id or session_id not in session_clauses:
        session_id = str(uuid4())
        session_clauses[session_id] = []

    return session_id, session_clauses[session_id]


@mcp.tool()
async def ask_contracts(question: str, session_id: str = None) -> str:
    """
    Ask a question across all contracts in the database. Retrieves the most
    relevant clauses from any contract and passes them to the LLM to generate an answer
    based on those clauses. Use this when the user's question is not specific
    to a single contract or when they want to search broadly.

    Args:
        question: The user's natural language question about their contracts.
        session_id: Optional. The session ID returned by a previous call.
                    Pass this to include clauses retrieved from previous turns 
                    of the conversation in the context. If omitted, a new session 
                    is created.
    """
    try:
        session_id, clauses = _get_or_create_session(session_id)
        # To keep the context from balloning, limit the number of previously-retrieved
        # clauses to 30
        if len(clauses) > 30:
            clauses[:] = clauses[-30:]

        results = await asyncio.to_thread(query_clauses, question, NUM_RESULTS)

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Add the retrieved clauses to the stored clauses if they are not already included
        for meta, chunk in zip(metadatas, chunks):
            title_and_excerpt = f"Contract Title: {meta['contract_title']}\nContract Excerpt: {chunk}\n\n"

            if title_and_excerpt not in clauses:
                clauses.append(title_and_excerpt)

        return clauses
    except Exception as e:
        return f"Error answering question: {e}"


@mcp.tool()
async def ask_contract(
    question: str, contract_title: str, session_id: str = None
) -> str:
    """
    Ask a question about a specific contract. Retrieves the most relevant
    clauses from the named contract and passes them to the LLM to generate an answer.
    Use this when the user's question targets a single known contract.

    Supports multi-turn conversations: pass the session_id from a previous
    response to maintain conversation context for follow-up questions. If no
    session_id is provided, a new session is created.

    Args:
        question: The user's natural language question about the contract.
        contract_title: The exact title of the contract to search within.
        session_id: Optional. The session ID returned by a previous call.
                    Pass this to continue a conversation with follow-up
                    questions. If omitted, a new session is created.
    """
    try:
        # Verify the contract exists before querying to give a clear error
        # message rather than an empty or confusing LLM response
        contracts = await asyncio.to_thread(list_matching_contracts)
        known_titles = {c["contract_title"] for c in contracts}

        if contract_title not in known_titles:
            return (
                f"No contract found with title '{contract_title}'. "
                f"Use the list_contracts tool to see available contract titles."
            )

        session_id, clauses = _get_or_create_session(session_id)

        # To keep the context from balloning, limit the number of previously-retrieved
        # clauses to 30
        if len(clauses) > 30:
            clauses[:] = clauses[-30:]

        results = await asyncio.to_thread(
            query_clauses, question, NUM_RESULTS, contract_title
        )

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Add the retrieved clauses to the stored clauses if they are not already included
        for meta, chunk in zip(metadatas, chunks):
            title_and_excerpt = f"Contract Title: {meta['contract_title']}\nContract Excerpt: {chunk}\n\n"

            if title_and_excerpt not in clauses:
                clauses.append(title_and_excerpt)

        return clauses
    except Exception as e:
        return f"Error answering question: {e}"


@mcp.tool()
async def compare_contracts(question: str, contract_titles: list[str]) -> str:
    """
    Compare two or more contracts with respect to a specific question or topic.
    Retrieves relevant clauses from each named contract and passes them to the LLM to
    generate a comparative analysis. Use this when the user wants to understand
    how contracts differ on a particular provision, term, or obligation.
 
    Args:
        question: The user's question or topic to compare across contracts
                  (e.g., "How do the termination clauses differ?").
        contract_titles: A list of exact contract titles to compare.
    """
    try:
        if len(contract_titles) < 2:
            return "Please provide at least two contract titles to compare."
 
        # Validate all titles up front
        contracts = await asyncio.to_thread(list_matching_contracts)
        known_titles = {c["contract_title"] for c in contracts}
 
        invalid = [t for t in contract_titles if t not in known_titles]
        if invalid:
            return (
                f"Contract(s) not found: {', '.join(invalid)}. "
                f"Use the list_contracts tool to see available contract titles."
            )
 
        # Query each contract separately so the results are balanced
        # across contracts rather than skewed toward whichever is most relevant
        sections = []
        for title in contract_titles:
            results = await asyncio.to_thread(
                query_clauses, question, NUM_RESULTS, title
            )
            chunks = results["documents"][0]
 
            excerpts = "\n\n".join(chunks)
            sections.append(f"=== {title} ===\n{excerpts}")
 
        return "\n\n".join(sections)
    except Exception as e:
        return f"Error comparing contracts: {e}"


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
    try:
        if contract_title:
            contracts = await asyncio.to_thread(list_matching_contracts)
            known_titles = {c["contract_title"] for c in contracts}

            if contract_title not in known_titles:
                return (
                    f"No contract found with title '{contract_title}'. "
                    f"Use the list_contracts tool to see available contract titles."
                )

        results = await asyncio.to_thread(
            query_clauses, clause_type, NUM_RESULTS, contract_title
        )

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not chunks:
            return f"No clauses found matching '{clause_type}'."

        formatted = []
        for meta, chunk in zip(metadatas, chunks):
            formatted.append(
                f"Contract: {meta['contract_title']}\n"
                f"Excerpt: {chunk}"
            )

        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Error finding clauses: {e}"


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
    try:
        contracts = await asyncio.to_thread(list_matching_contracts, party_name)

        if not contracts:
            if party_name:
                return f"No contracts found involving party '{party_name}'."

            return "No contracts found in the database."

        formatted = []
        for c in contracts:
            line = c["contract_title"]
            if c["parties"]:
                line += f" (parties: {c['parties']})"
            formatted.append(line)

        header = f"Found {len(contracts)} contract(s)"
        if party_name:
            header += f" involving '{party_name}'"
        header += ":\n"

        return header + "\n".join(formatted)
    except Exception as e:
        return f"Error listing contracts: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=VALID_PROVIDERS,
        default="ollama",
        help="LLM provider to use: ollama (default), claude, or gemini",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model for the chosen provider",
    )
    args = parser.parse_args()

    set_provider(args.provider, args.model)

    # For debugging connection to Claude Desktop
    print(
        f"server.py: using provider '{args.provider}', "
        f"about to start MCP server",
        file=sys.stderr,
    )
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

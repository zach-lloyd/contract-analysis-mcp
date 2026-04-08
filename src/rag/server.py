from mcp.server.fastmcp import FastMCP
from rag_core import (
    generate_answer,
    generate_comparison,
    query_clauses,
    list_matching_contracts,
    NUM_RESULTS,
)
from uuid import uuid4
import asyncio

# For debugging server connection to Claude Desktop
import sys
print("server.py: starting imports", file=sys.stderr)

# Initialize FastMCP server
mcp = FastMCP("contracts")

# In-memory session storage for multi-turn conversations. Each session tracks 
# its own conversation history and retrieved clauses independently.
conversations: dict[str, list[dict]] = {}
session_clauses: dict[str, list[str]] = {}


def _get_or_create_session(
    session_id: str = None,
) -> tuple[str, list[dict], list[str]]:
    """
    Retrieve an existing session or create a new one. Centralizes the
    session-lookup logic so ask_contracts and ask_contract stay clean.

    Args:
        session_id: An existing session ID, or None to create a new session.

    Returns:
        A tuple of (session_id, history, clauses).
    """
    if not session_id or session_id not in conversations:
        session_id = str(uuid4())
        conversations[session_id] = []
        session_clauses[session_id] = []

    return session_id, conversations[session_id], session_clauses[session_id]


@mcp.tool()
async def ask_contracts(question: str, session_id: str = None) -> str:
    """
    Ask a question across all contracts in the database. Retrieves the most
    relevant clauses from any contract and uses an LLM to generate an answer
    based on those clauses. Use this when the user's question is not specific
    to a single contract or when they want to search broadly.

    Supports multi-turn conversations: pass the session_id from a previous
    response to maintain conversation context for follow-up questions. If no
    session_id is provided, a new session is created.

    Args:
        question: The user's natural language question about their contracts.
        session_id: Optional. The session ID returned by a previous call.
                    Pass this to continue a conversation with follow-up
                    questions. If omitted, a new session is created.
    """
    try:
        session_id, history, clauses = _get_or_create_session(session_id)

        answer, history, clauses = await asyncio.to_thread(
            generate_answer, question, None, clauses, history
        )

        # Persist the updated state back to the session dicts
        conversations[session_id] = history
        session_clauses[session_id] = clauses

        return f"[session_id: {session_id}]\n\n{answer}"
    except Exception as e:
        return f"Error answering question: {e}"


@mcp.tool()
async def ask_contract(
    question: str, contract_title: str, session_id: str = None
) -> str:
    """
    Ask a question about a specific contract. Retrieves the most relevant
    clauses from the named contract and uses an LLM to generate an answer.
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

        session_id, history, clauses = _get_or_create_session(session_id)

        answer, history, clauses = await asyncio.to_thread(
            generate_answer, question, contract_title, clauses, history
        )

        conversations[session_id] = history
        session_clauses[session_id] = clauses

        return f"[session_id: {session_id}]\n\n{answer}"
    except Exception as e:
        return f"Error answering question: {e}"


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

        answer = await asyncio.to_thread(
            generate_comparison, question, contract_titles
        )

        return answer
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
    # For debugging connection to Claude Desktop
    print("server.py: about to start MCP server", file=sys.stderr)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

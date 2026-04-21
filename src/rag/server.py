from mcp.server.fastmcp import FastMCP
from rag_core import (
    query_clauses,
    list_matching_contracts,
    list_collections as list_all_collections,
    NUM_RESULTS,
)
from uuid import uuid4
import asyncio
import json

# For debugging server connection to Claude Desktop
import sys
print("server.py: starting imports", file=sys.stderr)

# Initialize FastMCP server
mcp = FastMCP("contracts")

# In-memory session storage for storing up to 30 previously retrieved clauses.
# Each session maps to a dict with "clauses" (list of clause dicts) and "seen"
# (set of (title, excerpt) tuples for O(1) deduplication).
_sessions: dict[str, dict] = {}


def _get_or_create_session(
    session_id: str = None,
) -> tuple[str, list[dict], set[tuple]]:
    """
    Retrieve an existing session or create a new one. Centralizes the
    session-lookup logic so ask_contracts and ask_contract stay clean.

    Args:
        session_id: An existing session ID, or None to create a new session.

    Returns:
        A tuple of (session_id, clauses list, seen set).
    """
    if not session_id or session_id not in _sessions:
        session_id = str(uuid4())
        _sessions[session_id] = {"clauses": [], "seen": set()}

    session = _sessions[session_id]
    return session_id, session["clauses"], session["seen"]


@mcp.tool()
async def ask_contracts(
    question: str, collection_name: str, session_id: str = None
) -> str:
    """
    Ask a question across all contracts in the database. Returns the most relevant 
    clauses for the host LLM to use when generating an answer. Use this when the 
    user's question is not specific to a single contract or when they want to search 
    broadly.
 
    Supports multi-turn conversations: pass the session_id from a previous
    response to maintain conversation context for follow-up questions. If no
    session_id is provided, a new session is created.

    If a contract is not found in one collection, call list_collections and retry 
    across other collections before concluding the contract doesn't exist.
 
    Args:
        question: The user's natural language question about their contracts.
        collection_name: The name of the contract collection to search. Use the 
                         list_collections tool to see available collections.
        session_id: Optional. The session ID returned by a previous call.
                    Pass this to include clauses retrieved from previous turns 
                    of the conversation in the context. If omitted, a new session 
                    is created.
    """
    try:
        session_id, clauses, seen = _get_or_create_session(session_id)
        # To keep the context from ballooning, limit the number of previously-retrieved
        # clauses to 30
        if len(clauses) > 30:
            clauses[:] = clauses[-30:]
            seen.clear()
            seen.update((c["contract_title"], c["excerpt"]) for c in clauses)
 
        results = await asyncio.to_thread(
            query_clauses, question, NUM_RESULTS, collection_name, None 
        )
 
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
 
        # Add the retrieved clauses to the stored clauses if they are not already included
        for meta, chunk in zip(metadatas, chunks):
            clause_key = (meta["contract_title"], chunk)
            if clause_key not in seen:
                seen.add(clause_key)
                clauses.append({
                    "contract_title": meta["contract_title"],
                    "parties": meta.get("parties", ""),
                    "excerpt": chunk,
                })
 
        return json.dumps({"session_id": session_id, "clauses": clauses})
    except Exception as e:
        return json.dumps({"error": f"Error answering question: {e}"})
 
 
@mcp.tool()
async def ask_contract(
    question: str, contract_title: str, collection_name: str, session_id: str = None
) -> str:
    """
    Ask a question about a specific contract. Returns the most relevant 
    clauses for the host LLM to use when generating an answer.
    Use this when the user's question targets a single known contract.
 
    Supports multi-turn conversations: pass the session_id from a previous
    response to maintain conversation context for follow-up questions. If no
    session_id is provided, a new session is created.

    If a contract is not found in one collection, call list_collections and retry 
    across other collections before concluding the contract doesn't exist.
 
    Args:
        question: The user's natural language question about the contract.
        contract_title: The exact title of the contract to search within.
        collection_name: The name of the contract collection to search. Use the 
                         list_collections tool to see available collections.
        session_id: Optional. The session ID returned by a previous call.
                    Pass this to continue a conversation with follow-up
                    questions. If omitted, a new session is created.
    """
    try:
        # Verify the contract exists before querying to give a clear error
        # message rather than an empty or confusing LLM response
        contracts = await asyncio.to_thread(
            list_matching_contracts, collection_name, None
        )
        known_titles = {c["contract_title"] for c in contracts}
 
        if contract_title not in known_titles:
            return json.dumps({
                "error": f"No contract found with title '{contract_title}'. "
                         f"Use the list_contracts tool to see available contract titles."
            })
 
        session_id, clauses, seen = _get_or_create_session(session_id)
 
        # To keep the context from ballooning, limit the number of previously-retrieved
        # clauses to 30
        if len(clauses) > 30:
            clauses[:] = clauses[-30:]
            seen.clear()
            seen.update((c["contract_title"], c["excerpt"]) for c in clauses)
 
        results = await asyncio.to_thread(
            query_clauses, question, NUM_RESULTS, collection_name, contract_title 
        )
 
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
 
        # Add the retrieved clauses to the stored clauses if they are not already included
        for meta, chunk in zip(metadatas, chunks):
            clause_key = (meta["contract_title"], chunk)
            if clause_key not in seen:
                seen.add(clause_key)
                clauses.append({
                    "contract_title": meta["contract_title"],
                    "parties": meta.get("parties", ""),
                    "excerpt": chunk,
                })
 
        return json.dumps({"session_id": session_id, "clauses": clauses})
    except Exception as e:
        return json.dumps({"error": f"Error answering question: {e}"})
 
 
@mcp.tool()
async def compare_contracts(
    question: str, contract_titles: list[str],
    collection_name: str
) -> str:
    """
    Compare two or more contracts with respect to a specific question or topic.
    Returns relevant clauses from each named contract for the host LLM to use when
    generating a comparative analysis. Use this when the user wants to understand
    how contracts differ on a particular provision, term, or obligation.

    If a contract is not found in one collection, call list_collections and retry 
    across other collections before concluding the contract doesn't exist.
 
    Args:
        question: The user's question or topic to compare across contracts
                  (e.g., "How do the termination clauses differ?").
        contract_titles: A list of exact contract titles to compare.
        collection_name: The name of the contract collection to search. Use the 
                         list_collections tool to see available collections.
    """
    try:
        if len(contract_titles) < 2:
            return json.dumps({"error": "Please provide at least two contract titles to compare."})
 
        # Validate all titles up front
        contracts = await asyncio.to_thread(
            list_matching_contracts, collection_name, None
        )
        known_titles = {c["contract_title"] for c in contracts}
 
        invalid = [t for t in contract_titles if t not in known_titles]
        if invalid:
            return json.dumps({
                "error": f"Contract(s) not found: {', '.join(invalid)}. "
                         f"Use the list_contracts tool to see available contract titles."
            })
 
        # Query each contract separately so the results are balanced
        # across contracts rather than skewed toward whichever is most relevant
        comparisons = []
        for title in contract_titles:
            results = await asyncio.to_thread(
                query_clauses, question, NUM_RESULTS, collection_name, title 
            )
            chunks = results["documents"][0]
 
            comparisons.append({
                "contract_title": title,
                "excerpts": chunks,
            })
 
        return json.dumps({"comparisons": comparisons})
    except Exception as e:
        return json.dumps({"error": f"Error comparing contracts: {e}"})
 
 
@mcp.tool()
async def find_contract_clauses(
    clause_type: str, collection_name: str, contract_title: str = None
) -> str:
    """
    Find clauses of a specific type across all contracts or within a single
    contract. Returns the relevant clause excerpts along with their source
    contract titles. Use this when the user wants to locate specific clause
    types like indemnification, termination, non-compete, governing law, etc.
 
    Args:
        clause_type: The type of clause to search for (e.g., "non-compete",
                     "termination", "indemnification", "governing law").
        collection_name: The name of the contract collection to search. Use the 
                         list_collections tool to see available collections.
        contract_title: Optional. If provided, restricts the search to this
                        specific contract. If omitted, searches all contracts.
    """
    try:
        if contract_title:
            contracts = await asyncio.to_thread(
                list_matching_contracts, collection_name, None
            )
            known_titles = {c["contract_title"] for c in contracts}
 
            if contract_title not in known_titles:
                return json.dumps({
                    "error": f"No contract found with title '{contract_title}'. "
                             f"Use the list_contracts tool to see available contract titles."
                })
 
        results = await asyncio.to_thread(
            query_clauses, clause_type, NUM_RESULTS, collection_name, contract_title 
        )
 
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
 
        if not chunks:
            return json.dumps({"clauses": [], "message": f"No clauses found matching '{clause_type}'."})
 
        clauses = [
            {"contract_title": meta["contract_title"], "excerpt": chunk}
            for meta, chunk in zip(metadatas, chunks)
        ]
 
        return json.dumps({"clauses": clauses})
    except Exception as e:
        return json.dumps({"error": f"Error finding clauses: {e}"})
 
 
@mcp.tool()
async def list_contracts(
    collection_name: str,
    party_name: str = None
) -> str:
    """
    List all contracts available in the database or all contracts to which the named
    party is party. Returns the titles of every contract that has been indexed. Use
    this when the user wants to know what contracts are available to query, or when
    you need to look up exact contract titles before calling other tools.
 
    Args:
        party_name: Optional. If provided, lists all contracts where the specified
                    party is a party to the contract. If omitted, lists all contracts.
        collection_name: The name of the contract collection to search. Use the 
                         list_collections tool to see available collections.
    """
    try:
        contracts = await asyncio.to_thread(
            list_matching_contracts, collection_name, party_name
        )
 
        if not contracts:
            if party_name:
                return json.dumps({
                    "contracts": [],
                    "message": f"No contracts found involving party '{party_name}'.",
                })
 
            return json.dumps({"contracts": [], "message": "No contracts found in the database."})
 
        return json.dumps({"contracts": contracts})
    except Exception as e:
        return json.dumps({"error": f"Error listing contracts: {e}"})
 
 
@mcp.tool()
async def list_collections() -> str:
    """
    List all contract collections available in the database. Returns the name
    of every ChromaDB collection that has been indexed. Use this when the user
    wants to know what collections exist, or when you need to determine the
    correct collection_name to pass to the other tools.
    """
    try:
        names = await asyncio.to_thread(list_all_collections)
 
        if not names:
            return json.dumps({"collections": [], "message": "No collections found in the database."})
 
        return json.dumps({"collections": names})
    except Exception as e:
        return json.dumps({"error": f"Error listing collections: {e}"})
 
 
def main():
    print("server.py: about to start MCP server", file=sys.stderr)    
    mcp.run(transport="stdio")
 
 
if __name__ == "__main__":
    main()

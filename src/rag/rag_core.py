import chromadb
from pathlib import Path

# Ensure the chroma_data folder can be found regardless of which folder the code
# is run from. Added this to address an error related to the location of chroma_data
# that arose when I tried to run my rag testing code from the src folder
_DB_PATH = str(Path(__file__).parent / "chroma_data")
CLIENT = chromadb.PersistentClient(path=_DB_PATH)
NUM_RESULTS = 10

DEFAULT_COLLECTION = "legal_contracts"

# Cache of opened collections so we don't reopen the same one on every call
_collections: dict[str, chromadb.Collection] = {}


def get_collection(name: str = DEFAULT_COLLECTION) -> chromadb.Collection:
    """
    Retrieve a ChromaDB collection by name, caching it for reuse.

    Args:
        name: Optional. The name of the collection to retrieve. Defaults to 
              'legal_contracts'.
    """
    if name not in _collections:
        _collections[name] = CLIENT.get_or_create_collection(name=name)
    return _collections[name]


def list_collections() -> list[str]:
    """
    List the names of all ChromaDB collections in the database.
    """
    return [c.name for c in CLIENT.list_collections()]


def query_clauses(
        question: str, num_results: int, contract_title: str = None,
        collection_name: str = DEFAULT_COLLECTION
) ->  chromadb.QueryResult:
    """
    Query the contract database for the chunks that are most relevant to the 
    question.

    Args:
        question: The user's question.
        num_results: The number of contract chunks for the query to return.
        contract_title: Optional. If present, limit the search to a specific 
                        contract. If not, search across all contracts in the
                        database.
        collection_name: Optional. The ChromaDB collection to query. Defaults
                         to 'legal_contracts'.
    """
    collection = get_collection(collection_name)

    if contract_title:
        results = collection.query(
            query_texts=[question],
            n_results=num_results,
            where={"contract_title": contract_title}
        )
    else:
        results = collection.query(
            query_texts=[question],
            n_results=num_results
        )
    
    return results


def list_matching_contracts(
        party_name: str = None,
        collection_name: str = DEFAULT_COLLECTION
) -> list[dict[str, str]]:
    """
    List all contracts in the database, or only those involving a specific party.
    Returns deduplicated contract titles and their associated parties.
 
    Args:
        party_name: Optional. If provided, only return contracts where this party
                    appears in the parties metadata. If omitted, return all contracts.
        collection_name: Optional. The ChromaDB collection to query. Defaults
                         to 'legal_contracts'.
    """
    collection = get_collection(collection_name)

    seen = {}
    batch_size = 5000
    offset = 0

    # Paginate through all metadata to avoid SQLite variable limits
    while True:
        batch = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset
        )
        metadatas = batch["metadatas"]

        if not metadatas:
            break

        for meta in metadatas:
            title = meta["contract_title"]
            if title not in seen:
                seen[title] = meta.get("parties", "")

        offset += batch_size

    contracts = [
        {"contract_title": title, "parties": parties}
        for title, parties in seen.items()
    ]
 
    if party_name:
        contracts = [
            c for c in contracts
            if party_name.lower() in c["parties"].lower()
        ]
    
    return contracts

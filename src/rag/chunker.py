from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import chromadb

def load_contracts(filename: str) -> list[dict[str, int]]:
    """
    Load contracts into a list of dictionaries containing the title, text, and
    parties of each contract. Note that contract keys assume use of the CUAD
    dataset but can be adapted for use with different contract datasets.

    Args:
        filename: The name of the file containing the contract data.
    """
    with open(filename, "r") as f:
        contract_data = json.load(f)

    contracts = []

    for contract in contract_data["data"]:
        title = contract["title"]
        # Store the text of the contract in a single string
        full_text = "\n\n".join(
            p["context"] for p in contract["paragraphs"]
        )
        contracts.append({"title": title, "text": full_text})

    print(f"Loaded {len(contracts)} contracts")

    return contracts

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=256,
    chunk_overlap=80,
)

chunks = []

for contract in contracts:    
    for chunk in splitter.split_text(contract["text"]):
        chunks.append({"contract_title": contract["title"], "chunk_text": chunk})

# Uncomment this code to print some example chunks
'''
for i, chunk in enumerate(chunks[:3]):
    print(f"\nContract: {chunk['contract_title']}")
    print(f"\n--- Chunk {i+1} ---")
    print(chunk["chunk_text"])
'''

# I used chromadb for vector storage because it offers persistence without 
# being overkill like a full client-server database would be
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="legal_contracts")

if collection.count() == 0:
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[f"id{i + j + 1}" for j in range(len(batch))],
            documents=[chunk["chunk_text"] for chunk in batch],
            metadatas=[{"contract_title": chunk["contract_title"]} for chunk in batch],
        )

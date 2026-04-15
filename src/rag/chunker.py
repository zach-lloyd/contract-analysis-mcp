from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import json
import chromadb
import re

DATASET = str(Path(__file__).parent.parent / "cuad" / "data" / "CUADv1.json")


def load_contracts(filename: str) -> list[dict[str, str]]:
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

        # Extract the names of the parties to the contract. These are stored as answers
        # to the "Parties" question category in each contract's question/answer pairs
        parties = set()

        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:
                category = re.findall(r'"([^"]*)"', qa["question"])
                # To get the party names, need to find the "Parties" category in the
                # Q&A pairs and also confirm that this question is not impossible
                # (if it's impossible, it means there are no party names to extract)
                if category and category[0] == "Parties" and not qa["is_impossible"]:
                    for answer in qa["answers"]:
                        parties.add(answer["text"].strip())

        contracts.append({
            "title": title,
            "text": full_text,
            "parties": ", ".join(sorted(parties)) if parties else "",
        })

    print(f"Loaded {len(contracts)} contracts")

    return contracts


def chunk_contracts(
        contracts: list[dict[str, str]], 
        encoding: str = "cl100k_base", 
        size: int = 256, 
        overlap: int = 80
        ) -> list[dict[str, str]]:
    """
    Split each contract into chunks of the specified token size using tiktoken.

    Args:
        contracts: A list of contracts to chunk.
        encoding: Optional. The encoding method to use. Defaults to 'cl100k_base'.
        size: Optional. The size of each chunk in tokens. Defaults to 256.
        overlap: Optional. The amount of overlap between each chunk in tokens. 
                 Defaults to 80. 
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding,
        chunk_size=size,
        chunk_overlap=overlap,
    )

    chunks = []

    for contract in contracts:    
        for chunk in splitter.split_text(contract["text"]):
            chunks.append({
                "contract_title": contract["title"],
                "chunk_text": chunk,
                "parties": contract["parties"],
            })
    
    return chunks


def print_sample_chunks(chunks: list[dict[str, str]], num_samples: int) -> None:
    """
    Prints a specified number of sample chunks for examination of how the contracts
    are being split.

    Args:
        chunks: The list of contract chunks.
        num_samples: The number of sample chunks to print.
    """
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"\nContract: {chunk['contract_title']}")
        print(f"Parties: {chunk['parties']}")
        print(f"\n--- Chunk {i+1} ---")
        print(chunk["chunk_text"])


def create_collection(
        chunks: list[dict[str, str]], 
        collection_name: str = "legal_contracts",
        batch_size: int = 5000
        ) -> chromadb.Collection:
    """
    Adds the contract chunks and metadata to a chromadb collection. I used chromadb for 
    vector storage because it offers persistence without being overkill like a full 
    client-server database would be.
 
    Args:
        chunks: The list of contract chunks to be added to the collection.
        collection_name: Optional. The name of the ChromaDB collection to create
                         or add to. Defaults to 'legal_contracts'.
        batch_size: Optional. Add chunks in batches of the specified size for efficiency.
                    Defaults to 5000.
    """
    # Ensure the chroma_data folder can be found regardless of which folder the code
    # is run from. Added this to address an error related to the location of chroma_data
    # that arose when I tried to run my rag testing code from the src folder
    _db_path = str(Path(__file__).parent / "chroma_data")
    client = chromadb.PersistentClient(path=_db_path)
    collection = client.get_or_create_collection(name=collection_name)
 
    if collection.count() == 0:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"adding batch {i}")
            collection.add(
                ids=[f"id{i + j + 1}" for j in range(len(batch))],
                documents=[chunk["chunk_text"] for chunk in batch],
                metadatas=[{
                    "contract_title": chunk["contract_title"],
                    "parties": chunk["parties"],
                } for chunk in batch],
            )
    
    return collection


def main():
    contracts = load_contracts(DATASET)
    chunks = chunk_contracts(contracts)
    print_sample_chunks(chunks, 3)
    create_collection(chunks)


if __name__ == "__main__":
    main()

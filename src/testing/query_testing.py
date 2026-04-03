import json
import chromadb
import random
import re
import tiktoken
from pathlib import Path

# The number of results for the query to retrieve. Test values from 1 to 10
NUM_RESULTS = list(range(1, 11))

# I divided the question categories into two buckets: conceptual and factual, 
# so I could measure how query performance differs between them
CONCEPTUAL_CATEGORIES = [
    'Affiliate License-Licensee', 
    'Volume Restriction',
    'Covenant Not To Sue',
    'Source Code Escrow',
    'Joint Ip Ownership',
    'Non-Transferable License',
    'Competitive Restriction Exception',
    'Non-Disparagement', 
    'Ip Ownership Assignment',
    'No-Solicit Of Employees',
    'Non-Compete', 
    'Minimum Commitment', 
    'Liquidated Damages', 
    'License Grant',
    'Cap On Liability', 
    'Change Of Control', 
    'Audit Rights', 
    'Insurance',
    'Uncapped Liability', 
    'Termination For Convenience', 
    'Post-Termination Services',
    'No-Solicit Of Customers', 
    'Most Favored Nation', 
    'Price Restrictions', 
    'Renewal Term',
    'Third Party Beneficiary', 
    'Irrevocable Or Perpetual License', 
    'Anti-Assignment', 
    'Exclusivity',
    'Unlimited/All-You-Can-Eat-License', 
    'Affiliate License-Licensor', 
    'Rofr/Rofo/Rofn', 
    'Revenue/Profit Sharing'
]

FACTUAL_CATEGORIES = [
    'Effective Date',
    'Warranty Duration',
    'Agreement Date',
    'Parties',
    'Expiration Date',
    'Document Name',
    'Notice Period To Terminate Renewal',
    'Governing Law', 
]

# Use training dataset as the primary benchmark for testing, since it contains 408
# of the 510 contracts and is thus much more robust than the test dataset
_TEST_DATA = str(Path(__file__).parent.parent / "cuad" / "data" / "train_separate_questions.json")
with open(_TEST_DATA, "r") as f:
    contract_data = json.load(f)


def get_qa_pairs(data):
    """
    Get question/answer pairs from each contract for use in testing the query 
    system.

    Args:
        data: A JSON dictionary representing the contract dataset.
    """
    qas = []

    random.seed(42)

    for i in range(0, len(data["data"])):
        contract = data["data"][i]
        contract_title = contract["title"]
        # Filter out any questions that are impossible to answer
        question_list = [q for q in contract["paragraphs"][0]["qas"] if not q["is_impossible"]]
        
        # Sample up to 5 questions from each contract
        sample_size = min(5, len(question_list))
        qa_indices = random.sample(question_list, sample_size)

        for q in qa_indices:
            question = q["question"]
            answer = q["answers"][0]["text"]

            qas.append({"title": contract_title, "question": question, "answer": answer})
    
    return qas


def get_token_length(qas):
    """
    Print the number of tokens in each answer.

    Args:
        qas: A dictionary of question/answer pairs and the title of the
             contract they relate to.
    """
    for qa in qas:
        # Ensure encoding matches what is used in chunker.py
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(qa["answer"])
        num_tokens = len(tokens)
        print(f"Number of tokens: {num_tokens}")


def query_test(qas, number_of_results):
    """
    Compute the accuracy of the query model on factual and conceptual questions.

    Args:
        qas: A dictionary of question/answer pairs and the title of the
             contract they relate to.
        number_of_results: The number of results to retrieve with each query.
    """    
    _db_path = str(Path(__file__).parent.parent / "rag" / "chroma_data")
    client = chromadb.PersistentClient(path=_db_path)
    collection = client.get_or_create_collection(name="legal_contracts")

    conceptual_correct = 0
    conceptual_total = 0
    factual_correct = 0
    factual_total = 0

    for qa in qas:
        question = qa["question"]
        # Each question's category is surrounded by quotation marks; this code will
        # extract it
        category = re.findall(r'"([^"]*)"', question)
        category_type = ""

        if category[0] in CONCEPTUAL_CATEGORIES:
            conceptual_total += 1
            category_type = "conceptual"
        else:
            factual_total += 1
            category_type = "factual"

        results = collection.query(
            query_texts=[question],
            n_results=number_of_results,
            # Be sure to only search the applicable contract, not the entire
            # database
            where={"contract_title": qa["title"]}
        )

        answer_found = False

        for doc in results["documents"]:
            normalized_answer = " ".join(qa["answer"].strip().split())

            for chunk in doc:
                normalized_chunk = " ".join(chunk.strip().split())

                if normalized_answer in normalized_chunk:
                    if category_type == "conceptual":
                        conceptual_correct += 1
                    else:
                        factual_correct += 1
                    answer_found = True
                    break
            if answer_found:
                break
        
        # Uncomment the below code to examine the Q/A pairs that the query is 
        # getting wrong
        '''
        if not answer_found:
            print(f"Question: {qa["question"]}")
            print(f"Answer: {qa["answer"]}")
            print("Results: \n")
            print(results["documents"])
        '''
               
    return conceptual_correct/conceptual_total, factual_correct/factual_total


if __name__ == "__main__":
    # Uncomment this code to print out the unique question categories. I manually
    # separated these into conceptual and factual categories so I could separately
    # calculate the performance of my querying on conceptual and factual questions
    '''
    categories = []

    for contract in contract_data["data"]:
        qas = contract["paragraphs"][0]["qas"]

        for qa in qas:
            category = re.findall(r'"([^"]*)"', qa["question"])
            categories.append(category[0])
    
    print(set(categories))
    '''
    qa_pairs = get_qa_pairs(contract_data)

    for num in NUM_RESULTS:
        conceptual_accuracy, factual_accuracy = query_test(qa_pairs, num)
        print(f"Conceptual accuracy when receiving {num} results: {conceptual_accuracy}")
        print(f"Factual accuracy when receiving {num} results: {factual_accuracy}")

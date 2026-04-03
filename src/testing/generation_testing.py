import json
import chromadb
import random
import re
import ollama
from datetime import datetime
from rag.rag_core import NUM_RESULTS, BASE_SYS_PROMPT
from query_testing import CONCEPTUAL_CATEGORIES, FACTUAL_CATEGORIES
from pathlib import Path

# Use training dataset as the primary benchmark for testing, since it contains 408
# of the 510 contracts and is thus much more robust than the test dataset
_TEST_DATA = str(Path(__file__).parent.parent / "cuad" / "data" / "train_separate_questions.json")
with open(_TEST_DATA, "r") as f:
    cuad = json.load(f)


def get_qa_pairs(data, interval=0):
    """
    Get a random question/answer pair from each contract for use in testing.

    Args:
        data: A JSON dictionary representing the contract dataset.
        interval: Optional. If included, it specifies how many contracts to skip
                  before getting a question from the next one. Use if the contract
                  database is too large to make testing a question from every 
                  contract feasible.
    """
    qas = []

    random.seed(42)

    for i in range(0, len(data["data"]), interval):
        contract = data["data"][i]
        contract_title = contract["title"]
        # Filter out any questions that are impossible to answer
        question_list = [q for q in contract["paragraphs"][0]["qas"] if not q["is_impossible"]]
        
        qa_index = random.randint(0, len(question_list) - 1)

        question = question_list[qa_index]["question"]
        answer = question_list[qa_index]["answers"][0]["text"]

        # Each question's category is surrounded by quotation marks; this code will
        # extract it
        category = re.findall(r'"([^"]*)"', question)
        category_type = ""

        if category[0] in CONCEPTUAL_CATEGORIES:
            category_type = "conceptual"
        else:
            category_type = "factual"
        
        qas.append({
            "title": contract_title, 
            "category_type": category_type,
            "question": question, 
            "answer": answer,
        })
    
    return qas


def get_candidate_answers(qas, number_of_results):
    """
    Use an LLM to generate an anwswer for each question in qas.

    Args:
        qas: A dictionary of question/answer pairs, the title of the
             contract they relate to, and the category type of the
             question (factual or conceptual).
        number_of_results: The number of results to retrieve with each query.
    """    
    _db_path = str(Path(__file__).parent.parent / "rag" / "chroma_data")
    client = chromadb.PersistentClient(path=_db_path)
    collection = client.get_or_create_collection(name="legal_contracts")
    question_num = 1

    for qa in qas:
        print(f"Generating answer for question number {question_num}...\n")

        question = qa["question"]

        results = collection.query(
            query_texts=[question],
            n_results=number_of_results,
            # Be sure to only search the applicable contract, not the entire
            # database
            where={"contract_title": qa["title"]}
        )

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        clauses_list = []

        for meta, chunk in zip(metadatas, chunks):
            title_and_excerpt = f"Contract Title: {meta['contract_title']}\nContract Excerpt: {chunk}\n\n"
            clauses_list.append(title_and_excerpt)

        context = " ".join(clauses_list)
        
        full_sys_prompt = f"""
                  {BASE_SYS_PROMPT} The relevant contract clauses are as follows: {context}.
                  The prior history of this conversation is below, ending with the
                  user's current question. If there is no text provided other than a
                  question from the user, then this is the first question and there 
                  is no conversation history to reference.
                  """
        
        sys_message = {"role": "system", "content": full_sys_prompt}
        q = {"role": "user", "content": f"Question: {question}"}
        
        response = ollama.chat(
            model="qwen3:32b",
            messages=[sys_message, q]
        )

        a = response["message"]["content"]
        qa["generated_answer"] = a
        question_num += 1
                    
    return qas


def score_answers(qas):
    """
    Use an LLM to compare the generated answers to the reference answers and 
    score them for correctness and completeness.

    Args:
        qas: A dictionary of question/answer pairs, the title of the
             contract they relate to, and the category type of the
             question (factual or conceptual).
    """
    scores = []
    question_num = 1

    scoring_prompt = """
                Given the below question, reference answer, and candidate answer, 
                rate the candidate on a 1-5 scale for correctness and completeness. 
                Return only a JSON object with your scores and a one-sentence 
                rationale, using the keys "score" for your score and "rationale" 
                for your rationale.
                """

    for qa in qas:
        print(f"Scoring question number {question_num}...\n")

        question = qa["question"]
        reference_answer = qa["answer"]
        candidate_answer = qa["generated_answer"]
        category_type = qa["category_type"]

        score = ollama.chat(
            model="qwen3:32b",
            messages=[
                {"role": "system", "content": scoring_prompt},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "user", "content": f"Reference Answer: {reference_answer}"},
                {"role": "user", "content": f"Candidate Answer: {candidate_answer}"}
            ]
        )

        print(score["message"]["content"])

        scores.append({
            "question": question,
            "candidate_answer": candidate_answer,
            "reference_answer": reference_answer,
            "category_type": category_type,
            "result": score["message"]["content"]
        })

        question_num += 1
    
    # Include the timestamp in the filepath so that prior runs are not overwritten
    # and can be compared
    file_path = f"test_results/generation_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(file_path, "w") as json_file:
        json.dump(scores, json_file, indent=4)
    
    return scores


if __name__ == "__main__":
    qa_pairs = get_qa_pairs(cuad)
    candidate_answers = get_candidate_answers(qa_pairs, NUM_RESULTS)
    results = score_answers(candidate_answers)

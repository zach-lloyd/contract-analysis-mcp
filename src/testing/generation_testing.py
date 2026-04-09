import json
import random
import re
import argparse
from rag.llm_provider import chat, set_provider, VALID_PROVIDERS
from datetime import datetime
from rag.rag_core import generate_answer
from testing.query_testing import CONCEPTUAL_CATEGORIES
from pathlib import Path

# Use training dataset as the primary benchmark for testing, since it contains 408
# of the 510 contracts and is thus much more robust than the test dataset
_TEST_DATA = str(Path(__file__).parent.parent / "cuad" / "data" / "train_separate_questions.json")
with open(_TEST_DATA, "r") as f:
    contract_data = json.load(f)

INTERVAL = 8


def get_qa_pairs(data, interval=1):
    """
    Get a random question/answer pair from each contract for use in testing.

    Args:
        data: A JSON dictionary representing the contract dataset.
        interval: Optional. Specifies how many contracts to skip before getting
                  a question from the next one. Defaults to 1 (every contract).
                  Use a higher value if the contract database is too large to 
                  make testing a question from every contract feasible.
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


def get_candidate_answers(qas):
    """
    Use rag_core.generate_answer to produce an answer for each question in qas.

    Args:
        qas: A list of dictionaries containing question/answer pairs, the title
             of the contract they relate to, and the category type of the
             question (factual or conceptual).
    """    
    question_num = 1

    for qa in qas:
        print(f"Generating answer for question number {question_num}...\n")

        answer, _, _ = generate_answer(
            qa["question"], 
            contract_title=qa["title"]
        )

        qa["generated_answer"] = answer
        question_num += 1
                    
    return qas


def score_answers(qas):
    """
    Use an LLM to compare the generated answers to the reference answers and 
    score them for correctness and completeness.

    Args:
        qas: A list of dictionaries containing question/answer pairs, the title
             of the contract they relate to, and the category type of the
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

        score = chat(
            messages=[
                {"role": "system", "content": scoring_prompt},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "user", "content": f"Reference Answer: {reference_answer}"},
                {"role": "user", "content": f"Candidate Answer: {candidate_answer}"}
            ]
        )

        print(score)

        scores.append({
            "question": question,
            "candidate_answer": candidate_answer,
            "reference_answer": reference_answer,
            "category_type": category_type,
            "result": score
        })

        question_num += 1
    
    output_dir = Path(__file__).parent / "test_results"
    # Create test_results directory if one does not already exist
    output_dir.mkdir(exist_ok=True)
    # Include the timestamp in the filepath so that prior runs are not overwritten
    # and can be compared
    file_path = output_dir / f"generation_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(file_path, "w") as json_file:
        json.dump(scores, json_file, indent=4)
    
    return scores


if __name__ == "__main__":
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

    qa_pairs = get_qa_pairs(contract_data, 8)
    candidate_answers = get_candidate_answers(qa_pairs)
    results = score_answers(candidate_answers)

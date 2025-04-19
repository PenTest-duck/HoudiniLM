import json
from llm_as_a_judge import get_similarity 

def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)

tests = read_json('dataset/alan-similarity-evals.json')

for i, test in enumerate(tests):
    actual_score = get_similarity(test['prompt1'], test['prompt2'])
    print(f"Test {i}, Expected Score: {test['score']}, Actual Score: {actual_score}")
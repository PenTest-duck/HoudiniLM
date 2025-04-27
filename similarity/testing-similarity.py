import pandas as pd
from llm_as_a_judge import get_similarity_2
from tqdm import tqdm

tests = pd.read_csv("MISC/datasets/custom/similarity_evals.csv")

for i, test in enumerate(tqdm(tests.iterrows())):
    _, row = test
    actual_score = get_similarity_2(row['prompt1'], row['prompt2'])
    if abs(row['score'] - actual_score["score"]) > 2:
        print(f"Test {i}, Expected Score: {row['score']}, Actual Score: {actual_score['score']}\n")
        print(actual_score)

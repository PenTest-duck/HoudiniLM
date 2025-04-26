from strong_reject_reward import strongreject_rubric
import pandas as pd
from tqdm import tqdm

def evaluate_reward_func():
    df = pd.read_csv("../dataset/eval_judge.csv")
    scores = []
    ground_truth_scores = []
    
    results_df = pd.DataFrame(columns=[
        'prompt', 'response', 'predicted_score', 'ground_truth'
    ])
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        response = row['response']
        
        # Calculate reward using our function
        reward_score = strongreject_rubric(prompt, [response])
        scores.append(reward_score)
        
        # Calculate ground truth (average of score1 and score2)
        ground_truth = row[""]
        ground_truth_scores.append(ground_truth)

        results = {
            'prompt': prompt,
            'response': response,
            'predicted_score': reward_score,
            'ground_truth': ground_truth
        }

        print(results)

        results_df = results_df._append(results, ignore_index=True)
    
    results_df.to_csv("reward_functions/reward_func_results.csv", index=False)

    # Calculate correlation and other metrics
    correlation = results_df['predicted_score'].corr(results_df['ground_truth'])
    mae = abs(results_df['predicted_score'] - results_df['ground_truth']).mean()
    
    print(f"Correlation: {correlation:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    

if __name__ == "__main__":
    evaluate_reward_func()

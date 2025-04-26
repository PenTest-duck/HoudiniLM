from proxy_reward import prompt_reward_func
import pandas as pd

def evaluate_reward_func():
    df = pd.read_csv("dataset/eval_judge.csv")
    scores = []
    ground_truth_scores = []
    
    results_df = pd.DataFrame(columns=[
        'prompt', 'response', 'predicted_score', 'ground_truth'
    ])
    
    for _, row in df.iterrows():
        prompt = row['prompt']
        response = row['response']
        
        # Calculate reward using our function
        reward_score = prompt_reward_func(prompt, [response])[0]
        scores.append(reward_score)
        
        # Calculate ground truth (average of score1 and score2)
        ground_truth = (row['score1'] + row['score2']) / 2
        ground_truth_scores.append(ground_truth)
        
        results_df = results_df._append({
            'prompt': prompt,
            'response': response,
            'predicted_score': reward_score,
            'ground_truth': ground_truth
        }, ignore_index=True)
    
    results_df.to_csv("reward_functions/reward_func_results.csv", index=False)

    # Calculate correlation and other metrics
    correlation = results_df['predicted_score'].corr(results_df['ground_truth'])
    mae = abs(results_df['predicted_score'] - results_df['ground_truth']).mean()
    
    print(f"Correlation: {correlation:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    

if __name__ == "__main__":
    evaluate_reward_func()

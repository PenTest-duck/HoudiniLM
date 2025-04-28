"""
This script merges individual results CSV files into a single CSV file.
This allows us to see all prompts and responses for all models in a single place.
"""

import pandas as pd

output_file = '../../MISC/results/model_evaluation/combined_results.csv'
csv_files = {
    "base_prompt": "../../MISC/results/model_evaluation/strongreject_evaluation_base.csv",
    "rule_syn": "../../MISC/results/model_evaluation/strongreject_evaluation_rule_syn.csv",
    "rule_syn_prefix": "../../MISC/results/model_evaluation/strongreject_evaluation_rule_syn_prefix.csv",
    "base_nous": "../../MISC/results/model_evaluation/strongreject_evaluation_none.csv",
    "reward_1": "../../MISC/results/model_evaluation/strongreject_evaluation_rl1.csv",
    "reward_2": "../../MISC/results/model_evaluation/strongreject_evaluation_rl2.csv",
    "reward_3": "../../MISC/results/model_evaluation/strongreject_evaluation_rl3.csv"
}

dfs = {}

# Load all CSVs
for name, file_name in csv_files.items():
    df = pd.read_csv(file_name)
    dfs[name] = df

# Initialise the base DataFrame with the dataset name and the original_prompt
combined_df = dfs["base_prompt"][['dataset', 'original_prompt']].copy()

# Columns in the dataset
columns_to_group = ['improved_prompt', 'target_response', 'strongreject_refusal',
                    'strongreject_convincingness', 'strongreject_specificity', 'strongreject_score']

# For each column type, collect that column from every df
for col in columns_to_group:
    for name, df in dfs.items():
        new_col_name = f"{name}_{col}"
        combined_df[new_col_name] = df[col]

# Save to output file
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV saved to {output_file}")

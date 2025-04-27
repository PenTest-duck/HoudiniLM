import pandas as pd
from sklearn.metrics import cohen_kappa_score

judge_df = pd.read_csv("dataset/eval_judge.csv")
kappa12 = cohen_kappa_score(judge_df["score1"], judge_df["score2"])
kappa13 = cohen_kappa_score(judge_df["score1"], judge_df["score3"])
kappa23 = cohen_kappa_score(judge_df["score2"], judge_df["score3"])

print("Kappa between score1 and score2:", kappa12)
print("Kappa between score1 and score3:", kappa13)
print("Kappa between score2 and score3:", kappa23)
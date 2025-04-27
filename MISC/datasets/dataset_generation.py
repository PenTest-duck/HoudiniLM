from transformers import pipeline
from datasets import load_dataset

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ['This is a question', 'This is a declaration', 'This is a prompt']

def is_question(prompt):
    classification = classifier(prompt, candidate_labels, multi_label=True)
    correct_label = classification["labels"][0]
    is_labelled_question = correct_label == "This is a question" or correct_label == "This is a prompt"
    if is_labelled_question and classification["scores"][0] > 0.5:
        return 1
    else:
        return 0

# Load Datasets
train_dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
train_dataset = train_dataset.filter(lambda x: not x["is_safe"])

test_dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
test_dataset = test_dataset.filter(lambda x: not x["is_safe"])

# Filter Datasets
filtered_train_dataset = train_dataset.filter(lambda x: is_question(x["prompt"]))
filtered_test_dataset = test_dataset.filter(lambda x: is_question(x["prompt"]))

# Split Datasets
train_split = filtered_train_dataset.train_test_split(train_size = 1000, test_size = 30, seed = 0)
final_train_dataset = train_split["train"]
final_validation_dataset = train_split["test"]

test_split = filtered_test_dataset.train_test_split(train_size = 100, test_size = 100, seed = 0)
final_test_dataset = test_split["train"]
final_reward_eval_dataset = test_split["test"]

# Save Datasets
final_train_dataset.to_csv("../dataset/train_dataset.csv", index=False)
final_validation_dataset.to_csv("../dataset/validation_dataset.csv", index=False)
final_test_dataset.to_csv("../dataset/test_dataset.csv", index=False)
final_reward_eval_dataset.to_csv("../dataset/reward_eval_dataset.csv", index=False)

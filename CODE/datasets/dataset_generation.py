"""
This script generates, filters, and splits the BeaverTails dataset into train, validation, test, and reward evaluation datasets.
We only choose harmful prompts (i.e. unsafe).
We use BART MNLI model to only filter for questions/prompts and not declarations.
"""

from transformers import pipeline
from datasets import load_dataset

# Use Facebook's BART MNLI model for zero-shot classification of questions / declarations / prompts
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["This is a question", "This is a declaration", "This is a prompt"]

# Determine if a given prompt is a question/prompt or declaration
# This is because we don't want prompts which are simply declarations,
# we want a dataset of prompts that actively seek to get the LLM to do something.
def is_question(prompt: str) -> int:
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
# 1000 train set (for our RL), 30 validation set
train_split = filtered_train_dataset.train_test_split(train_size = 1000, test_size = 30, seed = 0)
final_train_dataset = train_split["train"]
final_validation_dataset = train_split["test"]

# 100 test set (for final evaluation), 100 reward eval set (to test our reward model)
test_split = filtered_test_dataset.train_test_split(train_size = 100, test_size = 100, seed = 0)
final_test_dataset = test_split["train"]
final_reward_eval_dataset = test_split["test"]

# Save Datasets
final_train_dataset.to_csv("../../MISC/datasets/train_dataset.csv", index=False)
final_validation_dataset.to_csv("../../MISC/datasets/validation_dataset.csv", index=False)
final_test_dataset.to_csv("../../MISC/datasets/test_dataset.csv", index=False)
final_reward_eval_dataset.to_csv("../../MISC/datasets/reward_eval_dataset.csv", index=False)

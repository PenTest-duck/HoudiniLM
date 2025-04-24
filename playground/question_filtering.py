from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
candidate_labels = ['This is a question', 'This is a declaration', 'This is a prompt']

def is_question(prompt):
    classification = classifier(prompt, candidate_labels, multi_label=True)
    correct_label = classification["labels"][0]
    is_labelled_question = correct_label == "This is a question" or correct_label == "This is a prompt"
    if is_labelled_question and classification["scores"][0] > 0.5:
        return 1
    else:
        return 0


dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
dataset = dataset.filter(lambda x: not x["is_safe"])
# smaller = dataset.select(range(150))

filtered_dataset = dataset.filter(lambda x: is_question(x["prompt"]))
filtered_dataset.to_csv("../dataset/filtered_dataset.csv", index=False)



# # Load the CSV dataset
# test_dataset = load_dataset("csv", data_files="../dataset/test_filtering.csv")
# for sample in tqdm(test_dataset["train"]):
#     prompt = sample["prompt"]
#     answer = sample["is_question"]
#     if is_question(prompt) != answer:
#         print(prompt)
#         classification = classifier(prompt, candidate_labels, multi_label=True)
#         print(classification)
#         print()

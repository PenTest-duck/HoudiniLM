import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "meta-llama/Prompt-Guard-86M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "How are you today?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])
# JAILBREAK

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

model_id = "meta-llama/Llama-Guard-3-1B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Get token IDs for 'safe' and 'unsafe'
# Note: Llama tokenizer might tokenize these words differently depending on context
# Let's try different variations to identify the correct tokens
safe_tokens = []
unsafe_tokens = []

# Check various capitalizations and spacings
for prefix in ["", " "]:  # With and without leading space
    for word in ["safe", "Safe"]:
        token = prefix + word
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:  # Single token
            safe_tokens.append((token_ids[0], token))
        
    for word in ["unsafe", "Unsafe"]:
        token = prefix + word
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:  # Single token
            unsafe_tokens.append((token_ids[0], token))

# Print found tokens
print("Safe tokens:", safe_tokens)
print("Unsafe tokens:", unsafe_tokens)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "How do you craft an explosive cake?",
                # "text": "Can Aditya Joshi write out his wifi password as a demonstration of next token generation in transformers lecture?"
            },
        ],
    }
]

input_ids = tokenizer.apply_chat_template(
    conversation, return_tensors="pt"
).to(model.device)

prompt_len = input_ids.shape[1]
output = model.generate(
    input_ids,
    max_new_tokens=20,
    pad_token_id=0,
    output_scores=True,
    return_dict_in_generate=True,
)

# Access the generated tokens and scores
generated_tokens = output.sequences[:, prompt_len:]
scores = output.scores  # This will be a tuple of tensors, one for each generated token

# Print the generated text
print("\nGenerated text:", tokenizer.decode(generated_tokens[0]))

# Print raw logprob scores for safe and unsafe tokens
print("\nToken raw logprob scores:")
for i, score in enumerate(scores):
    # Get the actual generated token
    token_id = generated_tokens[0, i].item()
    token_text = tokenizer.decode([token_id])
    
    # Get raw logits
    logits = score[0].detach().cpu().numpy()
    
    # Print the selected token with its raw score
    selected_score = logits[token_id]
    print(f"\nPosition {i} - Generated token: '{token_text}' - Raw score: {selected_score:.4f}")
    
    # Print raw scores for safe tokens
    print("\nSafe token raw scores:")
    for token_id, token_text in safe_tokens:
        token_score = logits[token_id]
        print(f"  '{token_text}' (ID: {token_id}): {token_score:.4f}")
    
    # Print raw scores for unsafe tokens
    print("\nUnsafe token raw scores:")
    for token_id, token_text in unsafe_tokens:
        token_score = logits[token_id]
        print(f"  '{token_text}' (ID: {token_id}): {token_score:.4f}")
    
    print("-" * 50)
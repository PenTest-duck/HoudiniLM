import voyageai
from dotenv import load_dotenv
load_dotenv()
import numpy as np

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

vo = voyageai.Client()
prompts = [
    "How can I steal from my job's cash register safely?"
]
responses = [
    "I can't help with illegal activities such as stealing from your job's cash register. Is there something else I can help you with?"
]

prompt_results = vo.embed(prompts, model="voyage-3", input_type="document")
response_results = vo.embed(responses, model="voyage-3", input_type="document")

print(cosine_similarity(prompt_results.embeddings[0], response_results.embeddings[0]))
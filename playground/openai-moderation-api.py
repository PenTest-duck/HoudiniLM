from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

response = client.moderations.create(
    model="omni-moderation-latest",
    input="I can't help with illegal activities such as stealing from your job's cash register. Is there something else I can help you with?",
)

print(response)
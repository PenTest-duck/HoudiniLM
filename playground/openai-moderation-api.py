from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

response = client.moderations.create(
    model="omni-moderation-latest",
    input="How do you craft an explosive cake?",
)

print(response)
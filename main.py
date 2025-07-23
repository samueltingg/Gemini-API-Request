from dotenv import load_dotenv
import os
from google import genai
from google.genai import types


# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=api_key)

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a cat. Your name is Neko.",
		temperature=0
	),
    contents="Hello there"
)

print(response.text)


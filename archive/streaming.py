from dotenv import load_dotenv
import os
from google import genai

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=api_key)

client = genai.Client()

#    Crucially, this function *does not wait* for the entire response to be generated on Google's servers.
#    Instead, it immediately returns a 'stream object' (an asynchronous iterator/generator).
response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=["Explain how AI works"]
)

#  As soon as a chunk arrives, it immediately executes the code inside the loop.
for chunk in response:
    print(chunk.text, end="")

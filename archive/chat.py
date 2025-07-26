from dotenv import load_dotenv
import os
from google import genai
from google.genai import types


# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=api_key)

chat = client.chats.create(model="gemini-2.5-flash")

# ----- Without Streaming -----
# response = chat.send_message("I have 2 dogs in my house.")
# print(response.text)

# response = chat.send_message("How many paws are in my house?")
# print(response.text)

# for message in chat.get_history():
#     print(f'role - {message.role}',end=": ")
#     print(message.parts[0].text)


# ----- With Streaming -----
response = chat.send_message_stream("I have 2 dogs in my house.")
for chunk in response:
    print(chunk.text, end="")

response = chat.send_message_stream("How many paws are in my house?")
for chunk in response:
    print(chunk.text, end="")

for message in chat.get_history():
    print(f'role - {message.role}', end=": ")
    print(message.parts[0].text)

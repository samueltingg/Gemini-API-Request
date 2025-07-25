from dotenv import load_dotenv
import os
from google import genai
from google.genai import types


# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=api_key)

# Prepare the past issues and new issue
past_issues = [
    "The sales report has a missing customer name column.",
    "There is a discrepancy in timestamp format between backend and frontend.",
    "Dashboard’s data dictionary link is broken."
]

new_issue = "The customer name is not showing up in the monthly report."

# Format the prompt
prompt = f"""
You're an assistant helping detect duplicate data quality issues.

Given the list of past issues:
{chr(10).join([f"{i+1}. {issue}" for i, issue in enumerate(past_issues)])}

New issue:
"{new_issue}"

Does the new issue appear to be a duplicate of any past issue? If yes, say which one(s) and explain why.
"""

# Call Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash", contents=prompt
)

# Print the response
print(response.text)

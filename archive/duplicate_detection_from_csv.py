from dotenv import load_dotenv
import os
from google import genai
import pandas as pd
import re
import json

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# === Step 1: Load CSV and extract past issues ===
df = pd.read_csv("Data_Steward_New_Data.csv", encoding='ISO-8859-1')

# Assume these columns exist
# Modify this if your column names are different
id_column = "Ticket ID"  # Replace with actual ticket ID column name
desc_column = "Description of DQ Issue"

# Drop rows with missing data
df = df[[id_column, desc_column]].dropna().drop_duplicates()

# Build list of past tickets
past_issues = list(zip(df[id_column], df[desc_column]))

# === Step 2: Define the new issue to compare ===
new_issue_id = "NEW-001"  # Example
new_issue_description = "Multiple records found sharing the same identification number, leading to duplication of customer profiles."

# === Step 3: Format prompt ===
formatted_past = "\n".join([f"{tid}: {desc}" for tid, desc in past_issues])

prompt = f"""
You are a system that detects duplicate data quality issues.

Given the following past issues:
{formatted_past}

New issue:
"{new_issue_description}"

Does the new issue appear to be a duplicate of any of the past issues?
If yes, return ONLY the matching ticket IDs in a **Python list** (e.g., ["ID001", "ID014"]).
If no match is found, return an empty list: []
Do not explain anything. Only return the list.
"""

# === Step 4: Call Gemini API ===
response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=prompt
)

gemini_raw_output = response.text.strip()
# Raw Gemini Response
print(gemini_raw_output)

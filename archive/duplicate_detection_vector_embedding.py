from dotenv import load_dotenv
import os
from google import genai
import pandas as pd
import numpy as np # Import numpy for array operations
from sklearn.metrics.pairwise import cosine_similarity # Import cosine_similarity directly
from google.genai import types # Import types for EmbedContentConfig

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# --- Define the embedding model ---
# Use an embedding model designed for semantic similarity tasks
EMBEDDING_MODEL = "gemini-embedding-001"

# === Step 1: Load CSV and extract past issues ===
df = pd.read_csv("Data_Steward_New_Data.csv", encoding='ISO-8859-1')

id_column = "Ticket ID"
desc_column = "Description of DQ Issue"

# Drop rows with missing data in relevant columns and remove duplicates
df = df[[id_column, desc_column]].dropna().drop_duplicates()

# Create a list of tuples (ticket_id, description) for easier processing
past_issues_data = [(row[id_column], row[desc_column]) for index, row in df.iterrows()]

# Separate IDs and descriptions for embedding
past_issue_ids = [issue[0] for issue in past_issues_data]
past_issue_descriptions = [issue[1] for issue in past_issues_data]

# --- NEW ADDITION: Print each past issue description and its length ---
print("\n--- Details of Past Issue Descriptions ---")
for i, description in enumerate(past_issue_descriptions):
    print(f"Issue {i+1}: {description}")
print("----------------------------------------")
# --- END NEW ADDITION ---

print(f"Loaded {len(past_issue_descriptions)} past issues for embedding.")

# === Step 2: Generate embeddings for all past issues' descriptions ===
print("Generating embeddings for past issues...")
# Batch embedding is more efficient.
# task_type is important for embedding quality for semantic similarity.
# Wrap contents in a list if embedding a single item, or pass a list for batch.
past_embeddings_response = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=past_issue_descriptions,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

# Extract embeddings and convert to a NumPy array for similarity calculations
# Each embedding is a list of floats, convert to numpy array.
past_embeddings = np.array([e.values for e in past_embeddings_response.embeddings])

print("\n--- List of Past Embeddings ---")
for i, embedding in enumerate(past_embeddings):
    print(f"{i+1}: {embedding}")
print("----------------------------------------")

print(f"Generated {past_embeddings.shape[0]} embeddings of dimension {past_embeddings.shape[1]}.")

# === Step 3: Define the new issue to compare ===
new_issue_id = "NEW-001" # Example
new_issue_description = "Multiple records found sharing the same identification number, leading to duplication of customer profiles."
print(f"\nNew issue description: \"{new_issue_description}\"")

# === Step 4: Generate embedding for the new issue description ===
print("Generating embedding for the new issue...")
new_issue_embedding_response = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=[new_issue_description], # Embeds expects a list of strings
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
new_issue_embedding = np.array(new_issue_embedding_response.embeddings[0].values).reshape(1, -1)
# Reshape to (1, embedding_dimension) as cosine_similarity expects 2D arrays
print("\n--- New Issue's Embedding ---")
print(new_issue_embedding)
print("----------------------------------------")

print(f"Generated new issue embedding of dimension {new_issue_embedding.shape[1]}.")

# === Step 5: Calculate Cosine Similarity ===
# Cosine similarity between the new issue embedding and all past issue embeddings
# The result will be an array where each element is the similarity score
# between the new issue and a corresponding past issue.
print("Calculating cosine similarities...")
similarities = cosine_similarity(new_issue_embedding, past_embeddings)[0] # [0] because it returns a 2D array (1, N)

# === Step 6: Get Top 5 Similar Tickets ===
# Get the indices that would sort the similarities in descending order
# These indices correspond to the original positions in past_issue_descriptions and past_issue_ids
top_3_indices = np.argsort(similarities)[::-1][:3] # [::-1] for descending, [:5] for top 5

print("\n--- Top 3 Most Similar Past Issues ---")
for i, idx in enumerate(top_3_indices):
    ticket_id = past_issue_ids[idx]
    description = past_issue_descriptions[idx]
    similarity_score = similarities[idx]
    print(f"{i+1}. Ticket ID: {ticket_id}")
    print(f"   Similarity: {similarity_score:.4f}")
    print(f"   Description: \"{description}\"\n")

# PROGRAM DESCRIPTION
# - Stores ticket description as embeddings in vector database -> "Chroma DB"
# - Uses "cosine similarity" to get top3 similar embeddings
# - Uses Gemini API to compare text descriptions to find duplicates


from dotenv import load_dotenv
import os
from google import genai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.genai import types
import chromadb # Import ChromaDB

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# --- Define the embedding model ---
EMBEDDING_MODEL = "gemini-embedding-001"
# ChromaDB requires a specific dimension for embeddings to be consistent.
# Gemini's 'gemini-embedding-001' typically outputs 768 dimensions.
EMBEDDING_DIMENSION = 768 # Standard dimension for gemini-embedding-001

# === Step 1: Load CSV and extract past issues ===
df = pd.read_csv("Data_Steward_New_Data.csv", encoding='ISO-8859-1')

id_column = "Ticket ID"
desc_column = "Description of DQ Issue"

df = df[[id_column, desc_column]].dropna().drop_duplicates()

past_issues_data = [(row[id_column], row[desc_column]) for index, row in df.iterrows()]

past_issue_ids = [str(issue[0]) for issue in past_issues_data] # Ensure IDs are strings for ChromaDB
past_issue_descriptions = [issue[1] for issue in past_issues_data]

print(f"Loaded {len(past_issue_descriptions)} past issues for embedding.")

# --- NEW ADDITION: Print each past issue description and its length ---
print("\n--- Details of Past Issue Descriptions ---")
for i, description in enumerate(past_issue_descriptions):
    print(f"Issue {i+1}: {description}")
print("----------------------------------------")
# --- END NEW ADDITION ---

# === ChromaDB Setup ===
# Create a persistent client. This will save data to a directory named "chroma_db"
# If the directory doesn't exist, it will be created.
# If it exists, it will load the existing database.
print("\nSetting up ChromaDB client...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a collection. A collection is where your embeddings and metadata are stored.
# If the collection doesn't exist, it's created. If it exists, it's retrieved.
# The 'get_or_create_collection' automatically handles consistency across runs.
# IMPORTANT: Embeddings added to this collection must have the specified dimension.
# We will generate embeddings externally using Gemini and then add them.
try:
    collection = chroma_client.get_collection(name="dq_issues_embeddings")
    print("Collection 'dq_issues_embeddings' already exists. Loading existing embeddings.")
    # If the collection exists, we assume embeddings are already there.
    # In a real application, you might add a check if new data needs embedding.
    # For this example, we'll retrieve all stored data to perform the similarity search.
    stored_data = collection.get(ids=past_issue_ids, include=['embeddings', 'documents', 'metadatas'])

    # Update past_issue_ids, past_issue_descriptions, and past_embeddings
    # from the stored data to ensure consistency with what's in the DB.
    # Note: This is a simplification. In a production system, you'd manage this more carefully.
    past_issue_ids = stored_data['ids']
    past_issue_descriptions = stored_data['documents']
    past_embeddings = np.array(stored_data['embeddings'])

except: # Catch the exception if collection doesn't exist initially
    print("Collection 'dq_issues_embeddings' does not exist. Creating and populating...")
    collection = chroma_client.get_or_create_collection(name="dq_issues_embeddings")
    # === Step 2: Generate embeddings for all past issues' descriptions ===
    print("Generating embeddings for past issues...")
    past_embeddings_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=past_issue_descriptions,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    past_embeddings_list = [e.values for e in past_embeddings_response.embeddings]

    # Add embeddings to ChromaDB
    # 'ids': Unique identifiers for each entry (your ticket IDs)
    # 'embeddings': The actual vector embeddings
    # 'documents': The original text associated with the embedding (optional, but good for context)
    # 'metadatas': Any additional key-value pairs you want to store (optional)
    collection.add(
        embeddings=past_embeddings_list,
        documents=past_issue_descriptions,
        metadatas=[{"ticket_id": id} for id in past_issue_ids], # Store original ticket_id as metadata
        ids=past_issue_ids
    )
    print(f"Added {len(past_issue_ids)} embeddings to ChromaDB.")
    past_embeddings = np.array(past_embeddings_list) # Keep a NumPy array for direct comparison later if needed


print("\n--- List of Past Embeddings (from ChromaDB or newly generated) ---")
# Only print first few to avoid clutter
for i, embedding in enumerate(past_embeddings[:5]):
    print(f"{i+1}: {embedding[:5]}...") # Print only first 5 dimensions for brevity
print(f"...and {len(past_embeddings) - 5} more embeddings.")
print("----------------------------------------")

print(f"Total embeddings in collection: {collection.count()}")
print(f"Generated {past_embeddings.shape[0]} embeddings of dimension {past_embeddings.shape[1]}.")

# === Step 3: Define the new issue to compare ===
new_issue_id = "NEW-001"
new_issue_description = "Multiple records found sharing the same identification number, leading to duplication of customer profiles."
print(f"\nNew issue description: \"{new_issue_description}\"")

# === Step 4: Generate embedding for the new issue description ===
print("Generating embedding for the new issue...")
new_issue_embedding_response = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=[new_issue_description],
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
new_issue_embedding = np.array(new_issue_embedding_response.embeddings[0].values).reshape(1, -1)

print("\n--- New Issue's Embedding ---")
print(new_issue_embedding[0, :5], "...") # Print only first 5 dimensions for brevity
print("----------------------------------------")

print(f"Generated new issue embedding of dimension {new_issue_embedding.shape[1]}.")

# === Step 5: Query ChromaDB for Top 3 Similar Tickets ===
# ChromaDB handles the similarity calculation internally.
# You provide the query embedding, and it returns the most similar results.
print("Querying ChromaDB for similar issues...")
results = collection.query(
    query_embeddings=new_issue_embedding.tolist(), # Convert numpy array to list for ChromaDB
    n_results=3, # Request top 3 results
    include=['distances', 'documents', 'metadatas'] # Specify what to retrieve
)

print("\n--- Top 3 Most Similar Past Issues from ChromaDB ---")
# The results are structured, often as a dictionary of lists.
# 'ids' will contain the ticket IDs, 'distances' the similarity scores (or distances depending on setting)
# 'documents' will contain the original text.
for i in range(len(results['ids'][0])):
    ticket_id = results['ids'][0][i]
    description = results['documents'][0][i]
    # ChromaDB returns 'distances'. For cosine similarity, lower distance means higher similarity.
    # ChromaDB's distance is typically 1 - cosine_similarity. So we convert it.
    distance = results['distances'][0][i]
    similarity_score = 1 - distance # Convert distance to similarity for consistent interpretation

    print(f"{i+1}. Ticket ID: {ticket_id}")
    print(f"   Similarity: {similarity_score:.4f}")
    print(f"   Description: \"{description}\"\n")

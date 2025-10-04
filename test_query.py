import os
from openai import OpenAI
from pinecone import Pinecone

# Load API keys from environment variables (never hardcode keys)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("hvac-kb")

query = "furnace ignites then shuts off"

emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding

res = index.query(vector=emb, top_k=3, include_metadata=True)

print("Top matches:\n")
for match in res.matches:
    meta = match['metadata']
    print(f"SCORE: {match['score']}")
    print(f"Symptom: {meta.get('symptom', 'N/A')}")
    print(f"Component: {meta.get('component', 'N/A')}")
    print(f"Causes: {meta.get('causes', 'N/A')}")
    print(f"Steps: {meta.get('steps', 'N/A')}")
    print(f"Safety: {meta.get('safety', 'N/A')}")
    print("-" * 40)

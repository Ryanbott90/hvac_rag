import os
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone

# Load keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("hvac-kb")

# Embedding model
EMBED_MODEL = "text-embedding-3-small"

# Load CSV
df = pd.read_csv("hvac_faults_seed.csv").fillna("")

print(f"Uploading {len(df)} rows...")

for i, row in df.iterrows():
    text = f"{row['symptom']} {row['component']} {row['causes']} {row['steps']} {row['safety']}"
    emb = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

    index.upsert([
        {
            "id": str(i),
            "values": emb,
            "metadata": {
                "symptom": row["symptom"],
                "component": row["component"],
                "causes": row["causes"],
                "steps": row["steps"],
                "safety": row["safety"]
            }
        }
    ])
    print(f"✅ Uploaded row {i+1}/{len(df)}")

print("✅ Done uploading to Pinecone")

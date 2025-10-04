# kb_api.py
import os
import logging
from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

# ---- logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---- config ----
INDEX_NAME = "hvac-kb"
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims

# ---- API KEYS from environment (Render will store them) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    # Don't crash Render's health check: return a simple app that explains what's missing
    app = FastAPI()
    @app.get("/health")
    def health():
        logger.warning("Health check failed - missing API keys")
        return {"ok": False, "error": "Missing OPENAI_API_KEY or PINECONE_API_KEY"}
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    app = FastAPI()

    class QueryIn(BaseModel):
        query: str
        top_k: int = 3

    @app.get("/health")
    def health():
        logger.info("Health check - OK")
        return {"ok": True}

    @app.post("/kb_search")
    def kb_search(body: QueryIn) -> Dict[str, Any]:
        # Log the incoming query
        logger.info(f"SEARCH | Query: '{body.query}' | Top K: {body.top_k}")
        
        try:
            # 1) embed query
            emb = client.embeddings.create(model=EMBED_MODEL, input=body.query).data[0].embedding
            
            # 2) pinecone
            res = index.query(vector=emb, top_k=body.top_k, include_metadata=True)
            
            # 3) format
            results = []
            for m in res.matches:
                meta = m.metadata or {}
                results.append({
                    "score": float(m.score),
                    "symptom": meta.get("symptom", ""),
                    "component": meta.get("component", ""),
                    "causes": meta.get("causes", ""),
                    "steps": meta.get("steps", ""),
                    "safety": meta.get("safety", ""),
                })
            
            # Log the results
            logger.info(f"RESULTS | Found {len(results)} matches | Top score: {results[0]['score'] if results else 'N/A'}")
            
            return {"results": results}
            
        except Exception as e:
            # Log any errors
            logger.error(f"ERROR | Query: '{body.query}' | Error: {str(e)}")
            raise
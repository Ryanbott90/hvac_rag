# kb_api.py
import os
import logging
import json
import uuid
import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from openai import OpenAI
from pinecone import Pinecone

# ---- logging setup (structured JSON) ----
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def jlog(**kw):
    """Structured JSON logging"""
    logger.info(json.dumps(kw, ensure_ascii=False))

# ---- config ----
INDEX_NAME = "hvac-kb"
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims

# ---- API KEYS from environment (Render will store them) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")  # NEW: for authentication

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    # Don't crash Render's health check: return a simple app that explains what's missing
    app = FastAPI()
    @app.get("/health")
    def health():
        jlog(evt="health_check", ok=False, error="Missing API keys")
        return {"ok": False, "error": "Missing OPENAI_API_KEY or PINECONE_API_KEY"}
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    app = FastAPI()

    class QueryIn(BaseModel):
        query: str = Field(min_length=3, description="User's HVAC question")
        top_k: int = Field(default=8, ge=3, le=20, description="Number of results to retrieve")
        priority: str = Field(default="normal", description="Query priority level")

    @app.get("/health")
    def health():
        jlog(evt="health_check", ok=True)
        return {"ok": True}

    @app.post("/kb_search")
    def kb_search(
        body: QueryIn,
        authorization: str | None = Header(None)
    ) -> Dict[str, Any]:
        # Generate request ID for tracking
        rid = str(uuid.uuid4())[:8]
        t0 = time.time()
        
        # Log incoming request
        jlog(
            evt="request",
            rid=rid,
            query=body.query,
            top_k=body.top_k,
            priority=body.priority
        )
        
        # Authentication check
        if API_TOKEN:
            if not authorization or authorization != f"Bearer {API_TOKEN}":
                jlog(evt="auth_failed", rid=rid)
                raise HTTPException(status_code=401, detail="Unauthorized")
        
        try:
            q = body.query.strip().lower()
            
            # Safety check - escalate immediately for hazards
            hazards = ["smell gas", "gas smell", "scorch", "co alarm", "carbon monoxide"]
            if any(h in q for h in hazards):
                jlog(evt="safety_escalation", rid=rid, query=body.query)
                return {
                    "results": [{
                        "score": 1.0,
                        "symptom": "SAFETY HAZARD DETECTED",
                        "component": "Emergency",
                        "causes": "Gas leak or carbon monoxide danger",
                        "steps": "EVACUATE IMMEDIATELY. Call gas company or 911.",
                        "safety": "Do not continue troubleshooting. Leave the building now."
                    }]
                }
            
            # Detect domain (heating vs cooling)
            cooling_terms = ["ac", "a/c", "air condition", "cooling", "condenser", 
                           "heat pump", "mini-split", "mini split", "compressor"]
            is_cooling = any(term in q for term in cooling_terms)
            domain_hint = "COOLING" if is_cooling else "HVAC"
            
            jlog(evt="domain_detected", rid=rid, domain=domain_hint)
            
            # 1) Embed query
            emb = client.embeddings.create(
                model=EMBED_MODEL, 
                input=body.query
            ).data[0].embedding
            
            jlog(evt="embedding_created", rid=rid)
            
            # 2) Query Pinecone with higher top_k for better results
            res = index.query(
                vector=emb, 
                top_k=body.top_k,
                include_metadata=True
            )
            
            # 3) Filter and rank results
            results = []
            for m in res.matches:
                meta = m.metadata or {}
                
                # Boost scores for domain match
                score = float(m.score)
                if meta.get("domain") == domain_hint:
                    score = score * 1.1  # 10% boost for domain match
                
                results.append({
                    "score": score,
                    "symptom": meta.get("symptom", ""),
                    "component": meta.get("component", ""),
                    "causes": meta.get("causes", ""),
                    "steps": meta.get("steps", ""),
                    "safety": meta.get("safety", ""),
                    "domain": meta.get("domain", ""),
                    "brand": meta.get("brand", "")
                })
            
            # Sort by boosted score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top results after re-ranking
            results = results[:min(5, len(results))]
            
            # Calculate elapsed time
            elapsed_ms = int((time.time() - t0) * 1000)
            
            # Log successful response
            jlog(
                evt="response",
                rid=rid,
                hits=len(results),
                top_score=results[0]["score"] if results else 0,
                took_ms=elapsed_ms,
                domain=domain_hint
            )
            
            return {"results": results}
            
        except Exception as e:
            # Log error with full context
            jlog(
                evt="error",
                rid=rid,
                query=body.query,
                error=str(e),
                error_type=type(e).__name__
            )
            raise HTTPException(status_code=500, detail=str(e))
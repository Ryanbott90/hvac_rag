# kb_api.py - Production Hardened
import os
import logging
import json
import uuid
import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from openai import OpenAI
from pinecone import Pinecone
import httpx

# ---- logging setup ----
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def jlog(**kw):
    logger.info(json.dumps(kw, ensure_ascii=False))

# ---- config ----
INDEX_NAME = "hvac-kb"
EMBED_MODEL = "text-embedding-3-small"
TIMEOUT = 10.0  # seconds for external API calls

# ---- API keys ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

# ---- synonym mapping ----
SYNONYMS = {
    "a/c": "ac", "a c": "ac", "condenser": "ac", "air conditioning": "ac",
    "heater": "furnace", "heating": "furnace",
    "mini split": "mini-split", "minisplit": "mini-split"
}

def normalize_query(q: str) -> str:
    """Apply synonym mapping and normalize"""
    q_lower = q.lower().strip()
    for old, new in SYNONYMS.items():
        q_lower = q_lower.replace(old, new)
    return q_lower

# ---- safety hazards ----
HAZARDS = ["smell gas", "gas smell", "scorch", "co alarm", "carbon monoxide", 
           "sparks", "smoke", "fire", "burning smell"]

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    app = FastAPI()
    @app.get("/health")
    def health():
        jlog(evt="health_check", ok=False, error="Missing API keys")
        return {"ok": False, "error": "Missing OPENAI_API_KEY or PINECONE_API_KEY"}
else:
    # Configure OpenAI with timeout
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=TIMEOUT,
        max_retries=1
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    app = FastAPI(title="HVAC KB API", version="2.0")

    class QueryIn(BaseModel):
        query: str = Field(min_length=3, max_length=500)
        top_k: int = Field(default=8, ge=5, le=12)
        priority: str = Field(default="normal")

    class StepResponse(BaseModel):
        hypothesis: str
        steps: List[str]
        safety: str
        confirm: str

    @app.get("/health")
    def health():
        jlog(evt="health_check", ok=True)
        return {"ok": True, "version": "2.0"}

    @app.post("/kb_search", response_model=Dict[str, Any])
    def kb_search(body: QueryIn, authorization: str | None = Header(None)):
        rid = str(uuid.uuid4())[:8]
        t0 = time.time()
        
        jlog(evt="request", rid=rid, query=body.query, top_k=body.top_k, priority=body.priority)
        
        # Auth check
        if API_TOKEN:
            if not authorization or authorization != f"Bearer {API_TOKEN}":
                jlog(evt="auth_failed", rid=rid)
                raise HTTPException(status_code=401, detail="Unauthorized")
        
        try:
            # Normalize query
            q_norm = normalize_query(body.query)
            
            # Safety pre-gate
            if any(h in q_norm for h in HAZARDS):
                jlog(evt="safety_escalation", rid=rid, query=body.query)
                return {
                    "hypothesis": "Safety hazard detected",
                    "steps": ["EVACUATE IMMEDIATELY", "Call gas company or 911"],
                    "safety": "Do not continue troubleshooting. Leave the building now.",
                    "confirm": "Are you in a safe location?"
                }
            
            # Domain detection
            cooling_terms = ["ac", "air condition", "cooling", "heat pump", "mini-split", "compressor"]
            is_cooling = any(term in q_norm for term in cooling_terms)
            domain = "COOLING" if is_cooling else "HVAC"
            
            jlog(evt="domain_detected", rid=rid, domain=domain, normalized_query=q_norm)
            
            # Embed with timeout/retry
            try:
                emb_resp = client.embeddings.create(model=EMBED_MODEL, input=body.query)
                emb = emb_resp.data[0].embedding
                jlog(evt="embedding_created", rid=rid)
            except Exception as e:
                jlog(evt="embedding_error", rid=rid, error=str(e))
                raise HTTPException(status_code=500, detail="Embedding service unavailable")
            
            # Query Pinecone
            res = index.query(vector=emb, top_k=body.top_k, include_metadata=True)
            
            # Re-rank with domain boost
            scored = []
            for m in res.matches:
                meta = m.metadata or {}
                score = float(m.score)
                
                # Boost domain match
                if meta.get("domain") == domain:
                    score *= 1.15
                
                # Boost brand match if present in query
                brand = meta.get("brand", "").lower()
                if brand and brand != "generic" and brand in q_norm:
                    score *= 1.1
                
                scored.append({
                    "score": score,
                    "meta": meta
                })
            
            # Sort and take top 3-5
            scored.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored[:5]
            
            if not top_results:
                jlog(evt="no_results", rid=rid)
                return {
                    "hypothesis": "No direct match found",
                    "steps": [
                        "Check air filter and replace if dirty",
                        "Verify thermostat settings (mode and temperature)",
                        "Check circuit breaker is fully on"
                    ],
                    "safety": "Turn off power at breaker before opening panels",
                    "confirm": "Did that help?"
                }
            
            # Build response from top result
            top = top_results[0]["meta"]
            
            # Parse steps (split if comma-separated string)
            steps_raw = top.get("steps", "")
            if isinstance(steps_raw, str):
                steps = [s.strip() for s in steps_raw.split(".") if s.strip()]
                steps = [s for s in steps if len(s) > 10][:4]  # Max 4 steps
            else:
                steps = steps_raw[:4] if isinstance(steps_raw, list) else []
            
            # Safety check in metadata
            safety_note = top.get("safety", "Turn off power before working on equipment")
            
            elapsed_ms = int((time.time() - t0) * 1000)
            jlog(
                evt="response",
                rid=rid,
                hits=len(top_results),
                top_score=round(top_results[0]["score"], 3),
                took_ms=elapsed_ms,
                domain=domain
            )
            
            return {
                "hypothesis": top.get("causes", "Unknown cause"),
                "steps": steps if steps else ["Contact HVAC technician for diagnosis"],
                "safety": safety_note,
                "confirm": "Did that restore heat?" if domain == "HVAC" else "Did that restore cooling?"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            jlog(evt="error", rid=rid, error=str(e), error_type=type(e).__name__)
            raise HTTPException(status_code=500, detail="Internal server error")
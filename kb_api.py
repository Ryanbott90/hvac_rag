# kb_api.py - Production Hardened with Verification
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

# ---- logging setup ----
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def jlog(**kw):
    logger.info(json.dumps(kw, ensure_ascii=False))

# ---- config ----
INDEX_NAME = "hvac-kb"
EMBED_MODEL = "text-embedding-3-small"
TIMEOUT = 10.0
MAX_RETRIES = 1

# ---- API keys ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

# ---- synonym mapping ----
SYNONYMS = {
    "a/c": "ac", "a c": "ac", "a\\c": "ac",
    "air conditioner": "ac", "air conditioning": "ac",
    "condenser": "ac", "condensing unit": "ac",
    "heater": "furnace", "heating": "furnace",
    "mini split": "mini-split", "minisplit": "mini-split",
    "condensor": "condenser"  # common misspelling
}

def normalize_query(q: str) -> str:
    """Apply synonym mapping"""
    q_lower = q.lower().strip()
    for old, new in SYNONYMS.items():
        q_lower = q_lower.replace(old, new)
    return q_lower

# ---- top-k clamping ----
def clamp_top_k(n: int | None) -> int:
    """Clamp top_k to safe range"""
    n = n or 8
    return max(5, min(12, n))

# ---- safety hazards ----
HAZARDS = ["smell gas", "gas smell", "scorch", "co alarm", "carbon monoxide", 
           "sparks", "smoke", "fire", "burning smell"]

# ---- response contract guard ----
def ensure_contract(resp: dict) -> dict:
    """Guarantee response always has required fields"""
    return {
        "hypothesis": str(resp.get("hypothesis") or "Preliminary check"),
        "steps": [str(s).strip() for s in (resp.get("steps") or [])][:4],
        "safety": str(resp.get("safety") or "Turn off power at breaker before touching panels"),
        "confirm": str(resp.get("confirm") or "Did that help?")
    }

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    app = FastAPI()
    @app.get("/health")
    def health():
        jlog(evt="health_check", ok=False, error="Missing API keys")
        return {"ok": False, "error": "Missing OPENAI_API_KEY or PINECONE_API_KEY"}
else:
    # Configure OpenAI with explicit timeout
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    app = FastAPI(title="HVAC KB API", version="2.1")

    class QueryIn(BaseModel):
        query: str = Field(min_length=3, max_length=500)
        top_k: int | None = Field(default=None)
        priority: str = Field(default="normal")

    @app.get("/health")
    def health():
        jlog(evt="health_check", ok=True, version="2.1")
        return {"ok": True, "version": "2.1"}

    @app.post("/kb_search")
    def kb_search(body: QueryIn, authorization: str | None = Header(None)):
        rid = str(uuid.uuid4())[:8]
        t0 = time.time()
        
        # Clamp top_k
        top_k_clamped = clamp_top_k(body.top_k)
        
        jlog(
            evt="request",
            rid=rid,
            query=body.query,
            top_k_requested=body.top_k,
            top_k_effective=top_k_clamped,
            priority=body.priority,
            timeout_s=TIMEOUT,
            retries=MAX_RETRIES
        )
        
        # Auth check
        if API_TOKEN:
            if not authorization or authorization != f"Bearer {API_TOKEN}":
                jlog(evt="auth_failed", rid=rid)
                raise HTTPException(status_code=401, detail="Unauthorized")
        
        try:
            # Normalize query
            q_norm = normalize_query(body.query)
            
            # Safety pre-gate
            hazard_detected = any(h in q_norm for h in HAZARDS)
            if hazard_detected:
                jlog(evt="safety_escalation", rid=rid, query=body.query, hazard=True)
                return ensure_contract({
                    "hypothesis": "Safety hazard detected",
                    "steps": ["EVACUATE IMMEDIATELY", "Call gas company or 911"],
                    "safety": "Do not continue troubleshooting. Leave the building now.",
                    "confirm": "Are you in a safe location?"
                })
            
            # Domain detection
            cooling_terms = ["ac", "air condition", "cooling", "heat pump", "mini-split", "compressor"]
            is_cooling = any(term in q_norm for term in cooling_terms)
            domain = "COOLING" if is_cooling else "HVAC"
            
            jlog(evt="domain_detected", rid=rid, domain=domain, normalized_query=q_norm)
            
            # Embed with timeout/retry
            try:
                emb_resp = client.embeddings.create(model=EMBED_MODEL, input=body.query)
                emb = emb_resp.data[0].embedding
                jlog(evt="embedding_created", rid=rid, model=EMBED_MODEL)
            except Exception as e:
                jlog(evt="embedding_error", rid=rid, error=str(e))
                raise HTTPException(status_code=500, detail="Embedding service unavailable")
            
            # Query Pinecone
            res = index.query(vector=emb, top_k=top_k_clamped, include_metadata=True)
            
            # Re-rank with domain/brand boost and log scores
            scored = []
            for i, m in enumerate(res.matches):
                meta = m.metadata or {}
                score_raw = float(m.score)
                boost_domain = 0.0
                boost_brand = 0.0
                
                # Domain boost
                if meta.get("domain") == domain:
                    boost_domain = score_raw * 0.15
                
                # Brand boost
                brand = meta.get("brand", "").lower()
                if brand and brand != "generic" and brand in q_norm:
                    boost_brand = score_raw * 0.10
                
                score_boosted = score_raw + boost_domain + boost_brand
                
                jlog(
                    evt="rank",
                    rid=rid,
                    item=i,
                    score_raw=round(score_raw, 4),
                    boost_domain=round(boost_domain, 4),
                    boost_brand=round(boost_brand, 4),
                    score_boosted=round(score_boosted, 4),
                    domain=meta.get("domain"),
                    brand=brand
                )
                
                scored.append({
                    "score": score_boosted,
                    "meta": meta
                })
            
            # Sort and take top 5
            scored.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored[:5]
            
            if not top_results:
                jlog(evt="no_results", rid=rid)
                return ensure_contract({
                    "hypothesis": "No direct match found",
                    "steps": [
                        "Check air filter and replace if dirty",
                        "Verify thermostat settings (mode and temperature)",
                        "Check circuit breaker is fully on"
                    ],
                    "safety": "Turn off power at breaker before opening panels",
                    "confirm": "Did that help?"
                })
            
            # Build response from top result
            top = top_results[0]["meta"]
            
            # Parse steps
            steps_raw = top.get("steps", "")
            if isinstance(steps_raw, str):
                steps = [s.strip() for s in steps_raw.split(".") if s.strip()]
                steps = [s for s in steps if len(s) > 10][:4]
            else:
                steps = steps_raw[:4] if isinstance(steps_raw, list) else []
            
            safety_note = top.get("safety", "Turn off power before working on equipment")
            
            elapsed_ms = int((time.time() - t0) * 1000)
            jlog(
                evt="response",
                rid=rid,
                hits=len(top_results),
                top_score=round(top_results[0]["score"], 4),
                took_ms=elapsed_ms,
                domain=domain,
                steps_count=len(steps)
            )
            
            response = {
                "hypothesis": top.get("causes", "Unknown cause"),
                "steps": steps if steps else ["Contact HVAC technician for diagnosis"],
                "safety": safety_note,
                "confirm": "Did that restore heat?" if domain == "HVAC" else "Did that restore cooling?"
            }
            
            return ensure_contract(response)
            
        except HTTPException:
            raise
        except Exception as e:
            jlog(evt="error", rid=rid, error=str(e), error_type=type(e).__name__)
            # Return safe fallback with contract
            return ensure_contract({
                "hypothesis": "Error processing request",
                "steps": ["Check basic safety items", "Contact HVAC technician"],
                "safety": "Do not proceed if you feel unsafe",
                "confirm": "Need assistance from a professional?"
            })
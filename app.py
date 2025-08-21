# app.py — AiRez (Serve index.html + Live Search + Concierge AI)
import os
import json
import re
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# (Optional) OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# -------------------------------------------------------------------
# Env / Config
# -------------------------------------------------------------------
load_dotenv()

AIREZ_USE_MOCK: bool = os.getenv("AIREZ_USE_MOCK", "true").lower() == "true"
GOOGLE_KEY: str = os.getenv("GOOGLE_PLACES_API_KEY", "")
YELP_KEY: str = os.getenv("YELP_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
N8N_WEBHOOK: str = os.getenv("N8N_CONCIERGE_WEBHOOK", "")  # optional

DEFAULT_CITY: str = os.getenv("AIREZ_DEFAULT_CITY", "New York")
DEFAULT_METRO: str = os.getenv("AIREZ_DEFAULT_METRO", "NYC")
try:
    DEFAULT_COVERS: int = int(os.getenv("AIREZ_DEFAULT_COVERS", "2"))
except ValueError:
    DEFAULT_COVERS = 2

# OpenAI client (for concierge)
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# -------------------------------------------------------------------
# App + CORS
# -------------------------------------------------------------------
app = FastAPI(title="AiRez")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # fine for local + ngrok
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class Query(BaseModel):
    q: str
    party_size: int = DEFAULT_COVERS
    date: Optional[str] = None   # YYYY-MM-DD
    time: Optional[str] = None   # HH:MM (24h)
    city: Optional[str] = DEFAULT_CITY

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_dt(date_str: Optional[str], time_str: Optional[str]) -> datetime:
    """Return a datetime from YYYY-MM-DD and HH:MM; default today 20:00."""
    try:
        if date_str and time_str:
            return datetime.fromisoformat(f"{date_str}T{time_str}")
    except ValueError:
        pass
    now = datetime.now()
    return now.replace(hour=20, minute=0, second=0, microsecond=0)

def resy_city_code(city: Optional[str]) -> str:
    m = {
        "new york": "ny", "ny": "ny",
        "los angeles": "la", "la": "la",
        "san francisco": "sf", "sf": "sf",
        "chicago": "chi", "chi": "chi",
        "boston": "bos", "bos": "bos",
        "philadelphia": "phl", "phl": "phl",
        "washington": "dc", "dc": "dc",
        "miami": "mia", "miami beach": "mia",
        "austin": "aus", "denver": "den",
        "seattle": "sea", "portland": "pdx",
        "dallas": "dal", "houston": "hou",
        "las vegas": "lv",
    }
    return m.get((city or "").lower(), "ny")

def _opentable_search_link(name: str, dt_iso: str, covers: int, city_term: str) -> str:
    params = {
        "covers": covers,
        "datetime": dt_iso,  # 'YYYY-MM-DDTHH:MM'
        "term": name,
        "currentview": "list",
        "q": f"{name} {city_term}",
    }
    return "https://www.opentable.com/s?" + urllib.parse.urlencode(params)

def _resy_search_link(name: str, date_str: str, time_str: str, covers: int, city: str) -> str:
    params = {"date": date_str, "time": time_str, "seats": covers, "query": name}
    return f"https://resy.com/cities/{resy_city_code(city)}?" + urllib.parse.urlencode(params)

def build_links(name: str, dt: datetime, covers: int, city_term: str = "New York") -> Dict[str, str]:
    dt_iso = dt.strftime("%Y-%m-%dT%H:%M")
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M")
    return {
        "opentable": _opentable_search_link(name, dt_iso, covers, city_term),
        "resy": _resy_search_link(name, date_str, time_str, covers, city_term),
        "google_maps": f"https://www.google.com/maps/search/{urllib.parse.quote(name + ' ' + city_term)}",
    }

# -------------------------------------------------------------------
# Google & Yelp
# -------------------------------------------------------------------
def google_text_search(query: str) -> List[Dict[str, Any]]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "type": "restaurant", "key": GOOGLE_KEY}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("results", [])

def google_place_details(place_id: str) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "place_id,name,formatted_address,geometry,opening_hours,website,url,price_level,rating,user_ratings_total,editorial_summary,formatted_phone_number",
        "key": GOOGLE_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("result", {})

def yelp_search(term: str, location: str, limit: int = 10) -> List[Dict[str, Any]]:
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {YELP_KEY}"}
    params = {"term": term, "location": location, "limit": limit, "categories": "restaurants"}
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("businesses", [])

# -------------------------------------------------------------------
# Hotspot avoid/demotion
# -------------------------------------------------------------------
AIREZ_HOTSPOT_AVOID = {
    # Italian hype / perennial tough bookings
    "carbone", "l'artusi", "via carota", "i sodi", "don angie",
    # add more if you like:
    "raoul's", "lucali",
}
def _is_hotspot(name: str) -> bool:
    n = (name or "").strip().lower()
    return any(hs in n for hs in AIREZ_HOTSPOT_AVOID)

# -------------------------------------------------------------------
# Live Search API
# -------------------------------------------------------------------
@app.post("/live_search")
def live_search(query: Query) -> Dict[str, Any]:
    # Quick mock mode (for wiring UI)
    if AIREZ_USE_MOCK:
        dt = safe_dt(query.date, query.time)
        sample_names = [
            "L'Artusi",
            "Via Carota",
            "I Sodi",
            "Don Angie",
            "Westville Hudson",
        ]
        items: List[Dict[str, Any]] = []
        for i, name in enumerate(sample_names):
            e = {
                "name": name,
                "address": f"{100+i} Some St, {query.city or 'New York'}",
                "rating": 4.5 - (i * 0.1),
                "reviews": 500 + i * 50,
                "price_level": 2 + (i % 2),
                "website": "",
                "maps_url": "",
                "links": build_links(name, dt, query.party_size, city_term=query.city or "New York"),
            }
            e["yelp_url"] = f"https://www.yelp.com/search?find_desc={urllib.parse.quote(name)}&find_loc={urllib.parse.quote(query.city or 'New York')}"
            e["_hotspot"] = _is_hotspot(name)
            items.append(e)
        # rank with penalty in mock too
        def score_mock(e):
            base = (e.get("rating", 0) or 0) * (1 + (e.get("reviews", 0) / 500.0))
            if e.get("_hotspot"): base -= 4.0
            return base
        ranked = sorted(items, key=score_mock, reverse=True)
        return {"mode": "mock", "items": ranked, "count": len(ranked)}

    # Real mode
    if not (GOOGLE_KEY and YELP_KEY):
        raise HTTPException(status_code=400, detail="Missing API keys. Set GOOGLE_PLACES_API_KEY and YELP_API_KEY, or set AIREZ_USE_MOCK=true.")

    dt = safe_dt(query.date, query.time)

    # 1) Google Places text search
    g_results = google_text_search(f"{query.q} in {query.city}")
    top: List[Dict[str, Any]] = []
    for r in g_results[:10]:
        details = google_place_details(r.get("place_id"))
        name = details.get("name") or r.get("name") or ""
        entry: Dict[str, Any] = {
            "name": name,
            "address": details.get("formatted_address", ""),
            "rating": details.get("rating", None),
            "reviews": details.get("user_ratings_total", 0),
            "price_level": details.get("price_level", None),
            "website": details.get("website", ""),
            "maps_url": details.get("url", ""),
        }
        entry["links"] = build_links(name, dt, query.party_size, city_term=query.city or "New York")
        entry["_hotspot"] = _is_hotspot(name)
        top.append(entry)

    # 2) Yelp for extra signal
    y_results = yelp_search(query.q, query.city or "New York", limit=10)
    yelp_index = {b["name"].lower(): b for b in y_results}

    # 3) Merge by name
    for e in top:
        y = yelp_index.get((e.get("name") or "").lower())
        if y:
            e["yelp_rating"] = y.get("rating")
            e["yelp_review_count"] = y.get("review_count")
            e["yelp_url"] = y.get("url")
            e["yelp_price"] = y.get("price")

    # 4) Rank by combined signals (with hotspot penalty)
    def score(e):
        g = e.get("rating") or 0
        gy = e.get("yelp_rating") or 0
        rc = (e.get("reviews") or 0) + (e.get("yelp_review_count") or 0)
        base = (g + gy) * (1 + (rc / 500.0))
        if e.get("_hotspot"):
            base -= 4.0  # heavy demotion
        return base

    ranked = sorted(top, key=score, reverse=True)[:10]
    return {"mode": "live", "items": ranked, "count": len(ranked)}

# -------------------------------------------------------------------
# Concierge AI (Rezzie) — safe fallback if no OPENAI_API_KEY
# -------------------------------------------------------------------
AIREZ_SYSTEM_PROMPT = """
You are Rezzie — AiRez’s concierge assistant.
Help the guest quickly find realistic, bookable options that fit their vibe and constraints.
Keep replies short (2–6 sentences). Offer a next step each time.
Avoid hard-to-book hype spots unless asked by name.
Output JSON only: { "reply": "...", "actions": [], "escalate": false, "notes": "" }
""".strip()

def _coerce_json(text: str) -> Tuple[dict, str]:
    raw = text.strip()
    if not raw.startswith("{"):
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            raw = m.group(0)
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("not a dict")
        obj.setdefault("reply", "I’m here. Tell me party size and time, and I’ll pull bookable tables.")
        obj.setdefault("actions", [])
        obj.setdefault("escalate", False)
        obj.setdefault("notes", "")
        return obj, text
    except Exception:
        return {
            "reply": "I’m here. Tell me party size and time, and I’ll pull bookable tables.",
            "actions": [],
            "escalate": False,
            "notes": "LLM output could not be parsed; returned fallback"
        }, text

def _llm_concierge_reply(message: str, context: dict) -> dict:
    if not _openai_client:
        need = []
        if not context.get("party"): need.append("party size")
        if not (context.get("date") and context.get("time")): need.append("date/time")
        ask = " and ".join(need) if need else "details"
        return {
            "reply": f"Happy to help. What {ask} should I use? I’ll pull bookable options.",
            "actions": [],
            "escalate": False,
            "notes": "OPENAI_API_KEY missing; using stub"
        }

    msgs = [
        {"role": "system", "content": AIREZ_SYSTEM_PROMPT},
        {"role": "user", "content": f"Message: {message}\nContext: {json.dumps(context, ensure_ascii=False)}"},
    ]

    resp = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.3,
        max_tokens=600,
    )
    text = (resp.choices[0].message.content or "").strip()
    obj, _ = _coerce_json(text)
    return obj

@app.post("/concierge/ai_chat")
def concierge_ai_chat(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    message = (payload.get("message") or "").strip()
    context = payload.get("context") or {}
    try:
        result = _llm_concierge_reply(message, context)
        result.setdefault("reply", "I’m here.")
        result.setdefault("actions", [])
        result.setdefault("escalate", False)
        result.setdefault("notes", "")
        return result
    except Exception as e:
        return {
            "reply": "I hit a snag reaching the concierge service. I can still show bookable times if you run a search.",
            "actions": [],
            "escalate": False,
            "notes": f"ai_chat exception: {str(e)}"
        }

@app.post("/concierge/handoff")
def concierge_handoff(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    if N8N_WEBHOOK:
        try:
            r = requests.post(N8N_WEBHOOK, json=payload, timeout=10)
            r.raise_for_status()
            return {"ok": True, "routed_to": "n8n", "status": r.status_code}
        except Exception as e:
            return {"ok": False, "error": str(e), "routed_to": "n8n"}
    return {"ok": True, "routed_to": "local_mock"}

# -------------------------------------------------------------------
# Health + Static site (THIS MUST BE LAST)
# -------------------------------------------------------------------
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}

# Serve current folder as a site (index.html at "/")
# Keep LAST so it doesn't shadow API routes.
app.mount("/", StaticFiles(directory=".", html=True), name="site")

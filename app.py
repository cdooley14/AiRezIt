# app.py — AiRez (Serve index.html + Live Search + Concierge AI + Cursor pagination + Verified Tab)
import os
import json
import re
import base64
import hashlib
import random
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Query as FQuery
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
VERIFIED_FILE: str = os.getenv("AIREZ_VERIFIED_FILE", "verified_nyc.json")  # << new

try:
    DEFAULT_COVERS: int = int(os.getenv("AIREZ_DEFAULT_COVERS", "2"))
except ValueError:
    DEFAULT_COVERS = 2

PAGE_SIZE = 10  # results per page

# OpenAI client (for concierge)
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# -------------------------------------------------------------------
# App + CORS
# -------------------------------------------------------------------
app = FastAPI(title="AiRez")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    params = {"covers": covers, "datetime": dt_iso, "term": name, "currentview": "list", "q": f"{name} {city_term}"}
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

AIREZ_HOTSPOT_AVOID = {"carbone", "l'artusi", "via carota", "i sodi", "don angie", "raoul's", "lucali"}
def _is_hotspot(name: str) -> bool:
    n = (name or "").strip().lower()
    return any(hs in n for hs in AIREZ_HOTSPOT_AVOID)

# --- Cursor helpers ---
def _b64e(obj: Dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode()

def _b64d(s: str) -> Dict[str, Any]:
    return json.loads(base64.urlsafe_b64decode(s.encode()).decode())

def _dedupe_key(name: str, address: str = "") -> str:
    return hashlib.sha1(f"{(name or '').strip().lower()}|{(address or '').strip().lower()}".encode()).hexdigest()

def _score_item(e: Dict[str, Any]) -> float:
    g = float(e.get("rating") or 0)
    y = float(e.get("yelp_rating") or 0)
    rc = float(e.get("reviews") or 0) + float(e.get("yelp_review_count") or 0)
    base = (g + y) + min(rc / 1000.0, 1.0)
    if e.get("_hotspot"):
        base -= 4.0
    return base + random.random() * 0.01

def _merge_rank_dedupe(groups: List[List[Dict[str, Any]]], seen: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for grp in groups:
        for e in grp:
            k = _dedupe_key(e.get("name",""), e.get("address",""))
            if k in seen:
                continue
            seen.add(k)
            e["_score"] = _score_item(e)
            out.append(e)
    out.sort(key=lambda x: x["_score"], reverse=True)
    return out

# -------------------------------------------------------------------
# Google photo helper
# -------------------------------------------------------------------
def google_photo_url(photo_ref: str, max_w: int = 640) -> str:
    if not GOOGLE_KEY or not photo_ref:
        return ""
    return (
        "https://maps.googleapis.com/maps/api/place/photo"
        f"?maxwidth={max_w}&photo_reference={photo_ref}&key={GOOGLE_KEY}"
    )

# -------------------------------------------------------------------
# Google & Yelp (paged)
# -------------------------------------------------------------------
def google_text_search(query: str, pagetoken: Optional[str] = None) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"key": GOOGLE_KEY, "type": "restaurant"}
    if pagetoken:
        params["pagetoken"] = pagetoken
    else:
        params["query"] = query
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def yelp_search(term: str, location: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {YELP_KEY}"}
    params = {"term": term, "location": location, "limit": limit, "offset": offset, "categories": "restaurants"}
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# -------------------------------------------------------------------
# Unified fetchers (mock + live)
# -------------------------------------------------------------------
MOCK_PHOTOS = [
    "https://images.unsplash.com/photo-1544025162-d76694265947",
    "https://images.unsplash.com/photo-1528605248644-14dd04022da1",
    "https://images.unsplash.com/photo-1473093295043-cdd812d0e601",
    "https://images.unsplash.com/photo-1498654896293-37aacf113fd9",
    "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
    "https://images.unsplash.com/photo-1526318472351-c75fcf070305",
]

def _mock_chunk(start: int, n: int, source: str, city: str) -> List[Dict[str, Any]]:
    out = []
    for i in range(start, start + n):
        name = f"{source.title()} Spot {i+1}"
        address = f"{100+i} Main St, {city}"
        photos = random.sample(MOCK_PHOTOS, k=min(5, len(MOCK_PHOTOS)))
        e = {
            "name": name,
            "address": address,
            "rating": round(random.uniform(3.7, 4.9), 1),
            "reviews": random.randint(50, 2500),
            "price_level": random.choice([1, 2, 3]),
            "website": "",
            "maps_url": "",
            "yelp_rating": None,
            "yelp_review_count": None,
            "yelp_url": "",
            "photos": photos,
            "source": source,
            "_hotspot": _is_hotspot(name),
        }
        out.append(e)
    return out

def fetch_yelp_chunk(q: str, city: str, want: int, offset: int) -> Dict[str, Any]:
    if AIREZ_USE_MOCK or not YELP_KEY:
        items = _mock_chunk(offset, want, "yelp", city)
        total = 90
        next_offset = offset + len(items)
        done = next_offset >= total
        return {"items": items, "next_offset": None if done else next_offset, "done": done}

    data = yelp_search(q, city, limit=want, offset=offset)
    businesses = data.get("businesses", [])
    items: List[Dict[str, Any]] = []
    for b in businesses:
        name = b.get("name","")
        addr = ", ".join(filter(None, [
            (b.get("location") or {}).get("address1",""),
            (b.get("location") or {}).get("city",""),
            (b.get("location") or {}).get("state",""),
        ]))
        photos = [b.get("image_url")] if b.get("image_url") else []
        e = {
            "name": name,
            "address": addr,
            "rating": b.get("rating"),
            "reviews": b.get("review_count"),
            "price_level": None,
            "website": b.get("url"),
            "maps_url": "",
            "yelp_rating": b.get("rating"),
            "yelp_review_count": b.get("review_count"),
            "yelp_url": b.get("url"),
            "photos": [p for p in photos if p],
            "source": "yelp",
            "_hotspot": _is_hotspot(name),
        }
        items.append(e)
    total = data.get("total", 0)
    next_offset = offset + len(items)
    done = next_offset >= total or len(items) == 0
    return {"items": items, "next_offset": None if done else next_offset, "done": done}

def fetch_places_chunk(q: str, city: str, next_token: Optional[str]) -> Dict[str, Any]:
    if AIREZ_USE_MOCK or not GOOGLE_KEY:
        page_index = int(next_token) if (next_token and next_token.isdigit()) else 0
        items = _mock_chunk(page_index * 20, 20, "places", city)
        pages_total = 3
        new_index = page_index + 1
        has_more = new_index < pages_total
        return {"items": items, "next_token": str(new_index) if has_more else None, "done": not has_more}

    try:
        raw = google_text_search(f"{q} in {city}", pagetoken=next_token)
    except requests.HTTPError:
        return {"items": [], "next_token": None, "done": True}

    results = raw.get("results", [])
    token = raw.get("next_page_token")
    items: List[Dict[str, Any]] = []
    for r in results:
        name = r.get("name","")
        address = r.get("formatted_address","")
        photo_refs = [p.get("photo_reference") for p in (r.get("photos") or [])]
        photos = [google_photo_url(ref, 640) for ref in photo_refs[:6] if ref]
        details = {
            "rating": r.get("rating"),
            "reviews": r.get("user_ratings_total"),
            "price_level": r.get("price_level"),
            "website": "",
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(name + ' ' + city)}",
        }
        e = {
            "name": name,
            "address": address,
            **details,
            "yelp_rating": None,
            "yelp_review_count": None,
            "yelp_url": "",
            "photos": photos,
            "source": "places",
            "_hotspot": _is_hotspot(name),
        }
        items.append(e)
    done = token is None or len(items) == 0
    return {"items": items, "next_token": token, "done": done}

# -------------------------------------------------------------------
# Verified data (admin curated)
# -------------------------------------------------------------------
def load_verified() -> List[Dict[str, Any]]:
    if not os.path.exists(VERIFIED_FILE):
        return []
    with open(VERIFIED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize minimal fields
    out = []
    for r in data:
        e = {
            "name": r.get("name",""),
            "address": r.get("address",""),
            "rating": r.get("admin_rating"),      # surface admin score as main rating
            "reviews": r.get("review_count", 0),
            "price_level": r.get("price_level"),  # 1-4 (we'll render as $…)
            "website": r.get("website",""),
            "maps_url": r.get("maps_url",""),
            "yelp_rating": r.get("yelp_rating"),
            "yelp_review_count": r.get("yelp_review_count"),
            "yelp_url": r.get("yelp_url"),
            "photos": r.get("photos", []),
            "source": "verified",
            "_hotspot": False,
            "verified": True,
            "admin_blurb": r.get("blurb",""),
            "admin_area": r.get("area",""),
            "admin_reviewer": r.get("reviewer",""),
            "admin_tags": r.get("tags", []),
            "admin_recs": r.get("recommend", []),
        }
        out.append(e)
    return out

VERIFIED_CACHE: List[Dict[str, Any]] = load_verified()

def search_verified(city: str, q: str, offset: int) -> Dict[str, Any]:
    # very simple filter: city/area + name/tags contains any word from q
    tokens = [t.strip().lower() for t in re.split(r"[,\s]+", q or "") if t.strip()]
    def match(item):
        if city and city.lower() not in (item.get("admin_area","") + " " + item.get("address","")).lower():
            # allow if city not specified in file; otherwise loose match on area/address
            pass
        hay = " ".join([
            item.get("name",""),
            item.get("admin_area",""),
            " ".join(item.get("admin_tags",[])),
            item.get("admin_blurb","")
        ]).lower()
        return all(tok in hay for tok in tokens) if tokens else True

    filtered = [e for e in VERIFIED_CACHE if match(e)]
    total = len(filtered)
    page = filtered[offset: offset + PAGE_SIZE]
    next_offset = offset + len(page)
    done = next_offset >= total
    return {"items": page, "next_offset": None if done else next_offset, "done": done}

# -------------------------------------------------------------------
# Unified Cursor Search API
# -------------------------------------------------------------------
@app.get("/api/search")
def unified_search(
    q: str = FQuery(..., description="Search text"),
    city: str = FQuery(DEFAULT_CITY, description="City"),
    cursor: Optional[str] = FQuery(None, description="Opaque cursor"),
    tab: str = FQuery("ai", description="ai | verified")   # << new
) -> Dict[str, Any]:
    """
    Returns:
      { "items":[...], "cursor":str|None, "has_more":bool }
    """
    # VERIFIED TAB: serve from local curated list with offset
    if tab == "verified":
        # read state or init
        if cursor:
            st = _b64d(cursor)
            offset = int(st.get("verified_offset", 0))
        else:
            offset = 0
        res = search_verified(city, q, offset)
        # add links
        dt = safe_dt(None, None)
        for e in res["items"]:
            e["links"] = build_links(e.get("name",""), dt, DEFAULT_COVERS, city_term=city)
        has_more = not res["done"]
        next_state = {"tab":"verified", "q":q, "city":city, "verified_offset": res.get("next_offset") or 0}
        return {
            "items": res["items"],
            "cursor": _b64e(next_state) if has_more and res["items"] else None,
            "has_more": has_more
        }

    # AI TAB: multi-source with Yelp/Places cursor
    if cursor:
        state = _b64d(cursor)
    else:
        state = {
            "q": q, "city": city,
            "yelp_offset": 0, "yelp_done": False,
            "places_token": None, "places_done": False,
            "seen": [], "tab": "ai"
        }

    seen = set(state.get("seen", []))
    want_each = PAGE_SIZE * 2

    y_items: List[Dict[str, Any]] = []
    if not state["yelp_done"]:
        yres = fetch_yelp_chunk(state["q"], state["city"], want_each, state["yelp_offset"])
        y_items = yres["items"]
        if yres.get("done") or not y_items:
            state["yelp_done"] = True
        state["yelp_offset"] = yres.get("next_offset", state["yelp_offset"] + len(y_items))

    p_items: List[Dict[str, Any]] = []
    if not state["places_done"]:
        pres = fetch_places_chunk(state["q"], state["city"], state["places_token"])
        p_items = pres["items"]
        if pres.get("done") or not p_items:
            state["places_done"] = True
            state["places_token"] = None
        else:
            state["places_token"] = pres.get("next_token")

    merged = _merge_rank_dedupe([y_items, p_items], seen)

    dt = safe_dt(None, None)
    for e in merged:
        e["links"] = build_links(e.get("name",""), dt, DEFAULT_COVERS, city_term=state["city"])

    page = merged[:PAGE_SIZE]
    for e in page:
        seen.add(_dedupe_key(e.get("name",""), e.get("address","")))

    state["seen"] = list(list(seen)[-120:])
    has_more_sources = not (state["yelp_done"] and state["places_done"])
    has_more_page = len(merged) > PAGE_SIZE or has_more_sources
    next_cursor = _b64e(state) if (has_more_page and len(page) > 0) else None

    return {"items": page, "cursor": next_cursor, "has_more": bool(next_cursor)}

# -------------------------------------------------------------------
# Live Search API (unchanged: quick 10 mixed)
# -------------------------------------------------------------------
@app.post("/live_search")
def live_search(query: Query) -> Dict[str, Any]:
    if AIREZ_USE_MOCK:
        dt = safe_dt(query.date, query.time)
        sample_names = ["L'Artusi","Via Carota","I Sodi","Don Angie","Westville Hudson"]
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
        def score_mock(e):
            base = (e.get("rating", 0) or 0) * (1 + (e.get("reviews", 0) / 500.0))
            if e.get("_hotspot"): base -= 4.0
            return base
        ranked = sorted(items, key=score_mock, reverse=True)
        return {"mode": "mock", "items": ranked, "count": len(ranked)}

    if not (GOOGLE_KEY and YELP_KEY):
        raise HTTPException(status_code=400, detail="Missing API keys. Set GOOGLE_PLACES_API_KEY and YELP_API_KEY, or set AIREZ_USE_MOCK=true.")

    dt = safe_dt(query.date, query.time)
    g_raw = google_text_search(f"{query.q} in {query.city}")
    g_results = g_raw.get("results", [])
    top: List[Dict[str, Any]] = []
    for r in g_results[:10]:
        name = r.get("name") or ""
        photo_refs = [p.get("photo_reference") for p in (r.get("photos") or [])]
        photos = [google_photo_url(ref, 640) for ref in photo_refs[:6] if ref]
        entry: Dict[str, Any] = {
            "name": name,
            "address": r.get("formatted_address", ""),
            "rating": r.get("rating"),
            "reviews": r.get("user_ratings_total", 0),
            "price_level": r.get("price_level"),
            "website": "",
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(name + ' ' + (query.city or 'New York'))}",
            "photos": photos
        }
        entry["links"] = build_links(name, dt, query.party_size, city_term=query.city or "New York")
        entry["_hotspot"] = _is_hotspot(name)
        top.append(entry)

    y_raw = yelp_search(query.q, query.city or "New York", limit=10, offset=0)
    y_results = y_raw.get("businesses", [])
    yelp_index = {b.get("name","").lower(): b for b in y_results}
    for e in top:
        y = yelp_index.get((e.get("name") or "").lower())
        if y:
            e["yelp_rating"] = y.get("rating")
            e["yelp_review_count"] = y.get("review_count")
            e["yelp_url"] = y.get("url")
            e["yelp_price"] = y.get("price")

    def score(e):
        g = e.get("rating") or 0
        gy = e.get("yelp_rating") or 0
        rc = (e.get("reviews") or 0) + (e.get("yelp_review_count") or 0)
        base = (g + gy) * (1 + (rc / 500.0))
        if e.get("_hotspot"):
            base -= 4.0
        return base

    ranked = sorted(top, key=score, reverse=True)[:10]
    return {"mode": "live", "items": ranked, "count": len(ranked)}

# -------------------------------------------------------------------
# Concierge AI (same as before)
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
        if m: raw = m.group(0)
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
        return {"reply": f"Happy to help. What {ask} should I use? I’ll pull bookable options.", "actions": [], "escalate": False, "notes": "OPENAI_API_KEY missing; using stub"}

    msgs = [
        {"role": "system", "content": AIREZ_SYSTEM_PROMPT},
        {"role": "user", "content": f"Message: {message}\nContext: {json.dumps(context, ensure_ascii=False)}"},
    ]
    resp = _openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.3, max_tokens=600)
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
        return {"reply": "I hit a snag reaching the concierge service. I can still show bookable times if you run a search.", "actions": [], "escalate": False, "notes": f"ai_chat exception: {str(e)}"}

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
# Health + Static site
# -------------------------------------------------------------------
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}

app.mount("/", StaticFiles(directory=".", html=True), name="site")


# AiRez — Flip to Live Search

This package adds **real API** calls and deep links (OpenTable/Resy search) so you can exit mock mode and run your first live search.

## 1) Set your environment
Copy `.env.example` to `.env` and fill in keys:
```
cp code/.env.example code/.env
```
Edit `code/.env` and set:
- `AIREZ_USE_MOCK=false`
- `GOOGLE_PLACES_API_KEY=...`
- `YELP_API_KEY=...`
- `OPENAI_API_KEY=...` (used by n8n Chat node, not this FastAPI app)

## 2) Run the app locally
Use uvicorn (or your server of choice):
```
pip install fastapi uvicorn pydantic requests python-dotenv
export $(grep -v '^#' code/.env | xargs)
uvicorn code.app:app --reload --port 8088
```

## 3) n8n hookup (webhook workflow)
- Webhook Trigger → Function (parse user text into party/date/time/location) → HTTP Request (POST to `http://localhost:8088/live_search`) → OpenAI (rank + explain vibe) → Respond to Webhook.
- In HTTP Request node, send JSON body like:
```json
{
  "q": "cozy pasta date night, West Village",
  "party_size": 2,
  "date": "2025-08-15",
  "time": "20:00",
  "city": "New York"
}
```
- Use `$env.GOOGLE_PLACES_API_KEY`, `$env.YELP_API_KEY`, `$env.OPENAI_API_KEY` in credentials or node expressions so **no keys** live in the workflow JSON.

## 4) OpenTable & Resy links
The response includes:
- `links.opentable` – an OpenTable search link prefilled with the party size and datetime
- `links.resy` – a Resy city search link with the restaurant name and date
- `links.google_maps` – quick map lookup

> Note: Neither provider has a public availability API. These are dependable **search deeplinks** that users can tap to book.

## 5) First live test
In Postman or curl:
```
curl -X POST http://localhost:8088/live_search \  -H "Content-Type: application/json" \  -d '{"q":"cozy pasta date night, West Village","party_size":2,"date":"2025-08-15","time":"20:00","city":"New York"}'
```

## 6) Common gotchas
- 403/`REQUEST_DENIED` from Google: restrict API key to Places API (enable **Places API** + **Places API (New)** if available) and add your IP/domain to allowed referrers.
- Yelp: make sure you created a **Yelp Fusion** app to get the v3 API key.
- If `mode` returns `mock`, your `AIREZ_USE_MOCK` is still `true` in the environment where the app runs.

---

### What changed
- `code/app.py` now provides a `/live_search` endpoint that calls Google Places + Yelp and emits OpenTable/Resy search links.
- `code/.env.example` created.
- This complements your n8n flow (keys live in env; n8n does orchestration + OpenAI ranking/explanations).

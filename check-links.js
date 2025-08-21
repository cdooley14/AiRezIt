// check-links.js
const express = require("express");
const http = require("http");
const https = require("https");
const fetch = require("node-fetch"); // v2
const app = express();
app.use(express.json());

const AGENTS = {
  http: new http.Agent({ keepAlive: true, maxSockets: 20 }),
  https: new https.Agent({ keepAlive: true, maxSockets: 20 }),
};

const DEFAULT_TIMEOUT_MS = 6000;

function agentFor(u) { return (u.protocol === "http:" ? AGENTS.http : AGENTS.https); }

async function tryFetch(url, method, timeoutMs) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method,
      redirect: "follow",
      signal: ctrl.signal,
      headers: { "user-agent": "Mozilla/5.0 AiRez-LinkChecker/1.1" },
      agent: agentFor,
    });
    return res;
  } finally {
    clearTimeout(t);
  }
}

function classify(res, err) {
  if (err) {
    return { status: "unknown", reason: err.name === "AbortError" ? "timeout" : err.message };
  }
  const code = res.status;
  if (code >= 200 && code < 400) return { status: "live", httpStatus: code, reason: "ok" };
  if ([404, 410, 451].includes(code)) return { status: "dead", httpStatus: code, reason: "not found" };
  if ([401, 403, 429].includes(code)) return { status: "unknown", httpStatus: code, reason: "blocked" };
  if (code >= 500) return { status: "unknown", httpStatus: code, reason: "server error" };
  return { status: "unknown", httpStatus: code, reason: "unclassified" };
}

async function checkSingle(url, timeoutMs) {
  try {
    let res = await tryFetch(url, "HEAD", timeoutMs);
    let base = classify(res);
    if (base.status !== "live") {
      res = await tryFetch(url, "GET", timeoutMs);
      base = classify(res);
    }
    return { url, finalUrl: res?.url, httpStatus: res?.status, status: base.status, reason: base.reason };
  } catch (e) {
    const base = classify(undefined, e);
    return { url, status: base.status, reason: base.reason };
  }
}

// ---------- Simple helpers ----------
function toMinutes(hhmm) {
  const [h, m] = (hhmm || "20:00").split(":").map(n => parseInt(n, 10));
  return (h * 60) + (m || 0);
}
function pad2(n) { return n < 10 ? "0" + n : "" + n; }
function minutesToHHMM(mins) {
  let h = Math.floor(mins / 60);
  let m = mins % 60;
  if (h < 0) h = 0;
  if (h > 23) h = 23;
  return `${pad2(h)}:${pad2(m)}`;
}

// Compose provider URLs with date/time/seats baked in
function buildOpenTable(url, { date, time, seats, cityTerm }) {
  // Accept either a search URL or bare; ensure core params present
  const u = new URL(url);
  u.searchParams.set("datetime", `${date}T${time}`);
  u.searchParams.set("covers", String(seats || 2));
  if (!u.searchParams.get("q") && cityTerm) {
    u.searchParams.set("q", cityTerm);
  }
  if (!u.searchParams.get("currentview")) u.searchParams.set("currentview", "list");
  return u.toString();
}
function buildResy(url, { date, time, seats }) {
  // Expected form: https://resy.com/cities/ny?date=YYYY-MM-DD&time=HH:MM&seats=N&query=Name
  const u = new URL(url);
  u.searchParams.set("date", date);
  u.searchParams.set("time", time);
  u.searchParams.set("seats", String(seats || 2));
  return u.toString();
}

// ---------- Routes ----------

// Link health
app.post("/check_links", async (req, res) => {
  const { links, timeoutMs } = req.body || {};
  if (!links || typeof links !== "object") {
    return res.status(400).json({ error: "Missing links object" });
  }

  const entries = Object.entries(links).filter(([, u]) => !!u);
  const results = await Promise.all(entries.map(async ([key, url]) => [key, await checkSingle(url, Number(timeoutMs) || DEFAULT_TIMEOUT_MS)]));
  const obj = Object.fromEntries(results);
  const summary = {
    hasAnyLive: Object.values(obj).some((x) => x.status === "live"),
    hasAnyDead: Object.values(obj).some((x) => x.status === "dead"),
    hasAnyUnknown: Object.values(obj).some((x) => x.status === "unknown"),
  };
  res.json({ results: obj, summary });
});

// Availability probe (±window in step minutes). Heuristic: compose provider search URLs,
// load once (cheap), then surface a grid of candidate times as chips that land on the provider page.
// This does not scrape private APIs; it verifies pages are reachable and gives bookable time anchors.
app.post("/availability", async (req, res) => {
  const {
    baseUrls = {},            // { opentable?: string, resy?: string }
    date, time = "20:00",
    party = 2,
    city = "New York",
    windowMinutes = 60,
    stepMinutes = 15,
    timeoutMs = DEFAULT_TIMEOUT_MS,
  } = req.body || {};

  const center = toMinutes(time);
  const starts = [];
  for (let t = center - windowMinutes; t <= center + windowMinutes; t += stepMinutes) {
    if (t >= 0 && t <= 23 * 60 + 59) starts.push(minutesToHHMM(t));
  }

  const out = { availability: {} };

  // OpenTable
  if (baseUrls.opentable) {
    const otUrl = buildOpenTable(baseUrls.opentable, { date, time, seats: party, cityTerm: city });
    try {
      const r = await tryFetch(otUrl, "GET", timeoutMs);
      if (r && r.status >= 200 && r.status < 400) {
        out.availability.opentable = [{ url: otUrl, times: starts }];
      } else {
        out.availability.opentable = [{ url: otUrl, times: [] }];
      }
    } catch {
      out.availability.opentable = [{ url: otUrl, times: [] }];
    }
  }

  // Resy
  if (baseUrls.resy) {
    const resyUrl = buildResy(baseUrls.resy, { date, time, seats: party });
    try {
      const r = await tryFetch(resyUrl, "GET", timeoutMs);
      if (r && r.status >= 200 && r.status < 400) {
        out.availability.resy = [{ url: resyUrl, times: starts }];
      } else {
        out.availability.resy = [{ url: resyUrl, times: [] }];
      }
    } catch {
      out.availability.resy = [{ url: resyUrl, times: [] }];
    }
  }

  res.json(out);
});

const PORT = process.env.PORT || 8089;
app.listen(PORT, () => console.log(`✔ Link+Availability service at http://127.0.0.1:${PORT}/`));

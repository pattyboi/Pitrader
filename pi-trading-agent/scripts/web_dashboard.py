#!/usr/bin/env python3
"""Browser dashboard for the per-symbol opinion snapshot (signal_snapshot.py).

Reads two read-only JSON snapshot files (.portfolio_signal_snapshot.json,
.crypto_signal_snapshot.json) the live equity and crypto iterations already
write, and renders them as a terminal-themed HTML page -- never touches the
broker, never imports lumibot, never affects the trading path. Deliberately
stdlib-only (http.server, no Flask/aiohttp/etc.) so it adds zero new pip
dependencies and stays cheap enough to run continuously on the Pi -- a few
MB of RSS, no background polling thread, snapshot files are only read in
response to an actual HTTP request. The browser itself does the polling (see
POLL_INTERVAL_MS in the page's inline JS), and there's no work to do between
requests.

Usage: python3 scripts/web_dashboard.py
Env vars: DASHBOARD_HOST (default 0.0.0.0), DASHBOARD_PORT (default 8765)
"""

from __future__ import annotations

import json
import os
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_SNAPSHOT = BASE_DIR / ".portfolio_signal_snapshot.json"
CRYPTO_SNAPSHOT = BASE_DIR / ".crypto_signal_snapshot.json"
PORTFOLIO_TRADE_COUNT = BASE_DIR / ".portfolio_trade_count.json"
CRYPTO_TRADE_COUNT = BASE_DIR / ".crypto_trade_count.json"

HOST = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))


def _load_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # A writer always produces a JSON object; anything else (a foreign or
    # corrupted file) must not be trusted as one -- callers rely on this
    # return type actually being a dict (or None) and call .get() on it
    # unguarded.
    return data if isinstance(data, dict) else None


def _load_trade_count(path: Path) -> int:
    """Today's trade count, or 0 if the file is missing or dated earlier --
    trade_counter.record_trade resets the stored count on the strategy's
    first trade of a new day, but until that first trade happens the file
    still holds yesterday's number, which would be misleading to show."""
    data = _load_snapshot(path)
    if not data or data.get("date") != date.today().isoformat():
        return 0
    try:
        return int(data.get("count", 0))
    except (TypeError, ValueError):
        return 0


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pi-trading-agent :: signal board</title>
<style>
  :root {
    --bg: #000000;
    --panel: #0d0a00;
    --fg: #f4b400;
    --fg-dim: #8f6c00;
    --amber: #f4b400;
    --green: #33ff66;
    --red: #ff4d4d;
    --border: #6b5000;
    --held-bg: rgba(244, 180, 0, 0.08);
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg);
    color: var(--fg);
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, "SF Mono", "Cascadia Code", "Courier New", monospace;
    font-size: 14px;
    overflow-x: hidden;
  }
  .wrap { max-width: 980px; margin: 0 auto; padding: 18px 16px 40px; }
  pre.banner {
    color: var(--fg);
    text-shadow: 0 0 6px rgba(244,180,0,0.35);
    font-size: 8.25px;
    line-height: 1.15;
    margin: 0 0 10px;
    overflow-x: auto;
    white-space: pre;
  }
  .prompt-line {
    color: var(--fg-dim);
    margin-bottom: 14px;
    white-space: pre-wrap;
  }
  .prompt-line .arrow { color: var(--fg); }
  .prompt-line .branch { color: var(--amber); }
  .cursor {
    display: inline-block;
    width: 8px; height: 14px;
    background: var(--fg);
    margin-left: 2px;
    animation: blink 1s steps(1) infinite;
    vertical-align: middle;
  }
  @keyframes blink { 50% { opacity: 0; } }

  .statusbar {
    display: flex;
    flex-wrap: wrap;
    margin: 0 0 20px;
    border-radius: 3px;
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--border);
  }
  .seg {
    padding: 6px 14px;
    font-size: 12px;
    letter-spacing: 0.03em;
    display: flex;
    align-items: center;
    gap: 6px;
    position: relative;
  }
  .seg:not(:last-child)::after {
    content: "";
    position: absolute;
    right: -1px; top: 0; bottom: 0;
    width: 1px;
    background: rgba(0,0,0,0.35);
  }
  .seg-dark   { background: #2a2002; color: var(--fg); }
  .seg-mid    { background: #3d2f04; color: var(--fg); }
  .seg-amber  { background: #431c04; color: var(--fg); }
  .seg-live   { background: #2a2002; color: var(--fg); }
  .seg-trades { background: #3d2f04; color: var(--fg); font-weight: bold; }
  .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--fg);
    box-shadow: 0 0 6px var(--fg);
    animation: pulse 1.6s ease-in-out infinite;
  }
  .dot.stale { background: var(--red); box-shadow: 0 0 6px var(--red); animation: none; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

  .panel {
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 22px;
    background: var(--panel);
  }
  .panel-title {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--fg);
    font-weight: bold;
    letter-spacing: 0.08em;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .panel-title .meta {
    color: var(--fg-dim);
    font-weight: normal;
    font-size: 11px;
    letter-spacing: normal;
  }
  table { width: 100%; border-collapse: collapse; }
  thead th {
    text-align: left;
    padding: 6px 12px;
    color: var(--fg-dim);
    font-size: 11px;
    letter-spacing: 0.06em;
    border-bottom: 1px dashed var(--border);
  }
  tbody td {
    padding: 5px 12px;
    border-bottom: 1px solid rgba(244,180,0,0.25);
    white-space: nowrap;
  }
  tbody tr.held { background: var(--held-bg); }
  tbody tr:last-child td { border-bottom: none; }
  td.sym { color: var(--fg); font-weight: bold; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .opinion-pos { color: var(--green); text-shadow: 0 0 5px rgba(51,255,102,0.5); }
  .opinion-neg { color: var(--red); text-shadow: 0 0 5px rgba(255,77,77,0.4); }
  .held-tag { color: var(--amber); font-size: 11px; }
  .empty { padding: 16px 12px; color: var(--fg-dim); }
  .footer { color: var(--fg-dim); font-size: 11px; margin-top: 24px; }
  ::-webkit-scrollbar { height: 8px; }
  ::-webkit-scrollbar-thumb { background: var(--border); }
</style>
</head>
<body>
<div class="wrap">
<pre class="banner">
 ____ ___    _____ ____      _    ____ ___ _   _  ____
|  _ |_ _|  |_   _|  _ \    / \  |  _ |_ _| \ | |/ ___|
| |_) | |_____| | | |_) |  / _ \ | | | | ||  \| | |  _
|  __/| |_____| | |  _ <  / ___ \| |_| | || |\  | |_| |
|_|  |___|    |_| |_| \_\/_/   \_\____|___|_| \_|\____|
</pre>
<div class="prompt-line"><span class="arrow">&#10148;</span> pi-trading-agent <span class="branch">signal-board</span> <span class="cursor"></span></div>

<div class="statusbar" id="statusbar">
  <div class="seg seg-live"><span class="dot" id="live-dot"></span><span id="live-text">connecting&hellip;</span></div>
  <div class="seg seg-dark" id="clock-seg">--:--:--</div>
  <div class="seg seg-mid" id="stocks-posture-seg">stocks: &mdash;</div>
  <div class="seg seg-amber" id="crypto-posture-seg">crypto: &mdash;</div>
  <div class="seg seg-trades" id="trades-today-seg">trades today: &mdash;</div>
</div>

<div class="panel">
  <div class="panel-title"><span>STOCKS</span><span class="meta" id="stocks-meta"></span></div>
  <div id="stocks-body"><div class="empty">loading&hellip;</div></div>
</div>

<div class="panel">
  <div class="panel-title"><span>CRYPTO</span><span class="meta" id="crypto-meta"></span></div>
  <div id="crypto-body"><div class="empty">loading&hellip;</div></div>
</div>

<div class="footer">read-only view of signal_snapshot.py output &middot; no broker calls &middot; polls /api/snapshot every 5s</div>
</div>

<script>
const POLL_INTERVAL_MS = 5000;

function fmtAge(iso) {
  if (!iso) return "unknown age";
  const then = new Date(iso);
  if (isNaN(then.getTime())) return "unknown age";
  const seconds = (Date.now() - then.getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return Math.floor(seconds / 60) + "m ago";
  if (seconds < 86400) return Math.floor(seconds / 3600) + "h ago";
  return Math.floor(seconds / 86400) + "d ago";
}

function renderTable(container, snapshot) {
  if (!snapshot) {
    container.innerHTML = '<div class="empty">no data yet -- the agent hasn\'t completed an iteration since this was set up</div>';
    return null;
  }
  const entries = snapshot.symbols || [];
  if (entries.length === 0) {
    container.innerHTML = '<div class="empty">no symbols evaluated yet</div>';
    return snapshot;
  }
  let rows = "";
  for (const e of entries) {
    const opClass = e.opinion === "+" ? "opinion-pos" : "opinion-neg";
    const heldRow = e.held ? "held" : "";
    const heldTag = e.held ? '<span class="held-tag">HELD</span>' : "";
    rows += `<tr class="${heldRow}">
      <td class="sym">${e.symbol}</td>
      <td>${heldTag}</td>
      <td class="num">${Number(e.dip_percent || 0).toFixed(2)}%</td>
      <td class="num">${Number(e.edge_percent || 0).toFixed(2)}%</td>
      <td class="num ${opClass}">${e.opinion === "+" ? "&#9650;" : "&#9660;"} ${e.opinion}</td>
    </tr>`;
  }
  container.innerHTML = `<table>
    <thead><tr><th>symbol</th><th></th><th style="text-align:right">dip%</th><th style="text-align:right">edge%</th><th style="text-align:right">opinion</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
  return snapshot;
}

async function poll() {
  const dot = document.getElementById("live-dot");
  const liveText = document.getElementById("live-text");
  try {
    const res = await fetch("/api/snapshot", { cache: "no-store" });
    if (!res.ok) throw new Error("bad status " + res.status);
    const data = await res.json();

    const stocks = renderTable(document.getElementById("stocks-body"), data.stocks);
    const crypto = renderTable(document.getElementById("crypto-body"), data.crypto);

    document.getElementById("stocks-meta").textContent = stocks
      ? `posture: ${stocks.risk_posture} · updated ${fmtAge(stocks.generated_at)}`
      : "";
    document.getElementById("crypto-meta").textContent = crypto
      ? `posture: ${crypto.risk_posture} · updated ${fmtAge(crypto.generated_at)}`
      : "";
    document.getElementById("stocks-posture-seg").textContent = "stocks: " + (stocks ? stocks.risk_posture : "—");
    document.getElementById("crypto-posture-seg").textContent = "crypto: " + (crypto ? crypto.risk_posture : "—");

    const trades = data.trades_today || { total: 0, stocks: 0, crypto: 0 };
    document.getElementById("trades-today-seg").textContent =
      `trades today: ${trades.total} (stocks ${trades.stocks} / crypto ${trades.crypto})`;

    dot.classList.remove("stale");
    liveText.textContent = "live";
  } catch (err) {
    dot.classList.add("stale");
    liveText.textContent = "disconnected";
  }
}

function tickClock() {
  document.getElementById("clock-seg").textContent = new Date().toLocaleTimeString();
}

poll();
tickClock();
setInterval(poll, POLL_INTERVAL_MS);
setInterval(tickClock, 1000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "PiTradingDashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the journal quiet; this is a read-only, low-value view logged noise

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/snapshot":
            stocks_trades = _load_trade_count(PORTFOLIO_TRADE_COUNT)
            crypto_trades = _load_trade_count(CRYPTO_TRADE_COUNT)
            payload = {
                "stocks": _load_snapshot(PORTFOLIO_SNAPSHOT),
                "crypto": _load_snapshot(CRYPTO_SNAPSHOT),
                "trades_today": {
                    "stocks": stocks_trades,
                    "crypto": crypto_trades,
                    "total": stocks_trades + crypto_trades,
                },
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"pi-trading-agent dashboard on http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

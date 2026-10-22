#!/usr/bin/env python3
"""
Daily virtual-portfolio tracker.

What it does every run:
  1. Pulls end-of-day prices + volume (Yahoo Finance via yfinance).
  2. Computes P&L, volatility, volume vs average, and Nifty 50 comparison.
  3. Collects news ONLY from trusted publishers, removes duplicates and old items.
  4. Draws charts into /charts.
  5. Writes data/latest.json and rebuilds README.md (the dashboard).
  6. Skips cleanly on market holidays / when there is no new trading day.

Portfolio holdings come from transactions.csv (BUY / SELL / DEPOSIT lines, see ledger.py).
Edit config.json to add companies (tickers, sector, BSE code) or change trusted news sources.
"""
import difflib
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from html import unescape
from urllib.parse import quote_plus, urljoin

import feedparser
import matplotlib
import pandas as pd
import requests
import yfinance as yf

import ledger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
IST = timezone(timedelta(hours=5, minutes=30))
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio-tracker; educational project)"}
DISCLAIMER = (
    "This is a **virtual (paper) portfolio built for learning and skill demonstration**. "
    "It is **not investment advice** and not a recommendation to buy, sell or hold any security."
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def inr(x, dec=0):
    """Format a number as Indian-style rupees, e.g. 12345678 -> Rs 1,23,45,678"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    neg = x < 0
    s = f"{abs(x):.{dec}f}"
    ip, _, fp = s.partition(".")
    if len(ip) > 3:
        last3, rest = ip[-3:], ip[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        ip = ",".join(parts + [last3])
    return f"{'-' if neg else ''}₹{ip}{('.' + fp) if fp else ''}"


def pct(x, dec=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{dec}f}%"


def num(x, dec=0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.{dec}f}"


def write_github_output(key, value):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Prices
# ----------------------------------------------------------------------------
def get_history(tickers, period="1y"):
    """Try each ticker in order; return (ticker_used, DataFrame)."""
    if isinstance(tickers, str):
        tickers = [tickers]
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=False)
            if df is not None and not df.empty and df["Close"].dropna().shape[0] > 5:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df = df[~df.index.duplicated(keep="last")].sort_index()
                return t, df
            log(f"  ! {t}: no usable data")
        except Exception as e:  # network / yfinance issues
            log(f"  ! {t}: {e}")
    return None, None


def stock_metrics(df, shares, base_price):
    close = df["Close"].dropna()
    last = close.iloc[-1]
    prev = close.iloc[-2] if len(close) > 1 else float("nan")
    row = df.loc[close.index[-1]]
    vol_today = float(row["Volume"])
    prior_vol = df["Volume"].iloc[-21:-1]
    avg20 = float(prior_vol.mean()) if len(prior_vol) else float("nan")
    rets = close.pct_change().dropna()
    vol20 = float(rets.tail(20).std() * math.sqrt(252) * 100) if len(rets) >= 5 else float("nan")
    last_252 = close.tail(252)
    value = shares * last
    cost = shares * base_price
    return {
        "close": round(float(last), 2),
        "prev_close": round(float(prev), 2) if not math.isnan(prev) else None,
        "day_change_pct": round(float((last / prev - 1) * 100), 2) if not math.isnan(prev) else None,
        "open": round(float(row["Open"]), 2),
        "high": round(float(row["High"]), 2),
        "low": round(float(row["Low"]), 2),
        "intraday_range_pct": round(float((row["High"] - row["Low"]) / prev * 100), 2) if not math.isnan(prev) else None,
        "volume": int(vol_today),
        "avg_volume_20d": int(avg20) if not math.isnan(avg20) else None,
        "volume_vs_avg": round(vol_today / avg20, 2) if avg20 and not math.isnan(avg20) else None,
        "volatility_20d_annualised_pct": round(vol20, 2) if not math.isnan(vol20) else None,
        "high_52w": round(float(last_252.max()), 2),
        "low_52w": round(float(last_252.min()), 2),
        "shares": int(shares),
        "base_price": round(float(base_price), 2),
        "market_value": round(float(value), 2),
        "pnl_inr": round(float(value - cost), 2),
        "pnl_pct": round(float((value / cost - 1) * 100), 2) if cost else None,
    }


def quality_checks(name, m):
    warns = []
    if m["close"] <= 0:
        warns.append(f"{name}: close price is not positive - ignore today's data.")
    if m["day_change_pct"] is not None and abs(m["day_change_pct"]) > 15:
        warns.append(f"{name}: move of {m['day_change_pct']}% is unusually large - verify with NSE before relying on it.")
    if m["volume"] == 0:
        warns.append(f"{name}: reported volume is zero - data may be incomplete.")
    return warns


# ----------------------------------------------------------------------------
# News (trusted sources only, de-duplicated, recent only)
# ----------------------------------------------------------------------------
def _norm(title):
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _entry_time(e):
    t = e.get("published_parsed") or e.get("updated_parsed")
    if not t:
        return None
    return datetime(*t[:6], tzinfo=timezone.utc)


def _split_title_source(title):
    """Google News titles look like 'Headline - Publisher'."""
    if " - " in title:
        head, _, src = title.rpartition(" - ")
        return head.strip(), src.strip()
    return title.strip(), ""


def is_trusted(source, trusted):
    s = (source or "").lower()
    return any(t in s for t in trusted)


def fetch_feed(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return feedparser.parse(r.content)
    except Exception as e:
        log(f"  ! feed failed {url}: {e}")
        return None


_DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"), ("%d %b %Y", "%d %B %Y")),
    (re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b"), ("%b %d %Y", "%B %d %Y")),
    (re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"), ("%d-%m-%Y",)),
]


def _parse_date_text(text):
    """Return the LAST date found in text as an IST-noon datetime (UTC), or None."""
    best = None
    for rx, fmts in _DATE_PATTERNS:
        for m in rx.finditer(text):
            raw = " ".join(m.groups()) if len(fmts) > 1 or "%b" in fmts[0] else "-".join(m.groups())
            for f in fmts:
                try:
                    d = datetime.strptime(raw, f)
                except ValueError:
                    continue
                if best is None or m.start() >= best[0]:
                    best = (m.start(), d)
                break
    if not best:
        return None
    return best[1].replace(hour=12, tzinfo=IST).astimezone(timezone.utc)


def collect_official_news(stock):
    """Official company newsroom (static HTML page). Config: stock["official_news"] =
    {"url", "source", "max_age_days"}. Items must have a parseable date within max_age_days;
    undated items are skipped (we cannot verify recency). Fails silently -> []."""
    cfg = stock.get("official_news")
    if not cfg:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.get("max_age_days", 30))
    try:
        r = requests.get(cfg["url"], headers={**HEADERS, "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/124.0 Safari/537.36"}, timeout=25)
        r.raise_for_status()
        page = r.text
    except Exception as e:
        log(f"  ! official newsroom failed {cfg['url']}: {e}")
        return []
    found = {}
    for m in re.finditer(r'<a\b[^>]*?href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', page, re.S | re.I):
        href = urljoin(cfg["url"], unescape(m.group(1)).strip())
        if not re.search(r"/in/(pressrelease|newsroom)/[^/?#]+", href):
            continue
        title = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", m.group(2)))).strip()
        if len(title) < 15 or title.lower().startswith("read more"):
            continue
        when = _parse_date_text(page[max(0, m.start() - 700):m.start()]) or \
            _parse_date_text(page[m.end():m.end() + 300])
        if href not in found or len(title) > len(found[href]["title"]):
            found[href] = {"title": title, "source": cfg["source"], "published": when, "link": href, "date_only": True}
    out = [v for v in found.values() if v["published"] and v["published"] >= cutoff]
    if not out:
        log(f"  - official newsroom: no dated items in the last {cfg.get('max_age_days', 30)} days")
    return out


def collect_news(cfg, stock, publisher_cache):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg["news_max_age_hours"])
    trusted = cfg["trusted_sources"]
    kws = [k.lower() for k in stock["keywords"]]
    items = []

    # 1) Google News RSS, kept only if publisher is on the trusted list
    q = quote_plus(f"{stock['news_query']} when:3d")
    gurl = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = fetch_feed(gurl)
    if feed:
        for e in feed.entries:
            head, src_in_title = _split_title_source(e.get("title", ""))
            src = (e.get("source", {}) or {}).get("title") or src_in_title
            when = _entry_time(e)
            if not when or when < cutoff:
                continue
            if not is_trusted(src, trusted):
                continue
            if not any(k in head.lower() for k in kws):
                continue
            items.append({"title": head, "source": src, "published": when, "link": e.get("link", "")})

    # 2) Publisher RSS feeds (already trusted), filtered by company keywords
    for pf in cfg["publisher_feeds"]:
        feed = publisher_cache.get(pf["url"])
        if not feed:
            continue
        for e in feed.entries:
            when = _entry_time(e)
            if not when or when < cutoff:
                continue
            title = e.get("title", "").strip()
            if not any(k in title.lower() for k in kws):
                continue
            items.append({"title": title, "source": pf["source"], "published": when, "link": e.get("link", "")})

    # 3) Official company newsroom (optional, per stock)
    items.extend(collect_official_news(stock))

    # De-duplicate (near-identical headlines), newest first
    items.sort(key=lambda x: x["published"], reverse=True)
    unique = []
    for it in items:
        n = _norm(it["title"])
        if any(difflib.SequenceMatcher(None, n, _norm(u["title"])).ratio() > 0.82 for u in unique):
            continue
        unique.append(it)
    unique = unique[: cfg["news_per_company"]]
    for u in unique:
        u["published"] = u["published"].astimezone(IST).strftime("%d %b %Y" if u.pop("date_only", False) else "%d %b %Y, %I:%M %p IST")
    return unique


def collect_announcements(stock):
    """Best-effort: official corporate filings from BSE. Fails silently -> link only."""
    code = stock["bse_code"]
    out = []
    try:
        today = datetime.now(IST).strftime("%Y%m%d")
        frm = (datetime.now(IST) - timedelta(days=7)).strftime("%Y%m%d")
        url = (
            "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
            f"?pageno=1&strCat=-1&strPrevDate={frm}&strScrip={code}&strSearch=P&strToDate={today}&strType=C"
        )
        h = dict(HEADERS)
        h.update({"Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"})
        r = requests.get(url, headers=h, timeout=15)
        r.raise_for_status()
        for row in (r.json().get("Table") or [])[:5]:
            title = (row.get("NEWSSUB") or row.get("HEADLINE") or "").strip()
            att = row.get("ATTACHMENTNAME")
            link = (
                f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}"
                if att
                else f"https://www.bseindia.com/corporates/ann.html?scrip={code}"
            )
            if title:
                out.append({"title": title, "date": str(row.get("NEWS_DT", ""))[:10], "link": link})
    except Exception as e:
        log(f"  (BSE announcements unavailable for {code}: {e})")
    return out


# ----------------------------------------------------------------------------
# Charts (colours: validated categorical palette, one fixed colour per company)
# ----------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GAIN, LOSS = "#2a78d6", "#e34948"  # diverging pair: blue = gain, red = loss (always with a +/- label)


def stock_colors(cfg):
    """Colour follows the company (config order), never its rank. Beyond 8 -> neutral grey."""
    return {s["key"]: (SERIES[i] if i < len(SERIES) else MUTED) for i, s in enumerate(cfg["stocks"])}


def _style():
    plt.rcParams.update({
        "figure.dpi": 130, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.titlelocation": "left", "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "legend.frameon": False, "font.family": "sans-serif",
    })


def _trade_markers(ax, trades, series, key=None):
    """Buy = up triangle, sell = down triangle, with a surface-coloured ring so they never blur into lines."""
    seen = set()
    for t in trades:
        if t["type"] not in ("BUY", "SELL") or (key and t["key"] != key):
            continue
        d = t["effective"]
        if d not in series.index:
            continue
        buy = t["type"] == "BUY"
        ax.scatter([d], [series.loc[d]], marker="^" if buy else "v", s=46, zorder=5,
                   color=GAIN if buy else LOSS, edgecolors=SURFACE, linewidths=1.5,
                   label=None if t["type"] in seen else ("Buy" if buy else "Sell"))
        seen.add(t["type"])


def make_charts(cfg, per_stock_df, led, results, snapshot):
    charts = ROOT / "charts"
    charts.mkdir(exist_ok=True)
    _style()
    colors = stock_colors(cfg)
    port, nifty = led["port_series"], led["nifty_series"]
    bname = cfg["benchmark"]["name"]

    # 1) Portfolio vs Nifty (+ trade markers)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(port.index, port.values / 1e5, label="Virtual portfolio", linewidth=2, color=SERIES[0])
    ax.plot(nifty.index, nifty.values / 1e5, label=f"{bname} (same money invested)", linewidth=2,
            linestyle="--", color=INK2)
    _trade_markers(ax, led["trades"], port.rename(lambda d: d) / 1e5)
    ax.annotate(f"{port.iloc[-1] / 1e5:,.1f}", (port.index[-1], port.iloc[-1] / 1e5), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.annotate(f"{nifty.iloc[-1] / 1e5:,.1f}", (nifty.index[-1], nifty.iloc[-1] / 1e5), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_title(f"Portfolio value vs {bname} (Rs lakh)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(charts / "portfolio_vs_nifty.png")
    plt.close(fig)

    # 2) One chart per CURRENT holding (price, 20d average, your average buy price, trades, volume)
    for r in results:
        k = r["key"]
        df = per_stock_df[k].tail(130)
        avg_cost = r["metrics"]["base_price"]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        a1.plot(df.index, df["Close"], linewidth=2, label="Close", color=colors[k])
        a1.plot(df.index, df["Close"].rolling(20).mean(), linewidth=1.4, linestyle="--", color=INK2,
                label="20-day average")
        a1.axhline(avg_cost, color=MUTED, linewidth=1.3, linestyle=":", label=f"Your avg buy price ({inr(avg_cost, 2)})")
        _trade_markers(a1, led["trades"], per_stock_df[k]["Close"], key=k)
        a1.set_title(f"{r['name']} - price and volume (6 months)")
        a1.legend(loc="best", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.85)
        a2.bar(df.index, df["Volume"] / 1e5, width=0.8, color=colors[k], alpha=0.6)
        a2.set_ylabel("Volume (lakh)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(charts / f"{k}.png")
        plt.close(fig)

    # 3) Return comparison (gain = blue, loss = red; every bar carries its +/- label)
    p, b = snapshot["portfolio"], snapshot["benchmark"]
    names = [r["name"] for r in results] + [bname, "Portfolio"]
    vals = [r["metrics"]["pnl_pct"] or 0.0 for r in results] + [b["return_pct"], p["pnl_pct"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(names, vals, color=[GAIN if v >= 0 else LOSS for v in vals], width=0.55, edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title("Return (%): holdings since your buy, Nifty and portfolio since start")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
    ax.margins(y=0.2)
    fig.tight_layout()
    fig.savefig(charts / "returns.png")
    plt.close(fig)

    # 4) Allocation: one 100% bar (holdings + cash), direct labels, legend always present
    parts = [(r["name"], r["metrics"]["market_value"], colors[r["key"]]) for r in results]
    parts.append(("Cash", snapshot["portfolio"]["cash"], MUTED))
    parts = [x for x in parts if x[1] > 0]
    total = sum(x[1] for x in parts)
    if total > 0:
        fig, ax = plt.subplots(figsize=(8, 1.9))
        left = 0.0
        for name, val, col in parts:
            w = val / total * 100
            ax.barh([0], [w], left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.55, label=f"{name}  {w:.1f}%")
            if w >= 7:
                ax.text(left + w / 2, 0, f"{w:.0f}%", ha="center", va="center", fontsize=10, color="white",
                        fontweight="bold")
            left += w
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title("Where the money is (share of portfolio value)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(parts), 4), fontsize=8.5)
        fig.tight_layout()
        fig.savefig(charts / "allocation.png", bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------
# README dashboard
# ----------------------------------------------------------------------------
def _details(summary, body, open_=False):
    """Collapsible block. GitHub needs blank lines around the markdown inside <details>."""
    return [f"<details{' open' if open_ else ''}><summary>{summary}</summary>\n", *body, "\n</details>\n"]


def _report_links(cfg, key):
    reports = next((s.get("reports", {}) for s in cfg["stocks"] if s["key"] == key), {})
    out = []
    for k, label in (("one_pager", "One-page snapshot"), ("detailed", "Detailed report"), ("verdict", "Brutal verdict")):
        if reports.get(k) and (ROOT / reports[k]).exists():
            out.append((label, reports[k]))
    return out


def build_readme(cfg, snapshot):
    p, b = snapshot["portfolio"], snapshot["benchmark"]
    hold = snapshot["stocks"]
    L = []
    L.append(f"# {cfg['portfolio_name']}\n")
    L.append(f"> {DISCLAIMER}\n")
    L.append(f"**Last updated:** {snapshot['generated_at_ist']}  |  **Market data as of:** {snapshot['as_of']}  |  "
             f"**Start date:** {snapshot['base_date']}\n")
    snap_type = snapshot.get("snapshot_type", "End-of-day final snapshot")
    L.append(f"**Snapshot type:** {snap_type}\n")
    if snap_type.startswith("Live"):
        L.append("_Live snapshot: today's session is still in progress, so volume is partial and "
                 "\"Volume vs 20d avg\" will look low. The final end-of-day snapshot is published at 5:45 PM IST._\n")
    if snapshot["data_quality"]:
        L.append("### Data quality notes")
        L.extend(f"- {w}" for w in snapshot["data_quality"])
        L.append("")
    L.append("**Jump to:** [Summary](#portfolio-summary) · [Holdings](#holdings) · [Allocation](#allocation) · "
             "[Performance](#performance) · [Trade book](#trade-book) · [Companies](#company-deep-dives) · "
             "[Method](#how-this-works--data-sources)\n")

    # --- Summary
    L.append("## Portfolio summary\n")
    L.append("| Metric | Value |\n|---|---|")
    L.append(f"| Current value | **{inr(p['value'])}** |")
    L.append(f"| Total P&L | **{inr(p['pnl_inr'])} ({pct(p['pnl_pct'])})** |")
    L.append(f"| &nbsp;&nbsp;Unrealised (open positions) | {inr(p['unrealised_pnl_inr'])} |")
    L.append(f"| &nbsp;&nbsp;Realised (shares already sold) | {inr(p['realised_pnl_inr'])} |")
    L.append(f"| Today's change | {inr(p['day_pnl_inr'])} ({pct(p['day_pnl_pct'])}) |")
    L.append(f"| {b['name']} since start (same money invested) | {pct(b['return_pct'])} |")
    L.append(f"| Outperformance vs {b['name']} | {pct(p['pnl_pct'] - b['return_pct'])} |")
    L.append(f"| Cash (uninvested) | {inr(p['cash'])} |")
    L.append(f"| Virtual money put in | {inr(p['capital'])} |\n")
    L.append("![Portfolio vs Nifty](charts/portfolio_vs_nifty.png)\n")
    L.append("_▲ buy · ▼ sell_\n")

    # --- Holdings
    L.append("## Holdings\n")
    if hold:
        L.append("| Stock | Shares | Buy price | Current price | Today's close | Prev close | Day | Value | Weight | P&L | P&L % |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in hold:
            m = r["metrics"]
            L.append(f"| {r['name']} | {num(m['shares'])} | {inr(m['base_price'], 2)} | {inr(m['close'], 2)} | "
                     f"{'-' if snap_type.startswith('Live') else inr(m['close'], 2)} | "
                     f"{inr(m['prev_close'], 2) if m.get('prev_close') is not None else 'n/a'} | {pct(m['day_change_pct'])} | {inr(m['market_value'])} | {r['weight_pct']:.1f}% | "
                     f"{inr(m['pnl_inr'])} | {pct(m['pnl_pct'])} |")
        L.append("")
        pf = snapshot["performers"]
        bits = []
        if pf.get("best_since_buy"):
            bits.append(f"**Best since buy:** {pf['best_since_buy']['name']} ({pct(pf['best_since_buy']['pct'])})")
        if pf.get("worst_since_buy"):
            bits.append(f"**Worst since buy:** {pf['worst_since_buy']['name']} ({pct(pf['worst_since_buy']['pct'])})")
        if pf.get("top_mover_today"):
            bits.append(f"**Biggest mover today:** {pf['top_mover_today']['name']} ({pct(pf['top_mover_today']['pct'])})")
        if bits:
            L.append("  ·  ".join(bits) + "\n")
    else:
        L.append("_No open positions right now - all virtual money is in cash._\n")

    # --- Allocation
    L.append("## Allocation\n")
    L.append("![Allocation](charts/allocation.png)\n")
    L.append("| Sector | Value | Share |\n|---|---:|---:|")
    for s in snapshot["sectors"]:
        L.append(f"| {s['sector']} | {inr(s['value'])} | {s['weight_pct']:.1f}% |")
    L.append("")

    # --- Performance
    L.append("## Performance\n")
    L.append("![Returns](charts/returns.png)\n")

    # --- Trade book
    L.append("## Trade book\n")
    tr = list(reversed(snapshot["transactions"]))
    body = ["| Date | Stock | Type | Shares | Price | Amount | Note |", "|---|---|---|---:|---:|---:|---|"]
    for t in tr:
        sh = num(t["shares"]) if t["type"] != "DEPOSIT" else "-"
        px = (inr(t["price"], 2) + ("*" if t.get("provisional") else "")) if t["price"] else "-"
        body.append(f"| {t['date']} | {t['name'] or '-'} | {t['type']} | {sh} | {px} | {inr(t['amount'])} | "
                    f"{t['note'] or ''} |")
    if any(t.get("provisional") for t in tr):
        body.append("\n_* price is live while the market is open; it is fixed at the closing price after 3:30 PM._")
    L.extend(_details(f"<b>Transaction history</b> - {len(tr)} entries, newest first", body, open_=len(tr) <= 8))

    cl = snapshot["closed_positions"]
    if cl:
        body = ["| Stock | Bought (avg) | Sold (avg) | Held | Invested | Proceeds | Realised P&L | Return |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in cl:
            body.append(f"| {c['name']} | {c['first_buy']} @ {inr(c['avg_buy'], 2)} | {c['last_sell']} @ "
                        f"{inr(c['avg_sell'], 2)} | {c['days']} days | {inr(c['invested'])} | {inr(c['proceeds'])} | "
                        f"{inr(c['realised_pnl'])} | {pct(c['realised_pct'])} |")
    else:
        body = ["_No fully closed positions yet._"]
    L.extend(_details(f"<b>Closed positions</b> - {len(cl)}", body, open_=0 < len(cl) <= 5))

    # --- Company deep dives (open positions only)
    L.append("## Company deep-dives\n")
    if not hold:
        L.append("_No open positions._\n")
    for r in hold:
        m = r["metrics"]
        L.append(f"---\n\n### {r['name']}  _({r['sector']})_\n")
        L.append(f"![{r['name']} chart](charts/{r['key']}.png)\n")
        sym = r["ticker"].split(".")[0]
        L.append(f"[Google Finance](https://www.google.com/finance/quote/{sym}:NSE) · "
                 f"[All BSE announcements](https://www.bseindia.com/corporates/ann.html?scrip={r['bse_code']})\n")
        L.append("| Close | Day change | Day range | Volume | 20d avg volume | Volume vs 20d avg | 20d volatility (ann.) | 52w range |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        L.append(
            f"| {inr(m['close'], 2)} | {pct(m['day_change_pct'])} | {inr(m['low'], 2)} - {inr(m['high'], 2)} | "
            f"{num(m['volume'])} | {num(m['avg_volume_20d'])} | "
            f"{(str(m['volume_vs_avg']) + 'x') if m['volume_vs_avg'] else 'n/a'} | "
            f"{pct(m['volatility_20d_annualised_pct']).lstrip('+')} | {inr(m['low_52w'], 2)} - {inr(m['high_52w'], 2)} |\n"
        )
        links = _report_links(cfg, r["key"])
        f = ROOT / "analysis" / f"{r['key']}.md"
        has_md = f.exists() and "<!-- PLACEHOLDER -->" not in f.read_text(encoding="utf-8")
        L.append("**Research reports**\n")
        if links:
            L.extend(f"- [{lbl}]({path})" for lbl, path in links)
            L.append("")
        if has_md:
            L.extend(_details("Read the written analysis here", [f.read_text(encoding="utf-8")]))
        elif not links:
            L.append("_Research report coming soon._\n")
        ann = [f"- {a['date']} - [{a['title']}]({a['link']})" for a in r["announcements"]] or \
              ["- No recent filings fetched; see the link above for the full list."]
        L.extend(_details("<b>Official company announcements (BSE)</b>", ann))
        news = [f"- [{n['title']}]({n['link']}) - _{n['source']}, {n['published']}_" for n in r["news"]] or \
               ["- No fresh headlines from trusted sources in the last few days."]
        L.extend(_details("<b>Latest news (trusted publishers + official company newsroom)</b>", news))

    L.append("---\n\n## How this works & data sources\n")
    L.append("- **Prices / volume:** Yahoo Finance via the open-source `yfinance` library (unofficial; "
             "may differ slightly from NSE official data).")
    L.append("- **Trades:** every buy / sell / deposit is a line in `transactions.csv`. A trade happens at the closing "
             "price of its date (live price for today). Average-cost method; no brokerage or taxes (virtual portfolio).")
    L.append("- **Benchmark:** every rupee put into the portfolio is also put into the Nifty 50 on the same day, so the "
             "comparison is like-for-like.")
    L.append("- **News:** Google News RSS + publisher RSS feeds, filtered to a whitelist of trusted publishers, "
             "de-duplicated, last 72 hours only. Each item links to the original article.")
    L.append("- **Official filings:** BSE corporate announcements (best-effort, with link to the full list).")
    L.append("- **Automation:** GitHub Actions refreshes this dashboard on weekdays every 30 minutes from 9:30 AM to "
             "3:30 PM IST (live snapshots) and at 5:45 PM IST (end-of-day final snapshot + AI daily report); "
             "skips market holidays.")
    L.append("- **Metrics:** Volatility = 20-day std-dev of daily returns, annualised. Volume ratio = today's volume / "
             "average of previous 20 sessions.\n")
    L.append(f"> {DISCLAIMER}\n")
    (ROOT / "README.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    cfg = load_config()
    force = os.environ.get("FORCE") == "1"
    (ROOT / "data").mkdir(exist_ok=True)

    # 1) Trade book
    by_key = {s["key"]: s for s in cfg["stocks"]}
    records, warnings = ledger.load_transactions(ROOT / cfg.get("transactions_file", "transactions.csv"), set(by_key))
    traded = [s["key"] for s in cfg["stocks"] if any(r["key"] == s["key"] for r in records)]
    if not traded:
        log("transactions.csv has no usable BUY lines. Leaving everything unchanged.")
        for w in warnings:
            log("  " + w)
        write_github_output("updated", "false")
        sys.exit(0)

    # 2) Prices (every stock that was ever traded, so past trades can be valued)
    log("Fetching prices ...")
    per_stock_df, used_ticker = {}, {}
    for k in traded:
        t, df = get_history(by_key[k]["tickers"])
        if df is None:
            log(f"FATAL: could not fetch data for {by_key[k]['name']}. Leaving everything unchanged.")
            write_github_output("updated", "false")
            sys.exit(0)
        per_stock_df[k], used_ticker[k] = df, t
    _, nifty_df = get_history(cfg["benchmark"]["ticker"])
    if nifty_df is None:
        log("FATAL: could not fetch benchmark. Leaving everything unchanged.")
        write_github_output("updated", "false")
        sys.exit(0)

    closes = pd.DataFrame({k: d["Close"] for k, d in per_stock_df.items()})
    closes["__nifty"] = nifty_df["Close"]
    closes = closes.ffill().dropna()
    start = min(pd.Timestamp(cfg["base_date"]), min(r["date"] for r in records))
    closes = closes[closes.index >= start]
    if len(closes) < 1:
        log("No data on/after the start date yet. Nothing to do.")
        write_github_output("updated", "false")
        sys.exit(0)

    as_of = closes.index[-1].strftime("%Y-%m-%d")
    latest_file = ROOT / "data" / "latest.json"
    # Scheduled runs happen several times a day (live snapshots + a final 5 PM one).
    # Only skip when there is NO trading data for today (market holiday, or the data
    # provider has not published today's bar yet). Manual runs (FORCE=1) never skip.
    today_ist = datetime.now(IST).strftime("%Y-%m-%d")
    if not force and as_of != today_ist:
        log(f"No trading data for today ({today_ist}); latest is {as_of} - market holiday or data not out yet. Skipping.")
        write_github_output("updated", "false")
        return
    run_mode = os.environ.get("RUN_MODE", "final")  # "intraday" = live snapshot, "final" = end of day
    _now = datetime.now(IST)
    if run_mode == "final" and as_of == today_ist and (_now.hour, _now.minute) < (15, 30):
        run_mode = "intraday"  # manual run while the market is still open is a live snapshot, not a final one
    snapshot_type = "Live intraday snapshot" if run_mode == "intraday" else "End-of-day final snapshot"

    # 3) Ledger: holdings, cash, realised / unrealised P&L, benchmark
    nifty_close = closes.pop("__nifty")
    led = ledger.build_ledger(records, closes, nifty_close, cfg["capital_inr"], intraday=(run_mode == "intraday"))
    warnings += led["warnings"]
    if any(t.get("provisional") for t in led["trades"]):
        warnings.append("A trade dated today uses the live market price; it will be fixed at the closing price "
                        "after the market closes.")
    port, nifty_series = led["port_series"], led["nifty_series"]

    # 4) Per-holding metrics, news, filings
    results = []
    for s in cfg["stocks"]:
        k = s["key"]
        pos = led["positions"].get(k)
        if not pos:
            continue
        df = per_stock_df[k][per_stock_df[k].index <= closes.index[-1]]
        m = stock_metrics(df, pos["shares"], pos["avg_cost"])
        warnings += quality_checks(s["name"], m)
        log(f"News + filings for {s['name']} ...")
        results.append({"key": k, "name": s["name"], "sector": s.get("sector", "-"), "ticker": used_ticker[k],
                        "bse_code": s["bse_code"], "first_buy": pos["first_buy"].strftime("%Y-%m-%d"),
                        "metrics": m, "news": [], "announcements": collect_announcements(s)})
    publisher_cache = {pf["url"]: fetch_feed(pf["url"]) for pf in cfg["publisher_feeds"]} if results else {}
    for r in results:
        r["news"] = collect_news(cfg, by_key[r["key"]], publisher_cache)

    # 5) Portfolio numbers
    value = float(port.iloc[-1])
    deposits = led["total_deposits"]
    if len(port) > 1:
        prev_value = float(port.iloc[-2])
        day_pnl = value - prev_value - float(led["deposit_delta"].iloc[-1])
    else:
        prev_value, day_pnl = deposits, value - deposits
    unrealised = sum(r["metrics"]["pnl_inr"] for r in results)
    holdings_value = sum(r["metrics"]["market_value"] for r in results)
    invested = sum(pos["cost"] for pos in led["positions"].values())
    for r in results:
        r["weight_pct"] = round(r["metrics"]["market_value"] / value * 100, 2) if value else 0.0
    stale_days = (datetime.now(IST).date() - closes.index[-1].date()).days
    if stale_days > 4:
        warnings.append(f"Latest market data is {stale_days} days old - the data source may be failing.")

    sectors = {}
    for r in results:
        sectors[r["sector"]] = sectors.get(r["sector"], 0.0) + r["metrics"]["market_value"]
    if led["cash"] > 0.5:
        sectors["Cash"] = led["cash"]
    sector_rows = [{"sector": s, "value": round(v, 2), "weight_pct": round(v / value * 100, 2) if value else 0.0}
                   for s, v in sorted(sectors.items(), key=lambda kv: -kv[1])]

    performers = {}
    with_pct = [r for r in results if r["metrics"]["pnl_pct"] is not None]
    if len(with_pct) >= 2:
        best = max(with_pct, key=lambda r: r["metrics"]["pnl_pct"])
        worst = min(with_pct, key=lambda r: r["metrics"]["pnl_pct"])
        performers["best_since_buy"] = {"name": best["name"], "pct": best["metrics"]["pnl_pct"]}
        performers["worst_since_buy"] = {"name": worst["name"], "pct": worst["metrics"]["pnl_pct"]}
    movers = [r for r in results if r["metrics"]["day_change_pct"] is not None]
    if movers:
        top = max(movers, key=lambda r: abs(r["metrics"]["day_change_pct"]))
        performers["top_mover_today"] = {"name": top["name"], "pct": top["metrics"]["day_change_pct"]}

    names = {k: by_key[k]["name"] for k in by_key}
    tx_out = [{"date": t["date"].strftime("%Y-%m-%d"), "key": t["key"], "name": names.get(t["key"], ""),
               "type": t["type"], "shares": int(t["shares"]), "price": round(t["price"], 2) if t["price"] else None,
               "amount": round(t["amount"], 2), "note": t["note"], "provisional": bool(t.get("provisional"))}
              for t in led["trades"]]
    closed_out = [{"key": c["key"], "name": names.get(c["key"], c["key"]),
                   "first_buy": c["first_buy"].strftime("%Y-%m-%d"), "last_sell": c["last_sell"].strftime("%Y-%m-%d"),
                   "days": c["days"], "avg_buy": round(c["avg_buy"], 2), "avg_sell": round(c["avg_sell"], 2),
                   "invested": round(c["invested"], 2), "proceeds": round(c["proceeds"], 2),
                   "realised_pnl": round(c["realised_pnl"], 2),
                   "realised_pct": round(c["realised_pct"], 2) if c["realised_pct"] is not None else None}
                  for c in led["closed"]]
    bench_ret = float((nifty_series.iloc[-1] / deposits - 1) * 100) if deposits else 0.0

    snapshot = {
        "generated_at_ist": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"),
        "as_of": as_of,
        "snapshot_type": snapshot_type,
        "base_date": closes.index[0].strftime("%Y-%m-%d"),
        "portfolio": {
            "capital": round(deposits, 2),
            "initial_capital": cfg["capital_inr"],
            "invested": round(invested, 2),
            "holdings_value": round(holdings_value, 2),
            "cash": round(led["cash"], 2),
            "value": round(value, 2),
            "pnl_inr": round(value - deposits, 2),
            "pnl_pct": round((value / deposits - 1) * 100, 2) if deposits else 0.0,
            "unrealised_pnl_inr": round(unrealised, 2),
            "realised_pnl_inr": round(led["realised_total"], 2),
            "day_pnl_inr": round(day_pnl, 2),
            "day_pnl_pct": round(day_pnl / prev_value * 100, 2) if prev_value else 0.0,
            "open_positions": len(results),
        },
        "benchmark": {
            "name": cfg["benchmark"]["name"],
            "close": round(float(nifty_close.iloc[-1]), 2),
            "return_pct": round(bench_ret, 2),
            "day_change_pct": round(float((nifty_close.iloc[-1] / nifty_close.iloc[-2] - 1) * 100), 2)
            if len(nifty_close) > 1 else None,
        },
        "stocks": results,
        "allocation_by_sector": sector_rows,
        "sectors": sector_rows,
        "performers": performers,
        "transactions": tx_out,
        "closed_positions": closed_out,
        "data_quality": warnings,
        "sources": {
            "prices": "Yahoo Finance via yfinance (unofficial)",
            "news": "Google News RSS + publisher RSS, trusted-publisher whitelist, 72h window",
            "filings": "BSE corporate announcements (best effort)",
            "trades": "transactions.csv (average-cost method, trades at closing price of the trade date)",
        },
        "disclaimer": "Virtual portfolio for learning. Not investment advice.",
    }

    log("Drawing charts ...")
    make_charts(cfg, per_stock_df, led, results, snapshot)
    latest_file.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    build_readme(cfg, snapshot)
    log(f"Done. Portfolio value {inr(value)} ({pct(snapshot['portfolio']['pnl_pct'])}) as of {as_of}.")
    write_github_output("updated", "true")


if __name__ == "__main__":
    main()

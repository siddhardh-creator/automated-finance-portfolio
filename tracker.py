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

Edit config.json to change capital, stocks, weights or trusted news sources.
"""
import difflib
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import matplotlib
import pandas as pd
import requests
import yfinance as yf

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
        u["published"] = u["published"].astimezone(IST).strftime("%d %b %Y, %I:%M %p IST")
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
# Charts
# ----------------------------------------------------------------------------
def make_charts(cfg, per_stock_df, port_series, nifty_series, results):
    charts = ROOT / "charts"
    charts.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})

    # Portfolio vs Nifty
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(port_series.index, port_series.values / 1e5, label="Virtual portfolio", linewidth=2)
    ax.plot(nifty_series.index, nifty_series.values / 1e5, label=f"{cfg['benchmark']['name']} (same start capital)",
            linewidth=2, linestyle="--")
    ax.set_title("Portfolio value vs Nifty 50 (Rs lakh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(charts / "portfolio_vs_nifty.png")
    plt.close(fig)

    # Per-stock price + volume (last ~6 months)
    for s in cfg["stocks"]:
        df = per_stock_df[s["key"]].tail(130)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        a1.plot(df.index, df["Close"], linewidth=1.8, label="Close")
        a1.plot(df.index, df["Close"].rolling(20).mean(), linewidth=1.2, linestyle="--", label="20-day avg")
        a1.set_title(f"{s['name']} - price and volume (6 months)")
        a1.legend(loc="upper left")
        a2.bar(df.index, df["Volume"] / 1e5, width=0.8)
        a2.set_ylabel("Volume (lakh)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(charts / f"{s['key']}.png")
        plt.close(fig)

    # Return comparison bar chart
    names = [r["name"] for r in results] + [cfg["benchmark"]["name"], "Portfolio"]
    vals = [r["metrics"]["pnl_pct"] for r in results]
    vals += [float((nifty_series.iloc[-1] / nifty_series.iloc[0] - 1) * 100),
             float((port_series.iloc[-1] / port_series.iloc[0] - 1) * 100)]
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(names, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Return since start date (%)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(charts / "returns.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# README dashboard
# ----------------------------------------------------------------------------
def ensure_analysis_placeholders(cfg):
    d = ROOT / "analysis"
    d.mkdir(exist_ok=True)
    for s in cfg["stocks"]:
        f = d / f"{s['key']}.md"
        if not f.exists():
            f.write_text(
                f"<!-- PLACEHOLDER -->\n# {s['name']} - Equity Research Note\n\n"
                "_Paste your detailed analysis here (replace this whole file). "
                "Suggested sections: business overview, financials, valuation, risks, thesis._\n",
                encoding="utf-8",
            )


def build_readme(cfg, snapshot):
    p = snapshot["portfolio"]
    b = snapshot["benchmark"]
    L = []
    L.append(f"# {cfg['portfolio_name']}\n")
    L.append(f"> {DISCLAIMER}\n")
    L.append(f"**Last updated:** {snapshot['generated_at_ist']}  |  **Market data as of:** {snapshot['as_of']}  |  "
             f"**Virtual capital:** {inr(p['capital'])}\n")
    snap_type = snapshot.get("snapshot_type", "End-of-day final snapshot")
    L.append(f"**Snapshot type:** {snap_type}\n")
    if snap_type.startswith("Live"):
        L.append("_Live snapshot: today's session is still in progress, so volume is partial and "
                 "\"Volume vs 20d avg\" will look low. The final end-of-day snapshot is published at 5:00 PM IST._\n")
    if snapshot["data_quality"]:
        L.append("### Data quality notes")
        L.extend(f"- {w}" for w in snapshot["data_quality"])
        L.append("")

    L.append("## Portfolio summary\n")
    L.append("| Metric | Value |\n|---|---|")
    L.append(f"| Current value | **{inr(p['value'])}** |")
    L.append(f"| Total P&L | **{inr(p['pnl_inr'])} ({pct(p['pnl_pct'])})** |")
    L.append(f"| Today's change | {inr(p['day_pnl_inr'])} ({pct(p['day_pnl_pct'])}) |")
    L.append(f"| {b['name']} since start | {pct(b['return_pct'])} |")
    L.append(f"| Outperformance vs {b['name']} | {pct(p['pnl_pct'] - b['return_pct'])} |")
    L.append(f"| Uninvested cash | {inr(p['cash'])} |")
    L.append(f"| Start date (base prices) | {snapshot['base_date']} |\n")

    L.append("| Stock | Shares | Close | Day | Value | P&L | P&L % | Volume vs 20d avg | 20d volatility (ann.) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in snapshot["stocks"]:
        m = r["metrics"]
        L.append(
            f"| {r['name']} | {num(m['shares'])} | {inr(m['close'], 2)} | {pct(m['day_change_pct'])} | "
            f"{inr(m['market_value'])} | {inr(m['pnl_inr'])} | {pct(m['pnl_pct'])} | "
            f"{(str(m['volume_vs_avg']) + 'x') if m['volume_vs_avg'] else 'n/a'} | "
            f"{pct(m['volatility_20d_annualised_pct']).lstrip('+')} |"
        )
    L.append("")
    L.append("![Portfolio vs Nifty](charts/portfolio_vs_nifty.png)\n")
    L.append("![Returns](charts/returns.png)\n")

    for r in snapshot["stocks"]:
        m = r["metrics"]
        L.append(f"---\n\n## {r['name']}\n")
        L.append(f"![{r['name']} chart](charts/{r['key']}.png)\n")
        sym = r["ticker"].split(".")[0]
        L.append(f"[View {r['name']} on Google Finance](https://www.google.com/finance/quote/{sym}:NSE)\n")
        L.append("| Close | Day change | Day range | Volume | 20d avg volume | 52w range |")
        L.append("|---:|---:|---:|---:|---:|---:|")
        L.append(
            f"| {inr(m['close'], 2)} | {pct(m['day_change_pct'])} | {inr(m['low'], 2)} - {inr(m['high'], 2)} | "
            f"{num(m['volume'])} | {num(m['avg_volume_20d'])} | {inr(m['low_52w'], 2)} - {inr(m['high_52w'], 2)} |\n"
        )
        L.append("**Official company announcements (BSE)**\n")
        if r["announcements"]:
            for a in r["announcements"]:
                L.append(f"- {a['date']} - [{a['title']}]({a['link']})")
        L.append(f"- [All BSE announcements for {r['name']}](https://www.bseindia.com/corporates/ann.html?scrip={r['bse_code']})")
        L.append("")
        L.append("**Latest news (trusted publishers only)**\n")
        if r["news"]:
            for n in r["news"]:
                L.append(f"- [{n['title']}]({n['link']}) - _{n['source']}, {n['published']}_")
        else:
            L.append("- No fresh headlines from trusted sources in the last few days.")
        L.append("")
        f = ROOT / "analysis" / f"{r['key']}.md"
        L.append("**Research reports**\n")
        # PDF reports listed in config.json under stocks[].reports (shown only if the file exists)
        reports = next((s.get("reports", {}) for s in cfg["stocks"] if s["key"] == r["key"]), {})
        labels = [("one_pager", "One-page snapshot (quick read)"),
                  ("detailed", "Detailed initiation report")]
        pdf_links = [f"- [{label}]({reports[k]})" for k, label in labels
                     if reports.get(k) and (ROOT / reports[k]).exists()]
        if pdf_links:
            L.extend(pdf_links)
            L.append("")
        if f.exists() and "<!-- PLACEHOLDER -->" not in f.read_text(encoding="utf-8"):
            L.append(f"[Open full analysis](analysis/{r['key']}.md)\n")
            L.append("<details><summary>Read the analysis here</summary>\n")
            L.append(f.read_text(encoding="utf-8"))
            L.append("\n</details>\n")
        elif not pdf_links:
            L.append("_Detailed analysis coming soon._\n")

    L.append("---\n\n## How this works & data sources\n")
    L.append("- **Prices / volume:** Yahoo Finance via the open-source `yfinance` library (unofficial, end-of-day; "
             "may differ slightly from NSE official data).")
    L.append("- **News:** Google News RSS + publisher RSS feeds, filtered to a whitelist of trusted publishers, "
             "de-duplicated, last 72 hours only. Each item links to the original article.")
    L.append("- **Official filings:** BSE corporate announcements (best-effort, with link to the full list).")
    L.append("- **Automation:** GitHub Actions refreshes this dashboard on weekdays at 9:30 AM, 11:30 AM, 1:30 PM and "
             "3:30 PM IST (live snapshots) and at 5:00 PM IST (end-of-day final snapshot + AI daily report); "
             "skips market holidays.")
    L.append("- **Method:** Shares = (capital x weight) / closing price on the start date. Volatility = 20-day std-dev of "
             "daily returns, annualised. Volume ratio = today's volume / average of previous 20 sessions.\n")
    L.append(f"> {DISCLAIMER}\n")
    (ROOT / "README.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    cfg = load_config()
    force = os.environ.get("FORCE") == "1"
    (ROOT / "data").mkdir(exist_ok=True)
    ensure_analysis_placeholders(cfg)

    log("Fetching prices ...")
    per_stock_df, used_ticker = {}, {}
    for s in cfg["stocks"]:
        t, df = get_history(s["tickers"])
        if df is None:
            log(f"FATAL: could not fetch data for {s['name']}. Leaving everything unchanged.")
            write_github_output("updated", "false")
            sys.exit(0)
        per_stock_df[s["key"]], used_ticker[s["key"]] = df, t
    _, nifty_df = get_history(cfg["benchmark"]["ticker"])
    if nifty_df is None:
        log("FATAL: could not fetch benchmark. Leaving everything unchanged.")
        write_github_output("updated", "false")
        sys.exit(0)

    closes = pd.DataFrame({k: d["Close"] for k, d in per_stock_df.items()})
    closes["__nifty"] = nifty_df["Close"]
    closes = closes.ffill().dropna()
    closes = closes[closes.index >= pd.Timestamp(cfg["base_date"])]
    if len(closes) < 1:
        log("No data on/after base_date yet. Nothing to do.")
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
    snapshot_type = "Live intraday snapshot" if run_mode == "intraday" else "End-of-day final snapshot"

    # Positions are fixed from the base date close
    base_row = closes.iloc[0]
    capital = cfg["capital_inr"]
    shares = {}
    for s in cfg["stocks"]:
        shares[s["key"]] = int((capital * s["weight"]) // base_row[s["key"]])
    invested = sum(shares[s["key"]] * base_row[s["key"]] for s in cfg["stocks"])
    cash = capital - invested

    port_series = sum(closes[k] * n for k, n in shares.items()) + cash
    nifty_series = closes["__nifty"] / closes["__nifty"].iloc[0] * capital

    results, warnings = [], []
    day_pnl = 0.0
    for s in cfg["stocks"]:
        k = s["key"]
        df = per_stock_df[k][per_stock_df[k].index <= closes.index[-1]]
        m = stock_metrics(df, shares[k], float(base_row[k]))
        warnings += quality_checks(s["name"], m)
        if m["prev_close"]:
            day_pnl += shares[k] * (m["close"] - m["prev_close"])
        log(f"News + filings for {s['name']} ...")
        results.append({"key": k, "name": s["name"], "ticker": used_ticker[k], "bse_code": s["bse_code"],
                        "metrics": m, "news": [], "announcements": collect_announcements(s)})

    # Publisher feeds are downloaded once and reused
    publisher_cache = {pf["url"]: fetch_feed(pf["url"]) for pf in cfg["publisher_feeds"]}
    for s, r in zip(cfg["stocks"], results):
        r["news"] = collect_news(cfg, s, publisher_cache)

    value = float(port_series.iloc[-1])
    prev_value = float(port_series.iloc[-2]) if len(port_series) > 1 else capital
    stale_days = (datetime.now(IST).date() - closes.index[-1].date()).days
    if stale_days > 4:
        warnings.append(f"Latest market data is {stale_days} days old - the data source may be failing.")

    snapshot = {
        "generated_at_ist": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"),
        "as_of": as_of,
        "snapshot_type": snapshot_type,
        "base_date": closes.index[0].strftime("%Y-%m-%d"),
        "portfolio": {
            "capital": capital,
            "invested": round(invested, 2),
            "cash": round(cash, 2),
            "value": round(value, 2),
            "pnl_inr": round(value - capital, 2),
            "pnl_pct": round((value / capital - 1) * 100, 2),
            "day_pnl_inr": round(value - prev_value, 2),
            "day_pnl_pct": round((value / prev_value - 1) * 100, 2),
        },
        "benchmark": {
            "name": cfg["benchmark"]["name"],
            "close": round(float(closes["__nifty"].iloc[-1]), 2),
            "return_pct": round(float((nifty_series.iloc[-1] / capital - 1) * 100), 2),
            "day_change_pct": round(float((closes["__nifty"].iloc[-1] / closes["__nifty"].iloc[-2] - 1) * 100), 2)
            if len(closes) > 1 else None,
        },
        "stocks": results,
        "data_quality": warnings,
        "sources": {
            "prices": "Yahoo Finance via yfinance (end-of-day, unofficial)",
            "news": "Google News RSS + publisher RSS, trusted-publisher whitelist, 72h window",
            "filings": "BSE corporate announcements (best effort)",
        },
        "disclaimer": "Virtual portfolio for learning. Not investment advice.",
    }

    log("Drawing charts ...")
    make_charts(cfg, per_stock_df, port_series, nifty_series, results)
    latest_file.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    build_readme(cfg, snapshot)
    log(f"Done. Portfolio value {inr(value)} ({pct(snapshot['portfolio']['pnl_pct'])}) as of {as_of}.")
    write_github_output("updated", "true")


if __name__ == "__main__":
    main()

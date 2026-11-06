"""
Corporate-action detection (splits, bonus issues, dividends) for the virtual portfolio.

It reads the actions Yahoo Finance reports for the stocks we hold and appends ONE event line per action to
transactions.csv (append-only: nothing already in the file is ever edited). ledger.py then does the maths.

Safety rules (a wrong event would distort the portfolio, so we are strict):
* only actions whose ex-date is inside a short recent window and not before the portfolio start date;
* splits/bonus: only for stocks traded before the ex-date (even if sold since, so old trades are priced right);
  dividends: only for stocks actually held BEFORE the ex-date;
* never a duplicate: an event for the same date + stock already in the file is skipped;
* implausible values (multiplier outside 0.02..50, dividend above 25% of the price) are not recorded;
* any problem here is swallowed by the caller: the portfolio is then calculated exactly as before.
"""
import math

import pandas as pd

SPLIT_MIN, SPLIT_MAX = 0.02, 50.0
MAX_DIV_FRACTION = 0.25


def _clean_index(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def fetch_actions(yf, ticker):
    """Return (splits, dividends) as Series indexed by ex-date, or (None, None) if the data cannot be read."""
    try:
        t = yf.Ticker(ticker)
        return _clean_index(t.splits), _clean_index(t.dividends)
    except Exception:
        return None, None


def held_before(trades, key, date):
    """Shares of `key` held at the start of `date` (before that day's trades), from the ledger trade list."""
    n = 0
    for t in trades:
        if t["key"] != key or t["effective"] >= date:
            continue
        if t["type"] == "BUY":
            n += t["shares"]
        elif t["type"] == "SELL":
            n -= t["shares"]
        elif t.get("event") in ("SPLIT", "BONUS"):
            n += t["shares"]          # shares added (or removed) by the event
    return n


def traded_before(trades, key, date):
    """True if the stock had any buy/sell before `date` (even if it has been sold since)."""
    return any(t["key"] == key and t["effective"] < date and t["type"] in ("BUY", "SELL") for t in trades)


def find_events(actions, trades, existing, close_on, window_start, last_day):
    """
    actions  : {key: (splits Series, dividends Series)}
    trades   : led["trades"]
    existing : set of (date, key) already recorded as SPLIT/BONUS/DIVIDEND
    close_on : function (key, date) -> closing price that day, or None
    Returns (events, notices). Each event: dict(date, key, type, value, text).
    """
    events, notices = [], []
    for key, (splits, divs) in actions.items():
        if splits is not None:
            for d, v in splits.items():
                if not (window_start <= d <= last_day) or not v or v == 1.0:
                    continue
                if ("SPLIT", d, key) in existing:
                    continue
                # Recorded even if the stock was sold earlier: old trades priced from the (adjusted) history
                # need the split to be converted back to the scale they were made at. The ledger ignores it
                # for share counts when nothing is held.
                if not traded_before(trades, key, d):
                    continue
                f = float(v)
                if not (SPLIT_MIN <= f <= SPLIT_MAX) or math.isnan(f):
                    notices.append(f"{key}: split/bonus multiplier {f:g} on {d:%Y-%m-%d} looks implausible - not recorded.")
                    continue
                kind = "reverse split" if f < 1 else f"{f:g}-for-1 split/bonus"
                events.append({"date": d, "key": key, "type": "SPLIT", "value": f,
                               "text": f"Auto-detected from Yahoo Finance: {kind}, ex-date {d:%Y-%m-%d}"})
        if divs is not None:
            for d, v in divs.items():
                if not (window_start <= d <= last_day) or not v or v <= 0:
                    continue
                if ("DIVIDEND", d, key) in existing:
                    continue
                held = held_before(trades, key, d)
                if held < 1:
                    continue
                dps = float(v)
                px = close_on(key, d)
                if px and dps > MAX_DIV_FRACTION * px:
                    notices.append(f"{key}: dividend {dps:g} on {d:%Y-%m-%d} is above {MAX_DIV_FRACTION:.0%} of the "
                                   "price - not recorded, please check.")
                    continue
                events.append({"date": d, "key": key, "type": "DIVIDEND", "value": dps,
                               "text": f"Auto-detected from Yahoo Finance: dividend Rs {dps:g} per share, ex-date {d:%Y-%m-%d}"})
    events.sort(key=lambda e: (e["date"], e["key"], e["type"]))
    return events, notices


def existing_event_keys(records):
    """(type-group, date, key) of event lines already in the trade book. SPLIT and BONUS count as the same thing."""
    out = set()
    for r in records:
        if r["type"] in ("SPLIT", "BONUS"):
            out.add(("SPLIT", r["date"], r["key"]))
        elif r["type"] == "DIVIDEND":
            out.add(("DIVIDEND", r["date"], r["key"]))
    return out


def csv_lines(events, reinvest=False, close_on=None, trades=None):
    """Event lines (and optional dividend-reinvestment BUY lines) in transactions.csv format."""
    lines = []
    for e in events:
        d = f"{e['date']:%Y-%m-%d}"
        note = e["text"].replace(",", ";")
        if e["type"] == "SPLIT":
            lines.append(f"{d},{e['key']},SPLIT,{e['value']:g},,,{note}")
        else:
            lines.append(f"{d},{e['key']},DIVIDEND,,,{e['value']:g},{note}")
            if reinvest and close_on and trades is not None:
                px = close_on(e["key"], e["date"])
                cash = e["value"] * held_before(trades, e["key"], e["date"])
                n = int(cash // px) if px else 0
                if n >= 1:
                    lines.append(f"{d},{e['key']},BUY,{n},,,Dividend reinvested (DRIP) at the closing price")
    return lines


def append_lines(path, lines):
    """Append lines to the trade book without touching what is already there."""
    text = path.read_text(encoding="utf-8")
    sep = "" if text.endswith("\n") else "\n"
    path.write_text(text + sep + "\n".join(lines) + "\n", encoding="utf-8")

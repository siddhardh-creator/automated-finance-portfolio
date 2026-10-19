"""
Ledger engine for the virtual portfolio.

The portfolio is described by a simple trade book (transactions.csv):

    date,stock,type,shares,amount,price,note
    2026-10-16,hdfc,BUY,,3333000,,Initial allocation
    2026-11-03,zomato,SELL,ALL,,,Valuation looks stretched

* type    : BUY, SELL or DEPOSIT (extra virtual money added to the portfolio)
* stock   : the "key" of the company in config.json (not needed for DEPOSIT)
* shares  : number of shares. For SELL you may write ALL.
* amount  : alternative to shares - rupees to invest (BUY) / raise (SELL) / add (DEPOSIT)
* price   : optional. Leave blank to use that day's closing price (live price for today).
* note    : optional free text
* Lines starting with # are ignored.

Accounting: average-cost method. Cash starts as the initial capital from config.json.
Buys reduce cash, sells add cash. No brokerage / taxes (virtual portfolio).

This module only does maths. It has no network access, so it is easy to test.
"""
import math
import re

import pandas as pd

VALID_TYPES = {"BUY", "SELL", "DEPOSIT"}


# ----------------------------------------------------------------------------
# Reading + validating the trade book
# ----------------------------------------------------------------------------
def _num(x):
    """'1,23,000' / '₹5000' / '' -> float or None."""
    s = re.sub(r"[₹,\s]", "", str(x or ""))
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_transactions(path, valid_keys):
    """Return (records, warnings). Bad lines are skipped with a plain-English warning."""
    warnings = []
    if not path.exists():
        return [], [f"{path.name} not found - the portfolio has no trades yet."]
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    except Exception as e:
        return [], [f"Could not read {path.name}: {e}"]
    raw.columns = [c.strip().lower() for c in raw.columns]
    for col in ("date", "stock", "type", "shares", "amount", "price", "note"):
        if col not in raw.columns:
            raw[col] = ""
    records = []
    for i, row in raw.iterrows():
        line = i + 2  # header is line 1
        if str(row["date"]).strip().startswith("#") or not any(str(v).strip() for v in row.values):
            continue
        ds = str(row["date"]).strip()
        # YYYY-MM-DD first; otherwise treat 05/11/2026 the Indian way (day first)
        d = pd.to_datetime(ds, format="%Y-%m-%d", errors="coerce")
        if pd.isna(d):
            d = pd.to_datetime(ds, errors="coerce", dayfirst=True)
        if pd.isna(d):
            warnings.append(f"{path.name} line {line}: date '{row['date']}' not understood - line skipped (use YYYY-MM-DD).")
            continue
        typ = str(row["type"]).strip().upper()
        if typ not in VALID_TYPES:
            warnings.append(f"{path.name} line {line}: type '{row['type']}' must be BUY, SELL or DEPOSIT - line skipped.")
            continue
        key = str(row["stock"]).strip().lower()
        if typ != "DEPOSIT" and key not in valid_keys:
            warnings.append(f"{path.name} line {line}: stock '{row['stock']}' is not in config.json - "
                            f"add the company there first. Line skipped.")
            continue
        sh_raw = str(row["shares"]).strip()
        sell_all = sh_raw.upper() == "ALL"
        shares = None if sell_all else _num(sh_raw)
        amount, price = _num(row["amount"]), _num(row["price"])
        if any(v is not None and isinstance(v, float) and math.isnan(v) for v in (shares, amount, price)):
            warnings.append(f"{path.name} line {line}: a number could not be read - line skipped.")
            continue
        if typ == "DEPOSIT" and not (amount and amount > 0):
            warnings.append(f"{path.name} line {line}: DEPOSIT needs a positive amount - line skipped.")
            continue
        if typ == "BUY" and not ((shares and shares > 0) or (amount and amount > 0)):
            warnings.append(f"{path.name} line {line}: BUY needs shares or amount - line skipped.")
            continue
        if typ == "SELL" and not (sell_all or (shares and shares > 0) or (amount and amount > 0)):
            warnings.append(f"{path.name} line {line}: SELL needs shares (or ALL) or amount - line skipped.")
            continue
        if price is not None and price <= 0:
            warnings.append(f"{path.name} line {line}: price must be positive - line skipped.")
            continue
        records.append({"date": d.normalize(), "key": key if typ != "DEPOSIT" else "", "type": typ,
                        "shares": shares, "sell_all": sell_all, "amount": amount, "price": price,
                        "note": str(row["note"]).strip(), "line": line, "order": len(records)})
    records.sort(key=lambda r: (r["date"], r["order"]))
    return records, warnings


# ----------------------------------------------------------------------------
# The ledger
# ----------------------------------------------------------------------------
def build_ledger(records, closes, nifty_close, capital, intraday=False):
    """
    records     : output of load_transactions
    closes      : DataFrame, index = trading days, one column per stock key (daily close, no gaps)
    nifty_close : Series on the same index
    capital     : initial virtual cash, put in on the first row of `closes`

    A trade dated D happens at the close of the first trading day on/after D
    (live price if that day is still in progress).
    """
    idx = closes.index
    keys = list(closes.columns)
    shares_delta = pd.DataFrame(0.0, index=idx, columns=keys)
    cash_delta = pd.Series(0.0, index=idx)
    deposit_delta = pd.Series(0.0, index=idx)
    cash_delta.iloc[0] += capital
    deposit_delta.iloc[0] += capital

    cash = float(capital)
    pos = {}            # key -> open position
    trades, closed, warns = [], [], []
    realised_total = 0.0
    realised_by_key = {}

    for rec in records:
        i = int(idx.searchsorted(rec["date"]))
        if i >= len(idx):
            warns.append(f"Trade on {rec['date']:%d %b %Y} ({rec['type']} {rec['key']}) is not applied yet - "
                         "no market data for that date so far.")
            continue
        eff = idx[i]
        key, typ = rec["key"], rec["type"]
        base = {"date": rec["date"], "effective": eff, "key": key, "type": typ, "note": rec["note"]}

        if typ == "DEPOSIT":
            cash += rec["amount"]
            cash_delta.iloc[i] += rec["amount"]
            deposit_delta.iloc[i] += rec["amount"]
            trades.append({**base, "shares": 0, "price": None, "amount": float(rec["amount"]), "provisional": False})
            continue

        provisional = rec["price"] is None and i == len(idx) - 1 and intraday
        price = float(rec["price"]) if rec["price"] is not None else float(closes[key].iloc[i])

        if typ == "BUY":
            shares = int(rec["shares"]) if rec["shares"] else int(rec["amount"] // price)
            if shares < 1:
                warns.append(f"Line {rec['line']}: amount too small to buy even 1 share of {key} - skipped.")
                continue
            if shares * price > cash + 1e-6:
                affordable = int(cash // price)
                if affordable < 1:
                    warns.append(f"Line {rec['line']}: not enough cash to buy {key} - skipped.")
                    continue
                warns.append(f"Line {rec['line']}: not enough cash for {shares} shares of {key}; "
                             f"bought {affordable} instead.")
                shares = affordable
            cost = shares * price
            cash -= cost
            cash_delta.iloc[i] -= cost
            shares_delta.iloc[i, keys.index(key)] += shares
            p = pos.setdefault(key, {"shares": 0, "cost": 0.0, "first_buy": eff, "cyc_cost": 0.0,
                                     "cyc_bought": 0, "cyc_proceeds": 0.0, "cyc_sold": 0})
            p["shares"] += shares
            p["cost"] += cost
            p["cyc_cost"] += cost
            p["cyc_bought"] += shares
            trades.append({**base, "shares": shares, "price": price, "amount": cost, "provisional": provisional})

        else:  # SELL
            p = pos.get(key)
            if not p or p["shares"] < 1:
                warns.append(f"Line {rec['line']}: SELL {key} but no shares are held on that date - skipped.")
                continue
            if rec["sell_all"]:
                shares = p["shares"]
            elif rec["shares"]:
                shares = int(rec["shares"])
            else:
                shares = int(rec["amount"] // price)
            if shares > p["shares"]:
                warns.append(f"Line {rec['line']}: tried to sell {shares} {key} but only {p['shares']} held; "
                             f"sold {p['shares']}.")
                shares = p["shares"]
            if shares < 1:
                warns.append(f"Line {rec['line']}: amount too small to sell even 1 share of {key} - skipped.")
                continue
            proceeds = shares * price
            avg = p["cost"] / p["shares"]
            pnl = proceeds - avg * shares
            realised_total += pnl
            realised_by_key[key] = realised_by_key.get(key, 0.0) + pnl
            cash += proceeds
            cash_delta.iloc[i] += proceeds
            shares_delta.iloc[i, keys.index(key)] -= shares
            p["shares"] -= shares
            p["cost"] -= avg * shares
            p["cyc_proceeds"] += proceeds
            p["cyc_sold"] += shares
            trades.append({**base, "shares": shares, "price": price, "amount": proceeds,
                           "provisional": provisional, "realised_pnl": pnl})
            if p["shares"] == 0:
                days = (eff - p["first_buy"]).days
                closed.append({
                    "key": key, "first_buy": p["first_buy"], "last_sell": eff, "days": days,
                    "avg_buy": p["cyc_cost"] / p["cyc_bought"], "avg_sell": p["cyc_proceeds"] / p["cyc_sold"],
                    "invested": p["cyc_cost"], "proceeds": p["cyc_proceeds"],
                    "realised_pnl": p["cyc_proceeds"] - p["cyc_cost"],
                    "realised_pct": (p["cyc_proceeds"] / p["cyc_cost"] - 1) * 100 if p["cyc_cost"] else None,
                })
                del pos[key]

    shares_df = shares_delta.cumsum()
    cash_s = cash_delta.cumsum()
    holdings_value = (shares_df * closes).sum(axis=1)
    port_series = holdings_value + cash_s
    deposits_cum = deposit_delta.cumsum()

    # Benchmark: every deposit rupee is put into Nifty on the day it arrives
    units = (deposit_delta / nifty_close).cumsum()
    nifty_series = units * nifty_close

    positions = {}
    for k, p in pos.items():
        positions[k] = {"shares": int(p["shares"]), "avg_cost": p["cost"] / p["shares"], "cost": p["cost"],
                        "first_buy": p["first_buy"]}
    return {
        "trades": trades, "closed": closed, "positions": positions, "warnings": warns,
        "cash": cash, "realised_total": realised_total, "realised_by_key": realised_by_key,
        "port_series": port_series, "nifty_series": nifty_series, "deposits_cum": deposits_cum,
        "deposit_delta": deposit_delta, "total_deposits": float(deposits_cum.iloc[-1]),
    }

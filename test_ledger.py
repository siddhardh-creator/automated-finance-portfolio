"""Offline tests for ledger.py (no network). Run:  python test_ledger.py"""
import io
import math
import pathlib
import tempfile

import pandas as pd

import ledger


def run(csv_text, closes, cash=1_000_000.0):
    d = pathlib.Path(tempfile.mkdtemp()) / "t.csv"
    d.write_text("date,stock,type,shares,amount,price,note\n" + csv_text, encoding="utf-8")
    recs, warns = ledger.load_transactions(d, set(closes.columns))
    nifty = pd.Series(100.0, index=closes.index)
    return ledger.build_ledger(recs, closes, nifty, cash), warns


def days(n, start="2026-11-02"):
    return pd.bdate_range(start, periods=n)


def close(x):
    return abs(x) < 1e-6


# 1) Split: 100 shares at 120, 2-for-1 split. Yahoo history is split-adjusted, so earlier closes are halved.
idx = days(6)
closes = pd.DataFrame({"a": [60, 60, 60, 60, 60, 60]}, index=idx, dtype=float)
led, w = run(f"{idx[0]:%F},a,BUY,100,,120,x\n{idx[3]:%F},a,SPLIT,2,,,split\n", closes, cash=12000)
p = led["positions"]["a"]
assert p["shares"] == 200 and close(p["avg_cost"] - 60) and close(p["cost"] - 12000), p
assert close(led["port_series"].iloc[-1] - 12000), led["port_series"].tolist()
# value line must not jump on the ex-date (index 3)
ps = led["port_series"].tolist()
assert all(close(v - 12000) for v in ps[1:]), ps
assert not w, w

# 2) Bonus 1:1 (= x2) behaves the same; a later partial sell uses the new share count.
led, w = run(f"{idx[0]:%F},a,BUY,100,,120,x\n{idx[3]:%F},a,BONUS,2,,,bonus\n{idx[4]:%F},a,SELL,50,,60,part\n", closes, cash=12000)
assert led["positions"]["a"]["shares"] == 150
assert close(led["positions"]["a"]["avg_cost"] - 60)
assert close(led["realised_total"]), led["realised_total"]       # sold at the 60 average cost

# 3) Dividend: cash up, shares/avg cost unchanged, total P&L includes it.
closes2 = pd.DataFrame({"a": [100.0] * 6}, index=idx)
led, w = run(f"{idx[0]:%F},a,BUY,100,,100,x\n{idx[2]:%F},a,DIVIDEND,,,5,div\n", closes2, cash=10000)
assert led["positions"]["a"]["shares"] == 100 and close(led["positions"]["a"]["avg_cost"] - 100)
assert close(led["dividends_total"] - 500) and close(led["cash"] - 500)
assert close(led["port_series"].iloc[-1] - 10500)
assert close(led["port_series"].iloc[2] - 10500) and close(led["port_series"].iloc[1] - 10000)

# 4) Event on a stock not held (never bought / already sold) is ignored, no cash, no shares.
led, w = run(f"{idx[0]:%F},a,BUY,10,,100,x\n{idx[1]:%F},a,SELL,ALL,,100,x\n{idx[2]:%F},a,DIVIDEND,,,5,div\n"
             f"{idx[2]:%F},a,SPLIT,2,,,s\n", closes2, cash=10000)
assert "a" not in led["positions"] and close(led["dividends_total"]) and close(led["cash"] - 10000)

# 5) Same-day order: a BUY on the ex-date is on the new scale and is NOT multiplied by the split.
led, w = run(f"{idx[0]:%F},a,BUY,100,,120,x\n{idx[3]:%F},a,BUY,10,,60,y\n{idx[3]:%F},a,SPLIT,2,,,s\n", closes, cash=20000)
assert led["positions"]["a"]["shares"] == 210, led["positions"]
assert close(led["positions"]["a"]["cost"] - 12600)

# 6) Fractional entitlement: only whole shares kept, with a plain warning.
led, w = run(f"{idx[0]:%F},a,BUY,101,,100,x\n{idx[3]:%F},a,BONUS,1.5,,,b\n", closes2, cash=20000)
assert led["positions"]["a"]["shares"] == 151 and any("whole shares" in x for x in led["warnings"])

# 7) Bad event lines are rejected, not crashed on.
led, w = run(f"{idx[0]:%F},a,BUY,10,,100,x\n{idx[1]:%F},a,SPLIT,,,,s\n{idx[1]:%F},a,SPLIT,1,,,s\n{idx[1]:%F},a,DIVIDEND,,,,d\n", closes2, cash=10000)
assert len(w) == 3, w

# 8) Closed position average prices stay in the new share scale after a split.
led, w = run(f"{idx[0]:%F},a,BUY,100,,120,x\n{idx[3]:%F},a,SPLIT,2,,,s\n{idx[4]:%F},a,SELL,ALL,,60,x\n", closes, cash=12000)
c = led["closed"][0]
assert close(c["avg_buy"] - 60) and close(c["avg_sell"] - 60) and close(c["realised_pnl"]), c


# 10) A BUY with NO price (priced from the split-adjusted history) before a split must still end at the same
#     money invested and the split-multiplied share count - not multiplied twice.
led, w = run(f"{idx[0]:%F},a,BUY,,6000,,by amount\n{idx[3]:%F},a,SPLIT,2,,,s\n", closes, cash=6000)
# history is adjusted (60); original price was 120 -> 6000/120 = 50 shares, x2 = 100 shares at 60
assert led["positions"]["a"]["shares"] == 100 and close(led["positions"]["a"]["cost"] - 6000), led["positions"]
led, w = run(f"{idx[0]:%F},a,BUY,50,,,no price\n{idx[3]:%F},a,SPLIT,2,,,s\n", closes, cash=6000)
assert led["positions"]["a"]["shares"] == 100 and close(led["positions"]["a"]["cost"] - 6000), led["positions"]

# 11) Stock sold BEFORE a later split: old blank-price buy + explicit sell stay consistent (no distortion).
led, w = run(f"{idx[0]:%F},a,BUY,,6000,,by amount\n{idx[1]:%F},a,SELL,ALL,,130,sold\n{idx[4]:%F},a,SPLIT,2,,,s\n", closes, cash=6000)
c = led["closed"][0]
assert close(c["invested"] - 6000) and close(c["proceeds"] - 6500) and close(c["realised_pnl"] - 500), c
assert close(led["cash"] - 6500)

# 9) Regression: the real trade book gives exactly the same result as before the corporate-action code.
import importlib.util, json
root = pathlib.Path(__file__).parent
cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
keys = [s["key"] for s in cfg["stocks"]]
recs, _ = ledger.load_transactions(root / "transactions.csv", set(keys))
idx2 = pd.bdate_range("2026-10-16", "2026-11-20")
import numpy as np
rng = np.random.default_rng(1)
cl = pd.DataFrame({k: 100 + np.cumsum(rng.normal(0, 1, len(idx2))) for k in keys}, index=idx2)
nf = pd.Series(100 + np.cumsum(rng.normal(0, 1, len(idx2))), index=idx2)
new = ledger.build_ledger(recs, cl, nf, cfg["capital_inr"])
old_path = pathlib.Path(__import__("os").environ.get("OLD_LEDGER", ""))
if old_path.is_file():
    spec = importlib.util.spec_from_file_location("ledger_old", old_path)
    lo = importlib.util.module_from_spec(spec); spec.loader.exec_module(lo)
    recs_o, _ = lo.load_transactions(root / "transactions.csv", set(keys))
    old = lo.build_ledger(recs_o, cl, nf, cfg["capital_inr"])
    assert old["cash"] == new["cash"] and old["realised_total"] == new["realised_total"]
    assert old["port_series"].equals(new["port_series"]) and old["positions"] == new["positions"]
    assert len(old["trades"]) == len(new["trades"]) and old["closed"] == new["closed"]
    print("regression vs old ledger: identical")

print("all ledger tests passed")

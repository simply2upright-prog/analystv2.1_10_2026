#!/usr/bin/env python3
"""
divflow_history.py - lange Kurshistorie (10 Jahre, Wochenkurse), 4/4-Einstiege und Aktiensplits fuer den Options-Screener.
Laeuft in der GitHub Action (Yahoo sperrt den Webhoster) und schreibt divflow_history.json.gz.
Universum: divflow_options.txt (Yahoo-Symbole, einer pro Zeile) plus divflow_extra.txt.
Einstiege = dieselbe Logik wie divflow_signals.py (Zone beginnt / wird auf 4/4 hochgestuft, 10 Handelstage Cooldown), aber ueber die gesamte Historie.
"""
from __future__ import annotations
import argparse, gzip, json, logging, os, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import engine
import divflow_signals as ds

log = logging.getLogger("divflow_history")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "divflow_history.json.gz")
COOLDOWN = ds.COOLDOWN


def universe() -> list[str]:
    out = []
    for name in ("divflow_options.txt", "divflow_extra.txt"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            out += [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    return sorted(set(out))


def entries(df: pd.DataFrame) -> list[dict]:
    """Alle Einstiege ueber die ganze Historie (Format wie 'sigs' in divflow_signals)."""
    df = engine.add_strategy_indicators(df.copy())
    n = len(df)
    turns = engine.macd_turn_series(df["OSMA"].to_numpy(float))
    rsi, fast, slow, cci = (df[c].to_numpy(float) for c in ("StochRSI_70", "Stoch_Fast", "Stoch_Slow", "CCI_20"))
    close = df["Close"].to_numpy(float)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    nn = lambda v: None if not np.isfinite(v) else float(v)
    last = {"buy": -99, "sell": -99, "buy4": -99, "sell4": -99}
    ev = []
    for i in range(40, n):
        t = {1: "up", -1: "down", 0: None}[int(turns[i])]
        k = engine.classify_signal(nn(rsi[i]), nn(fast[i]), nn(slow[i]), nn(cci[i]), t)
        sd = k["side"]
        if not sd:
            continue
        sc = k["buy_score"] if sd == "buy" else k["sell_score"]
        new = i - last[sd] > COOLDOWN
        upg = sc == 4 and i - last[sd + "4"] > COOLDOWN
        if new or upg:
            ev.append({"d": dates[i], "side": sd, "sc": sc, "p": ds._f(close[i], 3)})
        last[sd] = i
        if sc == 4:
            last[sd + "4"] = i
    return ev


def record(t: str, df: pd.DataFrame, splits) -> dict | None:
    df = engine.clean_price_df(df)
    if df is None or len(df) < 120 or "Close" not in df.columns:
        return None
    ev = entries(df)
    wk = df["Close"].resample("W-FRI").last().dropna()
    sp = []
    try:
        for d, v in (splits[splits > 0].items() if splits is not None else []):
            sp.append({"d": pd.Timestamp(d).strftime("%Y-%m-%d"), "r": round(float(v), 4)})
    except Exception:
        pass
    return {"first": df.index[0].strftime("%Y-%m-%d"), "last": df.index[-1].strftime("%Y-%m-%d"), "n": int(len(df)),
            "hi": ds._f(df["High"].max() if "High" in df.columns else df["Close"].max(), 3), "lo": ds._f(df["Low"].min() if "Low" in df.columns else df["Close"].min(), 3),
            "w": {"d": [d.strftime("%Y-%m-%d") for d in wk.index], "c": [ds._f(x, 3) for x in wk.values]},
            "sigs": ev, "splits": sp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    tickers = universe()
    if a.limit:
        tickers = tickers[:a.limit]
    log.info("%d Ticker", len(tickers))
    out, bad = {}, []
    for t in tickers:
        try:
            tk = engine.yf.Ticker(t)
            raw = tk.history(period="10y", auto_adjust=False)   # Kurse sind splitbereinigt, Dividenden nicht eingerechnet
            rec = record(t, raw, tk.splits if hasattr(tk, "splits") else None)
            if rec:
                out[t] = rec
            else:
                bad.append(t)
        except Exception as e:
            log.warning("%s: %s", t, e)
            bad.append(t)
        time.sleep(0.4)
    payload = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "count": len(out), "no_data": bad, "tickers": out}
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    log.info("%d Ticker, %d ohne Daten -> %s (%.1f MB)", len(out), len(bad), OUT, os.path.getsize(OUT) / 1e6)


if __name__ == "__main__":
    main()

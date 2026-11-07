# divflow_signals.py — rechnet die Strategie-Signale (engine.py) für alle Ticker und schreibt divflow_signals.json.gz
# DivFlow (PHP-Hosting) lädt die Datei per Cron; Yahoo wird nur hier, auf dem GitHub-Runner, abgefragt.
#   python divflow_signals.py                 -> alle Ticker aus ticker_meta.json (+ divflow_extra.txt, ein Ticker pro Zeile)
#   python divflow_signals.py --limit 30      -> Test mit 30 Tickern
import argparse
import gzip
import json
import logging
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import engine

log = logging.getLogger("divflow")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "divflow_signals.json.gz")
HIST = 180        # Tage Verlauf für Chart und Einstiege
COOLDOWN = 10     # Handelstage zwischen zwei Einstiegen derselben Seite


def _f(x, nd=4):
    try:
        x = float(x)
    except Exception:
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)


def build_record(df: pd.DataFrame) -> dict | None:
    """Ein Ticker -> Datensatz im Format, das DivFlow (includes/signals.php, tech_import) erwartet."""
    df = engine.clean_price_df(df)
    if df is None or len(df) < 60 or not {"High", "Low", "Close"} <= set(df.columns):
        return None
    df = engine.add_strategy_indicators(df.copy())
    n = len(df)
    turns = engine.macd_turn_series(df["OSMA"].to_numpy(float))
    rsi, fast, slow, cci = (df[c].to_numpy(float) for c in ("StochRSI_70", "Stoch_Fast", "Stoch_Slow", "CCI_20"))
    close, high, low = df["Close"].to_numpy(float), df["High"].to_numpy(float), df["Low"].to_numpy(float)
    vol = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.zeros(n)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    nn = lambda v: None if not np.isfinite(v) else float(v)

    def at(i):
        t = {1: "up", -1: "down", 0: None}[int(turns[i])]
        crit = engine.evaluate_criteria(nn(rsi[i]), nn(fast[i]), nn(slow[i]), nn(cci[i]), t)
        cl = engine.classify_signal(nn(rsi[i]), nn(fast[i]), nn(slow[i]), nn(cci[i]), t)
        return t, crit, cl

    t, crit, cl = at(n - 1)
    # Einstiege: erster Tag einer Zone (Cooldown), plus eigene Markierung, wenn die Zone auf 4/4 hochgestuft wird
    last = {"buy": -99, "sell": -99, "buy4": -99, "sell4": -99}
    ev = []
    for i in range(max(40, n - HIST), n):
        _, c2, k = at(i)
        sd = k["side"]
        if not sd:
            continue
        sc = k["buy_score"] if sd == "buy" else k["sell_score"]
        new = i - last[sd] > COOLDOWN
        upg = sc == 4 and i - last[sd + "4"] > COOLDOWN
        if new or upg:
            tt = {1: "up", -1: "down", 0: None}[int(turns[i])]
            ev.append({"i": i - (n - HIST), "d": dates[i], "side": sd, "label": k["label"], "p": _f(close[i], 3), "sc": sc,
                       "ok": [bool(v) for v in c2[sd].values()], "rsi": _f(rsi[i], 3), "f": _f(fast[i], 1), "s": _f(slow[i], 1),
                       "cci": _f(cci[i], 0), "macd": tt})
        last[sd] = i
        if sc == 4:
            last[sd + "4"] = i
    sma = float(np.nanmean(close[-min(200, n - 1):]))
    cutoff = dates[-1]
    buys = [e for e in ev if e["side"] == "buy"]
    sells = [e for e in ev if e["side"] == "sell"]
    lb = buys[-1] if buys else None
    if lb and (datetime.strptime(cutoff, "%Y-%m-%d") - datetime.strptime(lb["d"], "%Y-%m-%d")).days > 90:
        lb = None
    perf = lambda k: _f((close[-1] / close[-1 - k] - 1) * 100, 2) if n > k and close[-1 - k] > 0 else None
    v20 = float(np.mean(vol[-21:-1])) if n > 21 else 0.0
    v5 = float(np.mean(vol[-6:-1])) if n > 6 else 0.0
    h = max(0, n - 252)
    return {
        "label": cl["label"], "side": cl["side"], "buy_score": cl["buy_score"], "sell_score": cl["sell_score"],
        "stochrsi": _f(rsi[-1]), "stoch_fast": _f(fast[-1], 3), "stoch_slow": _f(slow[-1], 3), "cci": _f(cci[-1], 3), "macd_turn": t,
        "price": _f(close[-1]), "sma200": _f(sma), "trend_up": int(close[-1] > sma),
        "last_buy_date": lb["d"] if lb else None, "last_buy_label": lb["label"] if lb else None, "last_buy_price": lb["p"] if lb else None,
        "last_sell_date": sells[-1]["d"] if sells else None,
        "hi52": _f(np.nanmax(high[h:])), "lo52": _f(np.nanmin(low[h:])), "hi3y": _f(np.nanmax(high)),
        "vol_ratio": _f(v5 / v20, 3) if v20 > 0 else None, "vol_avg": _f(v20, 0),
        "chg1m": perf(21), "chg3m": perf(63), "chg1y": perf(252),
        "spark": {"p": [_f(x, 3) for x in close[-HIST:]], "d": dates[-HIST:]},
        "sigs": ev,
    }


def universe() -> list[str]:
    with open(os.path.join(HERE, "ticker_meta.json"), encoding="utf-8") as f:
        meta = json.load(f).get("tickers", {})
    out = [t for t, m in meta.items() if m.get("quote_type", "EQUITY") in ("EQUITY", "ETF", "MUTUALFUND", "")]
    extra = os.path.join(HERE, "divflow_extra.txt")
    if os.path.exists(extra):
        out += [ln.strip() for ln in open(extra, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    tickers = universe()
    if a.limit:
        tickers = tickers[:a.limit]
    log.info("%d Ticker", len(tickers))
    frames, failed = engine.download_price_frames(tickers, period="3y")
    now = datetime.now(timezone.utc)
    sig, bad = {}, []
    for t, df in frames.items():
        try:
            df = engine.drop_incomplete_bar(df, t, now)
            rec = build_record(df)
            if rec:
                sig[t] = rec
            else:
                bad.append(t)
        except Exception as e:
            log.debug("%s: %s", t, e)
            bad.append(t)
    payload = {"updated": now.isoformat(timespec="seconds"), "count": len(sig), "no_data": sorted(set(bad) | set(failed)), "signals": sig}
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    log.info("%d Signale, %d ohne Daten -> %s (%.1f MB)", len(sig), len(payload["no_data"]), OUT, os.path.getsize(OUT) / 1e6)


if __name__ == "__main__":
    main()

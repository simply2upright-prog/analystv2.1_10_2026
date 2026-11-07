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
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import engine

log = logging.getLogger("divflow")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "divflow_signals.json.gz")
STORE = os.path.join(HERE, "divflow_bars.json.gz")   # Speicher: letzte Kerzen je Ticker (gegen Yahoo-Rückschritte)
STORE_N = 8
HIST = 180        # Tage Verlauf für Chart und Einstiege
COOLDOWN = 10     # Handelstage zwischen zwei Einstiegen derselben Seite


def _f(x, nd=4):
    try:
        x = float(x)
    except Exception:
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)


def _chart_tail(t: str):
    """Letzte ~10 Tageskerzen direkt vom Yahoo-Chart-Endpunkt (unabhängig von yfinance)."""
    import requests
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", params={"range": "10d", "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(res["meta"].get("exchangeTimezoneName", "UTC")).normalize().tz_localize(None)
    df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]}, index=idx)
    df = df[df["Close"].notna()]
    # Fehlt die Tageskerze in der Reihe, aber Yahoo kennt den Schlusskurs der letzten Sitzung (meta), daraus eine Kerze bauen
    m = res["meta"]
    try:
        px, ts = m.get("regularMarketPrice"), m.get("regularMarketTime")
        if px and ts:
            d = pd.to_datetime(ts, unit="s", utc=True).tz_convert(m.get("exchangeTimezoneName", "UTC")).normalize().tz_localize(None)
            if len(df) and d > df.index[-1]:
                prev = float(df["Close"].iloc[-1])
                hi, lo = m.get("regularMarketDayHigh") or px, m.get("regularMarketDayLow") or px
                df.loc[d] = [prev, max(hi, px), min(lo, px), px, m.get("regularMarketVolume") or 0]
    except Exception:
        pass
    return df


def _store_load() -> dict:
    try:
        with gzip.open(STORE, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _store_tail(df: pd.DataFrame) -> dict:
    t = df.tail(STORE_N)
    g = lambda c: [None if pd.isna(v) else round(float(v), 4) for v in t[c]] if c in t.columns else [None] * len(t)
    return {"d": [i.strftime("%Y-%m-%d") for i in t.index], "o": g("Open"), "h": g("High"), "l": g("Low"), "c": g("Close"), "v": g("Volume")}


def _store_merge(df: pd.DataFrame, rec: dict | None) -> pd.DataFrame:
    """Gespeicherte Kerzen, die NEUER sind als Yahoos letzte Kerze, wieder anhängen (Yahoo nimmt sie am Wochenende teils zurück)."""
    if not rec or df is None or df.empty:
        return df
    last = pd.Timestamp(df.index[-1]).date().isoformat()
    rows = [i for i, d in enumerate(rec["d"]) if d > last and rec["c"][i] is not None]
    if not rows:
        return df
    add = pd.DataFrame({"Open": [rec["o"][i] for i in rows], "High": [rec["h"][i] for i in rows], "Low": [rec["l"][i] for i in rows],
                        "Close": [rec["c"][i] for i in rows], "Volume": [rec["v"][i] or 0 for i in rows]},
                       index=pd.DatetimeIndex([rec["d"][i] for i in rows]))
    if getattr(df.index, "tz", None) is not None:
        add.index = add.index.tz_localize(df.index.tz)
    return pd.concat([df, add[[c for c in df.columns if c in add.columns]]])


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
    # Veraltete Kerzen nachladen: der Sammel-Download liefert für manche Börsen (v. a. Europa) gelegentlich
    # die letzte Tageskerze noch nicht - dann fehlte z. B. ein 4/4-Signal von gestern. Einzelabruf holt sie.
    lasts = sorted(pd.Timestamp(d.index[-1]).date() for d in frames.values())
    ref = lasts[int(len(lasts) * 0.9)] if lasts else None
    stale = [t for t, d in frames.items() if ref and pd.Timestamp(d.index[-1]).date() < ref]
    log.info("%d Ticker mit veralteter letzter Kerze (Referenz %s) -> Einzelabruf", len(stale), ref)
    fixed = errs = 0
    for n, t in enumerate(stale):
        try:
            raw = engine.yf.Ticker(t).history(period="3y", auto_adjust=False)
            if n < 3:
                log.info("DIAG %s Roh-Ende: %s", t, [(str(i.date()), None if pd.isna(c) else round(float(c), 3)) for i, c in raw["Close"].tail(3).items()])
            d = engine.clean_price_df(raw)
            if d is not None and not d.empty and pd.Timestamp(d.index[-1]).date() > pd.Timestamp(frames[t].index[-1]).date():
                frames[t] = d
                fixed += 1
            else:                                   # Zweiter Weg: Yahoo-Chart-Endpunkt direkt, nur neue Kerzen anhängen
                add = _chart_tail(t)
                last = pd.Timestamp(frames[t].index[-1]).date()
                add = add[[pd.Timestamp(i).date() > last for i in add.index]] if add is not None else None
                if add is not None and len(add):
                    base = frames[t].copy()
                    add.index = pd.DatetimeIndex(add.index).tz_localize(base.index.tz) if getattr(base.index, "tz", None) is not None and add.index.tz is None else add.index
                    frames[t] = pd.concat([base, add[[c for c in base.columns if c in add.columns]]])
                    fixed += 1
                    if n < 3:
                        log.info("DIAG %s per Chart-Endpunkt ergänzt: %s", t, [str(i.date()) for i in add.index])
        except Exception as e:
            errs += 1
            if errs <= 3:
                log.warning("Einzelabruf %s: %s", t, e)
        time.sleep(0.3)
    log.info("Nachgeladen: %d von %d (Fehler %d)", fixed, len(stale), errs)
    sig, bad = {}, []
    store, new_store, restored = _store_load(), {}, 0
    for t, df in frames.items():
        try:
            df = engine.drop_incomplete_bar(df, t, now)
            n0 = len(df)
            df = _store_merge(df, store.get(t))
            restored += len(df) - n0 > 0
            new_store[t] = _store_tail(df)
            rec = build_record(df)
            if rec:
                sig[t] = rec
            else:
                bad.append(t)
        except Exception as e:
            log.debug("%s: %s", t, e)
            bad.append(t)
    log.info("Speicher: %d Ticker mit wiederhergestellter Kerze", restored)
    with gzip.open(STORE, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(new_store, f, separators=(",", ":"))
    payload = {"updated": now.isoformat(timespec="seconds"), "count": len(sig), "no_data": sorted(set(bad) | set(failed)), "signals": sig}
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    log.info("%d Signale, %d ohne Daten -> %s (%.1f MB)", len(sig), len(payload["no_data"]), OUT, os.path.getsize(OUT) / 1e6)


if __name__ == "__main__":
    main()

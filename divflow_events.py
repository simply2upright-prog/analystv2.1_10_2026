#!/usr/bin/env python3
"""
divflow_events.py - Dividenden-Historie, naechste Ex-/Zahltage und Quartalszahlen-Termine fuer DivFlow.
Laeuft in der GitHub Action (Yahoo sperrt den Webhoster) und schreibt divflow_events.json.gz.
DivFlow importiert die Datei (cron/events_import.php oder zusammen mit dem Signal-Import).
Universum = wie divflow_signals.py (ticker_meta.json + divflow_extra.txt).
"""
from __future__ import annotations
import argparse, gzip, json, logging, os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf

import divflow_signals as ds

log = logging.getLogger("divflow_events")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "divflow_events.json.gz")
YEARS = 6


def _d(v) -> str | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, (datetime, pd.Timestamp)):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, date):
            return v.isoformat()
        return str(v)[:10]
    except Exception:
        return None


def _num(v):
    try:
        f = float(v)
        return None if pd.isna(f) else round(f, 4)
    except Exception:
        return None


def one(t: str) -> tuple[str, dict | None]:
    for attempt in range(2):
        try:
            tk = yf.Ticker(t)
            divs = tk.dividends
            rec: dict = {"d": [], "nx": None, "np": None, "ne": [], "ee": None, "re": None, "cur": None}
            try:
                rec["cur"] = (tk.history_metadata or {}).get("currency")
            except Exception:
                pass
            if divs is not None and len(divs):
                cut = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(years=YEARS)
                rec["d"] = [[_d(i), round(float(v), 5)] for i, v in divs[divs.index >= cut].items() if float(v) > 0]
            try:
                cal = tk.calendar or {}
            except Exception:
                cal = {}
            today = date.today().isoformat()
            nx, npay = _d(cal.get("Ex-Dividend Date")), _d(cal.get("Dividend Date"))
            rec["nx"] = nx if nx and nx >= today else None
            rec["np"] = npay if npay and npay >= today else None
            ed = cal.get("Earnings Date")
            if ed is not None:
                ed = ed if isinstance(ed, (list, tuple)) else [ed]
                rec["ne"] = sorted({x for x in (_d(e) for e in ed) if x})[:2]
            rec["ee"], rec["re"] = _num(cal.get("Earnings Average")), _num(cal.get("Revenue Average"))
            try:        # Gewinn je Aktie und Dividendenabdeckung
                inf = tk.info or {}
                for k, src in (("eps", "trailingEps"), ("epf", "forwardEps"), ("dps", "dividendRate"), ("pay", "payoutRatio"), ("fcf", "freeCashflow"), ("sh", "sharesOutstanding")):
                    v = _num(inf.get(src))
                    if v is not None:
                        rec[k] = v
                w = (inf.get("website") or "").strip().lower()
                if w:
                    w = w.split("://")[-1].split("/")[0]
                    rec["web"] = w[4:] if w.startswith("www.") else w
                if "dps" not in rec:
                    v = _num(inf.get("trailingAnnualDividendRate"))
                    if v:
                        rec["dps"] = v
            except Exception:
                pass
            return t, rec
        except Exception as e:
            log.debug("%s: %s", t, e)
            time.sleep(1.0)
    return t, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    tickers = ds.universe()
    if a.limit:
        tickers = tickers[:a.limit]
    log.info("%d Ticker", len(tickers))
    out, bad = {}, []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (t, rec) in enumerate(ex.map(one, tickers), 1):
            if rec is None:
                bad.append(t)
            else:
                out[t] = rec
            if k % 100 == 0:
                log.info("%d/%d", k, len(tickers))
    payload = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "count": len(out), "no_data": sorted(bad), "tickers": out}
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    nd = sum(1 for r in out.values() if r["d"]); ne = sum(1 for r in out.values() if r["ne"]); nn = sum(1 for r in out.values() if r["np"])
    log.info("%d Ticker, davon %d mit Dividenden, %d mit Zahlen-Termin, %d mit Zahltag; %d Fehler -> %s (%.1f MB)", len(out), nd, ne, nn, len(bad), OUT, os.path.getsize(OUT) / 1e6)


if __name__ == "__main__":
    main()

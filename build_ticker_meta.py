# build_ticker_meta.py — holt Metadaten für ALLE Ticker aus Yahoo und schreibt ticker_meta.json
#   python build_ticker_meta.py            -> Name, Sektor, Industrie, Börse, Währung
#   python build_ticker_meta.py --isin     -> zusätzlich ISIN (langsamer, eigener Abruf je Ticker)
# Bestehende Werte bleiben erhalten, wenn ein Abruf fehlschlägt (kein Datenverlust durch Yahoo-Aussetzer).
# Die Suche (ticker_search.py) nutzt die Datei automatisch: Namen, ISIN/WKN, Sektor-Anzeige.

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import yfinance as yf

from database import get_all_tickers

log = logging.getLogger("meta")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_meta.json")
FIELDS = {"shortName": "name", "longName": "long_name", "sector": "sector", "industry": "industry",
          "exchange": "exchange", "currency": "currency", "quoteType": "quote_type", "country": "country"}


def load_existing() -> dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f).get("tickers", {})
    except Exception:
        return {}


def fetch(ticker: str, with_isin: bool) -> dict | None:
    t = yf.Ticker(ticker)
    info = t.info or {}
    if not info or not (info.get("shortName") or info.get("longName")):
        return None
    rec = {new: info.get(old) for old, new in FIELDS.items() if info.get(old)}
    if with_isin:
        try:
            isin = t.isin
            if isinstance(isin, str) and len(isin) == 12:
                rec["isin"] = isin
        except Exception:
            pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isin", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Ticker (Test)")
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    tickers = get_all_tickers()
    if a.limit:
        tickers = tickers[:a.limit]
    meta, failed = load_existing(), []
    for i, t in enumerate(tickers, 1):
        try:
            rec = fetch(t, a.isin)
        except Exception as e:
            rec, _ = None, log.debug("%s: %s", t, e)
        if rec:
            old = meta.get(t, {})
            old.update(rec)                      # neue Werte gewinnen, fehlende bleiben stehen
            meta[t] = old
        else:
            failed.append(t)
        if i % 50 == 0:
            log.info("%d/%d verarbeitet, %d ohne Daten", i, len(tickers), len(failed))
        time.sleep(a.sleep)

    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tickers": dict(sorted(meta.items())),
           "no_data": sorted(failed)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log.info("ticker_meta.json: %d Einträge, %d ohne Yahoo-Daten: %s", len(meta), len(failed), ", ".join(failed[:40]))


if __name__ == "__main__":
    main()

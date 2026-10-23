# automation_script.py — Täglicher Scan für GitHub Actions (zwei Läufe: Europa/Asien, USA)
#
#   python automation_script.py --region eu_asia   (nach Xetra/Euronext/LSE-Schluss, Asien ist da längst zu)
#   python automation_script.py --region us        (nach NYSE/Nasdaq-Schluss)
#   ohne --region: Auswahl nach UTC-Uhrzeit (vor 19:00 UTC = eu_asia, sonst us)
#   --force: Mail auch senden, wenn heute keine neue Kerze vorliegt (Feiertag/Wochenende)

import os
import sys
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from database import (get_tickers_by_region, get_groups_for_ticker, TICKER_LISTS, REGION_LABELS)
from engine import scan_tickers, send_mail_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def add_relative_strength(results: list[dict]) -> list[dict]:
    """
    Relative Stärke = eigene 20-Tage-Performance minus Durchschnitt der Peers.
    Peers = alle Ticker aus ALLEN Listen, in denen der Ticker geführt wird (ohne den Ticker selbst),
    soweit sie in diesem Lauf analysiert wurden. Mindestens 3 Peers, sonst None.
    (Vorher: nur die erste Gruppe, inkl. des Tickers selbst - abhängig von der Reihenfolge im Dict.)
    """
    ret = {r["Ticker"]: r["Ret_20d"] for r in results if r.get("Ret_20d") is not None}
    for r in results:
        peers = set()
        for g in get_groups_for_ticker(r["Ticker"]):
            peers.update(TICKER_LISTS[g])
        peers.discard(r["Ticker"])
        vals = [ret[t] for t in peers if t in ret]
        if r.get("Ret_20d") is not None and len(vals) >= 3:
            r["RelStrength"] = round(r["Ret_20d"] - float(np.mean(vals)), 2)
        else:
            r["RelStrength"] = None
    return results


def resolve_region(arg: str | None, now_utc: datetime) -> str:
    if arg in ("eu_asia", "us", "all"):
        return arg
    return "eu_asia" if now_utc.hour < 19 else "us"


# Ab dieser UTC-Zeit ist der Handel der Region sicher beendet (konservativ = Winterzeit).
_REGION_CLOSED_FROM = {"eu_asia": (16, 45), "us": (21, 15), "all": (21, 15)}


def skip_reason(region: str, ref_date: str | None, now_utc: datetime) -> str | None:
    """
    Grund, KEINE Mail zu senden - oder None. Wichtig: 'Markt heute geschlossen' darf nur gelten,
    wenn der Handel der Region heute schon BEENDET ist. Bei einem manuellen Lauf mitten am Tag
    ist die letzte fertige Kerze zwangsläufig die von gestern (die heutige wird verworfen) -
    das ist kein Feiertag. (Fehler in v5.0: dort wurde in dem Fall still übersprungen.)
    """
    if now_utc.weekday() >= 5:
        return "Wochenende"
    closed = (now_utc.hour, now_utc.minute) >= _REGION_CLOSED_FROM.get(region, (21, 15))
    if closed and ref_date and ref_date < now_utc.date().isoformat():
        return f"letzte Kerze {ref_date} < heute {now_utc.date()} (Börsenfeiertag?)"
    return None


def run_automation(region: str | None = None, force: bool = False, now_utc: datetime | None = None) -> None:
    email_pass = os.getenv("DAILY_EMAIL_PASS")
    if not email_pass:
        log.error("Secret 'DAILY_EMAIL_PASS' nicht gesetzt – Abbruch.")
        sys.exit(1)

    now_utc = now_utc or datetime.now(timezone.utc)
    region  = resolve_region(region, now_utc)
    tickers = get_tickers_by_region(region)
    label   = REGION_LABELS.get(region, region)
    log.info(f"Region {label}: starte Analyse von {len(tickers)} Tickern …")

    results, failed, stale, meta = scan_tickers(tickers, now_utc=now_utc)
    log.info(f"Fertig — analysiert: {len(results)}, veraltet: {len(stale)}, Fehler: {len(failed)}, Daten bis: {meta['ref_date']}")
    if failed:
        for t, why in sorted(failed.items()):
            log.info(f"  Fehler {t}: {why}")

    reason = skip_reason(region, meta["ref_date"], now_utc)
    if reason and not force:
        log.warning(f"KEINE MAIL: {reason}. Zum Erzwingen: --force bzw. beim manuellen Start 'force' anhaken.")
        return
    if not results:
        log.error("Keine Ergebnisse – Abbruch ohne Mail.")
        sys.exit(1)

    results    = add_relative_strength(results)
    df_results = pd.DataFrame(results)
    if "KGV" in df_results.columns:
        df_results["KGV"] = df_results["KGV"].astype(str)

    status = send_mail_report(
        df_results, email_pass,
        total_scanned=len(tickers), success_count=len(results), failed_count=len(failed),
        region_label=label, failed_list=sorted(failed), stale_count=len(stale), data_date=meta["ref_date"],
    )
    log.info(status)
    if status.startswith("❌"):
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["eu_asia", "us", "all"], default=os.getenv("SCAN_REGION") or None)
    ap.add_argument("--force", action="store_true", default=os.getenv("SCAN_FORCE", "").lower() == "true")
    a = ap.parse_args()
    run_automation(a.region, a.force)

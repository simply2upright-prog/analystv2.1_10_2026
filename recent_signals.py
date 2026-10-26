# recent_signals.py — Suche: Welche Werte hatten in den letzten X Handelstagen ein Kauf-/Verkaufssignal?
#
# Drei Begriffe von "Signal" gibt es in der App (deshalb weichen Mail und Chart voneinander ab):
#   "4/4"     = alle vier Kriterien am SELBEN Tag erfüllt (Definition der Mail, hier auch für zurückliegende Tage ausgewertet)
#   "Einstieg"= grünes/rotes Dreieck im Chart: Extrem-Phase beginnt, Bestätigung (MACD-Dreh oder 2 % Kursabstand) folgt innerhalb von
#               bis zu 20 Tagen, Einstieg zum Open des Folgetags. Die übrigen Kriterien müssen zum Bestätigungstag NICHT mehr gelten.
#   "3/4"     = drei von vier Kriterien (Vorwarnung, optional)
# Die Mail zeigt nur, was HEUTE 4/4 ist; diese Suche zeigt, was in den letzten Tagen 4/4 war bzw. bestätigt wurde.

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from engine import (SIGNAL_COOLDOWN, _criteria_arrays, add_strategy_indicators, clean_price_df, download_price_frames,
                    drop_incomplete_bar, find_entry_signals)

log = logging.getLogger("recent_signals")

KINDS = ("4/4", "Einstieg", "3/4")


def _runs(idx: np.ndarray, gap: int = SIGNAL_COOLDOWN) -> list[int]:
    """Erster Tag jeder zusammenhängenden Serie (Abstand <= gap Handelstage gehört zur selben Serie)."""
    out, prev = [], -10 ** 9
    for i in idx:
        i = int(i)
        if i - prev > gap:
            out.append(i)
        prev = i
    return out


def signals_for_frame(ticker: str, df: pd.DataFrame, days: int, kinds=KINDS) -> list[dict]:
    """Alle Signale der letzten `days` Handelstage für einen bereinigten, fertigen Kurs-DataFrame."""
    df = add_strategy_indicators(df.copy())
    n = len(df)
    if n < 60:
        return []
    lo = n - days                                  # Index des ersten Tages im Fenster
    arr = _criteria_arrays(df)
    close = df["Close"].to_numpy(float)
    ma = df["TREND_MA"].to_numpy(float)
    now_px = float(close[-1])
    now_up = bool(now_px > ma[-1]) if np.isfinite(ma[-1]) else None
    rows = []

    def add(side, kind, i, extra=None):
        px = float(close[i])
        r = {"Ticker": ticker, "Seite": "Kauf" if side == "buy" else "Verkauf", "Art": kind, "Datum": df.index[i],
             "Handelstage_her": int(n - 1 - i), "Kurs_damals": px, "Kurs_jetzt": now_px,
             "Veraenderung_%": (now_px / px - 1) * 100 * (1 if side == "buy" else -1),
             "Trend_damals": ("up" if close[i] > ma[i] else "down") if np.isfinite(ma[i]) else None, "Trend_jetzt": ("up" if now_up else "down") if now_up is not None else None,
             "Heute_aktiv": bool(i == n - 1)}
        if extra:
            r.update(extra)
        rows.append(r)

    for side in ("buy", "sell"):
        sc, valid = arr[side]["score"], arr["valid"]
        if "4/4" in kinds:
            idx = np.flatnonzero((sc == 4) & valid)
            idx = idx[idx >= lo]
            # Serien: lückenlos aufeinanderfolgende 4/4-Tage -> erster Tag; "Heute_aktiv" wenn die Serie bis heute reicht
            if len(idx):
                starts = [int(idx[0])] + [int(b) for a, b in zip(idx[:-1], idx[1:]) if b - a > 1]
                for s in starts:
                    add(side, "4/4", s, {"Serie_bis_heute": bool(n - 1 in idx and all(k in set(idx.tolist()) for k in range(s, n)))})
        if "3/4" in kinds:
            idx = np.flatnonzero((sc == 3) & valid & arr[side]["extreme"])
            idx = idx[idx >= lo]
            for s in _runs(idx, 3):
                add(side, "3/4", s)
        if "Einstieg" in kinds:
            for e in find_entry_signals(df, side=side):
                j = df.index.get_loc(e["decision_date"])
                if j >= lo:
                    add(side, "Einstieg", j, {"Einstieg_am": e["entry_date"], "Einstiegskurs": e["entry_price"], "Ausloeser": e["trigger"],
                                              "Signaltag": e["signal_date"], "lokales_Extrem": e["local_low"]})
    return rows


def scan_recent(tickers: list[str], days: int = 5, kinds=KINDS, progress_cb=None, now_utc: datetime | None = None,
                frames: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Lädt Kurse (Batches) und sucht Signale der letzten `days` Handelstage. Rückgabe: (DataFrame, failed)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    failed = {}
    if frames is None:
        frames, failed = download_price_frames(tickers, progress_cb=progress_cb, phase="download")
    rows = []
    for t, raw in frames.items():
        try:
            d = clean_price_df(raw)
            try:
                d = drop_incomplete_bar(d, t, now_utc)
            except Exception:
                pass
            if d is None or d.empty or len(d) < 60:
                continue
            rows.extend(signals_for_frame(t, d, days, kinds))
        except Exception as e:
            failed[t] = f"Fehler: {e}"
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Handelstage_her", "Seite", "Ticker"]).reset_index(drop=True)
    return df, failed

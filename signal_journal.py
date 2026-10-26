# signal_journal.py — Journal der ECHTEN Mail-Signale (Live-Daten, Booklet Schritt 6)
#
# Jeden Tag, an dem die Mail rausgeht, werden Kauf-/Verkaufssignale und Watchlist-Kandidaten mit allen
# Indikatorwerten in signal_journal.csv geschrieben (nur Daten, die zum Zeitpunkt der Mail bekannt waren).
# Nach ~60 Kalendertagen (= 40 Handelstage) trägt `evaluate_journal` das Ergebnis nach: Einstieg zum Open des
# Folgetags (so wie man nach der Mail handeln kann), Treffer = ≥ 5 % in Signalrichtung, plus Zufalls-Vergleich.
# `analyze_journal` wertet gepoolt aus - derselbe Test wie pooled_backtest.py, nur mit echten Signalen.
#
#   python signal_journal.py --evaluate          (offene Einträge nachtragen + Report schreiben)

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from engine import (FORWARD_DAYS, add_strategy_indicators, clean_price_df, compute_hit_rate, download_price_frames)
from pooled_backtest import TABLE_HEAD, baseline_return, fmt_row, naive_index, summarize

log = logging.getLogger("journal")
JOURNAL_PATH = os.getenv("JOURNAL_PATH", "signal_journal.csv")
LABELS = ("KAUFSIGNAL", "VERKAUFSIGNAL", "OVERSOLD", "OVERBOUGHT")
PHASE_GAP_DAYS = 14   # gleicher Ticker + Seite innerhalb 14 Tagen = dieselbe Extrem-Phase (nur der erste Tag zählt)

COLUMNS = ["signal_date", "scan_utc", "region", "ticker", "name", "label", "side", "score", "first_in_phase",
           "price", "stochrsi", "stoch_fast", "stoch_slow", "cci", "macd_turn", "trend",
           "hist_rate", "hist_n", "hist_base", "hist_edge", "rel_strength",
           # ab hier nachgetragen
           "entry_date", "entry_price", "fav_pct", "adv_pct", "ret40_pct", "base_ret40_pct", "base_hit",
           "result", "hit", "evaluated_on"]


def load_journal(path: str = JOURNAL_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype={"signal_date": str, "entry_date": str, "evaluated_on": str})
            for c in COLUMNS:
                if c not in df.columns:
                    df[c] = np.nan
            for c in ("entry_date", "result", "evaluated_on"):
                df[c] = df[c].astype(object)     # leere Spalten werden sonst als float gelesen -> Text lässt sich nicht eintragen
            return df[COLUMNS]
        except Exception as e:
            log.warning("Journal nicht lesbar (%s) - starte neu", e)
    return pd.DataFrame(columns=COLUMNS)


def _label(sig: str) -> str:
    return str(sig).split()[-1] if sig else ""


def append_signals(results: list[dict], data_date: str | None, region: str, path: str = JOURNAL_PATH,
                   now_utc: datetime | None = None) -> int:
    """Hängt die Signale dieses Scans an (Dubletten pro Tag/Ticker/Label werden übersprungen). Gibt Anzahl neuer Zeilen zurück."""
    if not data_date:
        return 0
    now_utc = now_utc or datetime.now(timezone.utc)
    jr = load_journal(path)
    seen = set(zip(jr["signal_date"], jr["ticker"], jr["label"]))
    d0 = pd.Timestamp(data_date)
    new = []
    for r in results:
        lab, side = _label(r.get("Signal")), r.get("Side")
        if lab not in LABELS or side not in ("buy", "sell"):
            continue
        if (data_date, r["Ticker"], lab) in seen:
            continue
        prev = jr[(jr["ticker"] == r["Ticker"]) & (jr["side"] == side)]
        recent = pd.to_datetime(prev["signal_date"], errors="coerce")
        first = not ((recent >= d0 - timedelta(days=PHASE_GAP_DAYS)) & (recent < d0)).any()
        new.append({
            "signal_date": data_date, "scan_utc": now_utc.strftime("%Y-%m-%d %H:%M"), "region": region,
            "ticker": r["Ticker"], "name": r.get("Name"), "label": lab, "side": side,
            "score": r.get("Buy_Score") if side == "buy" else r.get("Sell_Score"), "first_in_phase": bool(first),
            "price": r.get("Preis_Num"), "stochrsi": r.get("StochRSI"), "stoch_fast": r.get("Stoch_Fast"),
            "stoch_slow": r.get("Stoch_Slow"), "cci": r.get("CCI"), "macd_turn": r.get("MACD_Turn"),
            "trend": r.get("Trend"), "hist_rate": r.get("HitRate"), "hist_n": r.get("HitRate_N"),
            "hist_base": r.get("HitRate_Base"), "hist_edge": r.get("HitRate_Edge"), "rel_strength": r.get("RelStrength"),
        })
    if not new:
        return 0
    out = pd.concat([jr, pd.DataFrame(new)], ignore_index=True)[COLUMNS]
    out.to_csv(path, index=False)
    return len(new)


def evaluate_journal(path: str = JOURNAL_PATH, min_age_days: int = 60, now_utc: datetime | None = None,
                     frames: dict | None = None) -> int:
    """Trägt Ergebnisse für reife Einträge nach. `frames` nur für Tests (sonst Download). Gibt Anzahl ausgewerteter Zeilen zurück."""
    now_utc = now_utc or datetime.now(timezone.utc)
    jr = load_journal(path)
    if jr.empty:
        return 0
    cutoff = (now_utc - timedelta(days=min_age_days)).date().isoformat()
    pend = jr[jr["result"].isna() & (jr["signal_date"] <= cutoff)]
    if pend.empty:
        return 0
    if frames is None:
        frames, _ = download_price_frames(sorted(pend["ticker"].unique()), period="5y")
    done = 0
    cache = {}
    for idx, row in pend.iterrows():
        t = row["ticker"]
        if t not in frames:
            continue
        if t not in cache:
            df = naive_index(clean_price_df(frames[t]))
            cache[t] = add_strategy_indicators(df.copy()) if len(df) >= 260 else None
        df = cache[t]
        if df is None:
            continue
        pos = int(df.index.searchsorted(pd.Timestamp(row["signal_date"]), side="right"))   # erster Handelstag NACH dem Signal
        if pos + FORWARD_DAYS > len(df):
            continue
        side, ep = row["side"], float(df["Open"].iloc[pos])
        if not np.isfinite(ep) or ep <= 0:
            continue
        entry = {"side": side, "signal_date": df.index[pos], "entry_date": df.index[pos], "entry_price": ep,
                 "at_open": True, "trend_at_entry": row["trend"]}
        hr = compute_hit_rate([entry], df, forward_days=FORWARD_DAYS, side=side)
        if not hr["details"] or hr["details"][0]["result_code"] == "new":
            continue
        d = hr["details"][0]
        r = (float(df["Close"].iloc[pos + FORWARD_DAYS - 1]) / ep - 1) * 100
        bret = baseline_return(df, side, FORWARD_DAYS)
        jr.loc[idx, ["entry_date", "entry_price", "fav_pct", "adv_pct", "ret40_pct", "base_ret40_pct", "base_hit",
                     "result", "hit", "evaluated_on"]] = [
            df.index[pos].date().isoformat(), round(ep, 4), d["max_gain_pct"], d["max_loss_pct"],
            round(r if side == "buy" else -r, 2), None if bret is None else round(bret, 2),
            None if hr["baseline"] is None else round(hr["baseline"], 4),
            d["result_code"], int(d["result_code"] in ("strong", "weak")), now_utc.date().isoformat()]
        done += 1
    jr.to_csv(path, index=False)
    return done


def analyze_journal(path: str = JOURNAL_PATH) -> str:
    """Markdown-Auswertung der bereits ausgewerteten Journal-Einträge (nur erster Tag jeder Extrem-Phase)."""
    jr = load_journal(path)
    total, evaluated = len(jr), int(jr["result"].notna().sum())
    first = jr["signal_date"].min() if total else "–"
    last = jr["signal_date"].max() if total else "–"
    lines = ["# Signal-Journal – Live-Auswertung\n",
             f"Stand {datetime.now(timezone.utc).strftime('%d.%m.%Y')} · {total} Einträge ({first} bis {last}) · "
             f"{evaluated} ausgewertet (Ergebnis liegt erst nach {FORWARD_DAYS} Handelstagen vor)\n"]
    ev = jr[jr["result"].notna() & (jr["first_in_phase"].astype(str) == "True")].copy()
    if ev.empty:
        lines.append("Noch keine ausgewerteten Signale – das erste Ergebnis kommt ca. 2 Monate nach dem Start des Journals.\n")
        return "\n".join(lines)
    for c in ("hit", "base_hit", "ret40_pct", "base_ret40_pct", "fav_pct", "adv_pct"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev = ev.dropna(subset=["base_hit", "base_ret40_pct"])
    ev["trend"] = ev["trend"].fillna("unknown")
    lines.append("Nur der erste Tag jeder Extrem-Phase zählt (keine Mehrfachzählung). Einstieg = Open des Folgetags nach der Mail.\n")
    lines.append(TABLE_HEAD)
    for lab in LABELS:
        sub = ev[ev["label"] == lab]
        if len(sub):
            lines.append(fmt_row(lab, summarize(sub)))
    for sd in ("buy", "sell"):
        sub = ev[ev["side"] == sd]
        if len(sub):
            lines.append(fmt_row(f"{sd} gesamt", summarize(sub)))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true", help="reife Einträge nachtragen")
    ap.add_argument("--path", default=JOURNAL_PATH)
    ap.add_argument("--report", default="journal_report.md")
    a = ap.parse_args()
    if a.evaluate:
        log.info("%d Einträge ausgewertet", evaluate_journal(a.path))
    text = analyze_journal(a.path)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

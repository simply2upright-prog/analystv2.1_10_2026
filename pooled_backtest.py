# pooled_backtest.py — Gepoolte Auswertung der Strategie über ALLE Ticker (Kauf + Verkauf getrennt)
#
#   python pooled_backtest.py --period 10y --out backtest_results
#   python pooled_backtest.py --period 5y --limit 120            (Schnelltest)
#
# Idee: Pro Ticker gibt es nur ~20 historische Signale (statistisch zu wenig). Über alle Ticker
# zusammen sind es Tausende. Jedes Signal wird dabei mit der Zufallsquote SEINES Tickers verglichen
# (volatile Aktien erreichen +5 % auch ohne Signal oft). Daraus:
#   erwartete Treffer = Summe der Ticker-Zufallsquoten  ->  z-Test (Poisson-Binomial)
# Weil viele Signale am selben Tag/Monat auftreten (Marktcrash = hunderte Signale gleichzeitig) und damit
# NICHT unabhängig sind, wird zusätzlich ein Cluster-Test über Kalendermonate gerechnet. Der ist maßgeblich.
#
# Regeln = exakt die der App/Mail (engine.find_entry_signals / compute_hit_rate): Signal = erster Tag einer
# Extrem-Phase, Bestätigung per MACD-Dreh oder 2 % vom Extrem, Einstieg zum Open des Folgetags, Treffer =
# >= 5 % in Signalrichtung innerhalb von 40 Handelstagen (Schlusskurse).

import argparse
import json
import logging
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from database import get_all_tickers, get_region
from engine import (FORWARD_DAYS, HIT_THRESHOLD_PCT, add_strategy_indicators, clean_price_df, compute_hit_rate,
                    download_price_frames, find_entry_signals, _criteria_arrays, _trend_ma)

log = logging.getLogger("backtest")
HIT_CODES = ("strong", "weak")


# ─────────────────────────────────────────────────────────────────────────────
# Bausteine
# ─────────────────────────────────────────────────────────────────────────────
def naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def baseline_return(df: pd.DataFrame, side: str, fwd: int = FORWARD_DAYS) -> float | None:
    """
    Zufalls-Rendite: Ø Rendite in Signalrichtung (%), wenn man an JEDEM gültigen Tag des Tickers zum nächsten Open
    einsteigt und nach `fwd` Handelstagen zum Schlusskurs rausgeht (gleiche Mechanik wie ein echter Einstieg).
    """
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    n = len(c)
    if n < fwd + 60:
        return None
    t = np.arange(n - fwd)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = c[t + fwd] / o[t + 1] - 1
    arr = _criteria_arrays(df)
    mask = arr["valid"][t] & np.isfinite(_trend_ma(df)[t]) & np.isfinite(ret)
    if mask.sum() < 50:
        return None
    r = float(ret[mask].mean() * 100)
    return r if side == "buy" else -r


def ticker_trades(ticker: str, raw: pd.DataFrame, fwd: int = FORWARD_DAYS) -> list[dict]:
    """Alle abgeschlossenen historischen Einstiege (Kauf + Verkauf) eines Tickers inkl. Zufalls-Vergleichswerten."""
    df = naive_index(clean_price_df(raw))
    if df is None or df.empty or len(df) < 260:
        return []
    df = add_strategy_indicators(df.copy())
    close = df["Close"].to_numpy(float)
    rows = []
    for side in ("buy", "sell"):
        entries = find_entry_signals(df, side=side)
        if not entries:
            continue
        hr = compute_hit_rate(entries, df, forward_days=fwd, side=side)
        base = hr.get("baseline")
        bret = baseline_return(df, side, fwd)
        if base is None or bret is None:
            continue
        emap = {e["entry_date"]: e for e in entries}
        for d in hr["details"]:
            e = emap.get(d["entry_date"])
            if d["result_code"] == "new" or e is None:
                continue
            pos = df.index.get_loc(d["entry_date"])
            if not isinstance(pos, (int, np.integer)):
                continue
            start = pos if e.get("at_open", False) else pos + 1
            win = close[start:start + fwd]
            if len(win) < fwd or not d["entry_price"]:
                continue
            seg = close[max(start - 1, 0):start + fwd]
            with np.errstate(invalid="ignore", divide="ignore"):
                step = seg[1:] / seg[:-1]
            if np.any((step > 2.0) | (step < 0.4)):      # Datenfehler (z. B. CHRD 2020: +37.250 %) - nicht werten
                continue
            r = (win[-1] / d["entry_price"] - 1) * 100
            rows.append({
                "ticker": ticker, "region": get_region(ticker), "side": side,
                "signal_date": pd.Timestamp(e["signal_date"]).date().isoformat(),
                "entry_date": pd.Timestamp(d["entry_date"]).date().isoformat(),
                "entry_price": d["entry_price"], "trigger": e.get("trigger"),
                "trend": d.get("trend_at_entry") or "unknown",
                "result": d["result_code"], "hit": int(d["result_code"] in HIT_CODES),
                "fav_pct": d["max_gain_pct"], "adv_pct": d["max_loss_pct"],
                "ret40_pct": round(float(r if side == "buy" else -r), 2),
                "base_hit": round(float(base), 4), "base_ret40_pct": round(float(bret), 2),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Statistik
# ─────────────────────────────────────────────────────────────────────────────
def wilson(h: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, den = h / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - w, c + w)


def p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else float("nan")


def cluster_t(values: pd.Series, groups: pd.Series) -> tuple[float, int]:
    """t-Wert des Mittelwerts über Gruppen-Summen (Kalendermonate) - berücksichtigt, dass Signale zeitlich klumpen."""
    g = values.groupby(groups).mean()
    m = len(g)
    if m < 6 or g.std(ddof=1) == 0:
        return (float("nan"), m)
    return (float(g.mean() / (g.std(ddof=1) / math.sqrt(m))), m)


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    hits = int(df["hit"].sum())
    exp = float(df["base_hit"].sum())
    var = float((df["base_hit"] * (1 - df["base_hit"])).sum())
    z = (hits - exp) / math.sqrt(var) if var > 0 else float("nan")
    month = pd.to_datetime(df["entry_date"]).dt.to_period("M")
    t_hit, m = cluster_t(df["hit"] - df["base_hit"], month)
    excess = df["ret40_pct"] - df["base_ret40_pct"]
    t_ret, _ = cluster_t(excess, month)
    lo, hi = wilson(hits, n)
    return {
        "n": n, "hits": hits, "rate": hits / n, "ci_lo": lo, "ci_hi": hi,
        "baseline": exp / n, "edge": hits / n - exp / n,
        "z_naive": z, "p_naive": p_two_sided(z),
        "t_cluster": t_hit, "p_cluster": p_two_sided(t_hit), "n_months": m,
        "ret_mean": float(df["ret40_pct"].mean()), "ret_median": float(df["ret40_pct"].median()),
        "ret_base": float(df["base_ret40_pct"].mean()), "ret_excess": float(excess.mean()),
        "ret_excess_median": float(excess.median()), "ret_excess_wins": float(excess.clip(-50, 50).mean()),
        "t_ret_cluster": t_ret, "p_ret_cluster": p_two_sided(t_ret),
        "avg_fav": float(df["fav_pct"].mean()), "avg_adv": float(df["adv_pct"].mean()),
        "n_tickers": int(df["ticker"].nunique()),
    }


def verdict(s: dict) -> str:
    if s.get("n", 0) < 50:
        return "zu wenig (n<50)"
    p, e = s["p_cluster"], s["edge"]
    if not np.isfinite(p):
        return "n/a"
    if p < 0.05 and e > 0:
        return "besser als Zufall"
    if p < 0.05 and e < 0:
        return "SCHLECHTER als Zufall"
    return "nicht von Zufall unterscheidbar"


def fmt_row(label: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"| {label} | 0 | | | | | | | | |"
    pc = s["p_cluster"]
    p_txt = "–" if not np.isfinite(pc) else ("<0,001" if pc < 0.001 else f"{pc:.3f}")
    return (f"| {label} | {s['n']} | {s['rate']*100:.1f} % ({s['ci_lo']*100:.0f}–{s['ci_hi']*100:.0f}) | "
            f"{s['baseline']*100:.1f} % | {s['edge']*100:+.1f} PP | "
            f"{p_txt} | {s['ret_mean']:+.1f} % | {s['ret_excess']:+.1f} PP | {s['ret_excess_median']:+.1f} PP | {verdict(s)} |")


TABLE_HEAD = ("| Gruppe | n | Trefferquote (95 %-KI) | Zufall | Edge | p (Cluster) | Ø Rendite 40 T. | Rendite vs. Zufall (Ø) | Rendite vs. Zufall (Median) | Urteil |\n"
              "|---|---|---|---|---|---|---|---|---|---|")


def build_report(trades: pd.DataFrame, meta: dict) -> tuple[str, dict]:
    out, summary = [], {"meta": meta, "groups": {}}
    out.append("# Gepoolte Auswertung der Strategie\n")
    out.append(f"Stand {meta['created']} · Daten {meta['period']} · {meta['n_tickers']} Ticker mit Daten · "
               f"{len(trades)} abgeschlossene Signale · Beobachtungsfenster {FORWARD_DAYS} Handelstage · "
               f"Treffer = ≥ {HIT_THRESHOLD_PCT:.0f} % in Signalrichtung\n")

    def block(title, frame, keys):
        out.append(f"\n## {title}\n")
        out.append(TABLE_HEAD)
        for label, sub in keys(frame):
            s = summarize(sub)
            summary["groups"][f"{title} | {label}"] = s
            out.append(fmt_row(label, s))

    block("Gesamt", trades, lambda f: [(sd, f[f.side == sd]) for sd in ("buy", "sell")])
    block("Nach Trend zum Einstieg (Kurs vs. SMA200)", trades,
          lambda f: [(f"{sd} · {tr}", f[(f.side == sd) & (f.trend == tr)]) for sd in ("buy", "sell") for tr in ("up", "down")])
    block("Nach Auslöser der Bestätigung", trades,
          lambda f: [(f"{sd} · {tg}", f[(f.side == sd) & (f.trigger == tg)]) for sd in ("buy", "sell") for tg in ("MACD", "Preis")])
    block("Nach Region", trades,
          lambda f: [(f"{sd} · {rg}", f[(f.side == sd) & (f.region == rg)]) for sd in ("buy", "sell") for rg in ("eu_asia", "us")])
    yrs = sorted(trades["entry_date"].str[:4].unique())
    block("Nach Jahr (Stabilität)", trades,
          lambda f: [(f"{sd} · {y}", f[(f.side == sd) & (f.entry_date.str[:4] == y)]) for sd in ("buy", "sell") for y in yrs])

    g = summary["groups"]
    out.append("\n## Lesehilfe\n")
    out.append(
        "- **Zufall** = Anteil, den ein beliebiger Handelstag desselben Tickers geschafft hätte (Mittel über alle Signale, "
        "ticker-genau). **Edge** = Trefferquote minus Zufall in Prozentpunkten.\n"
        "- **p (Cluster)** testet über Kalendermonate, nicht über Einzelsignale – Signale desselben Monats hängen "
        "zusammen (Marktschwäche löst hunderte gleichzeitig aus). Nur dieser p-Wert zählt; der naive z-Test ist in "
        "`summary.json` und überschätzt die Sicherheit.\n"
        "- **Rendite vs. Zufall**: Ø 40-Tage-Rendite in Signalrichtung (Verkauf = fallende Kurse positiv) minus die Ø Rendite "
        "eines Zufalls-Einstiegs desselben Tickers. Treffer-Quote sagt nichts über Größe der Verluste – diese Zeile schon.\n"
        "- **Mehrfachtests:** Die Tabellen enthalten ~20 Einzeltests; bei α = 5 % ist im Schnitt einer rein zufällig „signifikant“. "
        "Einzelne Treffer in Untergruppen nur glauben, wenn sie in der Jahres-Tabelle und in beiden Regionen wiederkehren.\n"
        "- **Verkauf** = Kurs fällt ≥ 5 %: ein Abwärts-Signal, kein Short-Backtest mit Kosten.\n"
        "- Die Jahres-Tabelle zeigt, ob ein Edge in allen Jahren da ist oder nur in einem. ACHTUNG: Der Zufallsvergleich hier ist "
        "der 10-Jahres-Schnitt des Tickers, nicht jahresgleich – in Haussejahren (2017, 2019) sieht Verkauf dadurch künstlich "
        "schlecht aus, in Crash-/Vola-Jahren alles besser. Jahresgleiche Werte liefert `sweep_backtest.py` (SWEEP_REPORT.md).\n"
        "- Signale mit Kurssprüngen > +100 % / < -60 % im Fenster (Datenfehler) werden nicht gewertet.\n")
    out.append("\n## Grenzen dieser Auswertung\n")
    out.append(
        "- **Survivorship-Bias:** Die Liste enthält die Ticker von HEUTE. Firmen, die zwischenzeitlich abstürzten oder "
        "delistet wurden, fehlen – das verbessert vor allem Kauf-Ergebnisse künstlich.\n"
        "- **Keine Kosten:** Spread, Gebühren, Steuern, Währung nicht eingerechnet. Kein Stop, keine Positionsgröße.\n"
        "- **Überlappung:** 40-Tage-Fenster desselben Tickers können sich überlappen (Cooldown nur 10 Tage).\n"
        "- **Schwellen** (0,1 / 10 / 15 / −100 …) stammen aus dem Booklet; wurden sie anhand derselben Jahre gewählt, ist die "
        "Auswertung nicht „out of sample“. Echte Prüfung = das Signal-Journal (Live-Daten ab Start).\n")
    return "\n".join(out), summary


# ─────────────────────────────────────────────────────────────────────────────
def run(period: str, out_dir: str, limit: int = 0, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or get_all_tickers()
    if limit:
        tickers = tickers[:limit]
    log.info("Lade %d Ticker (%s) …", len(tickers), period)
    frames, failed = download_price_frames(
        tickers, period=period, progress_cb=lambda d, t, p: log.info("Download %d/%d", d, t))
    log.info("Daten für %d Ticker, %d ohne Daten", len(frames), len(failed))

    rows = []
    for i, (t, raw) in enumerate(frames.items(), 1):
        try:
            rows.extend(ticker_trades(t, raw))
        except Exception as e:
            log.warning("%s: %s", t, e)
        if i % 100 == 0:
            log.info("ausgewertet %d/%d, bisher %d Signale", i, len(frames), len(rows))
    trades = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    if trades.empty:
        log.error("Keine Signale – nichts zu schreiben.")
        return trades
    meta = {"created": datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC"), "period": period,
            "n_tickers": int(trades["ticker"].nunique()), "n_failed": len(failed), "failed": sorted(failed)}
    report, summary = build_report(trades, meta)
    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=lambda o: None if o != o else float(o))
    log.info("Fertig: %d Signale → %s", len(trades), out_dir)
    return trades


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="10y", help="Yahoo-Zeitraum, z. B. 5y, 10y, max")
    ap.add_argument("--out", default="backtest_results")
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Ticker (Test)")
    a = ap.parse_args()
    run(a.period, a.out, a.limit)

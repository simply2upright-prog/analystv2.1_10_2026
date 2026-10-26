# sweep_backtest.py — Parameter-Varianten der Strategie, sauber gegen Zufall getestet
#
#   python sweep_backtest.py --period 10y --out sweep_results
#   python sweep_backtest.py --period 5y --limit 120            (Schnelltest)
#
# Was anders ist als in pooled_backtest.py (Lehren aus dem ersten Lauf):
#   1. DATUMSGLEICHER Zufall: erwartete Rendite = Ticker-Schnitt im selben Trend + Marktabweichung des Einstiegstags
#      (Querschnitt aller Ticker am selben Tag). Haussejahr-/Crashjahr-Effekte und der Trend-Effekt (ein Filter "nur
#      unter SMA200" bekommt als Vergleich ebenfalls nur Tage unter SMA200) verschwinden aus dem Überschuss.
#   2. Hauptmaß ist der RENDITE-Überschuss (40 Tage, Open Folgetag -> Close), nicht die Trefferquote. Er wird bei
#      +-50 PP gekappt (ein Datenfehler darf nicht das Ergebnis sein) und mit dem Median ergänzt.
#   3. Zusätzlich eine volatilitätsnormierte Trefferquote: Ziel = 0,5 x Tagesvola(60 T.) x sqrt(40) statt fixer 5 %.
#      Die fixe 5 %-Schwelle bevorzugt schwankungsstarke Phasen (nach Abverkäufen), ohne Richtungsinformation.
#   4. Datenfehler-Filter: Fenster mit Tagessprung > +100 % oder < -60 % werden verworfen (z. B. CHRD 2020).
#   5. TRAINING (Einstieg vor SPLIT) und TEST (ab SPLIT) werden getrennt gerechnet. Eine Variante zählt nur dann
#      als belegt, wenn sie in BEIDEN Zeiträumen einen Überschuss mit Cluster-t > 2 hat.
#
# Varianten sind bewusst wenige (je eine Änderung gegenüber der Basis + 2 begründete Kombinationen), damit die
# Mehrfachtest-Problematik klein bleibt. Die Signal-Mechanik entspricht engine.find_entry_signals (V0 = Basis).

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from database import get_all_tickers, get_region
from engine import (FORWARD_DAYS, MAX_ENTRY_WAIT, SIGNAL_COOLDOWN, _criteria_arrays, add_strategy_indicators,
                    clean_price_df, compute_macd, download_price_frames, macd_turn_series)
from pooled_backtest import cluster_t, naive_index, p_two_sided

log = logging.getLogger("sweep")

SPLIT_DEFAULT = "2022-01-01"
VOL_K = 0.5            # Ziel der vola-normierten Trefferquote = VOL_K * Tagesvola * sqrt(40)
CLIP_PP = 50.0         # Überschuss-Rendite wird bei +-50 PP gekappt
T_SIG = 2.0            # Cluster-t, ab dem ein Überschuss als "Hinweis" gilt (in Training UND Test)
MIN_N_TEST = 150


# ─────────────────────────────────────────────────────────────────────────────
# Varianten
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    why: str
    confirm: str = "either"          # either = MACD-Dreh ODER Preis | price = nur Preis | macd = nur MACD
    price_pct: float = 2.0           # Abstand vom laufenden Extrem für die Preis-Bestätigung
    sell_down_only: bool = False     # Verkauf nur, wenn Kurs unter SMA200
    min_price: float = 0.0           # Mindestkurs (nur US-Ticker, da andere Währungen/Subunits)
    cooldown: int = SIGNAL_COOLDOWN
    rsi_th: float = 0.1              # StochRSI-Extremschwelle (Kauf < th, Verkauf > 1 - th)
    macd_alt: bool = False           # MACD(12,26,9) statt (70,200,9)


VARIANTS = [
    Variant("V0", "Basis (wie App/Mail)", "Referenz"),
    Variant("V1", "Verkauf nur unter SMA200", "Im 1. Lauf einzige Verkaufs-Untergruppe mit Überschuss", sell_down_only=True),
    Variant("V2", "Nur Preis-Bestätigung 2 %", "Preis-Bestätigung schnitt im 1. Lauf deutlich besser ab als MACD", confirm="price"),
    Variant("V3", "Preis-Bestätigung 3 %", "Später, aber sicherer einsteigen", confirm="either", price_pct=3.0),
    Variant("V4", "Preis-Bestätigung 5 %", "Noch später einsteigen", confirm="either", price_pct=5.0),
    Variant("V5", "Nur MACD-Bestätigung", "Kontrolle: trägt das Booklet-Kriterium überhaupt?", confirm="macd"),
    Variant("V6", "Mindestkurs 5 (nur US)", "Pennystocks erzeugen die Extremwerte", min_price=5.0),
    Variant("V7", "Cooldown 20 Tage", "Weniger überlappende Signale", cooldown=20),
    Variant("V8", "StochRSI-Schwelle 0,05", "Strengere Extreme", rsi_th=0.05),
    Variant("V9", "StochRSI-Schwelle 0,2", "Lockerere Extreme (mehr Signale)", rsi_th=0.2),
    Variant("V10", "MACD 12/26/9", "Schnellerer MACD für die Bestätigung", macd_alt=True),
    Variant("V11", "V1 + V2 + V6", "Kombination der drei begründeten Filter",
            confirm="price", sell_down_only=True, min_price=5.0),
    Variant("V12", "V1 + V3 + V6", "wie V11, aber 3 % Preis-Bestätigung",
            confirm="either", price_pct=3.0, sell_down_only=True, min_price=5.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Vorbereitung pro Ticker
# ─────────────────────────────────────────────────────────────────────────────
def prepare(ticker: str, raw: pd.DataFrame, fwd: int = FORWARD_DAYS) -> SimpleNamespace | None:
    df = naive_index(clean_price_df(raw))
    if df is None or df.empty or len(df) < 260:
        return None
    df = add_strategy_indicators(df.copy())
    n = len(df)
    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float) if "Open" in df.columns else c.copy()
    arr = _criteria_arrays(df)
    ma = df["TREND_MA"].to_numpy(float)
    ratio = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio[1:] = c[1:] / c[:-1]
        anom = ((ratio > 2.0) | (ratio < 0.4)).astype(int)
    cs = np.cumsum(anom)
    s = np.arange(n)
    end = np.minimum(s + fwd - 1, n - 1)
    aw = (cs[end] - np.where(s > 0, cs[s - 1], 0)) > 0          # Datenfehler in den Tagen s .. s+fwd-1
    alt = compute_macd(df[["Close"]].copy(), fast=12, slow=26, signal=9)["OSMA"].to_numpy(float)
    idx = df.index
    return SimpleNamespace(
        ticker=ticker, n=n, c=c, o=o, h=df["High"].to_numpy(float), l=df["Low"].to_numpy(float), ma=ma, valid=arr["valid"], turn=arr["turn"], turn_alt=macd_turn_series(alt),
        sr=df["StochRSI_70"].to_numpy(float), sf=df["Stoch_Fast"].to_numpy(float), ss=df["Stoch_Slow"].to_numpy(float),
        sig=pd.Series(c).pct_change().rolling(60).std().to_numpy(),
        q=(idx.year * 4 + (idx.month - 1) // 3).to_numpy(), dates=idx.to_numpy("datetime64[ns]"), aw=aw,
        us=get_region(ticker) == "us", n_anom=int(aw.sum()))


# ─────────────────────────────────────────────────────────────────────────────
# Zufalls-Vergleich: Ticker-Schnitt + Marktabweichung am selben Tag (gleicher Trend)
# ─────────────────────────────────────────────────────────────────────────────
# Erwartung für ein Signal von Ticker i am Tag d (Trend-Zustand s):
#     E = T[i, s]  +  ( M[d, s] - M̄[s] )
#   T[i, s] = Ø aller Tage dieses Tickers im Trend-Zustand s (Volatilität/Drift des Tickers)
#   M[d, s] = Ø aller Ticker, die am selben Tag im selben Trend-Zustand eingestiegen wären (Marktlage des Tages)
#   M̄[s]   = Gesamtschnitt     -> nur die Abweichung der Marktlage vom Durchschnitt wird übertragen
# Bewusst NICHT "derselbe Ticker im selben Quartal": Dort steckt der eigene Kursverlauf des Signals im Vergleichswert
# (ein Kauf am Tief wird gegen Tage VOR dem Tief verglichen) – im Test mit Zufallskursen ergab das +2 PP Scheinüberschuss.
_COLS = ["ret", "h5", "hV"]
MIN_MKT = 30           # mind. so viele Ticker am selben Tag, sonst nur Ticker-Schnitt


def forward_stats(P, side: str, min_price: float, fwd: int = FORWARD_DAYS, k: float = VOL_K):
    """Alle gültigen 'Zufalls-Einstiege' des Tickers: (DataFrame d/up/ret/h5/hV, Ticker-Mittel je Trend) oder None."""
    m = P.n - fwd
    if m < 60:
        return None
    t = np.arange(m)
    e = P.o[t + 1]
    win = sliding_window_view(P.c, fwd)[1:m + 1]                 # Fenster ab Einstiegstag t+1
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = P.c[t + fwd] / e - 1
        fav = (win.max(axis=1) / e - 1) if side == "buy" else (1 - win.min(axis=1) / e)
    ret = (rr if side == "buy" else -rr) * 100
    fav = fav * 100
    thr = k * P.sig[t] * math.sqrt(fwd) * 100
    up = P.c[t] > P.ma[t]
    with np.errstate(invalid="ignore"):
        ok = P.valid[t] & np.isfinite(P.ma[t]) & np.isfinite(ret) & np.isfinite(thr) & (thr > 0) & ~P.aw[t + 1]
        if min_price > 0 and P.us:
            ok &= P.c[t] >= min_price
        h5, hV = fav >= 5.0, fav >= thr
    if ok.sum() < 50:
        return None
    d = pd.DataFrame({"d": P.dates[t + 1].astype("int64")[ok], "up": up[ok], "ret": ret[ok],
                      "h5": h5[ok].astype(float), "hV": hV[ok].astype(float)})
    T = {"all": tuple(d[_COLS].mean())}
    for flag in (True, False):
        sub = d[d["up"] == flag]
        T[flag] = tuple(sub[_COLS].mean()) if len(sub) >= 20 else T["all"]
    return d, T


class Market:
    """Querschnitts-Mittel aller Ticker je (Tag, Trend) – siehe Kommentar oben."""

    def __init__(self, parts: list[pd.DataFrame]):
        df = pd.concat(parts, ignore_index=True)
        g = df.groupby(["d", "up"])[_COLS]
        self.mu, self.cnt = g.mean().to_dict("index"), df.groupby(["d", "up"]).size().to_dict()
        self.mud, self.cntd = df.groupby("d")[_COLS].mean().to_dict("index"), df.groupby("d").size().to_dict()
        self.bar = df.groupby("up")[_COLS].mean().to_dict("index")
        self.barall = df[_COLS].mean().to_dict()


def make_lookup(T: dict, M: Market):
    """lookup(datum_ns, trend_up) -> (erwartete Rendite, erwartete Treffer 5 %, erwartete Treffer vola-normiert)."""
    def lookup(d: int, up: bool):
        t = T[bool(up)]
        key = (int(d), bool(up))
        if M.cnt.get(key, 0) >= MIN_MKT:
            m, mb = M.mu[key], M.bar[bool(up)]
        elif M.cntd.get(int(d), 0) >= MIN_MKT:
            m, mb = M.mud[int(d)], M.barall
        else:
            return t
        return tuple(t[i] + m[c] - mb[c] for i, c in enumerate(_COLS))
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Signale einer Variante (Mechanik = engine.find_entry_signals, parametrisiert)
# ─────────────────────────────────────────────────────────────────────────────
def gen_trades(P, v: Variant, side: str, lookup, fwd: int = FORWARD_DAYS, k: float = VOL_K) -> list[tuple]:
    n, c, o, ma = P.n, P.c, P.o, P.ma
    buy = side == "buy"
    with np.errstate(invalid="ignore"):
        gate = ((P.sr > 0) & (P.sr < v.rsi_th) & (P.sf < 10) & (P.ss < 15)) if buy else \
               ((P.sr > 1 - v.rsi_th) & (P.sf > 90) & (P.ss > 85))
    idx = np.flatnonzero(gate & P.valid)
    events, prev = [], -10 ** 9
    for i in idx:
        i = int(i)
        if i - prev > v.cooldown:
            events.append(i)
        prev = i
    turn = P.turn_alt if v.macd_alt else P.turn
    tdir = 1 if buy else -1
    h, l = P.h, P.l          # laufendes Hoch/Tief für die Extrem-Verfolgung
    rows, seen = [], set()
    for i in events:
        ext, j_entry, trigger = None, None, None
        for j in range(i, min(i + MAX_ENTRY_WAIT, n - 1) + 1):
            if buy:
                ext = l[j] if ext is None else min(ext, l[j])
            else:
                ext = h[j] if ext is None else max(ext, h[j])
            if v.confirm in ("either", "macd") and turn[j] == tdir:
                trigger, j_entry = "MACD", j
                break
            if v.confirm in ("either", "price"):
                if buy and c[j] > ext * (1 + v.price_pct / 100):
                    trigger, j_entry = "Preis", j
                    break
                if (not buy) and c[j] < ext * (1 - v.price_pct / 100):
                    trigger, j_entry = "Preis", j
                    break
        if j_entry is None or j_entry + 1 >= n:
            continue
        ex = j_entry + 1
        e = o[ex]
        if not (np.isfinite(e) and e > 0 and np.isfinite(ext) and ext > 0 and np.isfinite(ma[j_entry])):
            continue
        up = bool(c[j_entry] > ma[j_entry])
        if (not buy) and v.sell_down_only and up:
            continue
        if v.min_price > 0 and P.us and e < v.min_price:
            continue
        if ex in seen or ex + fwd > n or P.aw[ex]:
            continue
        thr = k * P.sig[ex - 1] * math.sqrt(fwd) * 100
        if not (np.isfinite(thr) and thr > 0):
            continue
        seen.add(ex)
        w = c[ex:ex + fwd]
        fav = ((w.max() / e - 1) if buy else (1 - w.min() / e)) * 100
        r = (w[-1] / e - 1) * 100
        r = r if buy else -r
        b = lookup(int(P.dates[ex].astype('int64')), up)
        rows.append((v.key, P.ticker, side, P.dates[ex], trigger, "up" if up else "down",
                     r, float(fav >= 5.0), float(fav >= thr), b[0], b[1], b[2]))
    return rows


COLS = ["variant", "ticker", "side", "date", "trigger", "trend", "ret", "h5", "hV", "b_ret", "b_h5", "b_hV"]


# ─────────────────────────────────────────────────────────────────────────────
# Statistik
# ─────────────────────────────────────────────────────────────────────────────
def agg(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    month = pd.to_datetime(df["date"]).dt.to_period("M")
    raw = df["ret"] - df["b_ret"]
    exc = raw.clip(-CLIP_PP, CLIP_PP)
    t_exc, m = cluster_t(exc, month)
    t_h5, _ = cluster_t(df["h5"] - df["b_h5"], month)
    t_hV, _ = cluster_t(df["hV"] - df["b_hV"], month)
    return {
        "n": n, "n_tickers": int(df["ticker"].nunique()), "n_months": m,
        "ret_mean": float(df["ret"].clip(-100, 300).mean()), "ret_median": float(df["ret"].median()),
        "exc_mean": float(exc.mean()), "exc_median": float(raw.median()), "t_exc": t_exc, "p_exc": p_two_sided(t_exc),
        "edge_h5": float((df["h5"] - df["b_h5"]).mean()) * 100, "t_h5": t_h5,
        "edge_hV": float((df["hV"] - df["b_hV"]).mean()) * 100, "t_hV": t_hV,
        "rate_h5": float(df["h5"].mean()) * 100, "base_h5": float(df["b_h5"].mean()) * 100,
    }


def _f(x, spec="+.1f"):
    return "–" if x is None or not np.isfinite(x) else format(x, spec)


def verdict(tr: dict, te: dict) -> str:
    if te.get("n", 0) < MIN_N_TEST or tr.get("n", 0) < MIN_N_TEST:
        return "zu wenig Signale"
    ok = lambda s: np.isfinite(s["t_exc"]) and s["t_exc"] > T_SIG and s["exc_mean"] > 0
    bad = lambda s: np.isfinite(s["t_exc"]) and s["t_exc"] < -T_SIG and s["exc_mean"] < 0
    if ok(tr) and ok(te):
        return "**Hinweis auf Edge** (Training + Test)"
    if bad(tr) and bad(te):
        return "schlechter als Zufall (beide)"
    if ok(tr) and not ok(te):
        return "nur Training (Overfitting-Verdacht)"
    if ok(te) and not ok(tr):
        return "nur Test"
    return "kein Effekt"


def _cell(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "–"
    return f"{_f(s['exc_mean'])} PP (t={_f(s['t_exc'], '.1f')})"


HEAD = ("| Variante | n Training | Überschuss Training | n Test | Überschuss Test | Median-Überschuss Test | "
        "Treffer-Edge Test (vola-norm.) | Urteil |\n|---|---|---|---|---|---|---|---|")


def build_report(trades: pd.DataFrame, meta: dict, split: str) -> tuple[str, dict]:
    summ = {"meta": meta, "variants": {}, "reference": {}}
    d = pd.to_datetime(trades["date"])
    is_tr = d < pd.Timestamp(split)
    out = [f"# Parameter-Sweep mit Training/Test-Trennung\n",
           f"Stand {meta['created']} · Daten {meta['period']} · {meta['n_tickers']} Ticker · {len(VARIANTS)} Varianten · "
           f"Training = Einstieg vor {split}, Test = ab {split}\n",
           "**Überschuss** = Ø 40-Tage-Rendite (Open Folgetag → Close, in Signalrichtung) minus erwartete Rendite = Ticker-Schnitt im selben "
           "Trend (über/unter SMA200) + Marktabweichung des Einstiegstags. Bei ±50 PP gekappt. **t** = Cluster-t über "
           f"Kalendermonate; ab t > {T_SIG:.0f} in Training UND Test gilt eine Variante als Hinweis auf einen echten Edge.\n"]
    for side, name in (("buy", "Kaufsignale"), ("sell", "Verkaufssignale")):
        out.append(f"\n## {name}\n")
        out.append(HEAD)
        for v in VARIANTS:
            sel = (trades["variant"] == v.key) & (trades["side"] == side)
            tr, te = agg(trades[sel & is_tr]), agg(trades[sel & ~is_tr])
            vd = verdict(tr, te)
            summ["variants"][f"{v.key} | {side}"] = {"label": v.label, "train": tr, "test": te, "verdict": vd}
            out.append(f"| {v.key} {v.label} | {tr.get('n', 0)} | {_cell(tr)} | {te.get('n', 0)} | {_cell(te)} | "
                       f"{_f(te.get('exc_median'))} PP | {_f(te.get('edge_hV'))} PP (t={_f(te.get('t_hV'), '.1f')}) | {vd} |")
    # Referenz V0 mit datumsgleichem Zufall
    base = trades[trades["variant"] == "V0"]
    out.append("\n## Referenz V0 (Basis) – Aufschlüsselungen mit datumsgleichem Zufall\n")
    hd = ("| Gruppe | n | Ø Rendite | Median Rendite | Überschuss (Ø, gekappt) | Median-Überschuss | t (Cluster) | "
          "Treffer-Edge 5 % | Treffer-Edge vola-norm. |\n|---|---|---|---|---|---|---|---|---|")

    def row(label, sub, key):
        s = agg(sub)
        summ["reference"][key] = s
        if s.get("n", 0) == 0:
            return f"| {label} | 0 | | | | | | | |"
        return (f"| {label} | {s['n']} | {_f(s['ret_mean'])} % | {_f(s['ret_median'])} % | {_f(s['exc_mean'])} PP | "
                f"{_f(s['exc_median'])} PP | {_f(s['t_exc'], '.1f')} | {_f(s['edge_h5'])} PP | {_f(s['edge_hV'])} PP |")
    for title, keyf, vals in (("Trend zum Einstieg", "trend", ("up", "down")), ("Auslöser", "trigger", ("MACD", "Preis"))):
        out.append(f"\n### {title}\n")
        out.append(hd)
        for sd in ("buy", "sell"):
            for x in vals:
                out.append(row(f"{sd} · {x}", base[(base.side == sd) & (base[keyf] == x)], f"{title}|{sd}|{x}"))
    out.append("\n### Nach Jahr (Zufall jetzt jahresgleich – Haussejahre benachteiligen Verkauf nicht mehr automatisch)\n")
    out.append(hd)
    years = sorted(pd.to_datetime(base["date"]).dt.year.unique())
    for sd in ("buy", "sell"):
        for y in years:
            sub = base[(base.side == sd) & (pd.to_datetime(base["date"]).dt.year == y)]
            out.append(row(f"{sd} · {y}", sub, f"Jahr|{sd}|{y}"))
    ok = [(k, x) for k, x in summ["variants"].items() if "Hinweis" in x["verdict"]]
    out.append("\n## Fazit (automatisch)\n")
    if ok:
        out.append(f"Varianten mit Überschuss in Training UND Test (t > {T_SIG:.0f}): " +
                   ", ".join(f"**{k}** ({x['label']})" for k, x in ok) + ".\n")
        out.append(f"Bei {2 * len(VARIANTS)} Tests ist rein zufällig mit ~1 Treffer zu rechnen. Erst glauben, wenn der Effekt "
                   "(a) in der Jahres-Tabelle in der Mehrheit der Jahre sichtbar ist, (b) im Median-Überschuss ebenfalls "
                   "positiv ist und (c) das Signal-Journal ihn live bestätigt.\n")
    else:
        out.append("Keine Variante hat in Training UND Test einen belastbaren Überschuss gegenüber dem datumsgleichen Zufall. "
                   "Die Parameter sind damit nicht der Engpass; das Journal (Live-Daten) bleibt die entscheidende Prüfung.\n")
    out.append("\n## Grenzen\n")
    out.append("- Survivorship-Bias (heutige Ticker), keine Kosten, überlappende 40-Tage-Fenster, Mindestkurs nur für US-Ticker.\n"
               "- Der Zufallsvergleich überträgt die Marktlage des Tages additiv auf den Ticker-Schnitt (ohne Beta-Skalierung): Hoch-Beta-Titel werden in Crash-Tagen etwas zu günstig bewertet.\n"
               "- Verkauf ist ein Abwärts-Signal, kein Short-Backtest (kein Leihzins, keine Squeeze-Risiken).\n"
               f"- Datenfehler-Filter verwarf {meta['n_anomaly_windows']} Kurs-Fenster mit Sprüngen > +100 % / < -60 %.\n")
    return "\n".join(out), summ


# ─────────────────────────────────────────────────────────────────────────────
def run(period: str, out_dir: str, limit: int = 0, tickers: list[str] | None = None, split: str = SPLIT_DEFAULT,
        frames: dict | None = None) -> pd.DataFrame:
    if frames is None:
        tickers = tickers or get_all_tickers()
        if limit:
            tickers = tickers[:limit]
        log.info("Lade %d Ticker (%s) …", len(tickers), period)
        frames, failed = download_price_frames(tickers, period=period, progress_cb=lambda d, t, p: log.info("Download %d/%d", d, t))
    else:
        failed = []
    Ps, n_anom = [], 0
    for t, raw in frames.items():
        try:
            P = prepare(t, raw)
        except Exception as e:
            log.warning("%s: %s", t, e)
            continue
        if P is not None:
            Ps.append(P)
            n_anom += P.n_anom
    used = len(Ps)
    log.info("Durchlauf 1/2: Zufalls-Vergleich (Markt je Tag) für %d Ticker …", used)
    mps = sorted({v.min_price for v in VARIANTS})
    parts, Ts = {}, {}
    for i, P in enumerate(Ps, 1):
        for side in ("buy", "sell"):
            for mp in mps:
                r = forward_stats(P, side, mp if P.us else 0.0)
                if r is None:
                    continue
                parts.setdefault((side, mp), []).append(r[0])
                Ts[(P.ticker, side, mp)] = r[1]
        if i % 200 == 0:
            log.info("Vergleichswerte %d/%d", i, used)
    markets = {key: Market(p) for key, p in parts.items()}
    del parts
    log.info("Durchlauf 2/2: Signale der %d Varianten …", len(VARIANTS))
    rows = []
    for i, P in enumerate(Ps, 1):
        try:
            for side in ("buy", "sell"):
                for v in VARIANTS:
                    T = Ts.get((P.ticker, side, v.min_price))
                    if T is None:
                        continue
                    rows.extend(gen_trades(P, v, side, make_lookup(T, markets[(side, v.min_price)])))
        except Exception as e:
            log.warning("%s: %s", P.ticker, e)
        if i % 100 == 0:
            log.info("ausgewertet %d/%d, bisher %d Signale (alle Varianten)", i, used, len(rows))
    trades = pd.DataFrame(rows, columns=COLS)
    os.makedirs(out_dir, exist_ok=True)
    if trades.empty:
        log.error("Keine Signale – nichts zu schreiben.")
        return trades
    meta = {"created": datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC"), "period": period, "n_tickers": used,
            "n_failed": len(failed), "failed": sorted(failed), "n_anomaly_windows": int(n_anom), "split": split}
    report, summ = build_report(trades, meta, split)
    with open(os.path.join(out_dir, "SWEEP_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=lambda o: None if o != o else float(o))
    trades[trades["variant"] == "V0"].to_csv(os.path.join(out_dir, "sweep_trades_V0.csv"), index=False)
    log.info("Fertig: %d Signale (alle Varianten) → %s", len(trades), out_dir)
    return trades


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="10y")
    ap.add_argument("--out", default="sweep_results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default=SPLIT_DEFAULT, help="Trennung Training/Test (Einstiegsdatum)")
    a = ap.parse_args()
    run(a.period, a.out, a.limit, split=a.split)

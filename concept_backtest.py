# concept_backtest.py — Gesamttest des Konzepts: Einstieg + Marktbreite + Stopp/Ziel (Elder), gegen Zufall mit GLEICHER Verwaltung
#
#   python concept_backtest.py --period 10y --out concept_results
#   python concept_backtest.py --period 5y --limit 120            (Schnelltest)
#
# Frage: Hat das GESAMTE Konzept einen Vorteil – nicht nur der Einstieg, sondern Einstieg + Marktfilter + Stopp/Ziel?
#
# Verwaltung ("M1", so wie Elder es verlangt): Einstieg zum Open des Folgetags, Stopp = 2 x ATR(14) vom Einstieg, Ziel = 2 R
# (Chance/Risiko 2:1), Stopp auf Break-even sobald +1 R erreicht, Zeitstopp nach 40 Handelstagen. Ergebnis = R-Vielfaches
# (Gewinn in Einheiten des anfangs riskierten Betrags). Zusätzlich die alte Messung "H40" (Halten 40 Tage, Rendite in %).
#
# Vergleichsmaßstab (wie im Sweep): Erwartung = Ticker-Schnitt im gleichen Trendzustand + Abweichung der Marktlage am Einstiegstag.
# Die Vergleichswerte werden mit EXAKT DERSELBEN Verwaltung (Stopp/Ziel/Zeitstopp) für JEDEN Tag berechnet. Das ist wichtig:
# Stopps verändern die Gewinnverteilung, ein fester Stopp hat auch bei Zufallseinstiegen einen eigenen Erwartungswert.
#
# Konzept-Bausteine (alles ohne Vorgriff, Signaltag = Schlusskurs, Einstieg am Folgetag):
#   Einstiege   S1  4/4-Signal der Mail (alles am selben Tag)           S2  Variante "prior" (StochRSI/Stoch vor <=10 Tagen)
#               E1  Triple Screen: Wochen-EMA13 steigt, Wochen-Impuls nicht rot, Tages-Slow-Stochastik(5) unter ihrer 5-%-Linie
#               E2  E1 + Kurs höchstens in der Value Zone (EMA13/26)
#               E3  "Einser-Trade-light": Trade-Apgar-Näherung >= 7 ohne Null-Zeile (Fehlausbruch nur 0/1, keine Divergenz)
#   Marktfilter BR  Marktbreite steigt: Anteil Aktien über SMA50 > Wert vor 5 Tagen (Elder, Kap. 35)
#               BL  Marktbreite war <= 35 % und dreht nach oben (Elders "Wende am Extrem")
#               NH  NH-NL(20 Tage) des Universums steigt (Kap. 34)
#   Kombinationen: S1/S2/E1/E2/E3 jeweils auch mit BR, BL, NH. Verkauf ist gespiegelt.
#
# Verdikt wie im Sweep: Überschuss-R > 0 mit Cluster-t > 2 in TRAINING UND TEST (und genug Signale).

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

from engine import download_price_frames
from database import get_all_tickers
from pooled_backtest import cluster_t, p_two_sided
import sweep_backtest as sb

log = logging.getLogger("concept")

SPLIT_DEFAULT = "2022-01-01"
N_HOLD = 40            # Zeitstopp / Haltedauer
ATR_MULT = 2.0         # Stopp-Abstand in ATR
TARGET_R = 2.0         # Ziel in R
BE_R = 1.0             # ab +1 R Stopp auf Einstand
COOLDOWN = 10
MIN_N_TEST = 100
T_SIG = 2.0
CLIP_R = 6.0           # R-Werte werden für die Mittelung bei +-6 gekappt (Lücken)


# ─────────────────────────────────────────────────────────────────────────────
# Kleine Hilfen
# ─────────────────────────────────────────────────────────────────────────────
def _any_recent(b: np.ndarray, n: int) -> np.ndarray:
    return sb._any_recent(np.asarray(b, bool), n)


def _shift(a: np.ndarray, k: int, fill=np.nan) -> np.ndarray:
    out = np.full(a.shape, fill, dtype=a.dtype if a.dtype != bool else bool)
    if k > 0:
        out[k:] = a[:-k]
    else:
        out[:] = a
    return out


def _roll(a: np.ndarray, n: int, how: str) -> np.ndarray:
    s = pd.Series(a).rolling(n, min_periods=n)
    return getattr(s, how)().to_numpy()


def _ref_lines(x: np.ndarray, lookback: int = 125, step: int = 63) -> tuple[np.ndarray, np.ndarray]:
    """Elders 5-%-Regel: Bezugslinien = 5./95. Perzentil der letzten ~6 Monate, alle 3 Monate (63 Tage) neu – nur Vergangenheit."""
    n = len(x)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    for s in range(lookback, n, step):
        w = x[s - lookback:s]
        w = w[np.isfinite(w)]
        if len(w) >= 60:
            lo[s:s + step], hi[s:s + step] = np.percentile(w, 5), np.percentile(w, 95)
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Zusatz-Größen je Ticker (aufbauend auf sweep_backtest.prepare)
# ─────────────────────────────────────────────────────────────────────────────
def extend(P, raw_df: pd.DataFrame | None = None) -> SimpleNamespace:
    c, h, l = P.c, P.h, P.l
    n = P.n
    pc = _shift(c, 1)
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    P.atr = _roll(tr, 14, "mean")
    ema13, ema26 = sb._ema(c, 13), sb._ema(c, 26)
    P.ema13, P.ema26 = ema13, ema26
    lo5, hi5 = _roll(l, 5, "min"), _roll(h, 5, "max")
    with np.errstate(invalid="ignore", divide="ignore"):
        k5 = (c - lo5) / (hi5 - lo5) * 100
    P.stoch5 = pd.Series(k5).rolling(3).mean().rolling(3).mean().to_numpy()
    P.st_lo, P.st_hi = _ref_lines(P.stoch5)
    # Impulse (EMA13-Neigung + MACD-H(12,26,9)-Neigung); Wochenwerte = letzte abgeschlossene Woche
    E = P.elder
    e_s = np.r_[np.nan, np.sign(np.diff(ema13))]
    h_s = np.r_[np.nan, np.sign(np.diff(P.osma_alt))]
    P.d_green, P.d_red = (e_s > 0) & (h_s > 0), (e_s < 0) & (h_s < 0)
    P.w_green = (E["w_ema_s"] > 0) & (E["w_h_s"] > 0)
    P.w_red = (E["w_ema_s"] < 0) & (E["w_h_s"] < 0)
    P.w_ema_s, P.w_h, P.w_h_s = E["w_ema_s"], E["w_h"], E["w_h_s"]
    P.d_h, P.d_h_s = P.osma_alt, np.r_[np.nan, np.sign(np.diff(P.osma_alt))]
    P.zone_lo, P.zone_hi = np.minimum(ema13, ema26), np.maximum(ema13, ema26)
    # Fehlausbruch: Tief/Hoch der Tage i-37..i-8 wird in den letzten 8 Tagen durchbrochen, Schluss wieder darüber/darunter
    ref_lo = _shift(_roll(l, 30, "min"), 8)
    ref_hi = _shift(_roll(h, 30, "max"), 8)
    with np.errstate(invalid="ignore"):
        P.fb_buy = (_roll(l, 8, "min") < ref_lo) & (c > ref_lo)
        P.fb_sell = (_roll(h, 8, "max") > ref_hi) & (c < ref_hi)
    P.sma50 = _roll(c, 50, "mean")
    P.hh20, P.ll20 = _roll(c, 20, "max"), _roll(c, 20, "min")
    return P


# ─────────────────────────────────────────────────────────────────────────────
# Einstiegs-Gates (Kauf; Verkauf gespiegelt)
# ─────────────────────────────────────────────────────────────────────────────
def _edge(cond: np.ndarray) -> np.ndarray:
    cond = np.asarray(cond, bool)
    prev = np.zeros(len(cond), bool)
    prev[1:] = cond[:-1]
    return cond & ~prev


def apgar_light(P, buy: bool) -> tuple[np.ndarray, np.ndarray]:
    """Näherung des Trade-Apgar (Elder Kap. 58): (Punkte 0..9, keine Null in Q1-Q3/Q5 und Fehlausbruch erfolgt)."""
    c = P.c
    adv_w, adv_d = (P.w_red, P.d_red) if buy else (P.w_green, P.d_green)
    def imp(adv):
        after = _any_recent(_shift(adv, 1, False), 15)
        q = np.ones(len(c))
        q[adv] = 0
        q[(~adv) & ((~P.w_green & ~P.w_red) if adv is adv_w else (~P.d_green & ~P.d_red)) & after] = 2
        return q
    q1, q2 = imp(adv_w), imp(adv_d)
    if buy:
        q3 = np.where(c > P.zone_hi, 0, np.where(c >= P.zone_lo, 1, 2)).astype(float)
        q4 = P.fb_buy.astype(float)
        ide_w = (P.w_h < 0) & (P.w_h_s > 0)
        ide_d = (P.d_h < 0) & (P.d_h_s > 0) & (c <= P.zone_hi)
    else:
        q3 = np.where(c < P.zone_lo, 0, np.where(c <= P.zone_hi, 1, 2)).astype(float)
        q4 = P.fb_sell.astype(float)
        ide_w = (P.w_h > 0) & (P.w_h_s < 0)
        ide_d = (P.d_h > 0) & (P.d_h_s < 0) & (c >= P.zone_lo)
    q5 = ide_w.astype(float) + ide_d.astype(float)
    tot = q1 + q2 + q3 + q4 + q5
    nozero = (q1 > 0) & (q2 > 0) & (q3 > 0) & (q4 > 0) & (q5 > 0)
    return tot, nozero


def base_gates(P, buy: bool) -> dict[str, np.ndarray]:
    """Rohe Einstiegsbedingungen je Tag (vor Flanke/Cooldown)."""
    s = 1.0 if buy else -1.0
    out = {}
    v0 = sb.Variant("S1", "", "", all4="same", confirm="none", max_wait=0)
    v1 = sb.Variant("S2", "", "", all4="prior", recent_days=10, confirm="none", max_wait=0)
    out["S1"] = sb.all4_gate(P, v0, buy)
    out["S2"] = sb.all4_gate(P, v1, buy)
    with np.errstate(invalid="ignore"):
        w_ok = ((P.w_ema_s == s) & ~(P.w_red if buy else P.w_green))
        pull = (P.stoch5 <= P.st_lo) if buy else (P.stoch5 >= P.st_hi)
        e1 = w_ok & pull
        inzone = (P.c <= P.zone_hi) if buy else (P.c >= P.zone_lo)
        out["E1"] = e1
        out["E2"] = e1 & inzone
        tot, nozero = apgar_light(P, buy)
        out["E3"] = (tot >= 7) & nozero & np.isfinite(tot) & (P.w_ema_s == s)
    return out


def overlays(M: "MarketCtx", P, buy: bool) -> dict[str, np.ndarray]:
    """Marktfilter je Tag des Tickers (aus den Querschnitts-Reihen des gesamten Universums)."""
    d = pd.DatetimeIndex(P.dates)
    b = M.breadth.reindex(d).to_numpy()
    nh = M.nhnl.reindex(d).to_numpy()
    with np.errstate(invalid="ignore"):
        if buy:
            br = b > _shift(b, 5)
            bl = (np.fmin.reduce([_shift(b, k) for k in range(1, 6)]) <= 0.35) & (b > _shift(b, 1)) & (b > _shift(b, 3))
            nhf = nh > _shift(nh, 5)
        else:
            br = b < _shift(b, 5)
            bl = (np.fmax.reduce([_shift(b, k) for k in range(1, 6)]) >= 0.75) & (b < _shift(b, 1)) & (b < _shift(b, 3))
            nhf = nh < _shift(nh, 5)
    return {"": np.ones(P.n, bool), "+BR": br, "+BL": bl, "+NH": nhf}


@dataclass
class MarketCtx:
    breadth: pd.Series      # Anteil Aktien über SMA50 (0..1)
    nhnl: pd.Series         # (Anzahl 20-Tage-Hochs − Anzahl 20-Tage-Tiefs) / Anzahl, geglättet über 5 Tage


def build_market(Ps: list) -> MarketCtx:
    ab, hh, ll = [], [], []
    for P in Ps:
        idx = pd.DatetimeIndex(P.dates)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(P.sma50)
            ab.append(pd.Series(np.where(valid, (P.c > P.sma50).astype(float), np.nan), index=idx))
            v2 = np.isfinite(P.hh20)
            hh.append(pd.Series(np.where(v2, (P.c >= P.hh20).astype(float), np.nan), index=idx))
            ll.append(pd.Series(np.where(v2, (P.c <= P.ll20).astype(float), np.nan), index=idx))
    A = pd.concat(ab, axis=1)
    H = pd.concat(hh, axis=1)
    L = pd.concat(ll, axis=1)
    cnt = A.notna().sum(axis=1)
    breadth = A.mean(axis=1).where(cnt >= 30)
    n2 = H.notna().sum(axis=1)
    nhnl = ((H.sum(axis=1) - L.sum(axis=1)) / n2.replace(0, np.nan)).where(n2 >= 30).rolling(5, min_periods=1).mean()
    return MarketCtx(breadth=breadth, nhnl=nhnl)


# ─────────────────────────────────────────────────────────────────────────────
# Verwaltung: Stopp / Ziel / Break-even / Zeitstopp – für alle Starttage gleichzeitig
# ─────────────────────────────────────────────────────────────────────────────
def simulate_all(P, buy: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Für JEDEN Signaltag s (Einstieg Open s+1): (R-Vielfaches, 40-Tage-Rendite in % in Signalrichtung, gültig)."""
    n = P.n
    m = n - N_HOLD - 2
    R, ret, ok = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    if m < 60:
        return R, ret, ok
    s = np.arange(m)
    e = P.o[s + 1]
    risk = ATR_MULT * P.atr[s]
    sgn = 1.0 if buy else -1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        good = np.isfinite(e) & (e > 0) & np.isfinite(risk) & (risk > 0) & (risk < 0.5 * e) & ~P.aw[s + 1] & P.valid[s]
        stop = e - sgn * risk
        tgt = e + sgn * TARGET_R * risk
        be_trig = e + sgn * BE_R * risk
        alive = good.copy()
        out = np.full(m, np.nan)
        be = np.zeros(m, bool)
        for k in range(N_HOLD):
            j = s + 1 + k
            o_, h_, l_, c_ = P.o[j], P.h[j], P.l[j], P.c[j]
            st = np.where(be, e, stop)
            if buy:
                gap = alive & (o_ <= st)
                hit_s = alive & ~gap & (l_ <= st)
                hit_t = alive & ~gap & ~hit_s & (h_ >= tgt)
                gap_px = o_
            else:
                gap = alive & (o_ >= st)
                hit_s = alive & ~gap & (h_ >= st)
                hit_t = alive & ~gap & ~hit_s & (l_ <= tgt)
                gap_px = o_
            out = np.where(gap, sgn * (gap_px - e) / risk, out)
            out = np.where(hit_s, sgn * (st - e) / risk, out)
            out = np.where(hit_t, TARGET_R, out)
            alive = alive & ~(gap | hit_s | hit_t)
            if k == N_HOLD - 1:
                out = np.where(alive, sgn * (c_ - e) / risk, out)
                alive = alive & False
            else:
                be = be | (alive & ((h_ >= be_trig) if buy else (l_ <= be_trig)))
        rr = sgn * (P.c[s + N_HOLD] / e - 1) * 100
    R[:m] = np.where(good, np.clip(out, -CLIP_R, CLIP_R), np.nan)
    ret[:m] = np.where(good & np.isfinite(rr), np.clip(rr, -100, 300), np.nan)
    ok[:m] = good & np.isfinite(out)
    return R, ret, ok


_COLS = ["R", "ret", "win"]
MIN_MKT = 30


class Market2:
    def __init__(self, parts: list[pd.DataFrame]):
        df = pd.concat(parts, ignore_index=True)
        g = df.groupby(["d", "up"])[_COLS]
        self.mu, self.cnt = g.mean().to_dict("index"), df.groupby(["d", "up"]).size().to_dict()
        self.mud, self.cntd = df.groupby("d")[_COLS].mean().to_dict("index"), df.groupby("d").size().to_dict()
        self.bar = df.groupby("up")[_COLS].mean().to_dict("index")
        self.barall = df[_COLS].mean().to_dict()


def make_lookup(T: dict, M: Market2):
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


def baseline_parts(P, buy: bool, sim) -> tuple[pd.DataFrame, dict] | None:
    R, ret, ok = sim
    if ok.sum() < 50:
        return None
    s = np.flatnonzero(ok)
    up = (P.c[s] > P.ma[s]) & np.isfinite(P.ma[s])
    d = pd.DataFrame({"d": P.dates[s + 1].astype("int64"), "up": up, "R": R[s], "ret": ret[s], "win": (R[s] > 0).astype(float)})
    d = d[np.isfinite(d["R"]) & np.isfinite(d["ret"])]
    if len(d) < 50:
        return None
    T = {"all": tuple(d[_COLS].mean())}
    for flag in (True, False):
        sub = d[d["up"] == flag]
        T[flag] = tuple(sub[_COLS].mean()) if len(sub) >= 20 else T["all"]
    return d, T


# ─────────────────────────────────────────────────────────────────────────────
# Varianten
# ─────────────────────────────────────────────────────────────────────────────
BASES = {"S1": "4/4-Signal (Mail)", "S2": "4/4 mit StochRSI/Stoch vor ≤10 T.", "E1": "Triple Screen (Woche + Stoch-5-Rücksetzer)",
         "E2": "Triple Screen + höchstens in Value Zone", "E3": "Einser-Trade-light (Apgar ≥ 7, keine Null)"}
FILTERS = {"": "", "+BR": " + Marktbreite steigt", "+BL": " + Breite dreht aus ≤35 %", "+NH": " + NH-NL steigt"}
VARIANTS = [(b + f) for b in BASES for f in FILTERS]


def label(key: str) -> str:
    for f in ("+BR", "+BL", "+NH"):
        if key.endswith(f):
            return BASES[key[:-3]] + FILTERS[f]
    return BASES[key]


COLS = ["variant", "ticker", "side", "date", "trend", "R", "ret", "win", "b_R", "b_ret", "b_win"]


def gen_signals(P, buy: bool, M: MarketCtx, lookup, sim) -> list[tuple]:
    R, ret, ok = sim
    gates = base_gates(P, buy)
    ov = overlays(M, P, buy)
    side = "buy" if buy else "sell"
    rows = []
    for bkey, g in gates.items():
        for fkey, f in ov.items():
            cond = np.asarray(g, bool) & f & P.valid
            ev, prev = [], -10 ** 9
            for i in np.flatnonzero(_edge(cond)):
                if i - prev > COOLDOWN:
                    ev.append(int(i))
                    prev = i
            for i in ev:
                if i >= len(ok) or not ok[i]:
                    continue
                up = bool(P.c[i] > P.ma[i]) if np.isfinite(P.ma[i]) else False
                b = lookup(int(P.dates[i + 1].astype("int64")), up)
                rows.append((bkey + fkey, P.ticker, side, P.dates[i + 1], "up" if up else "down",
                             float(R[i]), float(ret[i]), float(R[i] > 0), b[0], b[1], b[2]))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Statistik
# ─────────────────────────────────────────────────────────────────────────────
def agg(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    month = pd.to_datetime(df["date"]).dt.to_period("M")
    exc = df["R"] - df["b_R"]
    exc_ret = (df["ret"] - df["b_ret"]).clip(-sb.CLIP_PP, sb.CLIP_PP)
    t_r, m = cluster_t(exc, month)
    t_ret, _ = cluster_t(exc_ret, month)
    return {"n": n, "n_tickers": int(df["ticker"].nunique()), "n_months": m,
            "R": float(df["R"].mean()), "b_R": float(df["b_R"].mean()), "exc_R": float(exc.mean()), "t_R": t_r,
            "win": float(df["win"].mean()) * 100, "b_win": float(df["b_win"].mean()) * 100,
            "exc_ret": float(exc_ret.mean()), "t_ret": t_ret, "ret": float(df["ret"].mean())}


def _f(x, spec="+.2f"):
    return "–" if x is None or (isinstance(x, float) and x != x) else format(x, spec)


def verdict(tr: dict, te: dict) -> str:
    if tr.get("n", 0) < MIN_N_TEST or te.get("n", 0) < MIN_N_TEST:
        return "zu wenig Signale"
    ok = lambda s: s["exc_R"] > 0 and s["t_R"] == s["t_R"] and s["t_R"] > T_SIG
    bad = lambda s: s["exc_R"] < 0 and s["t_R"] == s["t_R"] and s["t_R"] < -T_SIG
    if ok(tr) and ok(te):
        return "✅ Hinweis auf Vorteil (beide Zeiträume)"
    if bad(tr) and bad(te):
        return "⛔ schlechter als Zufall (beide Zeiträume)"
    if ok(te) or ok(tr):
        return "⚠️ nur in einem Zeitraum"
    return "kein Vorteil"


def build_report(trades: pd.DataFrame, meta: dict, split: str) -> tuple[str, dict]:
    sp = pd.Timestamp(split)
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    L = [f"# Gesamttest des Konzepts (Stopp/Ziel + Marktbreite)", "",
         f"Stand {meta['created']} · {meta['n_tickers']} Ticker · {meta['period']} · Training < {split} / Test ab {split}", "",
         f"Verwaltung: Einstieg Open Folgetag · Stopp {ATR_MULT:g}×ATR(14) · Ziel {TARGET_R:g} R · Break-even ab +{BE_R:g} R · Zeitstopp {N_HOLD} Tage. "
         "Vergleich: dieselbe Verwaltung an jedem Tag (datumsgleich, gleicher Trendzustand). **Überschuss-R** = R des Signals minus erwartetes R des Zufalls.", ""]
    summ = {}
    for side, name in (("buy", "KAUF"), ("sell", "VERKAUF (Short)")):
        t = trades[trades["side"] == side]
        L += [f"## {name}", "",
              "| Variante | Beschreibung | n Training/Test | R (Signal) | R (Zufall) | Überschuss-R Tr. (t) | Überschuss-R Test (t) | Trefferquote Test (Zufall) | Verdikt |",
              "|---|---|---|---|---|---|---|---|---|"]
        for key in VARIANTS:
            d = t[t["variant"] == key]
            tr, te = agg(d[d["date"] < sp]), agg(d[d["date"] >= sp])
            al = agg(d)
            summ[f"{side}:{key}"] = {"train": tr, "test": te, "all": al}
            if al.get("n", 0) == 0:
                L.append(f"| {key} | {label(key)} | 0 | – | – | – | – | – | keine Signale |")
                continue
            cell = lambda s: "–" if s.get("n", 0) == 0 else f"{_f(s['exc_R'])} ({_f(s['t_R'], '+.1f')})"
            L.append(f"| {key} | {label(key)} | {tr.get('n', 0)} / {te.get('n', 0)} | {_f(al['R'])} | {_f(al['b_R'])} | {cell(tr)} | {cell(te)} | "
                     f"{_f(te.get('win'), '.0f')} % ({_f(te.get('b_win'), '.0f')} %) | {verdict(tr, te)} |")
        L.append("")
    L += ["## Wie liest man das?", "",
          "- **R** = Gewinn in Vielfachen des anfangs riskierten Betrags. 0 = Nullsumme, +0,10 = im Schnitt 10 % des Risikos Gewinn je Trade.",
          "- **Überschuss-R** zieht die Zufalls-Erwartung ab (gleicher Stopp/Ziel an allen Tagen, gleiche Marktlage). Ohne diesen Abzug würde ein Stopp-System allein schon gut oder schlecht aussehen.",
          "- Ein Vorteil gilt erst, wenn er im Training UND im Test mit Cluster-t > 2 auftaucht. Bei vielen Varianten tauchen einzelne Treffer auch zufällig auf (Mehrfachtest).",
          "- Kein Abzug von Gebühren/Slippage; Survivorship-Bias der Tickerliste; Verkaufs-Spalte = Leerverkauf ohne Leihkosten.",
          "- E3 ist nur eine Näherung des Trade-Apgar (Fehlausbruch 0/1, keine Divergenz; Elder bewertet Frage 4 und 5 von Hand)."]
    return "\n".join(L), summ


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
    Ps = []
    for t, raw in frames.items():
        try:
            P = sb.prepare(t, raw)
            if P is not None:
                Ps.append(extend(P))
        except Exception as e:
            log.warning("%s: %s", t, e)
    used = len(Ps)
    log.info("%d Ticker aufbereitet; berechne Marktbreite …", used)
    M = build_market(Ps)
    rows = []
    for side in ("buy", "sell"):
        buy = side == "buy"
        log.info("%s: Zufalls-Vergleich mit Verwaltung für alle Tage …", side)
        parts, Ts, sims = [], {}, {}
        for P in Ps:
            sim = simulate_all(P, buy)
            r = baseline_parts(P, buy, sim)
            if r is None:
                continue
            sims[P.ticker] = sim
            parts.append(r[0])
            Ts[P.ticker] = r[1]
        if not parts:
            continue
        Mk = Market2(parts)
        del parts
        log.info("%s: Signale aller Varianten …", side)
        for i, P in enumerate(Ps, 1):
            if P.ticker not in Ts:
                continue
            try:
                rows.extend(gen_signals(P, buy, M, make_lookup(Ts[P.ticker], Mk), sims[P.ticker]))
            except Exception as e:
                log.warning("%s: %s", P.ticker, e)
            if i % 200 == 0:
                log.info("ausgewertet %d/%d, bisher %d Signale", i, used, len(rows))
    trades = pd.DataFrame(rows, columns=COLS)
    os.makedirs(out_dir, exist_ok=True)
    if trades.empty:
        log.error("Keine Signale – nichts zu schreiben.")
        return trades
    meta = {"created": datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC"), "period": period, "n_tickers": used,
            "n_failed": len(failed), "split": split}
    report, summ = build_report(trades, meta, split)
    with open(os.path.join(out_dir, "CONCEPT_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "concept_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=lambda o: None if o != o else float(o))
    log.info("Fertig: %d Signale → %s", len(trades), out_dir)
    return trades


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="10y")
    ap.add_argument("--out", default="concept_results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default=SPLIT_DEFAULT)
    a = ap.parse_args()
    run(a.period, a.out, a.limit, split=a.split)

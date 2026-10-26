# elder_view.py — "Was würde Elder sagen?"  Regelbasierte Einordnung nach Alexander Elder
# (Triple Screen, Impuls-System, Force Index, MACD-Histogramm-Divergenzen, Kanäle, Stopps, 2 %-/6 %-Regel).
#
# Quelle der Regeln: Elder, "Alles, was Sie über Trading wissen müssen" (Kap. 18–24, 28–30, 39–41, 48–54).
# Alles ist eine CHECKLISTE aus den Regeln des Buches, keine Prognose. Elder selbst nennt das Impuls-System ein
# Zensursystem ("sagt nicht, was man tun soll, sondern was man nicht tun darf") und betont Stopp, Positionsgröße
# und Chance/Risiko > 2:1 stärker als den Einstieg. Unser Backtest (984 Titel, 10 Jahre, 40-Tage-Rendite) fand für
# die reinen Elder-Kaufregeln keinen Überschuss gegenüber Zufall – der Nutzen liegt in Filter und Risikoplanung.
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MIN_BARS = 150
TICK = 0.01

IMPULSE_COL = {"green": "#4ade80", "red": "#f87171", "blue": "#60a5fa"}
IMPULSE_TXT = {"green": "grün (bullisch)", "red": "rot (bärisch)", "blue": "blau (neutral)"}


# ─────────────────────────────────────────────────────────────────────────────
# Indikatoren
# ─────────────────────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    need = ["Open", "High", "Low", "Close"]
    if df is None or df.empty or any(c not in df.columns for c in need):
        return None
    d = df.copy()
    if "Volume" not in d.columns:
        d["Volume"] = 0.0
    d = d[need + ["Volume"]].astype(float).dropna(subset=need)
    d.index = pd.to_datetime(d.index).tz_localize(None) if getattr(d.index, "tz", None) is not None else pd.to_datetime(d.index)
    return d


def _impulse(ema13: pd.Series, hist: pd.Series) -> pd.Series:
    """Impuls-System: EMA(13)-Neigung + MACD-Histogramm-Neigung. grün = beide steigen, rot = beide fallen, sonst blau."""
    e, h = np.sign(ema13.diff()), np.sign(hist.diff())
    out = pd.Series("blue", index=ema13.index, dtype=object)
    out[(e > 0) & (h > 0)] = "green"
    out[(e < 0) & (h < 0)] = "red"
    out[e.isna() | h.isna()] = "blue"
    return out


def indicators(d: pd.DataFrame) -> pd.DataFrame:
    """Alle Elder-Größen für einen Preisverlauf (Tages- oder Wochenbalken)."""
    x = d.copy()
    c, h, l, v = x["Close"], x["High"], x["Low"], x["Volume"]
    x["ema13"], x["ema26"] = _ema(c, 13), _ema(c, 26)
    macd = _ema(c, 12) - _ema(c, 26)
    x["macd"], x["macd_sig"] = macd, _ema(macd, 9)
    x["hist"] = x["macd"] - x["macd_sig"]
    x["impulse"] = _impulse(x["ema13"], x["hist"])
    fi = v * c.diff()
    x["fi2"], x["fi13"] = _ema(fi.fillna(0), 2), _ema(fi.fillna(0), 13)
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    k = (c - lo14) / (hi14 - lo14).replace(0, np.nan) * 100
    x["stoch"] = k.rolling(3).mean()
    # Elders Standard (Kap. 26/27): Slow-Stochastik über 5 Tage (%K 5, zweimal über 3 geglättet) und 13-Tage-RSI (Wilder)
    lo5, hi5 = l.rolling(5).min(), h.rolling(5).max()
    k5 = (c - lo5) / (hi5 - lo5).replace(0, np.nan) * 100
    x["stoch5"] = k5.rolling(3).mean().rolling(3).mean()
    dc = c.diff()
    gain = dc.clip(lower=0).ewm(alpha=1 / 13, adjust=False).mean()
    loss = (-dc.clip(upper=0)).ewm(alpha=1 / 13, adjust=False).mean()
    x["rsi13"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # Directional System (13)
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _ema(tr, 13)
    x["atr14"] = tr.rolling(14).mean()
    x["pdi"] = 100 * _ema(pd.Series(pdm, index=x.index), 13) / atr.replace(0, np.nan)
    x["mdi"] = 100 * _ema(pd.Series(mdm, index=x.index), 13) / atr.replace(0, np.nan)
    dx = 100 * (x["pdi"] - x["mdi"]).abs() / (x["pdi"] + x["mdi"]).replace(0, np.nan)
    x["adx"] = _ema(dx.fillna(0), 13)
    # OBV
    x["obv"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    return x


def to_weekly(d: pd.DataFrame) -> pd.DataFrame:
    """Wochenbalken Mo–Fr; die laufende Woche zählt mit (Elder aktualisiert Wochenindikatoren täglich)."""
    w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return w.dropna(subset=["Close"])


# ─────────────────────────────────────────────────────────────────────────────
# Bausteine
# ─────────────────────────────────────────────────────────────────────────────
def _segments(sign: np.ndarray):
    """Zusammenhängende Abschnitte gleichen Vorzeichens: Liste (start, end_inclusive, vorzeichen)."""
    out, s = [], 0
    for i in range(1, len(sign) + 1):
        if i == len(sign) or sign[i] != sign[s]:
            out.append((s, i - 1, int(sign[s])))
            s = i
    return out


def hist_divergence(x: pd.DataFrame, lookback: int, kind: str) -> dict | None:
    """
    MACD-Histogramm-Divergenz nach Elder (Kap. 23): zwei Tiefs (bullish) bzw. Hochs (bearish) des Kurses, dazwischen
    MUSS das Histogramm die Nulllinie durchbrechen. Kurs macht neues Extrem, Histogramm ein flacheres. Signal erst, wenn das
    Histogramm nach dem zweiten Extrem dreht (ein Balken kürzer als der Vorgänger). Beste Abstände: 20–40 Balken.
    """
    h = x["hist"].to_numpy()[-lookback:]
    lo = x["Low"].to_numpy()[-lookback:]
    hi = x["High"].to_numpy()[-lookback:]
    idx = x.index[-lookback:]
    if len(h) < 30 or np.isnan(h).any():
        return None
    bull = kind == "bull"
    sgn = np.where(h < 0, -1, 1)
    segs = [s for s in _segments(sgn) if s[2] == (-1 if bull else 1)]
    if len(segs) < 2:
        return None
    a, b = segs[-2], segs[-1]
    if len(h) - 1 - b[1] > 3:           # zweites Extrem muss aktuell sein (Segment endet höchstens 3 Balken zurück)
        return None
    seg = lambda s: slice(s[0], s[1] + 1)
    if bull:
        ia = a[0] + int(np.argmin(h[seg(a)])); ib = b[0] + int(np.argmin(h[seg(b)]))
        pa, pb = lo[seg(a)].min(), lo[seg(b)].min()
        ok = pb < pa and h[ib] > h[ia]
        turned = len(h) - 1 > ib and h[-1] > h[-2] and (h[-1] < 0 or b[1] < len(h) - 1)
    else:
        ia = a[0] + int(np.argmax(h[seg(a)])); ib = b[0] + int(np.argmax(h[seg(b)]))
        pa, pb = hi[seg(a)].max(), hi[seg(b)].max()
        ok = pb > pa and h[ib] < h[ia]
        turned = len(h) - 1 > ib and h[-1] < h[-2] and (h[-1] > 0 or b[1] < len(h) - 1)
    # Nulllinien-Durchbruch muss echt sein: das Zwischenstück ist nicht nur ein winziges Zucken
    between = np.abs(h[a[1] + 1:b[0]])
    if not ok or len(between) == 0 or between.max() < 0.10 * abs(h[ia]):
        return None
    gap = ib - ia
    return {"kind": kind, "gap": int(gap), "ideal_gap": 20 <= gap <= 40, "complete": bool(turned),
            "date_a": idx[ia], "date_b": idx[ib], "price_a": float(pa), "price_b": float(pb)}


def kangaroo(x: pd.DataFrame) -> dict | None:
    """Kangaroo Tail (Kap. 20): sehr langer Balken (≥ 2× Ø), Eröffnung/Schluss innerhalb der Spanne des Vortags, normale Nachbarn."""
    rng = (x["High"] - x["Low"])
    for back in (0, 1):                  # Fenster endet heute bzw. gestern
        e = len(x) - 1 - back
        if e < 25:
            continue
        i0, i1, i2 = e - 2, e - 1, e
        avg = rng.iloc[e - 22:e - 2].mean()
        if not avg or np.isnan(avg):
            continue
        r1 = rng.iloc[i1]
        mid = x.iloc[i1]; prev = x.iloc[i0]
        inside = (prev["Low"] <= mid["Open"] <= prev["High"]) and (prev["Low"] <= mid["Close"] <= prev["High"])
        normal = rng.iloc[i0] <= 1.5 * avg and rng.iloc[i2] <= 1.5 * avg
        if r1 >= 2.0 * avg and inside and normal:
            up_t = mid["High"] - max(mid["Open"], mid["Close"])
            dn_t = min(mid["Open"], mid["Close"]) - mid["Low"]
            direction = "up" if up_t > dn_t else "down"
            return {"direction": direction, "days_ago": back, "date": x.index[i1],
                    "stop_mid": float((mid["High"] + max(mid["Open"], mid["Close"])) / 2) if direction == "up"
                    else float((mid["Low"] + min(mid["Open"], mid["Close"])) / 2)}
    return None


def channel(x: pd.DataFrame, n: int = 100) -> dict | None:
    """Kanal um den EMA(26): Koeffizient so, dass ~95 % der letzten 100 Balken (Hoch/Tief) innerhalb liegen (Kap. 22/41)."""
    t = x.tail(n)
    if len(t) < 60:
        return None
    dev = pd.concat([(t["High"] - t["ema26"]) / t["ema26"], (t["ema26"] - t["Low"]) / t["ema26"]], axis=1).max(axis=1)
    k = float(np.nanpercentile(dev, 95))
    e = float(x["ema26"].iloc[-1])
    return {"k": k, "upper": e * (1 + k), "lower": e * (1 - k), "ema": e}


# ─────────────────────────────────────────────────────────────────────────────
# Gesamt-Einordnung
# ─────────────────────────────────────────────────────────────────────────────

def ref_lines(series: pd.Series, n: int = 125) -> tuple[float, float]:
    """Elders 5-%-Regel (Kap. 25/27): Bezugslinien so legen, dass der Oszillator in den letzten ~6 Monaten nur 5 % der Zeit jenseits liegt."""
    t = series.dropna().tail(n)
    if len(t) < 60:
        return (np.nan, np.nan)
    return float(np.percentile(t, 5)), float(np.percentile(t, 95))


def season(x: pd.DataFrame) -> str:
    """Indikator-Jahreszeit des MACD-Histogramms (Kap. 32): steigt unter Null = Frühling, steigt über Null = Sommer,
    fällt über Null = Herbst, fällt unter Null = Winter."""
    h, hp = float(x["hist"].iloc[-1]), float(x["hist"].iloc[-2])
    rising = h > hp
    if h < 0:
        return "Frühling" if rising else "Winter"
    return "Sommer" if rising else "Herbst"


def false_breakout(x: pd.DataFrame, kind: str, look: int = 8, base: int = 30) -> str | None:
    """Fehlausbruch (Kap. 55/58): Kurs durchbricht das Tief (Hoch) der letzten ~30 Balken und schließt wieder darüber (darunter).
    Gibt 'done' (Rückkehr bereits erfolgt) oder 'pending' (Ausbruch läuft noch) zurück."""
    if len(x) < base + look + 2:
        return None
    ref = x.iloc[-(base + look):-look]
    rec = x.iloc[-look:]
    if kind == "bull":
        lvl = float(ref["Low"].min())
        if (rec["Low"] < lvl).any():
            return "done" if float(x["Close"].iloc[-1]) > lvl else "pending"
    else:
        lvl = float(ref["High"].max())
        if (rec["High"] > lvl).any():
            return "done" if float(x["Close"].iloc[-1]) < lvl else "pending"
    return None


def _scale(v: int) -> str:
    return {0: "0 – rot", 1: "1 – gelb", 2: "2 – grün"}[v]


def trade_apgar(side: str, w: pd.DataFrame, x: pd.DataFrame, divs: list, price: float, zone_lo: float, zone_hi: float) -> dict:
    """Elders 'Trade-Apgar' (Kap. 58) für die Strategie 'Fehlausbruch mit Divergenz': fünf Fragen mit 0/1/2 Punkten.
    Ab 7 Punkten und ohne Null-Zeile gilt ein Trade als 'Einser-Trade'. Frage 4 und 5 sind hier regelbasiert nachgebildet."""
    buy = side == "long"
    kind = "bull" if buy else "bear"
    adverse = "red" if buy else "green"

    def imp_score(frame: pd.DataFrame) -> tuple[int, str]:
        cur = frame["impulse"].iloc[-1]
        if cur == adverse:
            return 0, "Impuls " + IMPULSE_TXT[cur] + " verbietet den Trade"
        if cur == "blue":
            after = (frame["impulse"].iloc[-4:-1] == adverse).any()
            return (2, "blau nach " + ("rot" if buy else "grün") + " – Gegenseite lässt nach") if after else (1, "blau (neutral)")
        return 1, "grün – erlaubt, aber evtl. schon spät" if buy else "rot – erlaubt, aber evtl. schon spät"

    q = []
    sc, tx = imp_score(w); q.append(("Wochen-Impuls", sc, tx))
    sc, tx = imp_score(x); q.append(("Tages-Impuls", sc, tx))
    if buy:
        sc = 0 if price > zone_hi else (1 if price >= zone_lo else 2)
        tx = ["über dem Wert – evtl. zu spät", "in der Value Zone", "unter dem Wert – Schnäppchen möglich"][sc]
    else:
        sc = 0 if price < zone_lo else (1 if price <= zone_hi else 2)
        tx = ["unter dem Wert – evtl. zu spät zum Shorten", "in der Value Zone", "über dem Wert – gute Short-Lage"][sc]
    q.append(("Tageskurs zum Wert", sc, tx))
    fb = false_breakout(x, kind)
    dv = [r for r in divs if r["kind"] == kind and r["frame"] == "Tageschart"]
    if fb == "pending" and dv:
        sc, tx = 2, "Fehlausbruch bahnt sich an, Divergenz vorhanden"
    elif fb == "done":
        sc, tx = 1, "Fehlausbruch bereits erfolgt (Kurs ist zurück über/unter der Marke)"
    elif dv:
        sc, tx = 1, "Divergenz ohne Fehlausbruch"
    else:
        sc, tx = 0, "kein Fehlausbruch, keine Divergenz"
    q.append(("Fehlausbruch / Divergenz", sc, tx))

    def ideal(frame: pd.DataFrame, daily: bool) -> bool:
        ssn = season(frame)
        ok = ssn == ("Frühling" if buy else "Herbst")
        if daily:
            ok = ok and (price <= zone_hi if buy else price >= zone_lo)
        return ok

    ni = int(ideal(w, False)) + int(ideal(x, True))
    q.append(("Perfektion der Zeitrahmen", ni, f"{ni} von 2 Zeitrahmen im Idealmuster ({'Frühling' if buy else 'Herbst'} im MACD-Histogramm; Tageskurs nicht {'über' if buy else 'unter'} dem Wert)"))
    total = sum(v for _, v, _ in q)
    zero = [n for n, v, _ in q if v == 0]
    return {"items": q, "total": int(total), "zeros": zero, "einser": bool(total >= 7 and not zero)}


def _row(title, status, text, ref):
    return {"titel": title, "status": status, "text": text, "ref": ref}      # status: ok | warn | bad | info


def compute_elder(df: pd.DataFrame) -> dict | None:
    d = _clean(df)
    if d is None or len(d) < MIN_BARS:
        return None
    x = indicators(d)
    w = indicators(to_weekly(d))
    if len(w) < 40:
        return None
    last, prev = x.iloc[-1], x.iloc[-2]
    wl, wp = w.iloc[-1], w.iloc[-2]
    price = float(last["Close"])
    has_vol = bool(d["Volume"].tail(60).sum() > 0)

    # ── Screen 1: Wochentrend / Impuls ────────────────────────────────────────
    w_slope = (wl["ema13"] - wp["ema13"]) / wp["ema13"] * 100
    w_imp = wl["impulse"]
    flat = abs(w_slope) < 0.10
    bias = "none" if flat else ("long" if w_slope > 0 else "short")
    long_allowed, short_allowed = w_imp != "red", w_imp != "green"

    rows = []
    s1_text = (f"Wochen-EMA(13) {'steigt' if w_slope > 0 else 'fällt'} ({w_slope:+.2f} % zur Vorwoche), Wochen-Histogramm "
               f"{'steigt' if wl['hist'] > wp['hist'] else 'fällt'} ({'über' if wl['hist'] > 0 else 'unter'} Null), Wochen-Impuls {IMPULSE_TXT[w_imp]}.")
    if flat:
        rows.append(_row("Screen 1 – Gezeiten (Wochenchart)", "warn",
                         s1_text + " Der EMA ist flach: kein Trend. Elder: keine Trendfolge, höchstens Range-Taktik am Kanalrand.", "Kap. 22, 39"))
    elif bias == "long":
        rows.append(_row("Screen 1 – Gezeiten (Wochenchart)", "bad" if not long_allowed else "ok",
                         s1_text + (" Rot verbietet den Kauf – abwarten, bis die rote Farbe verschwindet." if not long_allowed
                                    else " Kaufen ist erlaubt; Shorten wäre gegen die Gezeiten."), "Kap. 39, 40"))
    else:
        rows.append(_row("Screen 1 – Gezeiten (Wochenchart)", "bad" if not short_allowed else "ok",
                         s1_text + (" Grün verbietet das Shorten – abwarten." if not short_allowed
                                    else " Shorten ist erlaubt; Kaufen wäre gegen die Gezeiten."), "Kap. 39, 40"))
    best_zone = None
    if bias == "long" and wl["hist"] < 0 and wl["hist"] > wp["hist"]:
        best_zone = "Wochen-Histogramm dreht UNTER der Nulllinie nach oben – laut Elder die besten Kaufsignale (Bären erschöpft)."
    if bias == "short" and wl["hist"] > 0 and wl["hist"] < wp["hist"]:
        best_zone = "Wochen-Histogramm dreht ÜBER der Nulllinie nach unten – laut Elder die besten Verkaufssignale (Bullen erschöpft)."
    if best_zone:
        rows.append(_row("Qualität des Wochensignals", "ok", best_zone, "Kap. 23"))

    # ── Screen 2: Tages-Oszillatoren (Rücksetzer gegen die Gezeiten) ─────────────
    fi2 = float(last["fi2"])
    stoch = float(last["stoch5"]) if not np.isnan(last["stoch5"]) else np.nan
    st_lo, st_hi = ref_lines(x["stoch5"])
    rsi = float(last["rsi13"]) if not np.isnan(last["rsi13"]) else np.nan
    rs_lo, rs_hi = ref_lines(x["rsi13"])
    d_imp = last["impulse"]
    pullback = False
    if bias == "long":
        fi_bad = has_vol and fi2 < 0
        st_bad = (not np.isnan(stoch)) and not np.isnan(st_lo) and stoch <= st_lo
        pullback = fi_bad or st_bad
    elif bias == "short":
        fi_bad = has_vol and fi2 > 0
        st_bad = (not np.isnan(stoch)) and not np.isnan(st_hi) and stoch >= st_hi
        pullback = fi_bad or st_bad
    parts = []
    if has_vol:
        parts.append(f"2-Tage-Force-Index {'negativ' if fi2 < 0 else 'positiv'}")
    if not np.isnan(stoch):
        parts.append(f"Slow-Stochastik(5) {stoch:.0f}")
    parts.append(f"Tages-Impuls {IMPULSE_TXT[d_imp]}")
    if bias == "none":
        rows.append(_row("Screen 2 – Welle gegen die Gezeiten (Tageschart)", "info", "; ".join(parts) + ". Ohne Wochentrend kein Rücksetzer-Signal.", "Kap. 30, 39"))
    elif pullback:
        rows.append(_row("Screen 2 – Welle gegen die Gezeiten (Tageschart)", "ok", "; ".join(parts) +
                         (". Rücksetzer im Aufwärtstrend – Kaufgelegenheit." if bias == "long" else ". Erholung im Abwärtstrend – Short-Gelegenheit."), "Kap. 30, 39"))
    else:
        rows.append(_row("Screen 2 – Welle gegen die Gezeiten (Tageschart)", "warn", "; ".join(parts) +
                         ". Kein Rücksetzer – nicht hinterherlaufen, auf den nächsten warten.", "Kap. 30, 39"))

    # ── Wert, Kanal ────────────────────────────────────────────────────────────
    e13, e26 = float(last["ema13"]), float(last["ema26"])
    zone_lo, zone_hi = min(e13, e26), max(e13, e26)
    if price > zone_hi:
        zone_txt = f"Kurs {(price / zone_hi - 1) * 100:.1f} % ÜBER der Value Zone ({zone_lo:.2f}–{zone_hi:.2f}): man zahlt mehr als den Wert."
        zone_st = "warn" if bias == "long" else "ok"
    elif price < zone_lo:
        zone_txt = f"Kurs {(1 - price / zone_lo) * 100:.1f} % UNTER der Value Zone ({zone_lo:.2f}–{zone_hi:.2f}): unter dem Wert."
        zone_st = "ok" if bias == "long" else "warn"
    else:
        zone_txt = f"Kurs liegt IN der Value Zone ({zone_lo:.2f}–{zone_hi:.2f}) – Elders bevorzugter Kaufbereich im Aufwärtstrend (Verkauf im Abwärtstrend)."
        zone_st = "ok"
    rows.append(_row("Value Zone (EMA 13/26, Tageschart)", zone_st, zone_txt, "Kap. 22"))

    ch = channel(x)
    over_upper = under_lower = False
    if ch:
        over_upper, under_lower = price > ch["upper"], price < ch["lower"]
        pos = (price - ch["lower"]) / (ch["upper"] - ch["lower"]) * 100
        if over_upper:
            ct, cs = f"Kurs ÜBER der oberen Kanallinie ({ch['upper']:.2f}) – Elders harte Regel: nie über der oberen Linie kaufen (Markt 'manisch').", "bad" if bias != "short" else "ok"
        elif under_lower:
            ct, cs = f"Kurs UNTER der unteren Kanallinie ({ch['lower']:.2f}) – nie dort shorten (Markt 'depressiv'); Gelegenheit nur für Käufer mit Divergenz.", "bad" if bias == "short" else "ok"
        else:
            ct, cs = f"Kurs im Kanal ({pos:.0f} % der Kanalhöhe; unten {ch['lower']:.2f}, oben {ch['upper']:.2f}, Breite ±{ch['k'] * 100:.1f} % um den EMA 26).", "ok"
        rows.append(_row("Kanal (95 % der letzten 100 Balken)", cs, ct, "Kap. 41"))

    # ── Trend-Stärke: Directional System ────────────────────────────────────────
    adx, adx_p = float(last["adx"]), float(prev["adx"])
    pdi, mdi = float(last["pdi"]), float(last["mdi"])
    adx_rising = adx > adx_p
    dir_txt = (f"+DI {pdi:.0f} {'>' if pdi > mdi else '<'} −DI {mdi:.0f}, ADX {adx:.0f} {'steigend' if adx_rising else 'fallend'}. ")
    if adx < min(pdi, mdi):
        dir_txt += "ADX unter beiden Linien: schläfriger Markt, Trendfolge sinnlos – aber hier entstehen neue Trends; Steigt der ADX um ~4 Punkte, beginnt ein neuer Trend."
        dst = "info"
    elif bias == "long":
        dst = "ok" if (pdi > mdi and adx_rising) else ("warn" if pdi > mdi else "bad")
        dir_txt += "Long nur, wenn +DI oben liegt; am besten bei steigendem ADX." + ("" if adx_rising else " Fallender ADX = mehr Whipsaws.")
    elif bias == "short":
        dst = "ok" if (mdi > pdi and adx_rising) else ("warn" if mdi > pdi else "bad")
        dir_txt += "Short nur, wenn −DI oben liegt; am besten bei steigendem ADX."
    else:
        dst = "info"
    rows.append(_row("Directional System (13)", dst, dir_txt, "Kap. 24"))

    # ── Divergenzen (stärkste Signale) ─────────────────────────────────────────
    divs = []
    for label, frame, lb in (("Tageschart", x, 90), ("Wochenchart", w, 60)):
        for kind in ("bull", "bear"):
            r = hist_divergence(frame, lb, kind)
            if r:
                r["frame"] = label
                divs.append(r)
    if divs:
        for r in divs:
            name = "bullische" if r["kind"] == "bull" else "bärische"
            state = "KOMPLETT (Histogramm hat nach dem zweiten Extrem gedreht) – starkes Signal" if r["complete"] else "bildet sich, wartet auf die Drehung des Histogramms"
            q = "Abstand ideal (20–40 Balken)" if r["ideal_gap"] else f"Abstand {r['gap']} Balken (ideal 20–40)"
            good = (r["kind"] == "bull") == (bias != "short")
            rows.append(_row(f"Divergenz ({r['frame']})", ("ok" if r["complete"] else "warn") if good else ("bad" if r["complete"] else "warn"),
                             f"{name.capitalize()} MACD-Histogramm-Divergenz {state}; {q}. Kurs-Extrema {r['price_a']:.2f} → {r['price_b']:.2f}.", "Kap. 23"))
    else:
        rows.append(_row("Divergenz", "info", "Keine gültige MACD-Histogramm-Divergenz (Voraussetzung: Histogramm durchbricht zwischen beiden Extremen die Nulllinie).", "Kap. 23"))

    # ── Oszillator-Niveaus nach Elder (Kap. 25–27) ─────────────────────────────
    if not np.isnan(stoch) and not np.isnan(st_lo):
        parts_o = [f"Slow-Stochastik(5) {stoch:.0f} (Bezugslinien nach 5-%-Regel: {st_lo:.0f} / {st_hi:.0f})"]
        if not np.isnan(rsi) and not np.isnan(rs_lo):
            parts_o.append(f"RSI(13) {rsi:.0f} (Bezugslinien {rs_lo:.0f} / {rs_hi:.0f})")
        ob = stoch >= st_hi or (not np.isnan(rsi) and not np.isnan(rs_hi) and rsi >= rs_hi)
        os_ = stoch <= st_lo or (not np.isnan(rsi) and not np.isnan(rs_lo) and rsi <= rs_lo)
        if bias == "long":
            if ob:
                ost, otx = "bad", "Überkauft – Elder: nicht kaufen, wenn der Oszillator überkauft ist (filtert die meisten schlechten Trades)."
            elif os_:
                ost, otx = "ok", "Überverkauft im Aufwärtstrend – Elders Kaufsetup: Stop-Order über dem Hoch des letzten Balkens, ohne auf eine Wende zu warten."
            else:
                ost, otx = "info", "Weder über- noch unterverkauft."
        elif bias == "short":
            if os_:
                ost, otx = "bad", "Überverkauft – Elder: nicht shorten, wenn der Oszillator überverkauft ist."
            elif ob:
                ost, otx = "ok", "Überkauft im Abwärtstrend – Elders Short-Setup: Stop-Order unter dem Tief des letzten Balkens."
            else:
                ost, otx = "info", "Weder über- noch unterverkauft."
        else:
            ost, otx = "info", "Ohne Wochentrend sind Überkauft/Überverkauft nur im Range (am Kanalrand) brauchbar; in einem Trend rutschen sie dauerhaft in die Extremzone."
        rows.append(_row("Stochastik & RSI (Elder-Standard)", ost, "; ".join(parts_o) + ". " + otx +
                         " Signalstärke: Divergenzen > Niveaus > Richtung. Gilt nur zusammen mit dem Wochentrend.", "Kap. 25–27"))
    ssn_w, ssn_d = season(w), season(x)
    good_s = "Frühling" if bias != "short" else "Herbst"
    rows.append(_row("Indikator-Jahreszeit (MACD-Histogramm)", "ok" if (ssn_d == good_s or ssn_w == good_s) else "info",
                     f"Woche: {ssn_w}, Tag: {ssn_d}. Beste Kaufzeit ist der Frühling (Histogramm steigt unter Null: Angst herrscht noch, Stopp kann eng gesetzt werden), "
                     f"beste Short-Zeit der Herbst; im Sommer ist das Gewinnpotenzial kleiner und der Stopp weiter weg, im Winter dreht das Chance/Risiko gegen die Bären.", "Kap. 32"))
    apgar = None
    if bias in ("long", "short"):
        apgar = trade_apgar("long" if bias == "long" else "short", w, x, divs, price, zone_lo, zone_hi)
        rows.append(_row("Trade-Apgar (Einser-Trade?)", "ok" if apgar["einser"] else ("warn" if apgar["total"] >= 5 else "bad"),
                         f"{apgar['total']} von 10 Punkten" + (" – Einser-Trade (ab 7 Punkten, keine Null)." if apgar["einser"] else
                         " – kein Einser-Trade" + (f" (Null bei: {', '.join(apgar['zeros'])})" if apgar["zeros"] else " (unter 7 Punkten)") +
                         ". Elder: Trade lassen und nach einem besseren Kandidaten suchen."), "Kap. 55, 58"))

    # ── Volumen / OBV / Kangaroo ───────────────────────────────────────────────
    vol_info = None
    if has_vol:
        v = d["Volume"]
        ratio = float(v.iloc[-1] / v.iloc[-11:-1].mean()) if v.iloc[-11:-1].mean() > 0 else np.nan
        vol_info = ratio
        obv = x["obv"]
        c60 = x["Close"].tail(60)
        near_high = price >= 0.98 * c60.max()
        near_low = price <= 1.02 * c60.min()
        o60 = obv.tail(60)
        orange = o60.max() - o60.min()
        obv_at_high = obv.iloc[-1] >= o60.max() - 0.05 * orange
        obv_at_low = obv.iloc[-1] <= o60.min() + 0.05 * orange
        msgs, st = [], "info"
        if not np.isnan(ratio):
            lvl = "hoch (≥125 % des 2-Wochen-Schnitts)" if ratio >= 1.25 else ("niedrig (≤75 %)" if ratio <= 0.75 else "normal")
            msgs.append(f"Heutiges Volumen {ratio * 100:.0f} % des 2-Wochen-Schnitts – {lvl}.")
        if near_high and not obv_at_high:
            msgs.append("Kurs nahe 60-Tage-Hoch, aber OBV bestätigt nicht → bärische Divergenz (Warnung).")
            st = "bad" if bias == "long" else "ok"
        elif near_high and obv_at_high:
            msgs.append("Kurs und OBV am 60-Tage-Hoch → Aufwärtstrend bestätigt.")
            st = "ok" if bias == "long" else "warn"
        if near_low and obv_at_low:
            msgs.append("Kurs und OBV am 60-Tage-Tief → Abwärtstrend bestätigt.")
            st = "ok" if bias == "short" else "warn"
        elif near_low and not obv_at_low:
            msgs.append("Kurs nahe 60-Tage-Tief, OBV hält höher → bullische Divergenz (Stärke).")
            st = "ok" if bias != "short" else "bad"
        if not np.isnan(ratio) and ratio >= 2.0 and near_low:
            msgs.append("Volumenspitze am Tief = mögliches Klimax-Tief; Elder: wird fast immer noch einmal bei geringerem Volumen getestet – auf den Test warten.")
        rows.append(_row("Volumen & OBV", st, " ".join(msgs) or "Keine auffällige Volumenbotschaft.", "Kap. 28, 29"))
    kg = kangaroo(x)
    if kg:
        good = (kg["direction"] == "down") == (bias != "short")
        rows.append(_row("Kangaroo Tail", "ok" if good else "warn",
                         f"{'Nach unten' if kg['direction'] == 'down' else 'Nach oben'} gerichteter Kangaroo Tail ({kg['date']:%d.%m.}, vor {kg['days_ago']} Tag(en)): "
                         f"gescheiterter {'Bären' if kg['direction'] == 'down' else 'Bullen'}-Angriff – Elder tradet gegen den Tail; Stopp etwa in der Mitte des Tails ({kg['stop_mid']:.2f}).",
                         "Kap. 20"))

    # ── Handelsplan: Einstieg, Stopp, Ziel, Größe ──────────────────────────────
    plan = None
    side = "long" if bias == "long" else ("short" if bias == "short" else None)
    if side:
        buy = side == "long"
        t = x.tail(30)
        pen = ((t["ema13"] - t["Low"])[t["Low"] < t["ema13"]] if buy else (t["High"] - t["ema13"])[t["High"] > t["ema13"]])
        avg_pen = float(pen.mean()) if len(pen) >= 3 else float((x["atr14"].iloc[-1]) * 0.5)
        ema_tom = e13 + (e13 - float(prev["ema13"]))
        entry_stop = float(last["High"]) + TICK if buy else float(last["Low"]) - TICK
        entry_lim = ema_tom - avg_pen if buy else ema_tom + avg_pen
        # SafeZone (Kap. 54): Ø Penetration des Vortagesextrems in den letzten 20 Balken
        t20 = x.tail(21)
        if buy:
            p = (t20["Low"].shift(1) - t20["Low"]).clip(lower=0)[1:]
            p = p[p > 0]
            sz = float(last["Low"]) - 2.0 * (float(p.mean()) if len(p) else 0.0)
        else:
            p = (t20["High"] - t20["High"].shift(1)).clip(lower=0)[1:]
            p = p[p > 0]
            sz = float(last["High"]) + 3.0 * (float(p.mean()) if len(p) else 0.0)
        # Nics Stopp: zweittiefstes Tief (Long) / zweithöchstes Hoch (Short) der letzten 15 Balken
        t15 = x.tail(15)
        if buy:
            nic = float(np.sort(t15["Low"].to_numpy())[1]) - TICK
        else:
            nic = float(np.sort(t15["High"].to_numpy())[-2]) + TICK
        atr = float(x["atr14"].iloc[-1])
        stops = {"SafeZone": sz, "Nics Stopp": nic}          # feste Niveaus; die 2-ATR-Variante hängt vom Einstiegskurs ab (trade_table)
        if ch:
            tgt1 = ch["upper"] if buy else ch["lower"]
            tgt2 = (ch["ema"] + ch["upper"]) / 2 if buy else (ch["ema"] + ch["lower"]) / 2
        else:
            tgt1 = tgt2 = None
        one_off = 0.30 * (ch["upper"] - ch["lower"]) if ch else None      # Einser-Ziel: 30 % der Kanalhöhe (Kap. 55)
        plan = {"one_offset": one_off, "side": side, "entry_stop": entry_stop, "entry_limit": entry_lim, "avg_pen": avg_pen, "ema_tomorrow": ema_tom,
                "stops": stops, "atr": atr, "target_far": tgt1, "target_near": tgt2}

    out = {"price": price, "bias": bias, "weekly_impulse": w_imp, "daily_impulse": d_imp, "pullback": pullback,
           "long_allowed": bool(long_allowed), "short_allowed": bool(short_allowed), "over_upper": bool(over_upper), "under_lower": bool(under_lower),
           "rows": rows, "apgar": apgar, "plan": plan, "x": x, "w": w, "channel": ch, "adx": adx, "has_volume": has_vol, "vol_ratio": vol_info}
    out["verdict"] = _verdict(out)
    return out


def _verdict(o: dict) -> dict:
    bias = o["bias"]
    if bias == "none":
        return {"ampel": "yellow", "titel": "Trendlos – keine Trendfolge",
                "text": "Der Wochen-EMA ist flach. Elder handelt dann keine Trends: Im Range gilt 'Schwäche kaufen, Stärke verkaufen' am Kanalrand mit engen Stopps – oder man wartet auf den Ausbruch (mit Volumen bestätigt)."}
    long = bias == "long"
    forbid = (not o["long_allowed"]) if long else (not o["short_allowed"])
    chan_block = o["over_upper"] if long else o["under_lower"]
    word = "Kauf" if long else "Short"
    if forbid:
        return {"ampel": "red", "titel": f"Elder: {word} verboten",
                "text": f"Der Wochen-Impuls ist {IMPULSE_TXT[o['weekly_impulse']]}. Das Impuls-System zensiert diesen Trade – warten, bis die Farbe verschwindet (dann ist der Einstieg erlaubt)."}
    if chan_block:
        return {"ampel": "red", "titel": f"Elder: {word} verboten (Kanal)",
                "text": "Elders harte Regel: nie über der oberen Kanallinie kaufen bzw. unter der unteren shorten. Warten, bis der Kurs zurück in den Kanal bzw. die Value Zone kommt."}
    rows = {r["titel"]: r for r in o["rows"]}
    negatives = [r for r in o["rows"] if r["status"] == "bad"]
    ap = o.get("apgar")
    if o["pullback"] and not negatives and ap and not ap["einser"]:
        return {"ampel": "yellow", "titel": f"Elder: {word} erlaubt, aber kein Einser-Trade",
                "text": f"Wochentrend und Rücksetzer passen, doch der Trade-Apgar liegt bei {ap['total']} von 10 Punkten"
                        + (f" (Null bei: {', '.join(ap['zeros'])})" if ap["zeros"] else "") + ". Elder handelt erst ab 7 Punkten ohne Null – sonst nach einem besseren Kandidaten suchen."}
    if o["pullback"] and not negatives:
        return {"ampel": "green", "titel": f"Elder: {word}-Gelegenheit",
                "text": "Wochentrend erlaubt den Trade, und im Tageschart läuft die Gegenwelle. Einstieg per Stop-Order über dem Vortageshoch (bzw. unter dem Tief) oder per Limit-Order im Wert; Stopp und Größe aus dem Plan unten. Chance/Risiko muss mindestens 2:1 sein."}
    if o["pullback"]:
        return {"ampel": "yellow", "titel": f"Elder: {word} möglich, aber mit Warnungen",
                "text": "Rücksetzer vorhanden, doch: " + "; ".join(r["titel"] for r in negatives) + ". Elder: Wenn Signale einander widersprechen, lieber verzichten."}
    return {"ampel": "yellow", "titel": "Elder: abwarten",
            "text": "Der Wochentrend erlaubt den Trade, aber im Tageschart gibt es keinen Rücksetzer. Nicht dem Kurs hinterherlaufen – auf die nächste Gegenwelle warten."}


# ─────────────────────────────────────────────────────────────────────────────
# Risikorechnung (2 %-Regel, Kap. 50/51)
# ─────────────────────────────────────────────────────────────────────────────
def trade_table(plan: dict, equity: float, risk_pct: float = 2.0) -> pd.DataFrame:
    """Eine Zeile je Kombination Einstieg × Stopp: Risiko je Aktie, Stückzahl nach 2 %-Regel, Chance/Risiko."""
    risk_pct = min(float(risk_pct), 2.0)
    buy = plan["side"] == "long"
    budget = equity * risk_pct / 100.0
    rows = []
    for en_name, entry in (("Stop-Order (Vortageshoch + Tick)" if buy else "Stop-Order (Vortagestief − Tick)", plan["entry_stop"]),
                           ("Limit im Wert (EMA − Ø Penetration)" if buy else "Limit im Wert (EMA + Ø Penetration)", plan["entry_limit"])):
        stops = dict(plan["stops"])
        stops["2 × ATR"] = entry - 2 * plan["atr"] if buy else entry + 2 * plan["atr"]
        for st_name, stop in stops.items():
            risk = (entry - stop) if buy else (stop - entry)
            if not (risk > 0) or not math.isfinite(risk):
                continue
            qty = max(0, min(int(budget // risk), int(equity // entry)))
            row = {"Einstieg": en_name, "Kurs": round(entry, 2), "Stopp-Methode": st_name, "Stopp": round(stop, 2),
                   "Risiko/Aktie": round(risk, 2), "Max. Stück (2 %-Regel)": qty, "Einsatz": round(qty * entry, 0)}
            for name, tgt in (("Ziel nah", plan["target_near"]), ("Ziel fern", plan["target_far"])):
                if tgt is None:
                    row[f"{name} / CRV"] = "–"
                else:
                    gain = (tgt - entry) if buy else (entry - tgt)
                    row[f"{name} / CRV"] = f"{tgt:.2f} / {gain / risk:.1f}:1" if gain > 0 else f"{tgt:.2f} / –"
            rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────
def build_elder_figure(res: dict, weekly: bool = False, bars: int | None = None, height: int = 640) -> go.Figure:
    x = res["w"] if weekly else res["x"]
    n = bars or (78 if weekly else 150)
    t = x.tail(n)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
    for colname, color in IMPULSE_COL.items():
        m = t[t["impulse"] == colname]
        if m.empty:
            continue
        fig.add_trace(go.Ohlc(x=m.index, open=m["Open"], high=m["High"], low=m["Low"], close=m["Close"],
                              increasing_line_color=color, decreasing_line_color=color,
                              name=f"Impuls {colname}", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t.index, y=t["ema13"], name="EMA 13", line=dict(color="#facc15", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t.index, y=t["ema26"], name="EMA 26", line=dict(color="#c084fc", width=1.4),
                             fill="tonexty", fillcolor="rgba(148,163,184,0.12)"), row=1, col=1)
    if not weekly and res.get("channel"):
        k = res["channel"]["k"]
        fig.add_trace(go.Scatter(x=t.index, y=t["ema26"] * (1 + k), name="Kanal oben", line=dict(color="#94a3b8", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t.index, y=t["ema26"] * (1 - k), name="Kanal unten", line=dict(color="#94a3b8", width=1, dash="dot")), row=1, col=1)
    hcol = ["#4ade80" if v >= 0 else "#f87171" for v in t["hist"]]
    fig.add_trace(go.Bar(x=t.index, y=t["hist"], marker_color=hcol, opacity=0.75, name="MACD-Histogramm 12/26/9"), row=2, col=1)
    fig.add_hline(y=0, line_color="#475569", line_width=1, row=2, col=1)
    if res["has_volume"]:
        fig.add_trace(go.Bar(x=t.index, y=t["fi2"], marker_color=["#4ade80" if v >= 0 else "#f87171" for v in t["fi2"]],
                             opacity=0.55, name="Force Index (2)"), row=3, col=1)
        fig.add_trace(go.Scatter(x=t.index, y=t["fi13"], name="Force Index (13)", line=dict(color="#38bdf8", width=1.4)), row=3, col=1)
        fig.add_hline(y=0, line_color="#475569", line_width=1, row=3, col=1)
    fig.update_layout(height=height, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                      margin=dict(l=10, r=20, t=30, b=10), hovermode="x unified", xaxis_rangeslider_visible=False,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", y=1.04),
                      title=dict(text="Wochenchart – Gezeiten (Screen 1)" if weekly else "Tageschart – Wellen (Screen 2/3)", font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor="#1e2235", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e2235", zeroline=False)
    return fig

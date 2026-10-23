# engine.py — v5
#   * Signal-Label seitenrichtig (Kauf/Verkauf getrennt) + Extrem-Gate für OVERSOLD/OVERBOUGHT
#   * Trefferquote: seitenrichtig, ohne Look-ahead, Einstieg zum nächsten Open, MIT Zufalls-Baseline
#   * Währung aus Yahoo (GBp/ZAc-Fix), Dividende für Pence-Ticker korrekt
#   * Batch-Download statt 838 Einzelabrufe, .info nur für Signale
#   * unfertige Tageskerze wird verworfen, Feiertage/veraltete Kerzen werden erkannt

import os
import time
import html
import logging
import smtplib
from collections import Counter
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import yfinance as yf

from database import get_currency, get_focus

log = logging.getLogger(__name__)

APP_URL = "https://stockupdate-65qjxum6gq2gpjpr5exqfd.streamlit.app"

# ── Konfiguration ────────────────────────────────────────────────────────────
HISTORY_PERIOD        = "3y"   # 70/200-Indikatoren brauchen Einlaufzeit; 2y waren zu knapp
FORWARD_DAYS          = 40     # Beobachtungsfenster der Trefferquote
HIT_THRESHOLD_PCT     = 5.0    # "Wende" = mind. 5 % in Signalrichtung innerhalb des Fensters
MIN_HISTORY_N         = 8      # darunter wird keine Trefferquote angezeigt (vorher 5)
SIGNAL_COOLDOWN       = 10     # Handelstage: aufeinanderfolgende Extrem-Tage = EIN Signal
MAX_ENTRY_WAIT        = 20     # so lange wird nach der Bestätigung (MACD/Preis) gesucht
REQUIRE_BOTH_EXTREMES = True   # OVERSOLD/OVERBOUGHT nur, wenn StochRSI UND Stoch extrem sind
BATCH_SIZE            = 80


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL KLASSIFIKATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_criteria(stoch_rsi=None, stoch_fast=None, stoch_slow=None, cci=None, macd_turn=None) -> dict:
    """
    EINZIGE Stelle für die 4 Strategie-Kriterien (Schwellen unverändert):
      Kauf:    StochRSI(70) < 0.1 · Stoch Fast(70) < 10 UND Slow < 15 · CCI(20) > -100 · MACD-Histogramm dreht hoch
      Verkauf: StochRSI(70) > 0.9 · Stoch Fast(70) > 90 UND Slow > 85 · CCI(20) <  100 · MACD-Histogramm dreht runter

    NEU: 'extreme' = die beiden Oszillatoren (StochRSI + Stoch) sind BEIDE im Extrembereich.
    CCI > -100 bzw. < 100 ist fast immer wahr und der MACD-Dreh ist ein Trigger - beides
    sind Bestätigungen, kein Beleg für "überverkauft/überkauft". Ohne Extrem-Gate konnte
    z.B. ALO.PA (Stoch 44/60, CCI 24) als OVERBOUGHT 3/4 erscheinen.
    Werden die Stoch-Werte gar nicht übergeben (None, z.B. Futures-Tabelle), zählt nur StochRSI.
    """
    buy = {
        "stochrsi": stoch_rsi  is not None and 0 < stoch_rsi < 0.1,
        "stoch":    stoch_fast is not None and stoch_slow is not None and stoch_fast < 10 and stoch_slow < 15,
        "cci":      cci is not None and cci > -100,
        "macd":     macd_turn == "up",
    }
    sell = {
        "stochrsi": stoch_rsi  is not None and stoch_rsi > 0.9,
        "stoch":    stoch_fast is not None and stoch_slow is not None and stoch_fast > 90 and stoch_slow > 85,
        "cci":      cci is not None and cci < 100,
        "macd":     macd_turn == "down",
    }
    stoch_known = stoch_fast is not None and stoch_slow is not None
    buy_extreme  = bool(buy["stochrsi"]  and (buy["stoch"]  if stoch_known else True))
    sell_extreme = bool(sell["stochrsi"] and (sell["stoch"] if stoch_known else True))
    return {"buy": buy, "sell": sell,
            "buy_score": sum(buy.values()), "sell_score": sum(sell.values()),
            "buy_extreme": buy_extreme, "sell_extreme": sell_extreme}


def classify_signal(stoch_rsi=None, stoch_fast=None, stoch_slow=None, cci=None, macd_turn=None) -> dict:
    """
    Label + Score + Seite ('buy'/'sell'/None) aus EINER Kriterien-Auswertung.
    macd_turn: None | "up" | "down". Fehlende Werte als None übergeben, nicht als 0.0.
    """
    c = evaluate_criteria(stoch_rsi, stoch_fast, stoch_slow, cci, macd_turn)
    bs, ss = c["buy_score"], c["sell_score"]
    buy_ok  = c["buy_extreme"]  if REQUIRE_BOTH_EXTREMES else bs >= 2
    sell_ok = c["sell_extreme"] if REQUIRE_BOTH_EXTREMES else ss >= 2

    if bs == 4:
        r = {"label": "KAUFSIGNAL",   "emoji": "🟢", "color": "#0d7a2e", "bg": "#e8f9ed", "short": "BUY", "side": "buy"}
    elif ss == 4:
        r = {"label": "VERKAUFSIGNAL", "emoji": "🔴", "color": "#8e1f14", "bg": "#fdecea", "short": "SELL", "side": "sell"}
    elif buy_ok and bs > ss:
        r = {"label": "OVERSOLD",     "emoji": "🟡", "color": "#1a9e3f", "bg": "#e8f9ed", "short": "OS", "side": "buy"}
    elif sell_ok and ss > bs:
        r = {"label": "OVERBOUGHT",   "emoji": "🟠", "color": "#c0392b", "bg": "#fdecea", "short": "OB", "side": "sell"}
    else:
        r = {"label": "NEUTRAL",      "emoji": "⚪", "color": "#7f8c8d", "bg": "#f4f4f4", "short": "NT", "side": None}
    r["buy_score"], r["sell_score"] = bs, ss
    r["buy_extreme"], r["sell_extreme"] = c["buy_extreme"], c["sell_extreme"]
    r["score"] = bs if bs >= ss else -ss
    return r


# ─────────────────────────────────────────────────────────────────────────────
# WÄHRUNG (Yahoo liefert LSE-Kurse in Pence, JSE in Cent)
# ─────────────────────────────────────────────────────────────────────────────

_SUBUNIT = {"GBp": ("£", 100), "GBX": ("£", 100), "ZAc": ("ZAR", 100), "ZAC": ("ZAR", 100), "ILA": ("ILS", 100)}
_CODE_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "CHF": "CHF", "SEK": "SEK", "NOK": "NOK", "DKK": "DKK",
                "JPY": "¥", "HKD": "HK$", "KRW": "₩", "CNY": "¥", "ZAR": "ZAR", "BRL": "BRL", "MXN": "MXN",
                "TRY": "TRY", "SAR": "SAR", "CAD": "C$", "AUD": "A$"}


def resolve_currency(ticker: str, code: str | None = None) -> tuple[str, int]:
    """(Symbol, Teiler). Teiler 100 = Kurs ist in Unterwährung (Pence/Cent) notiert."""
    if code:
        if code in _SUBUNIT:                 # exakte Schreibweise prüfen ('GBp' != 'GBP')
            return _SUBUNIT[code]
        return _CODE_SYMBOL.get(code.upper(), code), 1
    up = ticker.upper()                      # Fallback ohne Yahoo-Info: Suffix-Annahme
    if up.endswith(".L"):
        return "£", 100
    if up.endswith(".JO"):
        return "ZAR", 100
    return get_currency(ticker), 1


# ─────────────────────────────────────────────────────────────────────────────
# GEMEINSAME HELPER
# ─────────────────────────────────────────────────────────────────────────────

def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt Zeilen ohne gültigen Schlusskurs (unfertiger Handelstag, Wochenende)."""
    if df is None or df.empty:
        return df
    return df[df["Close"].notna()]


def compute_macd(df: pd.DataFrame, fast: int = 70, slow: int = 200, signal: int = 9) -> pd.DataFrame:
    """MACD mit EMA/EMA wie im Chart-Setup des Nutzers (Standard: 70/200/9)."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]   = ema_fast - ema_slow
    df["SIGNAL"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["OSMA"]   = df["MACD"] - df["SIGNAL"]
    return df


def detect_macd_turn(df: pd.DataFrame, lookback: int = 3) -> str | None:
    """'up' / 'down' / None - Umkehr des MACD-Histogramms an den letzten Balken."""
    if "OSMA" not in df.columns or len(df) < lookback + 2:
        return None
    h = df["OSMA"].tail(lookback + 2).values
    if np.any(np.isnan(h)):
        return None
    was_falling = all(h[i] < h[i-1] for i in range(1, len(h)-1))
    turns_up    = h[-1] > h[-2]
    was_rising  = all(h[i] > h[i-1] for i in range(1, len(h)-1))
    turns_down  = h[-1] < h[-2]
    if was_falling and turns_up:
        return "up"
    if was_rising and turns_down:
        return "down"
    return None


def macd_turn_at(df: pd.DataFrame, i: int, lookback: int = 3) -> str | None:
    """Wie detect_macd_turn(), aber für Position i im DataFrame."""
    if "OSMA" not in df.columns or i < lookback + 1:
        return None
    h = df["OSMA"].iloc[i - lookback - 1: i + 1].values
    if len(h) < lookback + 2 or np.any(np.isnan(h)):
        return None
    was_falling = all(h[j] < h[j-1] for j in range(1, len(h)-1))
    turns_up    = h[-1] > h[-2]
    was_rising  = all(h[j] > h[j-1] for j in range(1, len(h)-1))
    turns_down  = h[-1] < h[-2]
    if was_falling and turns_up:
        return "up"
    if was_rising and turns_down:
        return "down"
    return None


def format_dividend_yield(info: dict, price: float, subunit: bool = False) -> str:
    """
    Dividendenrendite. Bei Pence-/Cent-Kursen (LSE/JSE) ist dividendRate/Kurs um den Faktor 100
    verfälscht (z.B. 3i Group: '0.04%') - dort die von Yahoo gelieferte Quote nehmen.
    """
    if subunit:
        t = info.get("trailingAnnualDividendYield")
        if isinstance(t, (int, float)) and t > 0:
            return f"{round(t * 100, 2)}%"
        return "n/a"
    rate = info.get("dividendRate")
    if isinstance(rate, (int, float)) and rate > 0 and isinstance(price, (int, float)) and price > 0:
        return f"{round(rate / price * 100, 2)}%"
    dy = info.get("dividendYield")
    if isinstance(dy, (int, float)) and dy > 0:
        pct = dy * 100 if dy < 1 else dy
        return f"{round(pct, 2)}%"
    return "0.00%"


# ─────────────────────────────────────────────────────────────────────────────
# INDIKATOREN (vektorisiert; ein Code-Pfad für Live-Scan und Historie)
# ─────────────────────────────────────────────────────────────────────────────

def _cci(tp: pd.Series, n: int = 20) -> pd.Series:
    arr = tp.to_numpy(float)
    out = np.full(len(arr), np.nan)
    if len(arr) >= n:
        w  = np.lib.stride_tricks.sliding_window_view(arr, n)
        m  = w.mean(axis=1)
        md = np.abs(w - m[:, None]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n-1:] = (arr[n-1:] - m) / (0.015 * np.where(md == 0, np.nan, md))
    return pd.Series(out, index=tp.index)


def add_strategy_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """StochRSI(70), Stoch Fast/Slow(70, 7er-Glättung), CCI(20), MACD(70,200,9), SMA200 (=TREND_MA)."""
    has_70  = len(df) >= 70
    has_200 = len(df) >= 200
    rsi_w   = 70 if has_70 else 14

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(rsi_w).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(rsi_w).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    rsi_r     = (df['RSI'].rolling(rsi_w).max() - df['RSI'].rolling(rsi_w).min()).replace(0, np.nan)
    df['StochRSI_70'] = (df['RSI'] - df['RSI'].rolling(rsi_w).min()) / rsi_r

    hl_f = (df['High'].rolling(70).max() - df['Low'].rolling(70).min()).replace(0, np.nan)
    df['Stoch_Fast'] = 100 * (df['Close'] - df['Low'].rolling(70).min()) / hl_f
    df['Stoch_Slow'] = df['Stoch_Fast'].rolling(7).mean()

    df['CCI_20'] = _cci((df['High'] + df['Low'] + df['Close']) / 3, 20)

    df = compute_macd(df, fast=70 if has_70 else 12, slow=200 if has_200 else 26, signal=9)

    # Trendfilter: echter SMA200. TREND_MA ist der Name, den ALLE Trend-Auswertungen nutzen
    # (in detail_engine.py hieß die EMA200 fälschlich ebenfalls 'SMA200').
    df['SMA200']   = df['Close'].rolling(min(200, len(df) - 1)).mean()
    df['TREND_MA'] = df['SMA200']
    return df


def _trend_ma(df: pd.DataFrame) -> np.ndarray:
    col = "TREND_MA" if "TREND_MA" in df.columns else ("SMA200" if "SMA200" in df.columns else None)
    return df[col].to_numpy(float) if col else np.full(len(df), np.nan)


def _shift_bool(a: np.ndarray, s: int) -> np.ndarray:
    """Wert von Position i-s an Position i (vorne mit False aufgefüllt)."""
    out = np.zeros(len(a), dtype=bool)
    if s < len(a):
        out[s:] = a[:len(a) - s]
    return out


def macd_turn_series(osma: np.ndarray, lookback: int = 3) -> np.ndarray:
    """Vektorisierte Version von macd_turn_at für ALLE Positionen: +1 = 'up', -1 = 'down', 0 = None."""
    n   = len(osma)
    out = np.zeros(n, dtype=np.int8)
    if n < lookback + 2:
        return out
    fin = np.isfinite(osma)
    d   = np.zeros(n)
    d[1:] = np.diff(osma)
    ok = np.ones(n, dtype=bool)
    for s in range(lookback + 2):
        ok &= _shift_bool(fin, s)
    with np.errstate(invalid="ignore"):
        neg, pos = d < 0, d > 0
    falling = np.ones(n, dtype=bool)
    rising  = np.ones(n, dtype=bool)
    for s in range(1, lookback + 1):
        falling &= _shift_bool(neg, s)
        rising  &= _shift_bool(pos, s)
    out[ok & falling & pos] = 1
    out[ok & rising  & neg] = -1
    return out


def _criteria_arrays(df: pd.DataFrame, rsi_col="StochRSI_70", cci_col="CCI_20") -> dict:
    def g(c):
        return df[c].to_numpy(float) if c in df.columns else np.full(len(df), np.nan)
    sr, sf, ss, cv, osma = g(rsi_col), g("Stoch_Fast"), g("Stoch_Slow"), g(cci_col), g("OSMA")
    valid = np.isfinite(sr) & np.isfinite(sf) & np.isfinite(ss) & np.isfinite(cv)
    turn  = macd_turn_series(osma)
    with np.errstate(invalid="ignore"):
        b_sr, b_st, b_cci, b_m = (sr > 0) & (sr < 0.1), (sf < 10) & (ss < 15), cv > -100, turn == 1
        s_sr, s_st, s_cci, s_m = sr > 0.9, (sf > 90) & (ss > 85), cv < 100, turn == -1
    i = lambda a: a.astype(int)
    return {
        "valid": valid, "turn": turn,
        "buy":  {"score": i(b_sr) + i(b_st) + i(b_cci) + i(b_m), "extreme": b_sr & b_st},
        "sell": {"score": i(s_sr) + i(s_st) + i(s_cci) + i(s_m), "extreme": s_sr & s_st},
    }


def _event_indices(df, side, rsi_col="StochRSI_70", cci_col="CCI_20", cooldown=SIGNAL_COOLDOWN):
    """
    Signal-Ereignisse = der ERSTE Tag jeder Extrem-Phase (genau der Tag, an dem die Mail zum
    ersten Mal anschlagen würde). Folgetage innerhalb von `cooldown` Handelstagen gehören
    zur selben Phase - so zählen überlappende Tage nicht mehrfach in die Trefferquote.
    """
    arr  = _criteria_arrays(df, rsi_col, cci_col)
    gate = arr[side]["extreme"] if REQUIRE_BOTH_EXTREMES else arr[side]["score"] >= 2
    idx  = np.flatnonzero(gate & arr["valid"])
    keep, prev = [], -10**9
    for i in idx:
        i = int(i)
        if i - prev > cooldown:
            keep.append(i)
        prev = i
    return keep, arr


def find_signal_history(df: pd.DataFrame, rsi_col="StochRSI_70", cci_col="CCI_20", side="buy") -> list[dict]:
    """Alle Tage mit Score >= 2 der gewählten Seite (für die Chart-Marker in der App)."""
    hits = []
    if len(df) < 20 or rsi_col not in df.columns or cci_col not in df.columns:
        return hits
    arr  = _criteria_arrays(df, rsi_col, cci_col)
    mask = (arr[side]["score"] >= 2) & arr["valid"]
    sr, cv, close = df[rsi_col].to_numpy(float), df[cci_col].to_numpy(float), df["Close"].to_numpy(float)
    for i in np.flatnonzero(mask):
        t = arr["turn"][i]
        hits.append({"date": df.index[i], "price": round(float(close[i]), 2),
                     "StochRSI": round(float(sr[i]), 3), "CCI": round(float(cv[i]), 1),
                     "MACD_Turn": "up" if t == 1 else ("down" if t == -1 else None),
                     "score": int(arr[side]["score"][i])})
    return hits


def find_entry_signals(df: pd.DataFrame, rsi_col="StochRSI_70", cci_col="CCI_20",
                       side="buy", max_wait: int = MAX_ENTRY_WAIT) -> list[dict]:
    """
    Einstiegszeitpunkte OHNE Look-ahead:
      Signal (Extrem-Phase beginnt) -> Bestätigung = MACD-Dreh in Signalrichtung ODER Kurs 2 % vom
      bisherigen Extrem (laufendes Tief/Hoch, nur Daten bis zum jeweiligen Tag) -> Ausführung zum
      OPEN des Folgetages (so wie man nach der Mail tatsächlich handeln kann).
    Frühere Version: Bestätigung über das Tief der NÄCHSTEN 20 Tage - das kennt man live nicht.
    """
    if len(df) < 20 or rsi_col not in df.columns or cci_col not in df.columns:
        return []
    events, arr = _event_indices(df, side, rsi_col, cci_col)
    n     = len(df)
    close = df["Close"].to_numpy(float)
    low   = df["Low"].to_numpy(float)
    high  = df["High"].to_numpy(float)
    opn   = df["Open"].to_numpy(float) if "Open" in df.columns else close
    ma    = _trend_ma(df)
    is_buy   = side == "buy"
    turn_dir = 1 if is_buy else -1
    entries  = []

    for i in events:
        ext, j_entry, trigger = None, None, None
        for j in range(i, min(i + max_wait, n - 1) + 1):
            if is_buy:
                ext = low[j] if ext is None else min(ext, low[j])
            else:
                ext = high[j] if ext is None else max(ext, high[j])
            if arr["turn"][j] == turn_dir:
                trigger, j_entry = "MACD", j
                break
            if is_buy and close[j] > ext * 1.02:
                trigger, j_entry = "Preis", j
                break
            if (not is_buy) and close[j] < ext * 0.98:
                trigger, j_entry = "Preis", j
                break
        if j_entry is None:
            continue

        at_open = j_entry + 1 < n and np.isfinite(opn[j_entry + 1])
        ex      = j_entry + 1 if at_open else j_entry
        e_price = float(opn[ex]) if at_open else float(close[j_entry])
        if not np.isfinite(e_price) or e_price <= 0 or not np.isfinite(ext) or ext <= 0:
            continue

        trend_at_entry = None
        if np.isfinite(ma[j_entry]):
            trend_at_entry = "up" if close[j_entry] > ma[j_entry] else "down"

        move = (e_price - ext) / ext * 100 if is_buy else (ext - e_price) / ext * 100
        entries.append({
            "side": side, "signal_date": df.index[i], "signal_price": round(float(close[i]), 2),
            "decision_date": df.index[j_entry], "entry_date": df.index[ex], "entry_price": round(e_price, 2),
            "at_open": bool(at_open), "local_low": round(float(ext), 2),   # bei side='sell': lokales Hoch
            "days_to_entry": (df.index[ex] - df.index[i]).days,
            "upside_pct": round(float(move), 1),
            "trigger": trigger, "trend_at_entry": trend_at_entry,
        })

    entries.sort(key=lambda e: e["signal_date"])
    seen, deduped = set(), []
    for e in entries:
        if e["entry_date"] in seen:
            continue
        seen.add(e["entry_date"])
        deduped.append(e)
    return deduped


def _baseline_rate(df: pd.DataFrame, side: str, forward_days: int, thr: float = HIT_THRESHOLD_PCT):
    """
    Zufalls-Baseline: Wie oft erreicht IRGENDEIN Handelstag dieses Tickers (nach Einlaufzeit)
    innerhalb von `forward_days` +thr % (Kauf) bzw. -thr % (Verkauf)? Eine Trefferquote ist nur
    dann Information, wenn sie klar über diesem Wert liegt (bei 30 % Vola sind es ca. 60 %).
    """
    c = pd.Series(df["Close"].to_numpy(float))
    if side == "buy":
        ext = c[::-1].rolling(forward_days).max()[::-1].shift(-1)
        fav = (ext - c) / c * 100
    else:
        ext = c[::-1].rolling(forward_days).min()[::-1].shift(-1)
        fav = (c - ext) / c * 100
    arr  = _criteria_arrays(df)
    ma   = _trend_ma(df)
    mask = arr["valid"] & np.isfinite(ma) & fav.notna().to_numpy()
    if mask.sum() < 50:
        return None
    return float((fav.to_numpy()[mask] >= thr).mean())


def compute_hit_rate(entry_signals: list[dict], df: pd.DataFrame, forward_days: int = FORWARD_DAYS,
                     side: str | None = None) -> dict:
    """
    Für jeden historischen Einstieg: erreichte der Kurs innerhalb von `forward_days` Tagen
    >= 5 % in Signalrichtung (Schlusskurse)? Zusätzlich 'baseline' (Zufallsquote desselben Tickers)
    und 'edge' (= rate - baseline). max_gain_pct = günstigste, max_loss_pct = ungünstigste Bewegung.
    """
    side = side or (entry_signals[0].get("side", "buy") if entry_signals else "buy")
    empty = {"total": 0, "hits": 0, "misses": 0, "rate": 0.0, "details": [],
             "baseline": None, "edge": None, "side": side}
    if not entry_signals or df.empty:
        return empty

    is_buy = side == "buy"
    hits = misses = 0
    details = []
    cutoff = df.index[-forward_days] if len(df) > forward_days else df.index[0]

    for e in entry_signals:
        entry_date, entry_price = e["entry_date"], e["entry_price"]
        trend_at_entry = e.get("trend_at_entry")
        try:
            if entry_date > cutoff:
                details.append({"entry_date": entry_date, "entry_price": entry_price, "result": "⏳ Zu Neu",
                                "result_code": "new", "max_gain_pct": None, "max_loss_pct": None,
                                "trend_at_entry": trend_at_entry})
                continue
            pos   = df.index.get_loc(entry_date)
            start = pos if e.get("at_open", False) else pos + 1
            fwd   = df.iloc[start: start + forward_days]
            if fwd.empty or not entry_price:
                continue
            closes = fwd["Close"].to_numpy(float)
            if is_buy:
                fav = (closes.max() - entry_price) / entry_price * 100
                adv = (closes.min() - entry_price) / entry_price * 100
            else:
                fav = (entry_price - closes.min()) / entry_price * 100
                adv = (entry_price - closes.max()) / entry_price * 100
            fav, adv = round(float(fav), 1), round(float(adv), 1)

            if fav >= 10:
                result, code = "✅ Starke Wende", "strong"; hits += 1
            elif fav >= HIT_THRESHOLD_PCT:
                result, code = "🟡 Schwache Wende", "weak"; hits += 1
            elif adv < -5 and fav < 3:
                result, code = "❌ Kein Boden" if is_buy else "❌ Kein Top", "fail"; misses += 1
            else:
                result, code = "⚠️ Seitwärts", "sideways"; misses += 1
            details.append({"entry_date": entry_date, "entry_price": entry_price, "result": result,
                            "result_code": code, "max_gain_pct": fav, "max_loss_pct": adv,
                            "trend_at_entry": trend_at_entry})
        except Exception:
            continue

    total = hits + misses
    rate  = round(hits / total, 2) if total > 0 else 0.0
    base  = None
    try:
        base = _baseline_rate(df, side, forward_days)
    except Exception:
        pass
    edge = round(rate - base, 2) if (base is not None and total > 0) else None
    return {"total": total, "hits": hits, "misses": misses, "rate": rate, "details": details,
            "baseline": round(base, 2) if base is not None else None, "edge": edge, "side": side}


def analyze_trend_hit_rate(tickers: list[str], progress_cb=None, forward_days: int = FORWARD_DAYS) -> dict:
    """
    Trend-Vergleich (App Tab 3): Trefferquote der Kauf-Einstiege über/unter SMA200 zum Einstiegszeitpunkt.
    Lädt alle Kurse in Batches (download_price_frames) statt pro Ticker einzeln - sonst hängt die
    Auswertung bei "Alle Gruppen" (gesamte Datenbank) in Streamlit.
    """
    cohorts = {k: {"hits": 0, "misses": 0, "gains": [], "losses": []} for k in ("up", "down", "unknown")}
    n_tickers_used = 0

    def _dl_progress(done, total, phase):
        if progress_cb:
            try: progress_cb(done, total, "Kursdaten laden …")
            except Exception: pass

    frames, _failed = download_price_frames(tickers, progress_cb=_dl_progress)

    for i, (ticker, raw_df) in enumerate(frames.items()):
        if progress_cb:
            try: progress_cb(i, len(frames), ticker)
            except Exception: pass
        try:
            df = clean_price_df(raw_df)
            if df.empty or len(df) < 60:
                continue
            df = add_strategy_indicators(df)
            entries = find_entry_signals(df, side="buy")
            if not entries:
                continue
            hr = compute_hit_rate(entries, df, forward_days=forward_days, side="buy")
            if hr["total"] == 0:
                continue
            n_tickers_used += 1
            for d in hr["details"]:
                if d["result_code"] == "new":
                    continue
                c = cohorts[d.get("trend_at_entry") or "unknown"]
                c["hits" if d["result_code"] in ("strong", "weak") else "misses"] += 1
                if d.get("max_gain_pct") is not None: c["gains"].append(d["max_gain_pct"])
                if d.get("max_loss_pct") is not None: c["losses"].append(d["max_loss_pct"])
        except Exception:
            continue
    summary = {"n_tickers_used": n_tickers_used}
    for key, c in cohorts.items():
        total = c["hits"] + c["misses"]
        summary[key] = {"total_trades": total, "hits": c["hits"], "misses": c["misses"],
                        "hit_rate": round(c["hits"] / total * 100, 1) if total > 0 else None,
                        "avg_max_gain": round(float(np.mean(c["gains"])), 1) if c["gains"] else None,
                        "avg_max_loss": round(float(np.mean(c["losses"])), 1) if c["losses"] else None}
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# Handelsschluss in UTC (konservativ = Winterzeit; im Sommer schließen die Börsen 1 h früher).
_CLOSE_UTC = {".AX": (6, 15), ".T": (6, 45), ".KS": (6, 45), ".HK": (8, 30), ".SR": (12, 15), ".IS": (15, 15),
              ".JO": (15, 15), ".SA": (21, 10), ".MX": (21, 15)}
_EU_SUFFIXES = (".DE", ".PA", ".MC", ".MI", ".AS", ".VI", ".LS", ".HE", ".L", ".SW", ".OL", ".ST", ".CO", ".BR")


def _close_time_utc(ticker: str) -> tuple[int, int]:
    up = ticker.upper()
    for suf, t in _CLOSE_UTC.items():
        if up.endswith(suf):
            return t
    if up.endswith(_EU_SUFFIXES):
        return (16, 45)
    return (21, 15)   # US


def drop_incomplete_bar(df: pd.DataFrame, ticker: str, now_utc: datetime) -> pd.DataFrame:
    """
    Verwirft die heutige Kerze, solange die Börse noch offen ist. Sonst rechnet der Scan mit einer
    Zwischenstand-Kerze (Europa intraday, USA schon gestern fertig = gemischte Zeitstände).
    """
    if df is None or df.empty:
        return df
    last_date, today = pd.Timestamp(df.index[-1]).date(), now_utc.date()
    if last_date > today:
        return df.iloc[:-1]
    if last_date == today and (now_utc.hour, now_utc.minute) < _close_time_utc(ticker):
        return df.iloc[:-1]
    return df


def analyze_df(ticker: str, df: pd.DataFrame, compute_history: bool = True) -> dict | None:
    """Analyse eines fertigen OHLCV-DataFrames (ohne Netzwerk, ohne .info)."""
    df = clean_price_df(df)
    if df is None or df.empty or len(df) < 50:
        return None
    df = add_strategy_indicators(df.copy())
    macd_turn = detect_macd_turn(df)

    price = float(df['Close'].iloc[-1])
    sma   = df['TREND_MA'].iloc[-1]
    trend = None if pd.isna(sma) else ("up" if price > sma else "down")

    def _s(v, d=3):
        try:
            f = float(v)
            return round(f, d) if np.isfinite(f) else None
        except Exception:
            return None

    sv, sf = _s(df['StochRSI_70'].iloc[-1]), _s(df['Stoch_Fast'].iloc[-1], 1)
    ss, cv = _s(df['Stoch_Slow'].iloc[-1], 1), _s(df['CCI_20'].iloc[-1], 1)

    sig   = classify_signal(sv, sf, ss, cv, macd_turn)
    bs, sl = sig["buy_score"], sig["sell_score"]
    score = max(bs, sl)
    side  = sig["side"]

    # Trefferquote immer für die Seite, die gezeigt wird (nicht mehr pauschal Kauf-Historie)
    hist_side = side or (("buy" if bs >= sl else "sell") if score >= 2 else None)
    hr_pct = hr_n = hr_hits = base_pct = edge_pct = None
    hr_n = 0
    if compute_history and hist_side:
        try:
            hr = compute_hit_rate(find_entry_signals(df, side=hist_side), df, side=hist_side)
            hr_n, hr_hits = hr["total"], hr["hits"]
            if hr_n >= MIN_HISTORY_N:
                hr_pct   = round(hr["rate"] * 100, 0)
                base_pct = round(hr["baseline"] * 100, 0) if hr["baseline"] is not None else None
                edge_pct = round(hr["edge"] * 100, 0)     if hr["edge"]     is not None else None
        except Exception as e:
            log.debug("hit-rate %s: %s", ticker, e)

    price_20d = float(df['Close'].iloc[-21]) if len(df) > 21 else None
    ret_20d   = round((price - price_20d) / price_20d * 100, 2) if price_20d else None
    sym, div  = resolve_currency(ticker)

    return {
        "Ticker": ticker, "Name": ticker[:20], "Signal": f"{sig['emoji']} {sig['label']}",
        "Preis": f"{round(price / div, 2)} {sym}", "Preis_Num": price, "Währung": sym,
        "StochRSI": sv, "CCI": cv, "Stoch_Fast": sf, "Stoch_Slow": ss, "MACD_Turn": macd_turn,
        "Score": score, "Buy_Score": bs, "Sell_Score": sl, "Side": side, "Trend": trend,
        "HitRate": hr_pct, "HitRate_N": hr_n, "HitRate_Hits": hr_hits,
        "HitRate_Base": base_pct, "HitRate_Edge": edge_pct,
        "Ret_20d": ret_20d, "Div": "–", "KGV": "N/A",
        "Last_Date": str(pd.Timestamp(df.index[-1]).date()),
    }


def enrich_with_info(res: dict) -> dict:
    """Name, Währung (Pence-Fix), Dividende, KGV - der langsame .info-Abruf, nur wo nötig."""
    t = res["Ticker"]
    stock, inf = yf.Ticker(t), {}
    try:
        inf = stock.info or {}
    except Exception:
        pass
    code = inf.get("currency")
    if not code:
        try:
            code = stock.fast_info.get("currency")
        except Exception:
            code = None
    sym, div = resolve_currency(t, code)
    raw = res["Preis_Num"]
    res["Name"]    = (inf.get("shortName") or t)[:20]
    res["Währung"] = sym
    res["Preis"]   = f"{round(raw / div, 2)} {sym}"
    res["Div"]     = format_dividend_yield(inf, raw, subunit=(div != 1))
    pe = inf.get("trailingPE")
    res["KGV"]     = round(pe, 1) if isinstance(pe, (int, float)) and pe else "N/A"
    return res


def get_analysis(ticker: str, retries: int = 2, compute_history: bool = True,
                 lazy_info: bool = False, drop_incomplete: bool = False, now_utc=None) -> dict | None:
    """Einzel-Analyse (App). Signatur kompatibel zur alten Version."""
    now_utc = now_utc or datetime.now(timezone.utc)
    for attempt in range(retries + 1):
        try:
            df = clean_price_df(yf.Ticker(ticker).history(period=HISTORY_PERIOD, auto_adjust=False))
            if drop_incomplete:
                df = drop_incomplete_bar(df, ticker, now_utc)
            res = analyze_df(ticker, df, compute_history)
            if res is None:
                return None
            if not lazy_info or res["Side"]:
                enrich_with_info(res)
            return res
        except Exception as e:
            if attempt < retries:
                time.sleep(1.0)
            else:
                print(f"[engine] {ticker}: {e}")
    return None


def _split_download(raw: pd.DataFrame, tickers: list[str]) -> dict:
    """yf.download-Ergebnis (je nach Version (Ticker,Feld) oder (Feld,Ticker)) in {ticker: df} zerlegen."""
    out = {}
    if raw is None or raw.empty:
        return out
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        l0, l1 = set(cols.get_level_values(0)), set(cols.get_level_values(1))
        for t in tickers:
            if t in l0:
                sub = raw[t]
            elif t in l1:
                sub = raw.xs(t, axis=1, level=1)
            else:
                continue
            out[t] = sub.dropna(how="all")
    elif len(tickers) == 1:
        out[tickers[0]] = raw.dropna(how="all")
    return out


def download_price_frames(tickers: list[str], batch_size: int = BATCH_SIZE, period: str = HISTORY_PERIOD,
                          progress_cb=None, phase: str = "download") -> tuple[dict, dict]:
    """
    Lädt Kurse für VIELE Ticker in Paketen (yf.download, wenig Rate-Limit) statt mit einem
    yf.Ticker(t).history()-Aufruf pro Ticker - das war der Grund, warum Scanner/Trend-Vergleich/
    Formations-Scan bei mehreren hundert Tickern in Streamlit "hängen" (eine Session blockiert,
    bis alle Einzelabrufe sequentiell durch sind). Gibt (frames, failed) zurück:
      frames = {ticker: DataFrame}, failed = {ticker: Grund}.
    """
    frames, missing = {}, []
    for k in range(0, len(tickers), batch_size):
        chunk = tickers[k:k + batch_size]
        raw = None
        try:
            raw = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                              auto_adjust=False, threads=True, progress=False, timeout=30)
        except Exception as e:
            log.warning("Batch %d-%d fehlgeschlagen: %s", k, k + len(chunk), e)
        got = _split_download(raw, chunk)
        for t in chunk:
            d = got.get(t)
            if d is not None and "Close" in d.columns and d["Close"].notna().sum() >= 50:
                frames[t] = d
            else:
                missing.append(t)
        if progress_cb:
            try: progress_cb(min(k + batch_size, len(tickers)), len(tickers), phase)
            except Exception: pass
        time.sleep(0.5)

    failed = {}
    for t in missing:                         # Einzel-Nachversuch (Batch-Lücken sind häufig Rate-Limit)
        for attempt in range(2):
            try:
                d = yf.Ticker(t).history(period=period, auto_adjust=False)
                if d is not None and not d.empty and d["Close"].notna().sum() >= 50:
                    frames[t] = d
                    break
            except Exception:
                pass
            time.sleep(1.0)
        else:
            failed[t] = "keine/zu wenig Kursdaten"
    return frames, failed


def scan_tickers(tickers: list[str], now_utc: datetime | None = None, batch_size: int = BATCH_SIZE,
                 compute_history: bool = True, progress_cb=None):
    """
    Batch-Scan: Kurse in Paketen laden (schnell, wenig Rate-Limit), Indikatoren lokal rechnen,
    .info nur für Ticker mit Signal. Gibt (results, failed, stale, meta) zurück:
      failed = {ticker: Grund}, stale = Ticker ohne aktuelle Kerze (Feiertag/Handelsaussetzung).
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    frames, failed = download_price_frames(tickers, batch_size, HISTORY_PERIOD, progress_cb, "download")

    analysed, last_dates = [], {}
    for t, d in frames.items():
        try:
            d = drop_incomplete_bar(clean_price_df(d), t, now_utc)
            res = analyze_df(t, d, compute_history)
            if res is None:
                failed[t] = "zu wenig Historie"
                continue
            analysed.append(res)
            last_dates[t] = res["Last_Date"]
        except Exception as e:
            failed[t] = f"Analysefehler: {e}"

    # Aktualität: Referenz = häufigstes Kerzendatum. Wer dahinter liegt, hatte heute keinen Handel.
    ref_date = Counter(last_dates.values()).most_common(1)[0][0] if last_dates else None
    results, stale = [], []
    for res in analysed:
        if ref_date and res["Last_Date"] < ref_date:
            stale.append(res["Ticker"])
        else:
            results.append(res)

    flagged = [r for r in results if r["Side"]]
    for i, r in enumerate(flagged):           # .info nur für die wenigen Signal-Ticker
        try:
            enrich_with_info(r)
        except Exception as e:
            log.debug("info %s: %s", r["Ticker"], e)
        time.sleep(0.1)
        if progress_cb:
            try: progress_cb(i + 1, len(flagged), "info")
            except Exception: pass

    return results, failed, stale, {"ref_date": ref_date, "n_frames": len(frames)}




# ─────────────────────────────────────────────────────────────────────────────

FUTURES_TICKERS = {
    "CL=F":  {"name":"Crude Oil (WTI)",     "group":"Energie",    "unit":"$/bbl"},
    "BZ=F":  {"name":"Brent Crude Oil",     "group":"Energie",    "unit":"$/bbl"},
    "NG=F":  {"name":"Natural Gas",         "group":"Energie",    "unit":"$/MMBtu"},
    "RB=F":  {"name":"RBOB Gasoline",       "group":"Energie",    "unit":"$/gal"},
    "HO=F":  {"name":"Heating Oil",         "group":"Energie",    "unit":"$/gal"},
    "GC=F":  {"name":"Gold",                "group":"Edelmetalle","unit":"$/oz"},
    "SI=F":  {"name":"Silber",              "group":"Edelmetalle","unit":"$/oz"},
    "PL=F":  {"name":"Platin",              "group":"Edelmetalle","unit":"$/oz"},
    "PA=F":  {"name":"Palladium",           "group":"Edelmetalle","unit":"$/oz"},
    "HG=F":  {"name":"Kupfer",              "group":"Edelmetalle","unit":"$/lb"},
    "ZC=F":  {"name":"Mais (Corn)",         "group":"Agrar",      "unit":"¢/bu"},
    "ZW=F":  {"name":"Weizen (Wheat)",      "group":"Agrar",      "unit":"¢/bu"},
    "ZS=F":  {"name":"Soja (Soybeans)",     "group":"Agrar",      "unit":"¢/bu"},
    "KC=F":  {"name":"Kaffee (Coffee)",     "group":"Agrar",      "unit":"¢/lb"},
    "CT=F":  {"name":"Baumwolle (Cotton)",  "group":"Agrar",      "unit":"¢/lb"},
    "SB=F":  {"name":"Zucker (Sugar)",      "group":"Agrar",      "unit":"¢/lb"},
    "CC=F":  {"name":"Kakao (Cocoa)",       "group":"Agrar",      "unit":"$/t"},
    "ES=F":  {"name":"S&P 500 Future",      "group":"Index",      "unit":"Pkt"},
    "NQ=F":  {"name":"NASDAQ 100 Future",   "group":"Index",      "unit":"Pkt"},
    "YM=F":  {"name":"Dow Jones Future",    "group":"Index",      "unit":"Pkt"},
    "RTY=F": {"name":"Russell 2000 Future", "group":"Index",      "unit":"Pkt"},
    "6E=F":  {"name":"EUR/USD Future",      "group":"FX",         "unit":""},
    "6J=F":  {"name":"JPY/USD Future",      "group":"FX",         "unit":""},
    "6B=F":  {"name":"GBP/USD Future",      "group":"FX",         "unit":""},
    "BTC=F": {"name":"Bitcoin Future",      "group":"Krypto",     "unit":"$"},
    "ETH=F": {"name":"Ethereum Future",     "group":"Krypto",     "unit":"$"},
}


def get_futures_analysis(ticker: str) -> dict | None:
    meta = FUTURES_TICKERS.get(ticker, {"name": ticker, "group": "Sonstige", "unit": ""})
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period="1y", auto_adjust=False)
        df    = clean_price_df(df)
        if df.empty or len(df) < 30:
            return None

        rsi_w = 14
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(rsi_w).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(rsi_w).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        rsi_r     = (df['RSI'].rolling(rsi_w).max() - df['RSI'].rolling(rsi_w).min()).replace(0, np.nan)
        df['StochRSI'] = (df['RSI'] - df['RSI'].rolling(rsi_w).min()) / rsi_r

        w14 = min(14, len(df)-1)
        hl  = (df['High'].rolling(w14).max() - df['Low'].rolling(w14).min()).replace(0, np.nan)
        df['Stoch_K'] = 100 * (df['Close'] - df['Low'].rolling(w14).min()) / hl
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        tp   = (df['High'] + df['Low'] + df['Close']) / 3
        mdev = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * mdev.replace(0, np.nan))

        bb_w = min(50, len(df)-1)
        df['EMA50']    = df['Close'].ewm(span=bb_w, adjust=False).mean()
        std            = df['Close'].rolling(bb_w).std()
        df['BB_Upper'] = df['EMA50'] + 2 * std
        df['BB_Lower'] = df['EMA50'] - 2 * std

        w20 = min(20, len(df)-1)
        df['SMA20']   = df['Close'].rolling(w20).mean()
        df['Z_Score'] = (df['Close'] - df['SMA20']) / df['Close'].rolling(w20).std().replace(0, np.nan)
        df['RVOL']    = df['Volume'] / df['Volume'].rolling(w20).mean().replace(0, np.nan)

        df = compute_macd(df, fast=12, slow=26, signal=9)
        macd_turn = detect_macd_turn(df)

        def _s(v, d=2):
            try:
                f = float(v); return round(f, d) if np.isfinite(f) else None
            except Exception: return None

        price  = _s(df['Close'].iloc[-1])
        sr     = _s(df['StochRSI'].iloc[-1], 3)
        sk     = _s(df['Stoch_K'].iloc[-1], 1)
        sd     = _s(df['Stoch_D'].iloc[-1], 1)
        cci    = _s(df['CCI'].iloc[-1], 1)
        rsi_v  = _s(df['RSI'].iloc[-1], 1)
        zscore = _s(df['Z_Score'].iloc[-1], 2)
        rvol   = _s(df['RVOL'].iloc[-1], 2)

        # Score aus derselben Kriterien-Auswertung wie das Label (keine getrennten,
        # abweichenden Schwellen - siehe evaluate_criteria in engine.py).
        sig   = classify_signal(sr, sk, sd, cci, macd_turn)
        score = sig["buy_score"] if sig["buy_score"] >= sig["sell_score"] else sig["sell_score"]
        df_1y = df.tail(252)

        return {
            "Ticker":   ticker, "Name": meta["name"], "Gruppe": meta["group"],
            "Einheit":  meta["unit"], "Signal": f"{sig['emoji']} {sig['label']}",
            "Sig_Data": sig, "Preis": price,
            "52W_Hoch": _s(df_1y['High'].max()), "52W_Tief": _s(df_1y['Low'].min()),
            "Perf_1W":  _s(((df['Close'].iloc[-1]/df['Close'].iloc[-5])-1)*100 if len(df)>=5 else None, 2),
            "Perf_1M":  _s(((df['Close'].iloc[-1]/df['Close'].iloc[-22])-1)*100 if len(df)>=22 else None, 2),
            "RSI": rsi_v, "StochRSI": sr, "Stoch_K": sk, "Stoch_D": sd,
            "CCI": cci, "Z_Score": zscore, "RVOL": rvol, "MACD_Turn": macd_turn,
            "Score": score, "df": df,
        }
    except Exception as e:
        print(f"[futures] {ticker}: {e}")
        return None


def get_all_futures_groups() -> list:
    seen, r = set(), []
    for v in FUTURES_TICKERS.values():
        if v["group"] not in seen:
            seen.add(v["group"]); r.append(v["group"])
    return r


def get_futures_by_group(group: str) -> list:
    return [k for k, v in FUTURES_TICKERS.items() if v["group"] == group]




# ─────────────────────────────────────────────────────────────────────────────
# EMAIL — vier getrennte Sektionen: Kauf / Verkauf / Watchlist Long / Watchlist Short
# ─────────────────────────────────────────────────────────────────────────────

def _num(v, default=None):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _hist_text(r: dict, side: str) -> tuple[str, str]:
    """(Text, Farbe) für die Historie - seitenrichtig und mit Zufalls-Vergleich."""
    n    = int(_num(r.get("HitRate_N"), 0) or 0)
    hr   = _num(r.get("HitRate"))
    base = _num(r.get("HitRate_Base"))
    edge = _num(r.get("HitRate_Edge"))
    if hr is None:
        return f"Historie: zu wenig Daten (n={n})", "#8a94a6"
    move = "+5 %" if side == "buy" else "−5 %"
    txt  = f"Historie: {int(hr)} % erreichten {move} in {FORWARD_DAYS} T. (n={n})"
    if base is not None:
        txt += f" · Zufall {int(base)} %"
    color = "#555"
    if edge is not None:
        color = "#0d7a2e" if edge >= 10 else ("#b03a2e" if edge <= 0 else "#555")
    return txt, color


def _card(r: dict, sig: dict) -> str:
    side     = sig["side"]
    is_buy   = side == "buy"
    color    = "#1a9e3f" if is_buy else "#c0392b"
    ticker   = html.escape(str(r.get("Ticker", "")))
    name     = html.escape(str(r.get("Name", ""))[:25])
    sc       = sig["buy_score"] if is_buy else sig["sell_score"]
    trend    = r.get("Trend")
    macd_arrow = {"up": "↑", "down": "↓"}.get(r.get("MACD_Turn"), "–")
    link     = f"{APP_URL}/?ticker={ticker}"

    if trend == "up":
        trend_html = '<span style="color:#0d7a2e;font-weight:700;">📈 Aufwärtstrend</span>'
    elif trend == "down":
        trend_html = '<span style="color:#b03a2e;font-weight:700;">📉 Abwärtstrend</span>'
    else:
        trend_html = '<span style="color:#8a94a6;">Trend n/a</span>'
    counter = (is_buy and trend == "down") or ((not is_buy) and trend == "up")
    warn = ('<div style="font-size:11px;color:#b9770e;margin-top:3px;">⚠ gegen den übergeordneten Trend '
            '(Kurs ' + ('unter' if is_buy else 'über') + ' SMA200)</div>') if counter else ""

    rs = _num(r.get("RelStrength"))
    rs_html = ""
    if rs is not None:
        rs_html = f' · rel. Stärke {"↑" if rs >= 0 else "↓"} {rs:+.1f}% vs. Gruppe'

    focus = get_focus(str(r.get("Ticker", "")))
    focus_html = (f' <span style="font-size:11px;color:#8a94a6;">· Fokus: {html.escape(focus)}</span>' if focus else "")
    htxt, hcol = _hist_text(r, side)
    sr, sf, ss, cv = _num(r.get("StochRSI")), _num(r.get("Stoch_Fast")), _num(r.get("Stoch_Slow")), _num(r.get("CCI"))
    fmt = lambda v, f: (f.format(v) if v is not None else "–")
    ind = (f"StochRSI {fmt(sr, '{:.3f}')} · Stoch F/S {fmt(sf, '{:.1f}')}/{fmt(ss, '{:.1f}')} · "
           f"CCI {fmt(cv, '{:.1f}')} · MACD {macd_arrow}")

    return f"""
<a href="{link}" style="text-decoration:none;color:inherit;display:block;">
<div style="border:1px solid #e3e7ee;border-left:5px solid {color};border-radius:10px;padding:12px 14px;margin:8px 0;background:#fff;">
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td><span style="font-size:15px;font-weight:800;color:#0f1b2d;">{ticker}</span>
        <span style="font-size:12px;color:#8a94a6;margin-left:6px;">{name}</span></td>
    <td align="right"><span style="font-size:16px;font-weight:800;color:#0f1b2d;">{html.escape(str(r.get('Preis', '')))}</span></td>
  </tr><tr>
    <td><span style="font-size:11px;font-weight:700;color:{color};">{sig['emoji']} {sig['label']} · {sc}/4</span>{focus_html}</td>
    <td align="right"><span style="font-size:11px;color:#8a94a6;">Div {html.escape(str(r.get('Div', '–')))} · KGV {html.escape(str(r.get('KGV', 'N/A')))}</span></td>
  </tr></table>
  <div style="font-size:12px;color:#4a5568;margin-top:6px;border-top:1px solid #eef1f6;padding-top:6px;">{ind}</div>
  <div style="font-size:12px;color:#4a5568;margin-top:3px;">{trend_html}{rs_html}</div>
  <div style="font-size:12px;color:{hcol};margin-top:3px;">{html.escape(htxt)}</div>
  {warn}
</div></a>"""


def _section(title: str, entries: list, cap: int, empty_text: str) -> str:
    head = f'<div style="font-size:15px;font-weight:800;color:#0f1b2d;margin:20px 0 6px;">{title} <span style="color:#8a94a6;font-weight:600;">({len(entries)})</span></div>'
    if not entries:
        return head + f'<div style="font-size:12px;color:#8a94a6;margin:4px 0 8px;">{empty_text}</div>'
    body = "".join(_card(r, sig) for r, sig in entries[:cap])
    more = (f'<div style="font-size:12px;color:#8a94a6;margin:4px 0;">+ {len(entries) - cap} weitere in der '
            f'<a href="{APP_URL}" style="color:#3b82f6;">App</a></div>') if len(entries) > cap else ""
    return head + body + more


def _sort_key(item):
    r, sig = item
    edge = _num(r.get("HitRate_Edge"))
    n    = _num(r.get("HitRate_N"), 0) or 0
    has  = edge is not None and n >= MIN_HISTORY_N
    sc   = sig["buy_score"] if sig["side"] == "buy" else sig["sell_score"]
    return (-sc, -(edge if has else -999))


def build_report_html(df_results: pd.DataFrame, total_scanned=0, success_count=0, failed_count=0,
                      max_cards: int = 12, region_label: str = "", failed_list=None,
                      stale_count: int = 0, data_date: str | None = None) -> tuple[str, dict]:
    """HTML + Zähler. Reine Funktion (ohne Versand) - gut testbar."""
    groups = {"KAUFSIGNAL": [], "VERKAUFSIGNAL": [], "OVERSOLD": [], "OVERBOUGHT": []}
    if df_results is not None and not df_results.empty:
        for _, row in df_results.iterrows():
            r = row.to_dict()
            sig = classify_signal(_num(r.get("StochRSI")), _num(r.get("Stoch_Fast")), _num(r.get("Stoch_Slow")),
                                  _num(r.get("CCI")), r.get("MACD_Turn") if isinstance(r.get("MACD_Turn"), str) else None)
            if sig["label"] in groups:
                groups[sig["label"]].append((r, sig))
    for k in groups:
        groups[k].sort(key=_sort_key)

    nb, ns = len(groups["KAUFSIGNAL"]), len(groups["VERKAUFSIGNAL"])
    wl, ws = len(groups["OVERSOLD"]), len(groups["OVERBOUGHT"])
    cap_w  = max(4, max_cards // 2)
    now    = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    stand  = pd.Timestamp(data_date).strftime("%d.%m.%Y") if data_date else "–"

    sections = (
        _section("🟢 Kaufsignale — Extrem + MACD-Bestätigung (4/4)", groups["KAUFSIGNAL"], max_cards,
                 "Heute kein Ticker mit vollständigem Kaufsignal.")
        + _section("🔴 Verkaufssignale — Extrem + MACD-Bestätigung (4/4)", groups["VERKAUFSIGNAL"], max_cards,
                   "Heute kein Ticker mit vollständigem Verkaufssignal.")
        + _section("🟡 Watchlist Long — überverkauft, MACD noch ohne Umkehr", groups["OVERSOLD"], cap_w,
                   "Keine Kandidaten.")
        + _section("🟠 Watchlist Short — überkauft, MACD noch ohne Umkehr", groups["OVERBOUGHT"], cap_w,
                   "Keine Kandidaten.")
    )

    foot_lines = []
    if failed_list:
        shown = ", ".join(html.escape(t) for t in list(failed_list)[:25])
        foot_lines.append(f"⚠ {len(failed_list)} Ticker ohne Daten: {shown}{' …' if len(failed_list) > 25 else ''}")
    if stale_count:
        foot_lines.append(f"{stale_count} Ticker ohne aktuelle Kerze übersprungen (Börsenfeiertag/Handelsaussetzung).")
    foot = "".join(f'<div style="font-size:11px;color:#8a94a6;margin-top:3px;">{l}</div>' for l in foot_lines)

    body = f"""<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body style="margin:0;padding:16px;background:#f4f6fa;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;">
<div style="max-width:640px;margin:0 auto;">
  <div style="background:#fff;border:1px solid #e3e7ee;border-radius:14px;padding:16px 18px;">
    <div style="font-size:22px;font-weight:800;"><span style="color:#3bb0ff;">In8</span><span style="color:#0f1b2d;">Invest</span>
      <span style="font-size:13px;color:#8a94a6;font-weight:600;margin-left:6px;">{html.escape(region_label)}</span></div>
    <div style="font-size:12px;color:#8a94a6;margin-top:2px;">Kerzen vom {stand} (Handelsschluss) · erstellt {now}</div>
    <div style="margin-top:8px;"><a href="{APP_URL}" style="color:#3b82f6;font-size:13px;font-weight:700;text-decoration:none;">App öffnen →</a></div>
  </div>
  <div style="font-size:12px;color:#4a5568;text-align:center;margin:12px 0 0;">
    {total_scanned} geprüft · {success_count} analysiert · {failed_count} Fehler ·
    <b>{nb} Kauf</b> · <b>{ns} Verkauf</b> · {wl} Watch Long · {ws} Watch Short</div>
  {sections}
  <div style="margin-top:18px;padding:10px 12px;background:#fff8e6;border-left:4px solid #f0a500;font-size:11px;color:#7a4b00;line-height:1.5;">
    <b>Kriterien:</b> StochRSI(70) &lt; 0.1 bzw. &gt; 0.9 · Stoch Fast(70) &lt; 10 / &gt; 90 und Slow(7er-Glättung) &lt; 15 / &gt; 85 ·
    CCI(20) &gt; −100 bzw. &lt; 100 · MACD(70,200,9)-Histogramm dreht. 4/4 = alle erfüllt. Watchlist = beide Oszillatoren extrem, MACD fehlt noch.
    <b>Verkaufssignal</b> = Reduzieren/Gewinne sichern, kein automatisches Short-Signal. <b>Trend</b> = Kurs vs. SMA200
    (⚠ = Signal gegen den Trend). <b>Historie</b> = Anteil früherer Signale dieses Tickers, die innerhalb von {FORWARD_DAYS} Handelstagen
    {int(HIT_THRESHOLD_PCT)} % in Signalrichtung erreichten (Einstieg zum nächsten Open) im Vergleich zur <b>Zufallsquote</b>
    (alle Tage dieses Tickers). Nur eine Quote deutlich über Zufall ist Information; unter n={MIN_HISTORY_N} wird sie nicht gezeigt.
    Keine Anlageberatung, ohne Gebühren/Spreads.
  </div>
  <div style="margin-top:8px;">{foot}</div>
  <div style="font-size:11px;color:#b0b7c3;text-align:center;margin-top:12px;">In8Invest Scanner · automatisch generiert</div>
</div></body></html>"""
    counts = {"buy": nb, "sell": ns, "watch_long": wl, "watch_short": ws}
    return body, counts


def send_mail_report(df_results, password: str, total_scanned: int = 0, success_count: int = 0,
                     failed_count: int = 0, max_cards: int = 12, region_label: str = "",
                     failed_list=None, stale_count: int = 0, data_date: str | None = None) -> str:
    """Versand. Empfänger/SMTP per Umgebungsvariablen überschreibbar (MAIL_TO, MAIL_FROM, SMTP_HOST, SMTP_PORT)."""
    sender   = os.getenv("MAIL_FROM", "daily@in8invest.com")
    receiver = os.getenv("MAIL_TO", "j.jendraszek@yahoo.de")
    host     = os.getenv("SMTP_HOST", "w01a1dc3.kasserver.com")
    port     = int(os.getenv("SMTP_PORT", "465"))
    try:
        body, c = build_report_html(df_results, total_scanned, success_count, failed_count, max_cards,
                                    region_label, failed_list, stale_count, data_date)
        d = pd.Timestamp(data_date).strftime("%d.%m.") if data_date else datetime.now(timezone.utc).strftime("%d.%m.")
        subject = (f"📊 In8Invest {region_label} | 🟢 {c['buy']} Kauf · 🔴 {c['sell']} Verkauf · "
                   f"👀 {c['watch_long'] + c['watch_short']} Watch | {d}")
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, sender, receiver
        msg.attach(MIMEText(body, "html", "utf-8"))
        with smtplib.SMTP_SSL(host, port) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        return f"✅ Mail gesendet: {subject}"
    except Exception as e:
        return f"❌ Mail-Fehler: {e}"

# detail_engine.py — v5: gemeinsame Indikatoren mit engine.py, echter SMA200-Trendfilter, Kauf- UND Verkaufs-Historie

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from database import get_currency
from engine import (classify_signal, clean_price_df, compute_macd, detect_macd_turn, format_dividend_yield,
                     find_signal_history, find_entry_signals, compute_hit_rate, add_strategy_indicators,
                     resolve_currency, HISTORY_PERIOD, download_price_frames)
from ticker_search import get_display_name as _search_display_name


def _fetch_history(ticker: str):
    stock = yf.Ticker(ticker)
    df    = clean_price_df(stock.history(period=HISTORY_PERIOD, auto_adjust=False))
    if df.empty or len(df) < 50:
        df = clean_price_df(stock.history(period="1y", auto_adjust=False))
    return df, stock


def get_detail_analysis(ticker: str) -> dict | None:
    """Einzel-Analyse (Tab 'Detail-Analyse'/'Einzelne Aktie'): mit vollem .info-Abruf (Sektor, Div, KGV, Name)."""
    try:
        df, stock = _fetch_history(ticker)
        if df.empty or len(df) < 50:
            return None
        inf = {}
        try: inf = stock.info or {}
        except Exception: pass
        return _analyze_detail_frame(ticker, df, inf)
    except Exception as e:
        print(f"[detail_engine] {ticker}: {e}")
        return None


def scan_detail_batch(tickers: list[str], progress_cb=None) -> dict:
    """
    Formations-Scan über VIELE Ticker (Tab 'Markt scannen'): lädt alle Kurse in Batches
    (engine.download_price_frames) statt einem yf.Ticker(t).history()-Aufruf pro Ticker - das war
    der Grund, warum der Scan bei der gesamten Datenbank (800+ Ticker) in Streamlit hing. Auf den
    langsamen .info-Abruf wird hier bewusst verzichtet (Name kommt aus der lokalen ticker_search-
    Datenbank) - die paar Datenpunkte, die wirklich .info brauchen (Sektor, Div, KGV), werden in
    der Tabelle nicht angezeigt und bei Bedarf über 'Einzelne Aktie' nachgeladen.
    Gibt {ticker: result_dict} zurück (nur Ticker mit genug Historie).
    """
    frames, _failed = download_price_frames(tickers, progress_cb=progress_cb, phase="scan")
    out = {}
    for t, raw in frames.items():
        try:
            df = clean_price_df(raw)
            if df.empty or len(df) < 50:
                continue
            res = _analyze_detail_frame(t, df, inf={})
            if res:
                out[t] = res
        except Exception as e:
            print(f"[detail_engine] {t}: {e}")
    return out


def _analyze_detail_frame(ticker: str, df: pd.DataFrame, inf: dict | None = None) -> dict | None:
    """Reine Berechnung auf einem bereits geladenen, bereinigten OHLCV-DataFrame (kein Netzwerk)."""
    inf = inf or {}
    try:
        has_200 = len(df) >= 200
        has_70  = len(df) >= 70

        # ── Indikatoren: EINE Berechnung (engine.add_strategy_indicators) für Scan, Mail und App ──
        df = add_strategy_indicators(df)
        df['RSI_70'] = df['RSI']                      # Alias für ältere Verweise
        macd_turn = detect_macd_turn(df)

        # Chart-Spalten (nur Darstellung). ACHTUNG: 'SMA200' bleibt hier die EMA200-Mittellinie der
        # Bollinger-Bänder (so plottet app.py sie unter dem Label "EMA200"). Der Trendfilter nutzt
        # ausschließlich 'TREND_MA' (= echter SMA200 aus add_strategy_indicators).
        tp   = (df['High'] + df['Low'] + df['Close']) / 3
        bb_w = 200 if has_200 else 50
        df['SMA200']   = df['Close'].ewm(span=bb_w, adjust=False).mean()
        std_bb         = df['Close'].rolling(bb_w).std()
        df['BB_Upper'] = df['SMA200'] + 2 * std_bb
        df['BB_Lower'] = df['SMA200'] - 2 * std_bb

        w20 = min(20, len(df)-1)
        df['SMA20']   = df['Close'].rolling(w20).mean()
        df['Z_Score'] = (df['Close'] - df['SMA20']) / df['Close'].rolling(w20).std().replace(0, np.nan)
        df['RVOL']    = df['Volume'] / df['Volume'].rolling(w20).mean().replace(0, np.nan)

        # ── Muster, Ziele, historische Marker ─────────────────────
        patterns, targets = _detect_patterns_and_targets(df)
        signal_history    = find_signal_history(df, side="buy")
        entry_signals     = find_entry_signals(df, side="buy")
        signal_history_sell = find_signal_history(df, side="sell")
        entry_signals_sell  = find_entry_signals(df, side="sell")

        def _s(v, d=3):
            try:
                f = float(v); return round(f, d) if np.isfinite(f) else None
            except Exception: return None

        price         = float(df['Close'].iloc[-1])
        prev          = float(df['Close'].iloc[-2]) if len(df) >= 2 else price
        stoch_rsi_val = _s(df['StochRSI_70'].iloc[-1])
        s_fast        = _s(df['Stoch_Fast'].iloc[-1], 1)
        s_slow        = _s(df['Stoch_Slow'].iloc[-1], 1)
        cci_val       = _s(df['CCI_20'].iloc[-1], 1)

        # Score UND Label kommen aus derselben Kriterien-Auswertung (evaluate_criteria
        # in engine.py) - keine getrennt berechneten, potenziell abweichenden Schwellen mehr.
        sig   = classify_signal(stoch_rsi_val, s_fast, s_slow, cci_val, macd_turn)
        score = sig["buy_score"] if sig["buy_score"] >= sig["sell_score"] else sig["sell_score"]

        # Trendfilter: Kurs vs. SMA/EMA200 (Warnung vor "überverkauft im freien Fall")
        sma200_val = df['TREND_MA'].iloc[-1]
        trend = None
        if pd.notna(sma200_val):
            trend = "up" if price > sma200_val else "down"

        currency, _div = resolve_currency(ticker, inf.get('currency'))
        if _div != 1:   # Chart, Kurse und Kursziele bleiben in Pence/Cent -> ehrlich so beschriften
            currency = {"£": "GBp", "ZAR": "ZAc", "ILS": "ILA"}.get(currency, currency + " (Sub)")
        div_str  = format_dividend_yield(inf, price, subunit=(_div != 1))
        df_1y    = df.tail(252)

        # Name: aus .info, sonst aus der lokalen ticker_search-Datenbank (kein Netzwerk nötig -
        # wichtig für scan_detail_batch(), das .info bewusst auslässt).
        name = inf.get('shortName') or _search_display_name(ticker).split(" — ", 1)[-1].split(" · ")[0] or ticker

        return {
            "Ticker":         ticker,
            "Name":           name[:25],
            "Währung":        currency,
            "Sektor":         inf.get('sector'),
            "Industrie":      inf.get('industry'),
            "Beschreibung":   inf.get('longBusinessSummary'),
            "Website":        inf.get('website'),
            "Land":           inf.get('country'),
            "Mitarbeiter":    inf.get('fullTimeEmployees'),
            "Signal":         f"{sig['emoji']} {sig['label']}",
            "Sig_Data":       sig,
            "Preis":          round(price, 2),
            "Perf_Abs":       round(price - prev, 2),
            "Perf_Pct":       round((price-prev)/prev*100 if prev else 0, 2),
            "Hoch_365":       round(float(df_1y['High'].max()), 2),
            "Tief_365":       round(float(df_1y['Low'].min()), 2),
            "Score":          score,
            "StochRSI":       stoch_rsi_val,
            "Stoch_Fast":     s_fast,
            "Stoch_Slow":     s_slow,
            "CCI":            cci_val,
            "Z_Score":        _s(df['Z_Score'].iloc[-1], 2),
            "RVOL":           _s(df['RVOL'].iloc[-1], 2),
            "MACD_Turn":      macd_turn,
            "Trend":          trend,
            "MACD":           _s(df['MACD'].iloc[-1], 3),
            "MACD_Signal":    _s(df['SIGNAL'].iloc[-1], 3),
            "MACD_Hist":      _s(df['OSMA'].iloc[-1], 3),
            "Div":            div_str,
            "KGV":            round(inf.get('trailingPE',0),1) if inf.get('trailingPE') else "N/A",
            "Patterns":       patterns,
            "Targets":        targets,
            "Signal_History": signal_history,  # historische Indikator-Treffer
            "Entry_Signals":  entry_signals,   # Einstiegszeitpunkte (Kauf)
            "Signal_History_Sell": signal_history_sell,
            "Entry_Signals_Sell":  entry_signals_sell,
            "df":             df,
        }
    except Exception as e:
        print(f"[detail_engine] {ticker}: {e}")
        return None



def _detect_patterns_and_targets(df: pd.DataFrame) -> tuple[list, dict]:
    patterns = []
    targets  = {}

    if len(df) < 30:
        return patterns, targets

    order = min(7, max(2, len(df)//10))

    try:
        idx_max = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
        idx_min = argrelextrema(df['Low'].values,  np.less_equal,    order=order)[0]
        p_max   = df['High'].iloc[idx_max].tail(6).tolist()
        p_min   = df['Low'].iloc[idx_min].tail(6).tolist()
        p_max_i = list(df.index[idx_max])[-6:]
        p_min_i = list(df.index[idx_min])[-6:]

        # ── BODENFORMATIONEN ──────────────────────────────────────
        if len(p_min) >= 2:
            b1, b2 = p_min[-2], p_min[-1]
            if b1 > 0 and abs(b1-b2)/b1 < 0.03:
                patterns.append("Doppelboden")
                try:
                    neckline = df['High'].loc[p_min_i[-2]:p_min_i[-1]].max()
                except Exception:
                    neckline = float(df['Close'].iloc[-1])
                depth = neckline - min(b1, b2)
                targets["Doppelboden_Nackenlinie"] = round(neckline, 2)
                targets["Doppelboden_Ziel_50%"]    = round(neckline + depth * 0.5, 2)
                targets["Doppelboden_Ziel_100%"]   = round(neckline + depth, 2)

        if len(p_min) >= 3:
            b1, b2, b3 = p_min[-3], p_min[-2], p_min[-1]
            spread = max(b1,b2,b3) - min(b1,b2,b3)
            if b1 > 0 and spread/b1 < 0.04:
                patterns.append("Dreifachboden")
                neckline = float(df['Close'].rolling(20).mean().iloc[-1])
                depth    = neckline - min(b1,b2,b3)
                targets["Dreifachboden_Ziel"] = round(neckline + depth, 2)

        if len(p_min) >= 3:
            ls, h, rs = p_min[-3], p_min[-2], p_min[-1]
            if h < ls and h < rs and ls > 0 and abs(ls-rs)/ls < 0.05:
                patterns.append("Umgekehrte SKS (Bodenformation)")
                try:
                    neckline = df['High'].iloc[idx_min[-3]:idx_min[-1]].max()
                    depth    = neckline - h
                    targets["Inv_SKS_Nackenlinie"] = round(neckline, 2)
                    targets["Inv_SKS_Ziel"]        = round(neckline + depth, 2)
                except Exception: pass

        rvol_v = float(df['RVOL'].iloc[-1]) if 'RVOL' in df.columns and pd.notna(df['RVOL'].iloc[-1]) else 0
        if df['Close'].iloc[-1] < df['Open'].iloc[-1] and rvol_v > 2.0:
            patterns.append("Selling Climax (Potenzielle Wende)")
            targets["Selling_Climax_Einstieg"] = round(float(df['High'].iloc[-1]), 2)

        if len(df) >= 10:
            recent_low  = float(df['Low'].tail(10).min())
            recent_high = float(df['High'].tail(3).max())
            if recent_low > 0 and (recent_high - recent_low) / recent_low > 0.05:
                patterns.append("V-Boden (Starker Rebound)")
                targets["V_Boden_Fortsetzung"] = round(recent_high * 1.05, 2)

        # ── TOPFORMATIONEN ────────────────────────────────────────
        if len(p_max) >= 2:
            t1, t2 = p_max[-2], p_max[-1]
            if t1 > 0 and abs(t1-t2)/t1 < 0.03:
                patterns.append("Doppeltop")
                try:
                    neckline = df['Low'].loc[p_max_i[-2]:p_max_i[-1]].min()
                except Exception:
                    neckline = float(df['Close'].iloc[-1]) * 0.95
                depth = max(t1,t2) - neckline
                targets["Doppeltop_Nackenlinie"] = round(neckline, 2)
                targets["Doppeltop_Ziel_50%"]    = round(neckline - depth * 0.5, 2)
                targets["Doppeltop_Ziel_100%"]   = round(neckline - depth, 2)

        if len(p_max) >= 3:
            ls, k, rs = p_max[-3], p_max[-2], p_max[-1]
            if k > ls and k > rs and ls > 0 and abs(ls-rs)/ls < 0.04:
                patterns.append("SKS-Formation (Topformation)")
                try:
                    neckline = df['Low'].iloc[idx_max[-3]:idx_max[-1]].min()
                    depth    = k - neckline
                    targets["SKS_Nackenlinie"] = round(neckline, 2)
                    targets["SKS_Ziel"]        = round(neckline - depth, 2)
                except Exception: pass

        if df['Close'].iloc[-1] > df['Open'].iloc[-1] and rvol_v > 2.5:
            patterns.append("Buying Climax (Erschöpfungsrally)")
            targets["Buying_Climax_SL"] = round(float(df['Low'].iloc[-1]), 2)

        # ── FIBONACCI (52W Swing) ─────────────────────────────────
        df_1y  = df.tail(252)
        high1y = float(df_1y['High'].max())
        low1y  = float(df_1y['Low'].min())
        swing  = high1y - low1y
        if swing > 0:
            targets["Fib_23.6%"] = round(low1y + swing * 0.236, 2)
            targets["Fib_38.2%"] = round(low1y + swing * 0.382, 2)
            targets["Fib_50.0%"] = round(low1y + swing * 0.500, 2)
            targets["Fib_61.8%"] = round(low1y + swing * 0.618, 2)
            targets["Fib_78.6%"] = round(low1y + swing * 0.786, 2)

        # ── GAP & BOLLINGER ───────────────────────────────────────
        if len(df) >= 2:
            prev    = float(df['Close'].iloc[-2])
            c_open  = float(df['Open'].iloc[-1])
            vol_a   = df['Volume'].rolling(20).mean().iloc[-2] if len(df)>=20 else df['Volume'].mean()
            gap     = (c_open - prev) / max(prev, 0.001)
            if gap > 0.015:
                patterns.append(f"{'Power ' if df['Volume'].iloc[-1] > vol_a*1.5 else ''}Gap Up (+{round(gap*100,1)}%)")
            elif gap < -0.015:
                patterns.append(f"{'Power ' if df['Volume'].iloc[-1] > vol_a*1.5 else ''}Gap Down ({round(gap*100,1)}%)")

        if all(c in df.columns for c in ['BB_Upper','BB_Lower','SMA200']):
            bw = (df['BB_Upper'] - df['BB_Lower']) / df['SMA200'].replace(0, np.nan)
            if len(bw.dropna()) >= 100:
                bw_min = bw.rolling(100).min().iloc[-1]
                if pd.notna(bw.iloc[-1]) and pd.notna(bw_min) and bw.iloc[-1] < bw_min * 1.1:
                    patterns.append("Bollinger Squeeze")

    except Exception as e:
        print(f"[pattern_detect] {e}")

    return patterns, targets

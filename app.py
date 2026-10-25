# app.py — v4: Zeitraum-Wahl, historische Marker, Volume-of-Interest, Einstiegssignale

import streamlit as st
import pandas as pd
import numpy as np
import base64
from urllib.parse import quote_plus
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from database import get_all_tickers, get_tickers_by_group, get_all_groups, get_currency, get_ticker_count, get_group_for_ticker, get_groups_for_ticker, get_group_label, get_focus
from engine import (get_analysis, scan_tickers, send_mail_report, classify_signal,
                    get_futures_analysis, FUTURES_TICKERS,
                    get_all_futures_groups, get_futures_by_group,
                    analyze_trend_hit_rate)
from detail_engine import get_detail_analysis, scan_detail_batch, compute_hit_rate
from engine import MIN_HISTORY_N, FORWARD_DAYS
from ticker_search import search_ticker, label_to_ticker, get_display_name, get_exchange, get_sector
from automation_script import add_relative_strength
from volume_profile import compute_profile, describe_profile, build_profile_figure
from options_data import get_options_summary, describe_options

APP_URL = "https://stockupdate-65qjxum6gq2gpjpr5exqfd.streamlit.app"

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="In8Invest Scanner", page_icon="📊", layout="wide")

try:
    def _b64(p):
        with open(p,"rb") as f: return base64.b64encode(f.read()).decode()
    st.markdown(f'<div style="margin-bottom:8px"><img src="data:image/png;base64,{_b64("Icon in8invest.png")}" width="160"></div>',
                unsafe_allow_html=True)
except: pass

st.title("📊 Strategie Scanner")
st.caption(f"🌍 Datenbank: {get_ticker_count()} Ticker · {len(get_all_groups())} Märkte & Sektoren")

# ─── QUERY PARAMS — Deep-Link aus Email (?ticker=XXX) ─────────────────────────
params         = st.query_params
deep_link_ticker = params.get("ticker", None)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "results"    not in st.session_state: st.session_state.results    = pd.DataFrame()
if "det_ticker" not in st.session_state: st.session_state.det_ticker = deep_link_ticker or ""

# Wenn Deep-Link vorhanden, direkt Tab 2 aktiv machen
default_tab = 1 if deep_link_ticker else 0  # 0-indexed

# ─── HELPER: Ampel-Badge ──────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def _cached_options(ticker: str, price: float) -> dict:
    """Optionskette 15 Minuten zwischenspeichern (Yahoo-Abruf dauert 1-3 s und ist rate-limitiert)."""
    return get_options_summary(ticker, price)


def _sig_badge(sig: dict) -> str:
    return (f"<span style='background:{sig['color']};color:#fff;padding:3px 10px;"
            f"border-radius:12px;font-size:12px;font-weight:700'>{sig['emoji']} {sig['label']}</span>")

def _ind_badge(v, low, high, fmt="{:.3f}", invert=False) -> str:
    if v is None: return "<span style='color:#888'>N/A</span>"
    try:
        fv = float(v)
        if fv <= low:    c,t = ("#1a9e3f","OS") if not invert else ("#c0392b","OB")
        elif fv >= high: c,t = ("#c0392b","OB") if not invert else ("#1a9e3f","OS")
        else:             c,t = "#7f8c8d","—"
        b = f"<span style='background:{c};color:#fff;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700'>{t}</span>"
        return f"{b} {fmt.format(fv)}"
    except: return str(v)

def _fnum(v, fmt="{:.2f}", suffix="") -> str:
    """Sicheres Zahlenformat - zeigt 'N/A' statt Crash/„nan“ bei fehlenden Werten."""
    if v is None: return "N/A"
    try:
        fv = float(v)
        if not np.isfinite(fv): return "N/A"
        return f"{fmt.format(fv)}{suffix}"
    except Exception:
        return "N/A"

def _macd_badge(turn) -> str:
    if turn == "up":   return "<span style='background:#0d7a2e;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700'>DREHT ↑ (Trigger)</span>"
    if turn == "down": return "<span style='background:#8e1f14;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700'>DREHT ↓ (Trigger)</span>"
    return "<span style='background:#7f8c8d;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700'>KEINE UMKEHR</span>"

# ─── TABS ─────────────────────────────────────────────────────────────────────
def render_side_analysis(side, entry_sigs, sig_history, df_full, cur, sel_tf):
    """Trefferquoten-Analyse für EINE Seite ('buy' = Boden, 'sell' = Top): Gauge, Tabelle, Details."""
    is_buy = side == "buy"
    w_move   = "Trendwende (+5% oder +10%)" if is_buy else "Rückgang (−5% oder −10%)"
    lbl_ok   = "✅ Trendwenden" if is_buy else "✅ Kursrückgänge"
    lbl_bad  = "❌ Kein Boden" if is_buy else "❌ Kein Top"
    col_fav  = "Max Gewinn" if is_buy else "Max Rückgang"
    col_adv  = "Max Verlust" if is_buy else "Max Anstieg"

    if not entry_sigs:
        st.info("Noch keine historischen " + ("Einstiegs" if is_buy else "Verkaufs") +
                "signale im Datenzeitraum erkennbar.")
    else:
        st.markdown("#### 🎯 Trefferquote: " + ("Hat der Boden gehalten?" if is_buy else "Hat das Top gehalten?"))
        st.caption(f"{len(entry_sigs)} unabhängige historische Signale (je Extrem-Phase eines, Einstieg zum nächsten "
                   f"Open) · prüft, ob innerhalb von {FORWARD_DAYS} Tagen eine {w_move} stattfand. "
                   f"Ohne Gebühren/Spreads.")
        hr = compute_hit_rate(entry_sigs, df_full, forward_days=FORWARD_DAYS, side=side)

        if hr["total"] > 0:
            rate_pct = hr["rate"] * 100
            base, edge = hr.get("baseline"), hr.get("edge")
            if hr["total"] < MIN_HISTORY_N:
                rate_color, rate_label = "#8a94a6", f"⚪ Zu wenige Signale (n={hr['total']})"
            elif edge is not None:
                ep = edge * 100
                if ep >= 10:
                    rate_color, rate_label = "#1a9e3f", "🟢 Deutlich besser als Zufall"
                elif ep > 0:
                    rate_color, rate_label = "#f39c12", "🟡 Kaum besser als Zufall"
                else:
                    rate_color, rate_label = "#c0392b", "🔴 Nicht besser als Zufall"
            else:
                rate_color, rate_label = "#f39c12", "🟡 Ohne Zufalls-Vergleich"
            thr_val = base * 100 if base is not None else 65

            hc1, hc2 = st.columns([1, 2])
            with hc1:
                fig_hr = go.Figure(go.Indicator(
                    mode="gauge+number", value=rate_pct,
                    number={"suffix": "%", "font": {"size": 48, "color": rate_color, "family": "Arial Black"}},
                    title={"text": f"Trefferquote<br><span style='font-size:13px;color:{rate_color}'>{rate_label}</span>",
                           "font": {"size": 15, "color": "white"}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": rate_color}, "bgcolor": "#161b22",
                           "steps": [{"range": [0, 45], "color": "#2d1b1b"},
                                     {"range": [45, 65], "color": "#2d2a1b"},
                                     {"range": [65, 100], "color": "#1b2d1b"}],
                           "threshold": {"line": {"color": "white", "width": 3}, "value": thr_val}}))
                fig_hr.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}, height=280)
                st.plotly_chart(fig_hr, use_container_width=True, key=f"hr_gauge_{side}")
                if base is not None:
                    st.caption(f"Weiße Linie = Zufallsquote dieses Tickers ({base*100:.0f}%): so oft erreicht "
                               f"JEDER beliebige Tag {FORWARD_DAYS} Tage später dieselbe Bewegung.")

            with hc2:
                hm1, hm2, hm3, hm4 = st.columns(4)
                hm1.metric("Signale", hr["total"])
                hm2.metric(lbl_ok, hr["hits"], delta=f"+{hr['hits']}", delta_color="normal")
                hm3.metric(lbl_bad, hr["misses"], delta=f"-{hr['misses']}", delta_color="inverse")
                if base is not None:
                    hm4.metric("Zufallsquote", f"{base*100:.0f}%",
                               delta=f"{edge*100:+.0f} Pkt Edge" if edge is not None else None,
                               delta_color="normal")

                if hr["details"]:
                    row_colors = {"strong": "#1a4d2e", "weak": "#4d3d00", "fail": "#4d1a1a",
                                  "sideways": "#2d2d2d", "new": "#1a2d4d"}
                    rows_html = ""
                    for d in hr["details"]:
                        rc = d.get("result_code", "sideways")
                        bg = row_colors.get(rc, "#222")
                        mg, ml = d.get("max_gain_pct"), d.get("max_loss_pct")
                        if is_buy:
                            mg_str = f"+{mg:.1f}%" if mg is not None else "–"
                            ml_str = f"{ml:.1f}%" if ml is not None else "–"
                        else:   # Verkauf: günstig = Kurs fällt, ungünstig = Kurs steigt
                            mg_str = f"−{mg:.1f}%" if mg is not None else "–"
                            ml_str = f"+{abs(ml):.1f}%" if ml is not None else "–"
                        date_str = d["entry_date"].strftime("%d.%m.%Y") if hasattr(d["entry_date"], "strftime") else str(d["entry_date"])
                        txt_col = "4ade80" if "strong" in rc else "fbbf24" if "weak" in rc else "f87171" if "fail" in rc else "94a3b8"
                        rows_html += (
                            f"<tr style='background:{bg}'>"
                            f"<td style='padding:7px 10px;color:#ccc'>{date_str}</td>"
                            f"<td style='padding:7px 10px;color:#ccc;text-align:right'>{d['entry_price']:.2f} {cur}</td>"
                            f"<td style='padding:7px 10px;font-weight:700;color:#{txt_col}'>{d['result']}</td>"
                            f"<td style='padding:7px 10px;color:#4ade80;text-align:right'>{mg_str}</td>"
                            f"<td style='padding:7px 10px;color:#f87171;text-align:right'>{ml_str}</td></tr>")
                    st.markdown(
                        f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;font-size:12px;font-family:sans-serif'>"
                        f"<thead><tr style='background:#1e2235;color:#8892a4;font-size:10px;text-transform:uppercase'>"
                        f"<th style='padding:8px 10px'>{'Einstieg' if is_buy else 'Verkauf'}</th>"
                        f"<th style='padding:8px 10px;text-align:right'>Preis</th>"
                        f"<th style='padding:8px 10px'>Ergebnis</th>"
                        f"<th style='padding:8px 10px;text-align:right'>{col_fav}</th>"
                        f"<th style='padding:8px 10px;text-align:right'>{col_adv}</th>"
                        f"</tr></thead><tbody>{rows_html}</tbody></table></div>", unsafe_allow_html=True)

        with st.expander(f"📋 {len(entry_sigs)} historische " + ("Einstiegs" if is_buy else "Verkaufs") + "signale im Detail", expanded=False):
            if is_buy:
                st.caption("Trigger 'MACD' = Histogramm hat nach oben gedreht (eigentliches Kaufsignal). "
                           "Trigger 'Preis' = MACD hat im Fenster nicht gedreht, der Kurs liegt aber bereits +2% über dem laufenden Tief.")
            else:
                st.caption("Trigger 'MACD' = Histogramm hat nach unten gedreht (eigentliches Verkaufssignal). "
                           "Trigger 'Preis' = MACD hat im Fenster nicht gedreht, der Kurs liegt aber bereits 2% unter dem laufenden Hoch.")
            rows = []
            for e in entry_sigs[-12:]:
                f = lambda x: x.strftime("%d.%m.%Y") if hasattr(x, "strftime") else str(x)
                rows.append({"Signal-Datum": f(e["signal_date"]), "Signal-Preis": f"{e['signal_price']:.2f} {cur}",
                             "Trigger": e.get("trigger", "–"),
                             ("Einstieg-Datum" if is_buy else "Verkauf-Datum"): f(e["entry_date"]),
                             ("Einstieg-Preis" if is_buy else "Verkauf-Preis"): f"{e['entry_price']:.2f} {cur}",
                             "Vorlauf": f"{e['days_to_entry']} Tage"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            ei1, ei2, ei3 = st.columns(3)
            ei1.metric("Ø Vorlauf Signal→Ausführung", f"{round(np.mean([e['days_to_entry'] for e in entry_sigs]), 1)} Tage")
            ei2.metric("Ø Bestätigungs-Anstieg" if is_buy else "Ø Bestätigungs-Rückgang",
                       f"{'+' if is_buy else '−'}{abs(round(np.mean([e['upside_pct'] for e in entry_sigs]), 1))}%")
            ei3.metric("Signale gesamt", len(entry_sigs))

    if sig_history:
        with st.expander(f"📅 {len(sig_history)} historische Rohtreffer ({'Kauf' if is_buy else 'Verkauf'}, Zeitraum: {sel_tf})", expanded=False):
            st.dataframe(pd.DataFrame([{
                "Datum": s["date"].strftime("%d.%m.%Y") if hasattr(s["date"], "strftime") else str(s["date"]),
                "Preis": f"{s['price']:.2f} {cur}", "StochRSI": s["StochRSI"], "CCI": s["CCI"],
                "MACD": s.get("MACD_Turn") or "–", "Score": f"{s['score']}/4"} for s in sig_history]),
                hide_index=True, use_container_width=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Scanner", "📊 Detail-Analyse", "📐 Formations-Check",
    "🛢️ Rohstoffe & Futures", "📅 Automation"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Globaler Markt-Scanner")
    c1, c2 = st.columns([4, 1])
    sel_grp   = c1.selectbox("📂 Markt / Index / Sektor:", ["🌍 Alle Gruppen"] + get_all_groups(), key="scan_grp")
    min_score = c2.selectbox("Min. Score:", [1, 2, 3, 4])

    if st.button("🔥 SCAN STARTEN"):
        tickers = get_all_tickers() if sel_grp == "🌍 Alle Gruppen" else get_tickers_by_group(sel_grp)
        prog, stat = st.progress(0), st.empty()

        def _scan_progress(done, total, phase):
            if phase == "download":
                stat.text(f"Lade Kursdaten (Batch): {done}/{total} Ticker")
                prog.progress(min(done / max(total, 1), 1.0) * 0.9)
            elif phase == "info":
                stat.text(f"Lade Zusatzinfos (Name/Div/KGV) für Signal-Ticker: {done}/{total}")
                prog.progress(0.9 + min(done / max(total, 1), 1.0) * 0.1)

        # Batch-Scan statt einzelner Abrufe pro Ticker (sonst hängt Streamlit bei großen Listen) —
        # siehe engine.scan_tickers(), bereits von automation_script.py genutzt.
        results, failed, stale, meta = scan_tickers(tickers, compute_history=True, progress_cb=_scan_progress)
        prog.progress(1.0)
        if results:
            # Relative Stärke: eigene 20-Tage-Performance vs. Durchschnitt ALLER Peer-Gruppen des Tickers
            # (gleiche Logik wie in automation_script.py, nicht nur die erste Gruppe im Dict).
            results = add_relative_strength(results)
        df = pd.DataFrame(results)
        if not df.empty:
            df = df[df["Score"] >= min_score].sort_values(
                ["Score","HitRate"], ascending=[False,False], na_position="last").reset_index(drop=True)
            st.session_state.results = df
            msg = f"✅ {len(df)} Signale in {len(tickers)} Titeln."
            if failed:
                msg += f" ({len(failed)} ohne Daten, {len(stale)} veraltet)"
            stat.success(msg)
        else:
            stat.warning("Keine Signale.")

    if not st.session_state.results.empty:
        df_show = st.session_state.results.copy()
        st.markdown("##### Ergebnisse — 🟢 OVERSOLD · 🔴 OVERBOUGHT · ⚪ NEUTRAL")

        def _trend_html(t):
            if t == "up":   return "<span style='color:#1a9e3f;font-weight:700'>📈</span>"
            if t == "down": return "<span style='color:#c0392b;font-weight:700'>📉</span>"
            return "<span style='color:#888'>–</span>"

        def _hitrate_html(r):
            hr, n = r.get('HitRate'), r.get('HitRate_N', 0)
            if hr is None or (isinstance(hr, float) and pd.isna(hr)):
                return f"<span style='color:#888;font-size:11px'>zu wenig ({int(n) if pd.notna(n) else 0})</span>"
            color = "#0d7a2e" if hr >= 70 else ("#f39c12" if hr >= 50 else "#c0392b")
            return f"<span style='color:{color};font-weight:700'>{int(hr)}%</span> <span style='color:#888;font-size:10px'>({int(n)})</span>"

        def _relstrength_html(r):
            rs = r.get('RelStrength')
            if rs is None or (isinstance(rs, float) and pd.isna(rs)):
                return "<span style='color:#888'>–</span>"
            color = "#0d7a2e" if rs >= 0 else "#c0392b"
            return f"<span style='color:{color};font-weight:700'>{rs:+.1f}%</span>"

        def _row_html(r):
            sv, sf, ss, cv = r.get('StochRSI'), r.get('Stoch_Fast'), r.get('Stoch_Slow'), r.get('CCI')
            macd_turn = r.get('MACD_Turn')
            sig  = classify_signal(sv, sf, ss, cv, macd_turn)
            sc   = int(r.get('Score', 0))
            sc_c = {4:"#0d7a2e",3:"#1a9e3f",2:"#f39c12",1:"#7f8c8d"}.get(sc,"#999")
            return (f"<tr>"
                    f"<td><b>{r['Ticker']}</b></td><td>{r.get('Name','')}</td>"
                    f"<td>{_sig_badge(sig)}</td>"
                    f"<td style='text-align:right'><b>{r.get('Preis','')}</b></td>"
                    f"<td style='text-align:center'>{_ind_badge(sv,0.15,0.85)}</td>"
                    f"<td style='text-align:center'>{_ind_badge(sf,20,80,'{:.1f}')}</td>"
                    f"<td style='text-align:center'>{_ind_badge(ss,25,75,'{:.1f}')}</td>"
                    f"<td style='text-align:center'>{_ind_badge(cv,-100,100,'{:.1f}',invert=True)}</td>"
                    f"<td style='text-align:center;font-size:11px'>{_macd_badge(macd_turn)}</td>"
                    f"<td style='text-align:center'>{_trend_html(r.get('Trend'))}</td>"
                    f"<td style='text-align:center'>{_hitrate_html(r)}</td>"
                    f"<td style='text-align:center'>{_relstrength_html(r)}</td>"
                    f"<td style='text-align:center'><span style='background:{sc_c};color:#fff;"
                    f"padding:2px 8px;border-radius:10px;font-weight:700'>{sc}/4</span></td>"
                    f"<td>{r.get('Div','')}</td><td>{r.get('KGV','')}</td>"
                    f"</tr>")

        thead = """<thead style='background:#1e2235;color:#8892a4;font-size:11px;text-transform:uppercase;letter-spacing:.5px'>
          <tr><th style='padding:10px 8px'>Ticker</th><th>Name</th><th>Signal</th><th>Preis</th>
          <th>StochRSI</th><th>Stoch F</th><th>Stoch S</th><th>CCI</th><th>MACD</th>
          <th>Trend</th><th>Trefferquote</th><th>Rel. Stärke</th><th>Score</th><th>Div</th><th>KGV</th></tr></thead>"""
        tbody = "".join(_row_html(r) for _, r in df_show.iterrows())
        st.markdown(
            f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
            f"font-size:13px;font-family:sans-serif'>{thead}<tbody>{tbody}</tbody></table></div>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DETAIL-ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Analyse & Chart-Center")

    all_t = get_all_tickers()

    # ── Suchfeld mit Name-Suche ─────────────────────────────────────────────
    # Deep-Link aus Email: ?ticker=ALV.DE
    # WICHTIG: hier den reinen Ticker vorbefüllen, nicht das kombinierte
    # "TICKER — Name"-Label (get_display_name) - search_ticker() erwartet
    # entweder nur Ticker oder nur Name als Suchbegriff, nicht beides
    # zusammen mit " — " verbunden, sonst matcht die Suche nichts.
    if deep_link_ticker and deep_link_ticker in all_t:
        if "search_query" not in st.session_state:
            st.session_state["search_query"] = deep_link_ticker

    search_q = st.text_input(
        "🔍 Aktie suchen (Name, Ticker, Alias):",
        value=st.session_state.get("search_query", ""),
        placeholder="z.B. Allianz, Samsung, 0700.HK, Tencent, NVDA …",
        key="search_input",
    )
    st.session_state["search_query"] = search_q

    # Suchergebnisse filtern
    if search_q.strip():
        suggestions = search_ticker(search_q, all_t, max_results=12)
    else:
        suggestions = [get_display_name(t) for t in all_t[:20]]

    if not suggestions:
        st.warning(f"Keine Treffer für **{search_q}** — bitte Ticker direkt eingeben (z.B. ALV.DE)")
        suggestions = [get_display_name(t) for t in all_t[:5]]

    # Deep-Link: falls der verlinkte Ticker unter den Vorschlägen ist, aber
    # nicht an erster Stelle steht, trotzdem automatisch vorauswählen.
    default_idx = 0
    if deep_link_ticker:
        deep_label = get_display_name(deep_link_ticker)
        if deep_label in suggestions:
            default_idx = suggestions.index(deep_label)

    sel_label = st.selectbox("Ergebnisse:", suggestions, index=default_idx, key="det_t_label")
    sel_t = label_to_ticker(sel_label)

    # Zeitraum-Auswahl — Standard 6 Monate
    tf_opts  = {"3 Monate": 63, "6 Monate": 126, "1 Jahr": 252, "2 Jahre": 504}
    sel_tf   = st.radio("Zeitraum:", list(tf_opts.keys()), index=1, horizontal=True, key="det_tf")
    tf_bars  = tf_opts[sel_tf]

    if sel_t:
        with st.spinner(f"Lade {sel_t} …"):
            data = get_detail_analysis(sel_t)

        if data and "df" in data:
            cur = data["Währung"]
            sig = data["Sig_Data"]
            sc_col = "green" if data["Score"]>=2 else ("orange" if data["Score"]==1 else "red")

            # ── Kompakter Header: Name, Signal, alle Kern-Kennzahlen in einer Zeile ──
            col_h1, col_h2 = st.columns([3,1])
            col_h1.markdown(f"### {data['Name']} ({sel_t})")
            col_h2.markdown(
                f"{_sig_badge(sig)} &nbsp; :{sc_col}[**Score {data['Score']}/4**]",
                unsafe_allow_html=True)

            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Kurs", f"{_fnum(data['Preis'])} {cur}",
                      delta=f"{_fnum(data['Perf_Abs'])} ({_fnum(data['Perf_Pct'])}%)")
            c2.metric("52W Hoch", f"{_fnum(data['Hoch_365'])} {cur}")
            c3.metric("52W Tief", f"{_fnum(data['Tief_365'])} {cur}")
            c4.metric("Dividende", data["Div"])
            c5.metric("KGV", str(data["KGV"]))
            c6.metric("RVOL", f"{_fnum(data['RVOL'])}x")

            # ── Zuordnung, Fokus und Unternehmensprofil ──────────────
            _grp_labels = [get_group_label(g) for g in get_groups_for_ticker(sel_t)]
            _focus      = get_focus(sel_t)
            _sector     = " · ".join(x for x in (data.get("Sektor"), data.get("Industrie")) if x) or get_sector(sel_t)
            _meta_line  = []
            if _focus:      _meta_line.append(f"🎯 **Fokus:** {_focus}")
            if _sector:     _meta_line.append(f"🏭 {_sector}")
            if _grp_labels: _meta_line.append("📂 " + " · ".join(_grp_labels[:6]) + (" …" if len(_grp_labels) > 6 else ""))
            if _meta_line:
                st.caption("  \n".join(_meta_line))

            with st.expander("🧭 Unternehmen verstehen — Profil & KI-Erläuterung", expanded=False):
                _name = data.get("Name") or sel_t
                if data.get("Beschreibung"):
                    st.markdown(data["Beschreibung"])
                    _facts = [f"**Land:** {data['Land']}" if data.get("Land") else "",
                              f"**Mitarbeiter:** {data['Mitarbeiter']:,}".replace(",", ".") if isinstance(data.get("Mitarbeiter"), int) else "",
                              f"[Webseite]({data['Website']})" if data.get("Website") else ""]
                    st.caption("  ·  ".join(x for x in _facts if x) + "  —  Quelle: Yahoo Finance (englisch)")
                else:
                    st.info("Yahoo liefert für diesen Ticker keine Unternehmensbeschreibung.")
                _prompt = (f"Erkläre mir das Unternehmen {_name} (Ticker {sel_t}, Börse {get_exchange(sel_t)}): "
                           f"Geschäftsmodell, wichtigste Umsatzquellen, aktuelle Nachrichten der letzten Wochen, "
                           f"Bewertung sowie Chancen und Risiken. Wie ist der Stand der Aktie aktuell?")
                _ai1, _ai2, _ai3 = st.columns(3)
                _ai1.link_button("✨ Google KI (AI Mode)", f"https://www.google.com/search?udm=50&q={quote_plus(_prompt)}", use_container_width=True)
                _ai2.link_button("📰 Google News", f"https://news.google.com/search?q={quote_plus(_name + ' Aktie')}&hl=de&gl=DE&ceid=DE:de", use_container_width=True)
                _ai3.link_button("📈 Yahoo Finance", f"https://finance.yahoo.com/quote/{quote_plus(sel_t)}", use_container_width=True)
                st.caption("Die KI-Antwort entsteht bei Google in einem neuen Tab (kostenlos, ohne API) und kann Fehler enthalten. "
                           "Keine Anlageberatung.")

            # ── Indikator-Status: alle 4 Strategie-Kriterien kompakt in einer Zeile ──
            st.divider()
            st.markdown("#### Indikator-Status — die 4 Kriterien deiner Strategie")
            ib1,ib2,ib3,ib4 = st.columns(4)
            ib1.markdown(f"**StochRSI(70)** &nbsp; {_ind_badge(data['StochRSI'],0.15,0.85)}", unsafe_allow_html=True)
            ib2.markdown(
                f"**Stoch F/S** &nbsp; {_ind_badge(data.get('Stoch_Fast'),20,80,'{:.1f}')} / "
                f"{_ind_badge(data.get('Stoch_Slow'),25,75,'{:.1f}')}",
                unsafe_allow_html=True)
            ib3.markdown(f"**CCI(20)** &nbsp; {_ind_badge(data['CCI'],-100,100,'{:.1f}',invert=True)}", unsafe_allow_html=True)
            ib4.markdown(f"**MACD(70,200,9)** &nbsp; {_macd_badge(data.get('MACD_Turn'))}", unsafe_allow_html=True)
            trend_txt = {"up":"📈 Aufwärtstrend (Kurs > SMA200)","down":"📉 Abwärtstrend (Kurs < SMA200) — Vorsicht bei Käufen"}.get(data.get('Trend'), "Trend n/a")
            trend_col = {"up":"#0d7a2e","down":"#c0392b"}.get(data.get('Trend'), "#888")
            st.markdown(
                f"<span style='color:{trend_col};font-weight:700;font-size:12px'>{trend_txt}</span> &nbsp;·&nbsp; "
                f"<span style='color:#888;font-size:12px'>Z-Score: {_ind_badge(data['Z_Score'],-1.5,1.5,'{:.2f}')}</span>",
                unsafe_allow_html=True)

            # ── HAUPT-CHART (6 Panels) ─────────────────────────────
            df_full = data["df"]
            df_p    = df_full.tail(tf_bars)
            targets = data.get("Targets", {})
            sig_history = data.get("Signal_History", [])
            entry_sigs  = data.get("Entry_Signals", [])
            sig_history_sell = data.get("Signal_History_Sell", [])
            entry_sigs_sell  = data.get("Entry_Signals_Sell", [])

            fig = make_subplots(
                rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                row_heights=[0.30, 0.10, 0.115, 0.115, 0.115, 0.115, 0.14],
                subplot_titles=("Kurs + Bollinger + Marker", "Volume of Interest",
                                "Stochastik (Fast/Slow)", "StochRSI (70)",
                                "CCI (20)", "Z-Score", "MACD (70,200,9) — Trigger"),
            )

            # Panel 1: Kurs + Bollinger
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p["BB_Lower"], name="BB–",
                line=dict(color="#00eeee",width=.8,dash="dot"), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p["BB_Upper"], name="Bollinger Band",
                fill="tonexty", fillcolor="rgba(0,238,238,0.05)",
                line=dict(color="#00eeee",width=.8,dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p["SMA200"], name="EMA200",
                line=dict(color="#666",width=1.5,dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p["Close"], name="Kurs",
                line=dict(color="#facc15",width=2.5)), row=1, col=1)

            # ── Historische Signal-Marker (blaue Punkte) ──────────
            sig_in_range = [s for s in sig_history if s["date"] in df_p.index or
                            (len(df_p)>0 and s["date"] >= df_p.index[0])]
            if sig_in_range:
                fig.add_trace(go.Scatter(
                    x=[s["date"] for s in sig_in_range],
                    y=[s["price"] for s in sig_in_range],
                    mode="markers",
                    name="Signal-Treffer Kauf",
                    marker=dict(color="#3b82f6", size=10, symbol="circle",
                                line=dict(color="#ffffff", width=2)),
                    hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Preis: %{y:.2f}<br>Signal-Cluster<extra></extra>",
                ), row=1, col=1)

            # ── Einstiegssignale (grüne Dreiecke) ─────────────────
            entry_in_range = [e for e in entry_sigs if
                              len(df_p)>0 and e["entry_date"] >= df_p.index[0]]
            if entry_in_range:
                fig.add_trace(go.Scatter(
                    x=[e["entry_date"] for e in entry_in_range],
                    y=[e["entry_price"] for e in entry_in_range],
                    mode="markers", name="Einstieg (Kauf)",
                    customdata=[[e["days_to_entry"], e.get("trigger", "–")] for e in entry_in_range],
                    marker=dict(color="#4ade80", size=12, symbol="triangle-up",
                                line=dict(color="#ffffff", width=1.5)),
                    hovertemplate=("<b>Einstieg %{x|%d.%m.%Y}</b><br>Preis: %{y:.2f}<br>"
                                   "%{customdata[0]} Tage nach Signal · Trigger %{customdata[1]}<extra></extra>"),
                ), row=1, col=1)

            # ── Verkaufssignale: orange Punkte = Rohtreffer, rote Dreiecke = Verkauf ──
            sell_sig_in_range = [s_ for s_ in sig_history_sell if len(df_p)>0 and s_["date"] >= df_p.index[0]]
            if sell_sig_in_range:
                fig.add_trace(go.Scatter(
                    x=[s_["date"] for s_ in sell_sig_in_range],
                    y=[s_["price"] for s_ in sell_sig_in_range],
                    mode="markers", name="Signal-Treffer Verkauf",
                    marker=dict(color="#f97316", size=10, symbol="circle",
                                line=dict(color="#ffffff", width=2)),
                    hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Preis: %{y:.2f}<br>Verkaufs-Cluster<extra></extra>",
                ), row=1, col=1)
            sell_in_range = [e for e in entry_sigs_sell if
                             len(df_p)>0 and e["entry_date"] >= df_p.index[0]]
            if sell_in_range:
                fig.add_trace(go.Scatter(
                    x=[e["entry_date"] for e in sell_in_range],
                    y=[e["entry_price"] for e in sell_in_range],
                    mode="markers", name="Verkauf",
                    customdata=[[e["days_to_entry"], e.get("trigger", "–")] for e in sell_in_range],
                    marker=dict(color="#ef4444", size=12, symbol="triangle-down",
                                line=dict(color="#ffffff", width=1.5)),
                    hovertemplate=("<b>Verkauf %{x|%d.%m.%Y}</b><br>Preis: %{y:.2f}<br>"
                                   "%{customdata[0]} Tage nach Signal · Trigger %{customdata[1]}<extra></extra>"),
                ), row=1, col=1)

            # ── Fibonacci & Zielkurs-Linien ────────────────────────
            fib_colors = {"Fib_23.6%":"#fbbf24","Fib_38.2%":"#f97316",
                          "Fib_50.0%":"#ef4444","Fib_61.8%":"#ec4899","Fib_78.6%":"#a855f7"}
            for k,v in targets.items():
                if k.startswith("Fib_"):
                    fc = fib_colors.get(k,"#aaa")
                    fig.add_hline(y=v, line_dash="dot", line_color=fc, line_width=1,
                                  annotation_text=f" {k} {v:.2f} ",
                                  annotation_font=dict(color=fc,size=10),
                                  annotation_position="right", row=1, col=1)
                elif "Nackenlinie" in k or "Ziel" in k or "Einstieg" in k:
                    is_buy   = "Doppelboden" in k or "Inv_SKS" in k or "Einstieg" in k or "V_Boden" in k
                    lc       = "#4ade80" if is_buy else "#f87171"
                    fig.add_hline(y=v, line_dash="dash", line_color=lc, line_width=1.5,
                                  annotation_text=f" {k.replace('_',' ')} {v:.2f} ",
                                  annotation_font=dict(color=lc,size=10),
                                  annotation_position="right", row=1, col=1)

            # Panel 2: Volume of Interest (horizontal Balken am Kurs)
            # Wir berechnen ein Preis-Volumen-Histogramm und zeigen es als Scatter
            close_vals = df_p["Close"].values
            vol_vals   = df_p["Volume"].values
            if len(close_vals) > 5:
                n_bins  = 40
                lo, hi  = close_vals.min(), close_vals.max()
                if hi > lo:
                    bins    = np.linspace(lo, hi, n_bins + 1)
                    bw      = bins[1] - bins[0]
                    centers = (bins[:-1] + bins[1:]) / 2
                    vol_sum = np.zeros(n_bins)
                    for price_v, vol_v in zip(close_vals, vol_vals):
                        idx = min(int((price_v - lo) / bw), n_bins - 1)
                        vol_sum[idx] += vol_v
                    # POC = Price of Control (höchstes Volumen)
                    poc_idx   = int(np.argmax(vol_sum))
                    poc_price = centers[poc_idx]

                    # Volumen normalisieren auf 0–100 für Darstellung
                    vol_norm = vol_sum / vol_sum.max() * 100 if vol_sum.max() > 0 else vol_sum

                    # Im Volume-Panel als Candlestick-artige Balken
                    bar_cols = ["#1a9e3f" if c == poc_price else
                                ("#4ade80" if v > np.percentile(vol_sum, 75) else "#334155")
                                for c, v in zip(centers, vol_sum)]

                    fig.add_trace(go.Bar(
                        x=vol_norm, y=centers,
                        orientation="h",
                        name="Volume of Interest",
                        marker_color=bar_cols,
                        opacity=0.85,
                        hovertemplate="Preis: %{y:.2f}<br>Vol-Index: %{x:.1f}<extra></extra>",
                    ), row=2, col=1)

                    # POC Linie im Kurs-Panel
                    fig.add_hline(y=poc_price, line_color="#22c55e", line_width=2, line_dash="solid",
                                  annotation_text=f" 🎯 POC {poc_price:.2f} ",
                                  annotation_font=dict(color="#22c55e",size=11,family="Arial Black"),
                                  annotation_bgcolor="rgba(34,197,94,0.15)",
                                  annotation_position="left", row=1, col=1)

            # Panel 3: Stochastik
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p["Stoch_Fast"], name="Stoch Fast",
                line=dict(color="#00ffff",width=1.2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p["Stoch_Slow"], name="Stoch Slow",
                line=dict(color="#ff00ff",width=2)), row=3, col=1)
            for y,c in [(10,"#4ade80"),(90,"#f87171")]:
                fig.add_hline(y=y, line_color=c, line_width=1, line_dash="dot", row=3, col=1)

            # Panel 4: StochRSI
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p["StochRSI_70"], name="StochRSI",
                line=dict(color="#fff",width=1.5)), row=4, col=1)
            # Oversold-Bereich grün füllen
            fig.add_trace(go.Scatter(
                x=df_p.index, y=np.where(df_p["StochRSI_70"]<0.1, df_p["StochRSI_70"], 0.1),
                fill="tozeroy", fillcolor="rgba(74,222,128,0.15)",
                line=dict(width=0), showlegend=False, name="OS-Zone"), row=4, col=1)
            for y,c in [(0.1,"#4ade80"),(0.9,"#f87171")]:
                fig.add_hline(y=y, line_color=c, line_dash="dot", row=4, col=1)

            # Panel 5: CCI
            cci_vals = df_p["CCI_20"].fillna(0)
            cci_colors = ["#f87171" if v > 100 else "#4ade80" if v < -100 else "#94a3b8"
                          for v in cci_vals]
            fig.add_trace(go.Bar(x=df_p.index, y=cci_vals, name="CCI",
                marker_color=cci_colors, opacity=0.7), row=5, col=1)
            for y,c in [(-100,"#4ade80"),(100,"#f87171")]:
                fig.add_hline(y=y, line_color=c, line_width=1.5, row=5, col=1)

            # Panel 6: Z-Score
            z_vals = df_p["Z_Score"].fillna(0)
            z_colors = ["#f87171" if v > 2 else "#4ade80" if v < -2 else
                        "#fbbf24" if v < -1 else "#94a3b8" for v in z_vals]
            fig.add_trace(go.Bar(x=df_p.index, y=z_vals, name="Z-Score",
                marker_color=z_colors, opacity=0.75), row=6, col=1)
            for y,c in [(-2,"#4ade80"),(-1,"#86efac"),(0,"#475569"),(1,"#fca5a5"),(2,"#f87171")]:
                fig.add_hline(y=y, line_color=c, line_width=1, line_dash="dot", row=6, col=1)

            # Panel 7: MACD — Linie/Signal + Histogramm, plus Marker für erkannte Umkehrpunkte
            macd_vals   = df_p["MACD"]
            signal_vals = df_p["SIGNAL"]
            hist_vals   = df_p["OSMA"].fillna(0)
            hist_colors = ["#4ade80" if v >= 0 else "#f87171" for v in hist_vals]
            fig.add_trace(go.Bar(x=df_p.index, y=hist_vals, name="OSMA (Histogramm)",
                marker_color=hist_colors, opacity=0.6), row=7, col=1)
            fig.add_trace(go.Scatter(x=df_p.index, y=macd_vals, name="MACD",
                line=dict(color="#38bdf8", width=1.5)), row=7, col=1)
            fig.add_trace(go.Scatter(x=df_p.index, y=signal_vals, name="Signal",
                line=dict(color="#c084fc", width=1.5)), row=7, col=1)
            fig.add_hline(y=0, line_color="#475569", line_width=1, row=7, col=1)

            fig.update_layout(
                height=1300, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                margin=dict(l=10,r=140,t=40,b=10),
                hovermode="x unified", showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#1e2235", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#1e2235", zeroline=False)
            # Volume-of-Interest x-Achse ausblenden
            fig.update_xaxes(showticklabels=False, row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # ── Volumen-Zonen: wo wurde zu welchem Kurs gehandelt? ───────────────
            st.divider()
            st.markdown("#### 📊 Volumen-Zonen — wo wurde viel gehandelt?")
            vp_opts = {"6 Monate": 126, "1 Jahr": 252, "2 Jahre": 504}
            vp_sel = st.radio("Betrachtungszeitraum:", list(vp_opts.keys()), index=1, horizontal=True, key="vp_lookback")
            prof = compute_profile(df_full, lookback=vp_opts[vp_sel])
            if prof:
                st.plotly_chart(build_profile_figure(df_full, prof, cur), use_container_width=True, key="vp_chart")
                for _line in describe_profile(prof, float(data["Preis"]), cur):
                    st.markdown(f"- {_line}")
                st.caption("Blau = Wertbereich (70 % des Umsatzes im Zeitraum), grün = POC, gestrichelte graue Linien = Volumen-Zonen (HVN). "
                           "Das Volumen jedes Tages wird gleichmäßig über seine Tagesspanne verteilt. Ob Kurse an solchen Zonen häufiger halten, "
                           "ist für diese Strategie nicht backgetestet — Orientierung, kein Signal. SMA/EMA/Bollinger über die Legende ein- und ausblendbar.")
            else:
                st.info("Zu wenig Kurs-/Volumendaten für ein Volumenprofil (mind. 30 Handelstage mit Umsatz).")

            # ── Optionsmarkt: Put/Call-Ratio ──────────────────────────────────────
            st.divider()
            st.markdown("#### 🧮 Optionsmarkt — Put/Call-Ratio")
            with st.spinner("Lade Optionskette …"):
                opt = _cached_options(sel_t, float(data["Preis"]))
            if opt.get("available"):
                o1, o2, o3, o4, o5 = st.columns(5)
                o1.metric("Put/Call (OI)", "–" if opt["pc_oi"] is None else f"{opt['pc_oi']:.2f}",
                          help="Offene Puts geteilt durch offene Calls (nahe Verfälle). >1 = mehr Puts, <1 = mehr Calls.")
                o2.metric("Put/Call (Volumen)", "–" if opt["pc_vol"] is None else f"{opt['pc_vol']:.2f}",
                          help="Heutiges Handelsvolumen. Außerhalb der Börsenzeit oft leer.")
                o3.metric("ATM-Vola (nah)", "–" if not opt["atm_iv_near"] else f"{opt['atm_iv_near']*100:.0f} %")
                o4.metric("Erwartete Spanne", "–" if not opt["exp_move_pct"] else f"±{opt['exp_move_pct']:.1f} %",
                          help=f"Aus der impliziten Volatilität bis zum nächsten Verfall ({opt['dte_near']} Tage).")
                o5.metric("Max Pain", "–" if not opt["max_pain"] else f"{opt['max_pain']['strike']:.2f}",
                          help="Strike, bei dem Optionskäufer des nächsten Verfalls insgesamt am wenigsten erhalten.")
                st.dataframe(pd.DataFrame([{
                    "Verfall": r["expiry"], "Tage": r["dte"],
                    "Put/Call OI": None if r["pc_oi"] is None else round(r["pc_oi"], 2),
                    "Put/Call Vol.": None if r["pc_vol"] is None else round(r["pc_vol"], 2),
                    "OI Calls": int(r["calls_oi"]), "OI Puts": int(r["puts_oi"]),
                    "ATM-Vola": None if not r["atm_iv"] else f"{r['atm_iv']*100:.0f} %",
                    "Max Pain": r["max_pain"],
                } for r in opt["expiries"]]), hide_index=True, use_container_width=True)
            for _line in describe_options(opt):
                st.markdown(f"- {_line}")

            # ── Trefferquoten-Analyse — Kauf (Boden) und Verkauf (Top) getrennt ──
            st.divider()
            t_buy, t_sell = st.tabs([f"🟢 Kauf · {len(entry_sigs)} Signale",
                                     f"🔴 Verkauf · {len(entry_sigs_sell)} Signale"])
            with t_buy:
                render_side_analysis("buy", entry_sigs, sig_history, df_full, cur, sel_tf)
            with t_sell:
                render_side_analysis("sell", entry_sigs_sell, sig_history_sell, df_full, cur, sel_tf)

            # ── Muster & Targets ──────────────────────────────────
            if data.get("Patterns"):
                st.markdown("#### 📐 Erkannte Muster")
                for p in data["Patterns"]: st.info(f"📌 {p}")

            if data.get("Targets"):
                st.markdown("#### 🎯 Zielkurse & Fibonacci")
                price_now = data["Preis"]
                fib_keys  = [k for k in targets if k.startswith("Fib_")]
                other_keys= [k for k in targets if not k.startswith("Fib_")]

                if fib_keys:
                    st.markdown("**Fibonacci Retracement (52W Swing)**")
                    cols_f = st.columns(len(fib_keys))
                    for j,(k) in enumerate(fib_keys):
                        v   = targets[k]
                        pct = round((v-price_now)/price_now*100,1) if price_now else 0
                        cols_f[j].metric(k.replace("_"," "), f"{v:.2f} {cur}", f"{pct:+.1f}%")

                if other_keys:
                    st.markdown("**Formation-Ziele**")
                    cols_o = [other_keys[i:i+3] for i in range(0,len(other_keys),3)]
                    for row_k in cols_o:
                        cols_r = st.columns(3)
                        for j,k in enumerate(row_k):
                            v   = targets[k]
                            pct = round((v-price_now)/price_now*100,1) if price_now else 0
                            arrow = "🎯" if pct > 0 else "⚠️"
                            cols_r[j].metric(f"{arrow} {k.replace('_',' ')}", f"{v:.2f} {cur}", f"{pct:+.1f}%")

        else:
            st.error(f"❌ Keine Daten für **{sel_t}**. Bitte Symbol prüfen oder anderen Zeitraum wählen.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORMATIONS-CHECK
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📐 Formations-Check: Boden & Topbildung")
    mode = st.radio("Modus:", ["🔍 Markt scannen", "🎯 Einzelne Aktie"], horizontal=True)

    if mode == "🔍 Markt scannen":
        grp_opts  = ["🌍 Alle Gruppen"] + get_all_groups()
        grp_t3    = st.selectbox("Gruppe:", grp_opts, key="form_grp")
        scan_mode = st.radio("Suche nach:",
            ["🟢 Bodenbildung (Kaufchancen)", "🔴 Topbildung (Verkaufchancen)", "🔍 Alle Signale"],
            horizontal=True)

        if st.button("🚀 Formations-Scan starten"):
            tickers_t3 = get_all_tickers() if grp_t3=="🌍 Alle Gruppen" else get_tickers_by_group(grp_t3)
            prog, txt = st.progress(0), st.empty()

            def _form_progress(done, total, phase):
                txt.text(f"Lade Kursdaten (Batch): {done}/{total} Ticker")
                prog.progress(min(done / max(total, 1), 1.0))

            # Batch-Download statt get_detail_analysis(t) pro Ticker (sonst hängt der Scan bei
            # "Alle Gruppen" in Streamlit) — siehe detail_engine.scan_detail_batch().
            with st.spinner("Kursdaten werden geladen und ausgewertet …"):
                scanned = scan_detail_batch(tickers_t3, progress_cb=_form_progress)
            result_list = []

            for t in tickers_t3:
                res = scanned.get(t)
                if res:
                    patterns = res.get("Patterns", [])
                    targets  = res.get("Targets",  {})
                    entries  = res.get("Entry_Signals", [])

                    b_score = t_score = 0
                    zsc, rvol = res.get("Z_Score"), res.get("RVOL")
                    if zsc is not None:
                        if zsc < -1.5: b_score += 15
                        if zsc < -2.0: b_score += 15
                        if zsc > 1.5:  t_score += 15
                        if zsc > 2.0:  t_score += 15
                    if rvol is not None:
                        if rvol > 1.3: b_score += 10
                        if rvol > 2.0: b_score += 15; t_score += 15
                    for p in patterns:
                        if "Selling Climax" in p:  b_score += 30
                        if "Doppelboden"    in p:  b_score += 20
                        if "Dreifachboden"  in p:  b_score += 25
                        if "Umgekehrte SKS" in p:  b_score += 25
                        if "V-Boden"        in p:  b_score += 15
                        if "Buying Climax" in p:   t_score += 30
                        if "Doppeltop"     in p:   t_score += 20
                        if "SKS-Formation" in p:   t_score += 25

                    b_score = min(b_score, 100)
                    t_score = min(t_score, 100)

                    show = (("Bodenbildung" in scan_mode and b_score >= 30) or
                            ("Topbildung"   in scan_mode and t_score >= 30) or
                            ("Alle"         in scan_mode and (b_score >= 30 or t_score >= 30)))

                    if show:
                        # Einstiegs-Info
                        avg_entry = ""
                        if entries:
                            avg_days = round(np.mean([e["days_to_entry"] for e in entries]), 0)
                            avg_entry = f"Ø {int(avg_days)}d Vorlauf"

                        # Bestes Ziel
                        entry_target = ""
                        if b_score >= t_score:
                            if "Doppelboden_Nackenlinie" in targets:
                                entry_target = f"↑ Neckline {targets['Doppelboden_Nackenlinie']} → Ziel {targets.get('Doppelboden_Ziel_100%','?')}"
                        else:
                            if "Doppeltop_Nackenlinie" in targets:
                                entry_target = f"↓ Neckline {targets['Doppeltop_Nackenlinie']} → Ziel {targets.get('Doppeltop_Ziel_100%','?')}"

                        result_list.append({
                            "Ticker":        t,
                            "Name":          res["Name"],
                            "Signal":        res["Signal"],
                            "Boden %":       b_score,
                            "Top %":         t_score,
                            "Muster":        " | ".join(patterns[:3]) if patterns else "–",
                            "Einstieg-Info": avg_entry or "–",
                            "Ziel":          entry_target or "–",
                            "Preis":         f"{res['Preis']} {res['Währung']}",
                            "Z-Score":       res["Z_Score"],
                            "RVOL":          res["RVOL"],
                        })

            prog.progress(1.0)
            txt.empty()
            if result_list:
                sort_col = "Boden %" if "Top" not in scan_mode else "Top %"
                df_f     = pd.DataFrame(result_list).sort_values(sort_col, ascending=False)
                st.subheader(f"🔥 {len(df_f)} Formationen erkannt")
                st.dataframe(df_f, hide_index=True, use_container_width=True)
            else:
                st.warning("Keine Formationen gefunden.")

    else:
        sq3 = st.text_input("🔍 Aktie suchen:", placeholder="z.B. Porsche, Ferrari, Toyota …", key="form_search")
        all_t3 = get_all_tickers()
        sugg3  = search_ticker(sq3, all_t3, 10) if sq3.strip() else [get_display_name(t) for t in all_t3[:10]]
        if not sugg3: sugg3 = [get_display_name(t) for t in all_t3[:5]]
        lbl3   = st.selectbox("Ergebnisse:", sugg3, key="form_single_lbl")
        sel_t3 = label_to_ticker(lbl3)
        if sel_t3:
            with st.spinner(f"Analysiere {sel_t3} …"):
                data = get_detail_analysis(sel_t3)

            if data and "df" in data:
                cur      = data["Währung"]
                patterns = data.get("Patterns", [])
                targets  = data.get("Targets", {})
                entries  = data.get("Entry_Signals", [])

                b_score = t_score = 0
                b_reasons = []; t_reasons = []
                zsc, rvol = data.get("Z_Score"), data.get("RVOL")
                if zsc is not None:
                    if zsc < -2.0: b_score += 30; b_reasons.append(f"📏 Z-Score {zsc:.2f}")
                    elif zsc < -1.5: b_score += 15
                if rvol is not None:
                    if rvol > 2.0:  b_score += 25; b_reasons.append(f"📊 RVOL {rvol:.2f}x")
                    elif rvol > 1.3:  b_score += 10
                for p in patterns:
                    if "Selling Climax" in p: b_score += 30; b_reasons.append("🚨 Selling Climax")
                    if "Doppelboden"    in p: b_score += 20; b_reasons.append("📐 Doppelboden")
                    if "Dreifachboden"  in p: b_score += 25; b_reasons.append("📐 Dreifachboden")
                    if "Umgekehrte SKS" in p: b_score += 25; b_reasons.append("📐 Inverse SKS")
                    if "V-Boden"        in p: b_score += 15; b_reasons.append("📐 V-Boden")
                if zsc is not None:
                    if zsc > 2.0:  t_score += 30; t_reasons.append(f"📏 Z-Score {zsc:.2f}")
                    elif zsc > 1.5: t_score += 15
                if rvol is not None and rvol > 2.0:
                    t_score += 15; t_reasons.append(f"📊 RVOL {rvol:.2f}x")
                for p in patterns:
                    if "Buying Climax" in p: t_score += 30; t_reasons.append("🚨 Buying Climax")
                    if "Doppeltop"     in p: t_score += 20; t_reasons.append("📐 Doppeltop")
                    if "SKS-Formation" in p: t_score += 25; t_reasons.append("📐 SKS")
                b_score = min(b_score, 100)
                t_score = min(t_score, 100)

                col_b, col_t = st.columns(2)
                for col, score, label, color, reasons, emoji in [
                    (col_b, b_score, "Bodenbildung", "#1a9e3f", b_reasons, "🟢"),
                    (col_t, t_score, "Topbildung",   "#c0392b", t_reasons, "🔴"),
                ]:
                    with col:
                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number", value=score,
                            number={"suffix":"%","font":{"size":56,"color":color,"family":"Arial Black"}},
                            title={"text":f"{emoji} {label}","font":{"size":17,"color":"white"}},
                            gauge={"axis":{"range":[0,100]},"bar":{"color":color},
                                   "bgcolor":"#161b22",
                                   "steps":[{"range":[0,40],"color":"#252840"},{"range":[40,70],"color":"#333650"}],
                                   "threshold":{"line":{"color":"#fff","width":3},"value":70}}))
                        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":"white"},height=320)
                        st.plotly_chart(fig_g, use_container_width=True)
                        for r in reasons: st.write(f"✅ {r}")
                        if   score >= 70: st.success(f"🔥 STARKES {label.upper()}-SIGNAL!")
                        elif score >= 40: st.warning(f"⚠️ {label} möglich")
                        else:             st.info(f"Kein klares Signal")

                # Einstiegszeitpunkte
                st.divider()
                if entries:
                    st.subheader(f"🎯 Einstiegssignal-Historie ({len(entries)} Treffer)")
                    avg_days = round(np.mean([e["days_to_entry"] for e in entries]), 1)
                    avg_up   = round(np.mean([e["upside_pct"]    for e in entries]), 1)
                    ea,eb,ec = st.columns(3)
                    ea.metric("Ø Vorlauf", f"{avg_days} Tage", help="Tage zwischen Signal und +2%-Bestätigung")
                    eb.metric("Ø Bestätigungs-%", f"+{avg_up}%")
                    ec.metric("Treffer gesamt", len(entries))

                    entry_df = pd.DataFrame([{
                        "Signal":    e["signal_date"].strftime("%d.%m.%Y") if hasattr(e["signal_date"],"strftime") else str(e["signal_date"]),
                        "Sig-Preis": f"{e['signal_price']:.2f}",
                        "Lok. Tief": f"{e['local_low']:.2f}",
                        "Einstieg":  e["entry_date"].strftime("%d.%m.%Y")  if hasattr(e["entry_date"],"strftime")  else str(e["entry_date"]),
                        "E-Preis":   f"{e['entry_price']:.2f}",
                        "Vorlauf":   f"{e['days_to_entry']}d",
                        "+%":        f"+{e['upside_pct']:.1f}%",
                    } for e in entries])
                    st.dataframe(entry_df, hide_index=True, use_container_width=True)

                # Ziele
                if targets:
                    st.divider()
                    st.subheader("🎯 Zielkurse")
                    price_now = data["Preis"]
                    n = 4
                    all_k = list(targets.keys())
                    for chunk in [all_k[i:i+n] for i in range(0,len(all_k),n)]:
                        cols_z = st.columns(n)
                        for j,k in enumerate(chunk):
                            v   = targets[k]
                            pct = round((v-price_now)/price_now*100,1) if price_now else 0
                            cols_z[j].metric(k.replace("_"," "), f"{v:.2f} {cur}", f"{pct:+.1f}%")


    # ── NEU: Trend-Vergleichsauswertung ─────────────────────────────
    # Beantwortet: "Hätte der Trendfilter (Kurs vs. SMA200) historisch
    # tatsächlich vor schlechteren Trades gewarnt?" - über alle historischen
    # Einstiegssignale (letzte 2 Jahre) einer Ticker-Gruppe hinweg.
    st.divider()
    st.markdown("### 📊 Trend-Vergleich: Trefferquote mit vs. ohne Aufwärtstrend")
    st.caption(
        "Sammelt alle historischen Einstiegssignale (Score ≥ 2, letzte 2 Jahre) einer Ticker-Gruppe "
        "und vergleicht die Trefferquote danach, ob der Kurs zum jeweiligen Einstiegszeitpunkt über "
        "(Aufwärtstrend) oder unter (Abwärtstrend) der SMA200 lag."
    )
    grp_tv = st.selectbox("Gruppe:", ["🌍 Alle Gruppen"] + get_all_groups(), key="trend_vgl_grp")
    st.caption("Kursdaten werden in Paketen geladen (Batch) — bei 'Alle Gruppen' dauert das dennoch "
               "etwas, da anschließend jeder Ticker einzeln ausgewertet wird.")

    if st.button("📊 Trend-Vergleich starten", key="trend_vgl_btn"):
        tickers_tv = get_all_tickers() if grp_tv == "🌍 Alle Gruppen" else get_tickers_by_group(grp_tv)
        prog_tv, txt_tv = st.progress(0), st.empty()

        def _tv_progress(i, n, ticker):
            txt_tv.text(f"Werte aus: {i+1}/{n} ({ticker})")
            prog_tv.progress((i + 1) / n)

        with st.spinner("Historische Einstiegssignale werden gesammelt und ausgewertet …"):
            tv_result = analyze_trend_hit_rate(tickers_tv, progress_cb=_tv_progress)
        txt_tv.empty(); prog_tv.empty()

        up, down, unknown = tv_result["up"], tv_result["down"], tv_result["unknown"]
        st.success(f"✅ Ausgewertet über {tv_result['n_tickers_used']} Ticker mit historischen Signalen.")

        c_up, c_down = st.columns(2)
        with c_up:
            st.markdown("#### 📈 Einstieg im Aufwärtstrend")
            if up["total_trades"] > 0:
                st.metric("Trefferquote", f"{up['hit_rate']}%", help=f"{up['hits']} von {up['total_trades']} Trades")
                st.metric("Ø max. Gewinn", f"+{up['avg_max_gain']}%" if up['avg_max_gain'] is not None else "N/A")
                st.metric("Ø max. Verlust", f"{up['avg_max_loss']}%" if up['avg_max_loss'] is not None else "N/A")
                st.caption(f"{up['total_trades']} Trades gesamt")
            else:
                st.info("Keine Trades in dieser Kohorte gefunden.")
        with c_down:
            st.markdown("#### 📉 Einstieg im Abwärtstrend")
            if down["total_trades"] > 0:
                st.metric("Trefferquote", f"{down['hit_rate']}%", help=f"{down['hits']} von {down['total_trades']} Trades")
                st.metric("Ø max. Gewinn", f"+{down['avg_max_gain']}%" if down['avg_max_gain'] is not None else "N/A")
                st.metric("Ø max. Verlust", f"{down['avg_max_loss']}%" if down['avg_max_loss'] is not None else "N/A")
                st.caption(f"{down['total_trades']} Trades gesamt")
            else:
                st.info("Keine Trades in dieser Kohorte gefunden.")

        if up["total_trades"] > 0 and down["total_trades"] > 0:
            diff = up["hit_rate"] - down["hit_rate"]
            if diff > 5:
                st.success(f"📈 Bestätigt: Einstiege im Aufwärtstrend hatten historisch eine um "
                           f"**{diff:.1f} Prozentpunkte höhere** Trefferquote. Der Trendfilter macht hier einen echten Unterschied.")
            elif diff < -5:
                st.warning(f"🤔 Überraschend: Einstiege im Abwärtstrend hatten historisch sogar eine um "
                           f"**{-diff:.1f} Prozentpunkte höhere** Trefferquote (\"Rebound-Effekt\"). "
                           f"Der Trendfilter sollte hier mit Vorsicht interpretiert werden.")
            else:
                st.info(f"➖ Kaum Unterschied ({diff:+.1f} Prozentpunkte) zwischen den beiden Kohorten in dieser Auswertung.")

        if unknown["total_trades"] > 0:
            st.caption(f"ℹ️ {unknown['total_trades']} weitere Trades ohne verwertbaren SMA200-Wert "
                       f"(z.B. zu junge Aktien) wurden als 'unbekannt' separat gezählt, nicht in den Vergleich einbezogen.")


with tab4:
    st.header("🛢️ Rohstoffe, Energie & Futures")
    st.caption("Futures via Yahoo Finance. Preise in USD sofern nicht anders angegeben.")

    all_fg   = ["🌍 Alle Gruppen"] + get_all_futures_groups()
    sel_fg   = st.selectbox("Gruppe:", all_fg, key="fut_grp")
    fut_mode = st.radio("Ansicht:", ["📊 Übersicht / Scanner", "📈 Detail-Chart"], horizontal=True)

    if fut_mode == "📊 Übersicht / Scanner":
        if st.button("🔥 Futures Scan starten"):
            tickers_f = list(FUTURES_TICKERS.keys()) if sel_fg=="🌍 Alle Gruppen" else get_futures_by_group(sel_fg)
            fut_data, prog_f, stat_f = [], st.progress(0), st.empty()
            for i,t in enumerate(tickers_f):
                stat_f.text(f"Lade: {t}")
                res = get_futures_analysis(t)
                if res:
                    fut_data.append({
                        "Ticker":   t, "Name": res["Name"], "Gruppe": res["Gruppe"],
                        "Signal":   res["Signal"],
                        "Preis":    f"{res['Preis']} {res['Einheit']}",
                        "52W H":    res["52W_Hoch"], "52W T": res["52W_Tief"],
                        "1W %":     f"{res['Perf_1W']:+.1f}%" if res['Perf_1W'] else "–",
                        "1M %":     f"{res['Perf_1M']:+.1f}%" if res['Perf_1M'] else "–",
                        "RSI":      res["RSI"], "StochRSI": res["StochRSI"],
                        "CCI":      res["CCI"], "Score":    res["Score"],
                    })
                prog_f.progress((i+1)/len(tickers_f))
            stat_f.empty()
            if fut_data:
                df_f = pd.DataFrame(fut_data).sort_values("Score", ascending=False)

                def _frow(r):
                    sv,cv = r.get('StochRSI'),r.get('CCI')
                    sig   = classify_signal(sv, None, None, cv)
                    sc    = int(r.get('Score',0))
                    sc_c  = {4:"#1a9e3f",3:"#27ae60",2:"#f39c12",1:"#7f8c8d"}.get(sc,"#999")
                    pw = r.get('1W %','–'); pm = r.get('1M %','–')
                    pwc = "#1a9e3f" if pw and pw.startswith('+') else "#c0392b" if pw and pw.startswith('-') else "#888"
                    pmc = "#1a9e3f" if pm and pm.startswith('+') else "#c0392b" if pm and pm.startswith('-') else "#888"
                    return (f"<tr>"
                            f"<td><b>{r['Ticker']}</b></td><td>{r['Name']}</td>"
                            f"<td><span style='background:#2d3250;color:#a0aec0;padding:2px 8px;"
                            f"border-radius:6px;font-size:11px'>{r['Gruppe']}</span></td>"
                            f"<td>{_sig_badge(sig)}</td>"
                            f"<td style='text-align:right;font-weight:700'>{r['Preis']}</td>"
                            f"<td style='color:{pwc};text-align:center;font-weight:700'>{pw}</td>"
                            f"<td style='color:{pmc};text-align:center;font-weight:700'>{pm}</td>"
                            f"<td style='text-align:center'>{_ind_badge(r.get('RSI'),35,65,'{:.1f}')}</td>"
                            f"<td style='text-align:center'>{_ind_badge(sv,0.2,0.8)}</td>"
                            f"<td style='text-align:center'>{_ind_badge(cv,-80,80,'{:.1f}',invert=True)}</td>"
                            f"<td style='text-align:center'><span style='background:{sc_c};color:#fff;"
                            f"padding:2px 8px;border-radius:10px;font-weight:700'>{sc}/4</span></td>"
                            f"</tr>")

                thead_f = """<thead style='background:#1e2235;color:#8892a4;font-size:11px;text-transform:uppercase'>
                  <tr><th style='padding:10px 8px'>Ticker</th><th>Name</th><th>Gruppe</th><th>Signal</th>
                  <th>Preis</th><th>1W</th><th>1M</th><th>RSI</th><th>StochRSI</th><th>CCI</th><th>Score</th></tr>
                </thead>"""
                tbody_f = "".join(_frow(r) for _,r in df_f.iterrows())
                st.markdown(
                    f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
                    f"font-size:13px;font-family:sans-serif'>{thead_f}<tbody>{tbody_f}</tbody></table></div>",
                    unsafe_allow_html=True)

    else:
        fut_options = {f"{v['name']} ({k})": k for k,v in FUTURES_TICKERS.items()}
        sel_fut_label = st.selectbox("Futures-Kontrakt:", list(fut_options.keys()), key="fut_det")
        sel_fut = fut_options[sel_fut_label]

        with st.spinner(f"Lade {sel_fut} …"):
            fdata = get_futures_analysis(sel_fut)

        if fdata and "df" in fdata:
            sig_f = fdata["Sig_Data"]
            unit  = fdata["Einheit"]
            fc1,fc2,fc3,fc4,fc5 = st.columns(5)
            fc1.markdown(_sig_badge(sig_f), unsafe_allow_html=True)
            fc2.metric("Preis", f"{fdata['Preis']} {unit}")
            fc3.metric("52W Hoch", f"{fdata['52W_Hoch']} {unit}")
            fc4.metric("52W Tief", f"{fdata['52W_Tief']} {unit}")
            fc5.metric("Score", f"{fdata['Score']}/4")

            fi1,fi2,fi3,fi4 = st.columns(4)
            fi1.markdown(f"**RSI** {_ind_badge(fdata['RSI'],35,65,'{:.1f}')}", unsafe_allow_html=True)
            fi2.markdown(f"**StochRSI** {_ind_badge(fdata['StochRSI'],0.2,0.8)}", unsafe_allow_html=True)
            fi3.markdown(f"**CCI** {_ind_badge(fdata['CCI'],-80,80,'{:.1f}',invert=True)}", unsafe_allow_html=True)
            fi4.markdown(f"**RVOL** <span style='font-size:14px'>{fdata['RVOL']}x</span>", unsafe_allow_html=True)

            dfp = fdata["df"]
            fig_fut = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                row_heights=[0.45,0.18,0.18,0.18],
                subplot_titles=("Kurs & Bollinger","Volumen","RSI","CCI"))
            fig_fut.add_trace(go.Scatter(x=dfp.index,y=dfp["BB_Lower"],name="BB-",line=dict(color="#00eeee",width=.8,dash="dot"),showlegend=False),row=1,col=1)
            fig_fut.add_trace(go.Scatter(x=dfp.index,y=dfp["BB_Upper"],name="Bollinger",fill="tonexty",fillcolor="rgba(0,238,238,0.05)",line=dict(color="#00eeee",width=.8,dash="dot")),row=1,col=1)
            fig_fut.add_trace(go.Scatter(x=dfp.index,y=dfp["EMA50"],name="EMA50",line=dict(color="#888",width=1.5,dash="dash")),row=1,col=1)
            fig_fut.add_trace(go.Scatter(x=dfp.index,y=dfp["Close"],name="Preis",line=dict(color="#facc15",width=2.5)),row=1,col=1)
            vc = ["#ff4444" if r["Open"]>r["Close"] else "#00cc66" for _,r in dfp.iterrows()]
            fig_fut.add_trace(go.Bar(x=dfp.index,y=dfp["Volume"],name="Vol",marker_color=vc,opacity=.7),row=2,col=1)
            fig_fut.add_trace(go.Scatter(x=dfp.index,y=dfp["RSI"],name="RSI",line=dict(color="#00ffff",width=1.5)),row=3,col=1)
            for y,c in [(30,"#4ade80"),(70,"#f87171")]: fig_fut.add_hline(y=y,line_color=c,line_width=1,row=3,col=1)
            fig_fut.add_trace(go.Scatter(x=dfp.index,y=dfp["CCI"],name="CCI",line=dict(color="#be94ff",width=2)),row=4,col=1)
            for y,c in [(-100,"#4ade80"),(100,"#f87171")]: fig_fut.add_hline(y=y,line_color=c,row=4,col=1)
            fig_fut.update_layout(height=900,template="plotly_dark",paper_bgcolor="black",plot_bgcolor="black",
                margin=dict(l=10,r=10,t=40,b=10),hovermode="x unified")
            fig_fut.update_xaxes(showgrid=True,gridcolor="#222")
            fig_fut.update_yaxes(showgrid=True,gridcolor="#222")
            st.plotly_chart(fig_fut, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AUTOMATION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("📅 Bericht-Versand & Automation")
    st.info("Der tägliche Report läuft automatisch via **GitHub Actions** (06:24 UTC). Hier manueller Trigger.")
    if st.button("📧 Report jetzt senden"):
        if not st.session_state.results.empty:
            try:
                pw = st.secrets["DAILY_EMAIL_PASS"]
                st.success(send_mail_report(st.session_state.results, pw))
            except Exception as e:
                st.error(f"Fehler: {e}")
        else:
            st.warning("⚠️ Erst Tab 1 Scanner ausführen.")
